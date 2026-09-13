"""Fixed MZ117 source and analytic finite-footprint mixture admission audit.

No rendering, predictions or outcome files are read. The unchanged MZ115 sensor
wrapper runs against source AABB segment intersections solely to precheck the
intended source condition, including its original packet RNG consumption. These
are analytic source-design results; native capture must independently verify them.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import mz115_zonal_sensors as zonal

SEED = 117013
TEXTURE_SEED = 117917
DT = .25
STEPS = 12
WEARER_SPEED_MPS = .025
FAMILY_EPISODES = dict(near_rod_farwall_mixture=6, substantial_offroute=4,
    suspended_head=4, occlusion=2, clear_ghost=2, boundary_1cm_stress=2)


def object_definition(start, size, reflectance, velocity=(0., 0., 0.), role='obstacle'):
    return dict(start_m=list(start), size_m=list(size), velocity_mps=list(velocity),
                tof_reflectance_proxy=reflectance, source_role=role)


def intersects(frame, obj):
    lo = [c-s/2-b for c, s, b in zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    hi = [c+s/2-b for c, s, b in zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    return hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3 and hi[2] >= .4 and lo[2] <= 2.05


def source(texture_seed=TEXTURE_SEED):
    sensor_rng = random.Random(SEED)
    texture_rng = random.Random(texture_seed)
    scenes = []

    def add(name, family, objects, yaw=0., ghost=None):
        scenes.append(dict(episode='mz117_'+name, family=family, objects=objects,
                           yaw_amplitude=yaw, ghost=ghost))

    # Front-plane gaps .42--.54m exercise the existing <.60m reducer without
    # changing its peak/threshold rules. Sparse ray phases intentionally differ.
    rods = ((3.20, .052, .018, .04, 3.73, .90, 0.),
            (3.23, -.053, .026, .10, 3.75, .85, .8),
            (3.27, .058, .046, .25, 3.74, .70, -.6),
            (3.31, -.058, .074, .65, 3.78, .40, 1.),
            (3.34, 0., .130, .90, 3.76, .22, 0.),
            (3.25, .085, .014, .06, 3.77, .95, .8))
    for k, (front, side, width, rho, wall_front, wall_rho, yaw) in enumerate(rods):
        add(f'mixture_{k}', 'near_rod_farwall_mixture', [
            object_definition((front+.03, side, 1.12), (.06, width, 2.24), rho, role='near_rod'),
            object_definition((wall_front+.04, 0., 1.8), (.08, 5.4, 3.6), wall_rho, role='farwall_reference'),
        ], yaw=yaw)

    for k, (front, side, width, height, rho) in enumerate(((2.54, 1.01, 1.08, 2.48, .16),
            (2.98, -1.15, 1.24, 2.62, .81), (2.72, .89, .80, 1.42, .61),
            (3.12, -1.09, .92, 1.66, .08))):
        add(f'offroute_{k}', 'substantial_offroute', [
            object_definition((front+.13, side, height/2), (.26, width, height), rho, role='substantial_offroute'),
        ], yaw=(3., -3., 1., -2.)[k], ghost=dict(x=.075, z=2.86) if k == 3 else None)

    add('head_weak_wall', 'suspended_head', [
        object_definition((3.25, -.035, 1.86), (.12, .32, .16), .045, role='near_weak_head'),
        object_definition((3.79, 0., 1.8), (.08, 5.4, 3.6), .88, role='farwall_reference'),
    ], yaw=-.7)
    add('head_static', 'suspended_head', [
        object_definition((2.87, .095, 1.93), (.18, .43, .20), .73, role='suspended_head'),
    ], yaw=2.4)
    add('head_cross_left', 'suspended_head', [
        object_definition((2.66, .91, 1.81), (.14, .34, .20), .27, (0., -.68, 0.), 'suspended_head'),
    ], yaw=3.2)
    add('head_cross_right', 'suspended_head', [
        object_definition((3.06, -.88, 1.90), (.18, .48, .22), .86, (0., .69, 0.), 'suspended_head'),
    ], yaw=-2.8)

    add('occluded_head', 'occlusion', [
        object_definition((2.12, -.24, 1.04), (.22, .46, 1.74), .80, (0., .17, 0.), 'moving_occluder'),
        object_definition((3.15, .055, 1.87), (.16, .39, .24), .12, role='partially_occluded_head'),
    ], yaw=-2.1, ghost=dict(x=-.065, z=2.81))
    add('occluded_rod', 'occlusion', [
        object_definition((2.48, .72, 1.02), (.20, .40, 1.84), .68, (0., -.49, 0.), 'moving_occluder'),
        object_definition((3.30, -.045, 1.13), (.06, .030, 2.26), .09, role='partially_occluded_rod'),
        object_definition((3.80, 0., 1.8), (.08, 5.4, 3.6), .79, role='farwall_reference'),
    ], yaw=1.4)

    add('clear_reference', 'clear_ghost', [])
    add('clear_persistent_ghost', 'clear_ghost', [], yaw=2.6, ghost=dict(x=.065, z=2.81))
    add('stress_enter_1cm', 'boundary_1cm_stress', [
        object_definition((2.94, .355, 1.29), (.14, .09, 1.46), .46,
                          (0., -.02/(11*DT), 0.), 'lateral_1cm_stress'),
    ])
    add('stress_exit_1cm', 'boundary_1cm_stress', [
        object_definition((3.11, -.335, 1.33), (.16, .09, 1.42), .46,
                          (0., -.02/(11*DT), 0.), 'lateral_1cm_stress'),
    ])

    frames = []; audits = []; counts = Counter(); transitions = Counter(); family_counts = Counter()
    for scene in scenes:
        sensor_seed = sensor_rng.randrange(2**30)
        definitions = [dict(o, name=f'shape{k}', texture_seed=texture_rng.randrange(2**30))
                       for k, o in enumerate(scene['objects'])]
        labels = []
        for step in range(STEPS):
            t = step*DT; x = WEARER_SPEED_MPS*t
            objects = [dict(name=o['name'], center_m=[c+v*t for c, v in zip(o['start_m'], o['velocity_mps'])],
                size_m=list(o['size_m']), texture_seed=o['texture_seed'], texture_grid=[3, 6],
                tof_reflectance_proxy=o['tof_reflectance_proxy'], source_role=o['source_role']) for o in definitions]
            frame = dict(id=f"{scene['episode']}_{step:02d}", episode=scene['episode'], family=scene['family'], time_s=t,
                camera=dict(x=x, y=0., z=1.7, yaw=scene['yaw_amplitude']*math.sin(.83*t), pitch=-3., roll=0.),
                body_origin_m=[x, 0., 0.], objects=objects, sensor_seed=sensor_seed,
                wearer_speed=WEARER_SPEED_MPS, radar_ghost=scene['ghost'])
            positive = any(intersects(frame, o) for o in objects); labels.append(positive)
            counts['positive' if positive else 'negative'] += 1
            family_counts[(scene['family'], positive)] += 1; frames.append(frame)
        enters = sum(not a and b for a, b in zip(labels, labels[1:]))
        exits = sum(a and not b for a, b in zip(labels, labels[1:])); transitions.update(enter=enters, exit=exits)
        audits.append(dict(episode=scene['episode'], family=scene['family'], objects=definitions,
            camera_velocity_mps=[WEARER_SPEED_MPS, 0., 0.], source_aabb_labels=labels, enters=enters, exits=exits))
    spec = dict(schema='mz117-real-actor-mixture-source-v1', seed=SEED, texture_seed=texture_seed,
        authority='CONTROLLED_UE_RGB_NATIVE_COLLISION_FINITE_FOOTPRINT_TOF_PROXY_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640, height=360, hfov_deg=70., tof_hfov_deg=45., tof_rows=8, tof_columns=8, rgb_camera_count=1),
        background=dict(center_m=[12.7, 0., 1.75], size_m=[.14, 18.6, 8.5], texture=False, tof_reflectance_proxy=.50),
        floor=dict(center_m=[4., 0., -.05], size_m=[24., 20., .1], tof_reflectance_proxy=.30),
        fixed_before_capture=True, dt_s=DT, frames_per_episode=STEPS, family_episode_counts=FAMILY_EPISODES,
        designated_mixture_episodes=[a['episode'] for a in audits if a['family'] == 'near_rod_farwall_mixture'],
        designated_mixture_minimum_frames=12,
        source_design_aabb_frame_counts=dict(counts), source_design_transition_counts=dict(transitions),
        family_source_aabb_counts={f:dict(positive=family_counts[(f, True)], negative=family_counts[(f, False)]) for f in FAMILY_EPISODES},
        source_audit_authority='SOURCE_DESIGN_ONLY_NOT_NATIVE_ENGINE_MEASUREMENTS_OR_OBSERVABLE_LABELS', source_audit=audits,
        initial_body_origin_m=[0., 0., 0.], corridor_m=dict(forward=[.2, 3.6], lateral=[-.3, .3], height=[.4, 2.05]),
        reflectance_authority='EXPLICIT_LATENT_SENSOR_PROXY_INDEPENDENT_OF_RGB_ALBEDO_TEXTURE',
        limitations=[
            'All object-list obstacles and far walls participate in current-frame native obstacle truth.',
            'Unchanged MZ115 global background and floor are context, excluded from target truth only after every-frame AABB nonintersection checks; they are not Radar target-list actors.',
            'Analytic ray/AABB source admission is not an engine measurement, rendered observation, prediction or performance result. Native collision/range/packet parity must be audited later.',
            'The sparse 3x3 quadrature and reflectance/range-squared proxy are hypothetical, uncalibrated choices; neither continuous-beam coverage nor physical photon/RF behavior is claimed.',
            'SIM_MERGED alone does not imply multiple objects. Admission requires returned lineage to include distinct real near-rod and far-wall actors, excluding context.',
            'All reducer distances and weighted means are slant ranges; far-wall corridor exclusion uses forward AABB distance separately.',
            'Thin rods may be unsampled, weak components may disappear, packets may be missing, and more oblique wall rays may exceed 4m. All such frames are retained.',
            'RGB texture and latent ToF reflectance are independently specified. Sensor packets are stochastic simulations with fixed source seeds, not guaranteed hardware observations.',
            'Camera moves 6.875cm per episode; yaw is observation rotation, not walking intention. Separate 1cm stress episodes do not define ordinary cohort margins.',
            'All 20 episodes and 240 frames are constructed Development, without natural-domain, hardware, safety or deployment authority.'], frames=frames)
    spec['analytic_mixture_precheck'] = analytic_precheck(spec)
    check_source(spec)
    return spec


class Vector:
    """Minimal scalar native-interface stand-in, in the same centimetre units."""
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    def __add__(self, other): return Vector(self.x+other.x, self.y+other.y, self.z+other.z)


class AnalyticActor:
    def __init__(self, obj):
        self.obj = obj
        self.static_mesh_component = object()

    def get_actor_bounds(self, _):
        return Vector(*(v*100 for v in self.obj['center_m'])), Vector(*(v*50 for v in self.obj['size_m']))


def segment_aabb_fraction(start, end, center, size):
    """First closed segment/AABB intersection; None for no hit."""
    lower, upper = 0., 1.
    for a, b, c, s in zip(start, end, center, size):
        delta = b-a; lo, hi = c-s/2, c+s/2
        if abs(delta) < 1e-15:
            if a < lo or a > hi: return None
            continue
        entry, leave = sorted(((lo-a)/delta, (hi-a)/delta))
        lower, upper = max(lower, entry), min(upper, leave)
        if lower > upper: return None
    return lower


def analytic_trace(world, origin, end, *_):
    start = [origin.x/100, origin.y/100, origin.z/100]
    finish = [end.x/100, end.y/100, end.z/100]
    found = []
    for index, actor in enumerate(world):
        fraction = segment_aabb_fraction(start, finish, actor.obj['center_m'], actor.obj['size_m'])
        if fraction is not None: found.append((fraction, index, actor))
    if not found: return None
    fraction, _, actor = min(found, key=lambda h:(h[0], h[1]))
    fields = [True]+[None]*10
    fields[5] = Vector(*((a+fraction*(b-a))*100 for a, b in zip(start, finish)))
    fields[10] = actor.static_mesh_component
    return SimpleNamespace(to_tuple=lambda:tuple(fields))


ANALYTIC_NATIVE = SimpleNamespace(Vector=Vector, SystemLibrary=SimpleNamespace(line_trace_single=analytic_trace),
    TraceTypeQuery=SimpleNamespace(TRACE_TYPE_QUERY1=1), DrawDebugTrace=SimpleNamespace(NONE=0))


def analytic_precheck(spec):
    """Run frozen sensors on source-only intersections; persist audit, never raw."""
    contexts = [AnalyticActor(spec[k]) for k in ('background', 'floor')]
    state = {'mz115_environment_reflectance':[(a.static_mesh_component, a.obj['tof_reflectance_proxy']) for a in contexts]}
    records = []; episode_counts = {}; total = Counter()
    for frame in spec['frames']:
        actors = [AnalyticActor(o) for o in frame['objects']]
        raw, details, _ = zonal.sensors(ANALYTIC_NATIVE, actors+contexts, frame, actors, state)
        roles = {frame['episode']+'/'+o['name']:o['source_role'] for o in frame['objects']}
        reflectances = {frame['episode']+'/'+o['name']:o['tof_reflectance_proxy'] for o in frame['objects']}
        near_ids = {i for i, role in roles.items() if role == 'near_rod'}
        far_ids = {i for i, role in roles.items() if role == 'farwall_reference'}
        sampled = set(); returned_mixtures = []; possible_mixtures = 0
        for zone in details['zonal_tof_native']:
            hits = zone['private_rays']
            sampled.update(h.get('actor_id') for h in hits if h.get('actor_id') in near_ids)
            # Packet-independent selected component availability (same strongest
            # order/max2 as the unchanged reducer; no extra noise draws).
            selected = sorted([c for c in zone['components'] if c['detected']],
                              key=lambda c:(-c['signal_strength_proxy'], c['pre_noise_range_m']))[:zonal.MAX_TARGETS]
            for comp in selected:
                ids = {hits[i].get('actor_id') for i in comp['hit_indices']}
                if comp['status'] == 'SIM_MERGED' and ids & near_ids and ids & far_ids:
                    possible_mixtures += 1
            for lineage in zone['returned_lineage']:
                target = raw['tof_zones'][zone['zone_id']]['targets'][lineage['target_index']]
                component_hits = [hits[i] for i in lineage['hit_indices']]
                ids = {h.get('actor_id') for h in component_hits}
                if target['status'] != 'SIM_MERGED' or not (ids & near_ids and ids & far_ids): continue
                weights = Counter()
                for hit in component_hits:
                    weights[hit.get('actor_id')] += hit['reflectance_proxy']/(9*max(hit['range_m'], .2)**2)
                near_weight = sum(weights[i] for i in near_ids); far_weight = sum(weights[i] for i in far_ids)
                near_ranges = [h['range_m'] for h in component_hits if h.get('actor_id') in near_ids]
                far_ranges = [h['range_m'] for h in component_hits if h.get('actor_id') in far_ids]
                def group_mean(group):
                    group_hits = [h for h in component_hits if h.get('actor_id') in group]
                    weighted = [(h['range_m'], h['reflectance_proxy']/(9*max(h['range_m'], .2)**2)) for h in group_hits]
                    return sum(r*w for r, w in weighted)/sum(w for _, w in weighted)
                near_mean, far_mean = group_mean(near_ids), group_mean(far_ids)
                mean = lineage['pre_noise_range_m']
                returned_mixtures.append(dict(zone_id=zone['zone_id'], target_index=lineage['target_index'],
                    real_actor_ids=sorted(i for i in ids if i in roles), status=target['status'],
                    near_slant_range_m=[min(near_ranges), max(near_ranges)],
                    far_slant_range_m=[min(far_ranges), max(far_ranges)],
                    pre_noise_slant_mean_m=lineage['pre_noise_range_m'], observed_slant_distance_m=target['distance_m'],
                    near_weighted_slant_mean_m=near_mean, far_weighted_slant_mean_m=far_mean,
                    far_signal_fraction=far_weight/(far_weight+near_weight), far_signal_dominant=far_weight>near_weight,
                    mean_closer_to_far=abs(mean-far_mean)<abs(mean-near_mean),
                    weak_near_reflectance=max(reflectances[i] for i in ids & near_ids)<=.10))
        counts = episode_counts.setdefault(frame['episode'], Counter())
        counts['frames'] += 1
        counts['packet_received_frames'] += bool(raw['tof_packet_received'])
        counts['near_rod_sampled_frames'] += bool(sampled)
        counts['near_rod_unsampled_frames'] += bool(near_ids) and not sampled
        counts['packet_independent_real_mixture_frames'] += possible_mixtures > 0
        counts['returned_real_mixture_frames'] += bool(returned_mixtures)
        counts['returned_real_mixture_targets'] += len(returned_mixtures)
        counts['far_dominant_real_mixture_targets'] += sum(m['far_signal_dominant'] for m in returned_mixtures)
        counts['far_biased_mean_real_mixture_targets'] += sum(m['mean_closer_to_far'] for m in returned_mixtures)
        counts['weak_near_far_biased_mean_targets'] += sum(m['weak_near_reflectance'] and m['mean_closer_to_far'] for m in returned_mixtures)
        records.append(dict(id=frame['id'], episode=frame['episode'], packet_received=raw['tof_packet_received'],
            near_rod_sampled=bool(sampled), packet_independent_real_mixture_targets=possible_mixtures,
            returned_real_mixtures=returned_mixtures))
    for counts in episode_counts.values(): total.update(counts)
    modules = [Path(zonal.__file__), Path(zonal.original.__file__), Path(zonal.__file__).with_name('mz115_zonal_tof.py'),
               Path(zonal.__file__).with_name('mz99_angle_information_capture.py')]
    return dict(authority='ANALYTIC_SOURCE_AABB_ONLY_NOT_NATIVE_CAPTURE_OR_MODEL_OUTCOMES',
        method='Unchanged MZ115 wrapper, MZ113 packet/Radar RNG consumption and sensor basis, source AABB first-hit traces; environment kept separate from Radar actor list.',
        gate='Each designated near_rod_farwall_mixture episode has a returned SIM_MERGED target with near rod and far wall real actor lineage; broad single-wall and environmental mixtures do not qualify.',
        source_only_audit_fields_not_observations=True,
        unchanged_sensor_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in modules},
        counts=dict(total), episode_counts={e:dict(c) for e, c in episode_counts.items()}, frames=records)


def check_source(spec):
    frames = spec['frames']; assert len(frames) == 240 and len({f['id'] for f in frames}) == 240
    episodes = {f['episode'] for f in frames}; assert len(episodes) == 20
    assert Counter(f['episode'] for f in frames) == {e:12 for e in episodes}
    assert Counter(a['family'] for a in spec['source_audit']) == FAMILY_EPISODES
    assert len({f['sensor_seed'] for f in frames}) == 20
    assert spec['source_design_aabb_frame_counts'] == dict(positive=145, negative=95)
    assert spec['source_design_transition_counts'] == dict(enter=3, exit=3)
    analytic = spec['analytic_mixture_precheck']; assert len(analytic['frames']) == 240
    mixture_episodes = [a['episode'] for a in spec['source_audit'] if a['family'] == 'near_rod_farwall_mixture']
    assert len(mixture_episodes) == 6
    assert spec['designated_mixture_episodes'] == mixture_episodes
    for ep in mixture_episodes:
        assert analytic['episode_counts'][ep]['returned_real_mixture_frames'] > 0, ep
    assert analytic['counts']['near_rod_unsampled_frames'] > 0, 'Retain sparse thin-rod misses'
    assert analytic['counts']['returned_real_mixture_frames'] >= spec['designated_mixture_minimum_frames']
    assert analytic['counts']['weak_near_far_biased_mean_targets'] > 0
    for frame in frames:
        assert frame['camera']['pitch'] == -3. and frame['camera']['roll'] == 0.
        assert frame['body_origin_m'] == [frame['camera']['x'], 0., 0.]
        assert abs(frame['camera']['x']-frame['time_s']*WEARER_SPEED_MPS) < 1e-12
        for context in (spec['background'], spec['floor']): assert not intersects(frame, context)
        for obj in frame['objects']:
            assert 0 < obj['tof_reflectance_proxy'] <= 1
            if obj['source_role'] == 'farwall_reference':
                front = obj['center_m'][0]-obj['size_m'][0]/2-frame['body_origin_m'][0]
                assert 3.6 < front <= 3.8 and not intersects(frame, obj)
            if obj['source_role'] == 'substantial_offroute':
                assert abs(obj['center_m'][1])-obj['size_m'][1]/2 > .4 and not intersects(frame, obj)
        if frame['family'] == 'near_rod_farwall_mixture':
            rod, wall = frame['objects']
            gap = (wall['center_m'][0]-wall['size_m'][0]/2)-(rod['center_m'][0]-rod['size_m'][0]/2)
            assert 0 < gap < .6 and intersects(frame, rod)
    for audit in spec['source_audit']:
        rows = [f for f in frames if f['episode'] == audit['episode']]
        assert [r['time_s'] for r in rows] == [i*DT for i in range(STEPS)]
        assert all([o['name'] for o in r['objects']] == [o['name'] for o in audit['objects']] for r in rows)
        assert audit['source_aabb_labels'] == [any(intersects(r, o) for o in r['objects']) for r in rows]
        if audit['family'] in ('substantial_offroute', 'clear_ghost'): assert not any(audit['source_aabb_labels'])
        if audit['family'] == 'boundary_1cm_stress': assert audit['enters']+audit['exits'] == 1
    return dict(status='PASS', frames=240, episodes=20, source_aabb_counts=spec['source_design_aabb_frame_counts'],
                transitions=spec['source_design_transition_counts'], analytic_mixture_counts=analytic['counts'])


def self_test():
    spec = source()
    assert json.dumps(spec, sort_keys=True) == json.dumps(source(), sort_keys=True)
    different = source(TEXTURE_SEED+1)
    assert spec['analytic_mixture_precheck'] == different['analytic_mixture_precheck']
    assert any(a['texture_seed'] != b['texture_seed'] for f, g in zip(spec['frames'], different['frames']) for a, b in zip(f['objects'], g['objects']))
    return dict(check_source(spec), deterministic_source=True, texture_rng_independence=True,
        family_source_aabb_counts=spec['family_source_aabb_counts'], analytic_episode_counts=spec['analytic_mixture_precheck']['episode_counts'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true'); parser.add_argument('--output', type=Path)
    parser.add_argument('--capture-source', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.self_test: print(json.dumps(self_test(), indent=2)); return
    if args.output is None or not args.capture_source: parser.error('--output and at least one --capture-source required')
    root = Path(__file__).resolve().parents[4]; output = args.output.resolve(); freeze = output.with_name('freeze.json')
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output == freeze or output.exists() or freeze.exists():
        raise ValueError('Fresh canonical source/freeze files required')
    helpers = [p.resolve() for p in args.capture_source]
    if not all(p.is_file() for p in helpers): raise ValueError('Capture helpers must be ready before freeze')
    spec = source(); payload = (json.dumps(spec, indent=2, allow_nan=False)+'\n').encode('utf-8')
    receipt = dict(spec_sha256=hashlib.sha256(payload).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}, frames=240, episodes=20,
        source_design_aabb_frame_counts=spec['source_design_aabb_frame_counts'], source_design_transition_counts=spec['source_design_transition_counts'],
        analytic_mixture_counts=spec['analytic_mixture_precheck']['counts'],
        selection='FIXED_SEED117013_ALL20_EPISODES_ALL240_FRAMES_NO_MODEL_OUTCOME_SELECTION')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle: handle.write(payload)
    with freeze.open('x', encoding='utf-8') as handle: handle.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
