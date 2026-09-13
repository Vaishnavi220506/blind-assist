"""Fixed complete paired source for a frozen-model test, without model access.

Only source geometry and prior source specifications are inspected. Native
capture/evaluation remain separate; analytic labels are not sensor observations.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from mz117_mixture_source import object_definition, intersects

SEED = 123013
TEXTURE_SEED = 123917
DT = .25
STEPS = 12
FAMILIES = dict(suspended_head=6, near_rod_farwall=6, substantial_body=6,
                shallow_boundary_stress=6)
ROOT = Path(__file__).resolve().parents[4]
HISTORY = ('mz115-zonal-allocation-20260913', 'mz117-surface-mixtures-20260913',
           'mz119-tof-scaled-parallax-20260913')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def trajectory_signature(frames):
    """Exclude IDs, noise/texture seeds and annotations; retain actual geometry."""
    return digest([dict(time_s=f['time_s'], camera=f['camera'],
                        body_origin_m=f['body_origin_m'],
                        objects=[dict(center_m=o['center_m'], size_m=o['size_m'])
                                 for o in f['objects']]) for f in frames])


def camera_motion(t, pair_index):
    sign = 1. if pair_index % 2 == 0 else -1.
    vx = .013 + .0004*pair_index
    amplitude = .019 + .0005*pair_index
    frequency = .93 + .013*pair_index
    camera = dict(x=.037+vx*t,
                  y=sign*(.014+amplitude*math.sin(frequency*t)), z=1.7,
                  yaw=sign*(.51+.047*pair_index)*math.sin(.67*t),
                  pitch=-3., roll=0.)
    return camera, [vx, sign*amplitude*frequency*math.cos(frequency*t), 0.]


def source(texture_seed=TEXTURE_SEED):
    geometry_rng = random.Random(SEED)
    noise_rng = random.Random(SEED+41)
    texture_rng = random.Random(texture_seed)
    frames = []; audits = []; pairs = []; counts = Counter(); transitions = Counter()
    family_counts = Counter()
    categories = ('suspended_head', 'near_rod_farwall', 'substantial_body',
                  'shallow_boundary_stress')
    for category_index, family in enumerate(categories):
        for k in range(3):
            pair_index = category_index*3+k
            pair_id = f'mz123_{family}_pair{k}'
            sign = 1. if k % 2 == 0 else -1.
            jitter = geometry_rng.uniform(-.018, .018)
            front = (2.37, 2.91, 3.31)[k]+jitter
            if family == 'suspended_head':
                size = [( .137, .353, .163), (.193, .427, .207), (.159, .287, .183)][k]
                z = (1.781, 1.919, 1.863)[k]
                role = 'suspended_head'; reflectance = (.13, .57, .84)[k]
            elif family == 'near_rod_farwall':
                size = [(.047, .023, 2.213), (.061, .039, 2.287), (.053, .031, 2.173)][k]
                z = size[2]/2; role = 'near_rod'; reflectance = (.075, .31, .69)[k]
            elif family == 'substantial_body':
                size = [(.263, .913, 2.413), (.227, 1.073, 1.813), (.291, .797, 2.573)][k]
                z = size[2]/2; role = 'substantial_body'; reflectance = (.22, .76, .43)[k]
            else:
                size = [(.143, .083, 1.537), (.177, .117, 1.683), (.151, .097, 1.459)][k]
                z = (.997, 1.127, 1.193)[k]; role = 'shallow_boundary_actor'
                reflectance = (.34, .62, .47)[k]
            # Appearance and object shape are identical within a pair. Only the
            # designated actor's lateral placement/trajectory differs.
            texture = texture_rng.randrange(2**30)
            wall_texture = texture_rng.randrange(2**30)
            pair_episodes = []
            variants = ('enter', 'exit') if category_index == 3 else ('in', 'out')
            for variant in variants:
                episode = pair_id+'_'+variant; pair_episodes.append(episode)
                noise_seed = noise_rng.randrange(2**30)
                stress = family == 'shallow_boundary_stress'
                # Ordinary out cases have a 0.17--0.22m true lateral clearance;
                # the source checker enforces the margin at every sampled frame.
                clearance = (.173, .197, .221)[k]
                side0 = sign*(.071+.013*k) if variant == 'in' else sign*(.3+size[1]/2+clearance)
                stress_span = (.016, .021, .024)[k]
                if stress:
                    side0 = sign*(.3+size[1]/2+(stress_span if variant == 'enter' else -stress_span))
                cam0, _ = camera_motion(0., pair_index)
                definition = object_definition((front+size[0]/2+cam0['x'],
                    side0+cam0['y'], z), size, reflectance, role=role)
                definition.update(name='shape0', texture_seed=texture)
                definitions = [definition]
                if family == 'near_rod_farwall':
                    # A genuinely separate far wall, outside the 3.6m query for
                    # all frames. No promise of returned multi-actor mixtures.
                    wall = object_definition((3.887+.011*k+cam0['x'], -.041+.019*k, 1.837),
                        (.093, 5.317+.083*k, 3.674), (.83, .66, .47)[k], role='farwall_reference')
                    wall.update(name='shape1', texture_seed=wall_texture); definitions.append(wall)
                labels = []; positions = []; velocities = []
                for step in range(STEPS):
                    t = DT*step; cam, velocity = camera_motion(t, pair_index)
                    objects = []
                    for definition in definitions:
                        center = list(definition['start_m'])
                        if definition['name'] == 'shape0':
                            # Small common world-space lateral motion compensates
                            # camera sway; the paired query-relative margin is known.
                            side = side0
                            if stress:
                                ramp = stress_span*(1-2*step/(STEPS-1))
                                side = sign*(.3+size[1]/2+(ramp if variant == 'enter' else -ramp))
                            center[1] = cam['y']+side
                        objects.append(dict(name=definition['name'], center_m=center,
                            size_m=list(definition['size_m']), texture_seed=definition['texture_seed'],
                            texture_grid=[5, 8], tof_reflectance_proxy=definition['tof_reflectance_proxy'],
                            source_role=definition['source_role']))
                    frame = dict(id=f'{episode}_{step:02d}', episode=episode, family=family,
                        pair_id=pair_id, pair_variant=variant, pair_member=variant, category=family,
                        group='SHALLOW_BOUNDARY_STRESS' if stress else 'ORDINARY_MARGIN',
                        shallow_boundary_stress=stress, time_s=t, camera=cam,
                        body_origin_m=[cam['x'], cam['y'], 0.], objects=objects,
                        sensor_seed=noise_seed, wearer_speed=math.hypot(*velocity[:2]), radar_ghost=None)
                    positive = any(intersects(frame, o) for o in objects)
                    labels.append(positive); frames.append(frame)
                    positions.append([cam[a] for a in ('x', 'y', 'z')]); velocities.append(velocity)
                    counts['positive' if positive else 'negative'] += 1
                    family_counts[(family, positive)] += 1
                enters = sum(not a and b for a,b in zip(labels,labels[1:]))
                exits = sum(a and not b for a,b in zip(labels,labels[1:])); transitions.update(enter=enters,exit=exits)
                audits.append(dict(episode=episode, pair_id=pair_id, pair_variant=variant, pair_member=variant, family=family,
                    category=family, group='SHALLOW_BOUNDARY_STRESS' if stress else 'ORDINARY_MARGIN',
                    shallow_boundary_stress=stress, objects=definitions,
                    object_motion_authority='EXPLICIT_PER_FRAME_CENTERS_OVERRIDE_STATIC_DEFINITION_VELOCITY',
                    camera_positions_m=positions, instantaneous_camera_velocity_mps=velocities,
                    source_aabb_labels=labels, enters=enters, exits=exits,
                    ordinary_out_clearance_m=None if stress else clearance,
                    stress_edge_half_sweep_m=stress_span if stress else None))
            pairs.append(dict(pair_id=pair_id, category=family, episodes=pair_episodes,
                intended_difference='OPPOSITE_SHALLOW_CROSSING' if category_index==3 else 'IN_VS_CLEARLY_OFFROUTE_LATERAL_PLACEMENT',
                shared_shape_texture_camera=True, independent_episode_sensor_noise=True))
    spec = dict(schema='mz123-frozen-model-fresh-paired-source-v1', seed=SEED, texture_seed=texture_seed,
        authority='CONTROLLED_UE_RGB_NATIVE_COLLISION_FINITE_FOOTPRINT_TOF_PROXY_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,tof_rows=8,tof_columns=8,rgb_camera_count=1),
        background=dict(center_m=[13.17,0.,1.83],size_m=[.157,19.31,8.73],texture=False,tof_reflectance_proxy=.50),
        floor=dict(center_m=[4.27,0.,-.05],size_m=[25.17,21.31,.1],tof_reflectance_proxy=.30),
        fixed_before_capture=True, dt_s=DT, frames_per_episode=STEPS, family_episode_counts=FAMILIES,
        source_design_aabb_frame_counts=dict(counts), source_design_transition_counts=dict(transitions),
        family_source_aabb_counts={f:dict(positive=family_counts[(f,True)],negative=family_counts[(f,False)]) for f in FAMILIES},
        source_audit_authority='SOURCE_DESIGN_ONLY_NOT_NATIVE_ENGINE_MEASUREMENTS_OR_OBSERVABLE_LABELS',
        source_audit=audits, pairs=pairs,
        initial_body_origin_m=frames[0]['body_origin_m'],
        corridor_m=dict(forward=[.2,3.6],lateral=[-.3,.3],height=[.4,2.05]),
        translation_authority='SOURCE_EVALUATOR_ONLY_NO_METRIC_TRANSLATION_IN_RAW',
        limitations=[
            'Source AABB labels are design checks, not native truth or model measurements.',
            'All 24 episodes and 288 frames are retained, including shallow stress and missing/weak sensor observations.',
            'Pair identifiers, categories, source roles, native positions and motion parameters never enter inference.',
            'Current corridor remains world-X aligned relative to camera XY; yaw is not walking intention.',
            'Within-pair shapes, textures and camera motion are shared intentionally; episode noise seeds differ.',
            'Target lateral motion partly follows camera sway to keep margins fixed; this is a constructed controlled scenario.',
            'Far walls, background and floor have new geometry; context is excluded from obstacle truth only after nonintersection checks.',
            'Rod/wall configurations do not guarantee returned multi-actor mixtures; unsampled or absent near returns are retained.',
            'No new sensor interface, calibrated hardware, natural-domain, safety, or scene-universal claims.'],frames=frames)
    check_source(spec)
    return spec


def check_source(spec, historical_specs=()):
    frames=spec['frames'];audits=spec['source_audit'];by_episode={a['episode']:[] for a in audits}
    assert len(frames)==len({f['id'] for f in frames})==288 and len(audits)==24
    assert len(spec['pairs'])==12 and Counter(a['family'] for a in audits)==FAMILIES
    assert len({f['sensor_seed'] for f in frames})==24
    assert spec['source_design_aabb_frame_counts']==dict(positive=144,negative=144)
    assert spec['source_design_transition_counts']==dict(enter=3,exit=3)
    for f in frames:
        by_episode[f['episode']].append(f);cam=f['camera']
        assert cam['z']==1.7 and cam['pitch']==-3. and cam['roll']==0. and abs(cam['yaw'])<1.3
        assert f['body_origin_m']==[cam['x'],cam['y'],0.]
        assert f['group']==('SHALLOW_BOUNDARY_STRESS' if f['shallow_boundary_stress'] else 'ORDINARY_MARGIN')
        for obj in (spec['background'],spec['floor']):assert not intersects(f,obj)
        for obj in f['objects']:
            assert 0<obj['tof_reflectance_proxy']<=1
            if obj['source_role']=='farwall_reference':assert not intersects(f,obj) and obj['center_m'][0]-obj['size_m'][0]/2-cam['x']>3.6
        actor=f['objects'][0];margin=abs(actor['center_m'][1]-cam['y'])-actor['size_m'][1]/2-.3
        if f['pair_variant']=='out':assert margin>=.17-1e-9 and not intersects(f,actor)
        if f['pair_variant']=='in':assert intersects(f,actor) and margin<=-.15
        if f['shallow_boundary_stress']:assert 0<abs(margin)<=.024+1e-9
    for a in audits:
        fs=by_episode[a['episode']]
        assert len(fs)==12 and [f['time_s'] for f in fs]==[DT*i for i in range(STEPS)]
        assert fs[0]['camera']['yaw']==0., 'Existing IMU adapter assumes zero initial yaw'
        actual=[any(intersects(f,o) for o in f['objects']) for f in fs]
        assert actual==a['source_aabb_labels']
        assert all([o['name'] for o in f['objects']]==[o['name'] for o in a['objects']] for f in fs)
    actual_counts=Counter('positive' if any(intersects(f,o) for o in f['objects']) else 'negative' for f in frames)
    assert dict(actual_counts)==spec['source_design_aabb_frame_counts']
    actual_transitions=Counter(enter=0,exit=0)
    for a in audits:
        flags=a['source_aabb_labels']
        actual_transitions['enter']+=sum(not x and z for x,z in zip(flags,flags[1:]))
        actual_transitions['exit']+=sum(x and not z for x,z in zip(flags,flags[1:]))
    assert dict(actual_transitions)==spec['source_design_transition_counts']
    for pair in spec['pairs']:
        first,second=[by_episode[e] for e in pair['episodes']]
        for a,b in zip(first,second):
            assert a['camera']==b['camera'] and a['time_s']==b['time_s']
            assert any(intersects(a,o) for o in a['objects']) != any(intersects(b,o) for o in b['objects'])
            for x,y in zip(a['objects'],b['objects']):
                assert x['size_m']==y['size_m'] and x['texture_seed']==y['texture_seed']
                assert x['center_m'][::2]==y['center_m'][::2]
    signatures={trajectory_signature(fs) for fs in by_episode.values()}
    camera_signatures={digest([f['camera'] for f in fs]) for fs in by_episode.values()}
    assert len(signatures)==24
    history=[]
    for path in historical_specs:
        path=Path(path);past=json.loads(path.read_text(encoding='utf-8'));groups={}
        for f in past['frames']:groups.setdefault(f['episode'],[]).append(f)
        overlap=signatures & {trajectory_signature(fs) for fs in groups.values()}
        assert not overlap, 'Historical geometry/camera trajectory duplicated'
        assert not camera_signatures & {digest([f['camera'] for f in fs]) for fs in groups.values()}, 'Historical camera trajectory duplicated'
        history.append(dict(spec=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            episodes=len(groups),identical_episode_geometry_trajectories=0,
                            identical_camera_trajectories=0))
    return dict(status='PASS',frames=288,episodes=24,pairs=12,
        source_aabb_counts=spec['source_design_aabb_frame_counts'],transitions=spec['source_design_transition_counts'],
        ordinary_frames=216,stress_frames=72,source_positive_segments=15,
        all_frames_retained=True,historical_geometry_checks=history)


def self_test():
    spec=source();assert spec==source()
    different=source(TEXTURE_SEED+1)
    for fs in (spec['frames'],different['frames']):
        for f in fs:
            for o in f['objects']:o.pop('texture_seed')
    assert spec['frames']==different['frames'], 'Texture changes altered geometry/sensor source'
    return dict(check_source(source()),deterministic_source=True,texture_rng_independent=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--capture-source',type=Path,action='append',default=[])
    p.add_argument('--historical-spec',type=Path,action='append',default=[])
    args=p.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    if args.output is None or not args.capture_source:p.error('--output and --capture-source required')
    spec=source();history=args.historical_spec or [ROOT/'artifacts.local/work'/name/'capture-v1/spec.json' for name in HISTORY]
    audit=check_source(spec,history);output=args.output.resolve();freeze=output.with_name('freeze.json');auditpath=output.with_name('source-audit.json')
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()) or len({output,freeze,auditpath})!=3 or any(x.exists() for x in (output,freeze,auditpath)):
        raise ValueError('Fresh canonical spec, freeze and audit paths required')
    helpers=[p.resolve() for p in args.capture_source]
    if not all(p.is_file() for p in helpers):raise ValueError('Capture helpers must exist before freeze')
    payload=(json.dumps(spec,indent=2,allow_nan=False)+'\n').encode()
    receipt=dict(spec_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers},
        selection='FIXED_SEED123013_ALL24_EPISODES_ALL288_FRAMES_NO_MODEL_OUTPUT_ACCESS',**audit)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:f.write(payload)
    with auditpath.open('x',encoding='utf-8') as f:f.write(json.dumps(dict(check=audit,pairs=spec['pairs'],episodes=spec['source_audit']),indent=2)+'\n')
    receipt['source_audit_sha256']=hashlib.sha256(auditpath.read_bytes()).hexdigest()
    with freeze.open('x',encoding='utf-8') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
