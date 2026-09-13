"""Fresh fixed source for causal monocular-parallax Development investigation.

Camera translation is source/evaluator-only. Existing capture exposes no body
velocity or native trajectory to inference. MZ117 analytic intersection helpers
and the MZ115 sensor stack are reused unchanged; no model outcomes are read.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import mz117_mixture_source as analytic

SEED = 119013
TEXTURE_SEED = 119917
DT = .25
STEPS = 12
FAMILIES = dict(near_rod_farwall_mixture=6, substantial_offroute=4, suspended_head=4,
                moving_occlusion=2, clear_ghost=2, translation_controls=2)
CAMERA_COHORTS = dict(lateral=13, rotation_only=5, stationary=1, low_translation_rotation=1)
obj = analytic.object_definition
intersects = analytic.intersects


def camera_motion(t, mode, sign, yaw_amplitude):
    amplitude = .12*sign if mode == 'lateral' else .003*sign if mode == 'low_translation_rotation' else 0.
    vx = .01 if mode == 'lateral' else .001 if mode == 'low_translation_rotation' else 0.
    camera = dict(x=vx*t, y=amplitude*math.sin(1.1*t), z=1.7,
                  yaw=yaw_amplitude*math.sin(.79*t), pitch=-3., roll=0.)
    velocity = [vx, 1.1*amplitude*math.cos(1.1*t), 0.]
    return camera, velocity


def source(texture_seed=TEXTURE_SEED):
    scenes = []; sensor_rng = random.Random(SEED); texture_rng = random.Random(texture_seed)
    def add(name, family, objects, mode='lateral', sign=1., yaw=0., ghost=None):
        scenes.append(dict(episode='mz119_'+name, family=family, objects=objects,
                           camera_motion_mode=mode, lateral_sign=sign, yaw_amplitude_deg=yaw, ghost=ghost))

    rods = ((3.215, .066, .021, .05, 3.735, .87), (3.245, -.078, .035, .14, 3.765, .76),
            (3.285, .105, .057, .39, 3.745, .63), (3.325, -.115, .094, .78, 3.795, .29),
            (3.265, .047, .025, .08, 3.755, .93), (3.305, -.061, .017, .045, 3.775, .83))
    for k, (front, side, width, rho, wall_front, wall_rho) in enumerate(rods):
        depth = .052+.002*k; height = (2.18, 2.30, 2.26, 2.32, 2.20, 2.28)[k]
        add(f'rod_wall_{k}', 'near_rod_farwall_mixture', [
            obj((front+depth/2, side, height/2), (depth, width, height), rho, role='near_rod'),
            obj((wall_front+.035, .018*(k-2), 1.84), (.07, 5.64, 3.68), wall_rho, role='farwall_reference'),
        ], mode='lateral' if k < 4 else 'rotation_only' if k == 4 else 'stationary',
            sign=1. if k % 2 == 0 else -1., yaw=(0., .9, -.6, 1.3, 3.4, 0.)[k])

    for k, (x, side, width, height, rho) in enumerate(((2.69, 1.13, 1.02, 2.58, .19),
            (3.16, -1.19, 1.14, 2.76, .79), (2.91, 1.02, .84, 1.54, .56),
            (3.29, -1.16, 1.04, 1.72, .11))):
        add(f'offroute_{k}', 'substantial_offroute', [
            obj((x, side, height/2), (.24+.014*k, width, height), rho, role='substantial_offroute'),
        ], mode='lateral' if k < 3 else 'rotation_only', sign=(-1.)**k,
            yaw=(.7, -.8, 1.1, -3.6)[k], ghost=dict(x=.082, z=2.97) if k == 2 else None)

    add('head_weak_wall', 'suspended_head', [
        obj((3.295, -.084, 1.89), (.11, .29, .15), .065, role='near_weak_head'),
        obj((3.815, .03, 1.86), (.07, 5.52, 3.72), .89, role='farwall_reference'),
    ], yaw=.8, sign=-1.)
    add('head_static_lateral', 'suspended_head', [
        obj((2.795, .135, 1.96), (.19, .47, .17), .67, role='suspended_head'),
    ], yaw=-.9)
    add('head_side_margin', 'suspended_head', [
        obj((3.075, .535, 1.83), (.15, .31, .21), .38, role='suspended_head'),
    ], yaw=.4)
    add('head_rotation_only', 'suspended_head', [
        obj((2.945, -.105, 1.87), (.22, .53, .23), .82, role='suspended_head'),
    ], mode='rotation_only', yaw=3.8)

    add('moving_barrier_head', 'moving_occlusion', [
        obj((2.225, -.33, 1.06), (.21, .49, 1.78), .74, (0., .23, 0.), 'moving_occluder'),
        obj((3.205, .075, 1.92), (.15, .41, .19), .16, role='partially_occluded_head'),
    ], yaw=-1.2, ghost=dict(x=-.085, z=2.89))
    add('moving_crossing_rod', 'moving_occlusion', [
        obj((2.765, .76, 1.09), (.064, .055, 2.18), .32, (0., -.59, 0.), 'moving_crossing_rod'),
        obj((3.795, -.025, 1.83), (.07, 5.48, 3.66), .72, role='farwall_reference'),
    ], sign=-1., yaw=.6)

    add('clear_lateral_ghost', 'clear_ghost', [], yaw=.5, ghost=dict(x=.09, z=2.92))
    add('clear_rotation_ghost', 'clear_ghost', [], mode='rotation_only', yaw=-4.1, ghost=dict(x=-.07, z=3.08))
    add('zero_translation_rotation', 'translation_controls', [
        obj((3.125, .095, 1.17), (.075, .067, 2.34), .28, role='control_rod'),
        obj((3.825, -.045, 1.87), (.07, 5.58, 3.74), .81, role='farwall_reference'),
    ], mode='rotation_only', yaw=4.3)
    add('low_translation_rotation', 'translation_controls', [
        obj((2.875, -.14, 1.88), (.17, .37, .18), .44, role='control_head'),
        obj((3.805, .045, 1.85), (.07, 5.62, 3.70), .69, role='farwall_reference'),
    ], mode='low_translation_rotation', sign=-1., yaw=-3.9)

    frames = []; audits = []; counts = Counter(); transitions = Counter(); family_counts = Counter()
    for scene in scenes:
        sensor_seed = sensor_rng.randrange(2**30)
        definitions = [dict(o, name=f'shape{k}', texture_seed=texture_rng.randrange(2**30)) for k, o in enumerate(scene['objects'])]
        labels = []; camera_velocities = []; camera_positions = []
        for step in range(STEPS):
            t = step*DT
            camera, velocity = camera_motion(t, scene['camera_motion_mode'], scene['lateral_sign'], scene['yaw_amplitude_deg'])
            objects = [dict(name=o['name'], center_m=[c+t*v for c, v in zip(o['start_m'], o['velocity_mps'])],
                size_m=o['size_m'], texture_seed=o['texture_seed'], texture_grid=[4, 7],
                tof_reflectance_proxy=o['tof_reflectance_proxy'], source_role=o['source_role']) for o in definitions]
            frame = dict(id=f"{scene['episode']}_{step:02d}", episode=scene['episode'], family=scene['family'], time_s=t,
                camera=camera, body_origin_m=[camera['x'], camera['y'], 0.], objects=objects,
                sensor_seed=sensor_seed, wearer_speed=math.hypot(*velocity[:2]), radar_ghost=scene['ghost'])
            label = any(intersects(frame, o) for o in objects); labels.append(label)
            counts['positive' if label else 'negative'] += 1; family_counts[(scene['family'], label)] += 1
            frames.append(frame); camera_velocities.append(velocity); camera_positions.append([camera[k] for k in ('x', 'y', 'z')])
        enters = sum(not a and b for a, b in zip(labels, labels[1:])); exits = sum(a and not b for a, b in zip(labels, labels[1:]))
        transitions.update(enter=enters, exit=exits)
        audits.append(dict(episode=scene['episode'], family=scene['family'], objects=definitions,
            camera_motion_mode=scene['camera_motion_mode'], camera_motion_parameters=dict(lateral_sign=scene['lateral_sign'],
                lateral_amplitude_m=.12 if scene['camera_motion_mode'] == 'lateral' else .003 if scene['camera_motion_mode'] == 'low_translation_rotation' else 0.,
                forward_speed_mps=.01 if scene['camera_motion_mode'] == 'lateral' else .001 if scene['camera_motion_mode'] == 'low_translation_rotation' else 0.,
                lateral_angular_frequency_rad_s=1.1, yaw_angular_frequency_rad_s=.79, yaw_amplitude_deg=scene['yaw_amplitude_deg']),
            camera_positions_m=camera_positions, instantaneous_camera_velocity_mps=camera_velocities,
            source_aabb_labels=labels, enters=enters, exits=exits))
    spec = dict(schema='mz119-causal-parallax-source-v1', seed=SEED, texture_seed=texture_seed,
        authority='CONTROLLED_UE_RGB_NATIVE_COLLISION_FINITE_FOOTPRINT_TOF_PROXY_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640, height=360, hfov_deg=70., tof_hfov_deg=45., tof_rows=8, tof_columns=8, rgb_camera_count=1),
        background=dict(center_m=[12.7, 0., 1.75], size_m=[.14, 18.6, 8.5], texture=False, tof_reflectance_proxy=.50),
        floor=dict(center_m=[4., 0., -.05], size_m=[24., 20., .1], tof_reflectance_proxy=.30),
        fixed_before_capture=True, dt_s=DT, frames_per_episode=STEPS, family_episode_counts=FAMILIES,
        camera_motion_episode_counts=dict(Counter(a['camera_motion_mode'] for a in audits)),
        object_motion_episode_counts=dict(static=sum(not any(any(v != 0 for v in o['velocity_mps']) for o in a['objects']) for a in audits),
            moving=sum(any(any(v != 0 for v in o['velocity_mps']) for o in a['objects']) for a in audits)),
        designated_mixture_episodes=[a['episode'] for a in audits if a['family'] == 'near_rod_farwall_mixture'],
        designated_mixture_minimum_frames=12,
        source_design_aabb_frame_counts=dict(counts), source_design_transition_counts=dict(transitions),
        family_source_aabb_counts={f:dict(positive=family_counts[(f, True)], negative=family_counts[(f, False)]) for f in FAMILIES},
        source_audit_authority='SOURCE_DESIGN_ONLY_NOT_NATIVE_ENGINE_MEASUREMENTS_OR_OBSERVABLE_LABELS', source_audit=audits,
        initial_body_origin_m=[0., 0., 0.], corridor_m=dict(forward=[.2, 3.6], lateral=[-.3, .3], height=[.4, 2.05]),
        translation_authority='SOURCE_AND_EVALUATOR_ONLY_CAMERA_TRAJECTORY_NEVER_RAW_OR_INFERENCE_INPUT',
        reflectance_authority='EXPLICIT_LATENT_SENSOR_PROXY_INDEPENDENT_OF_RGB_ALBEDO_TEXTURE',
        limitations=[
            'Body origin follows camera XY; the query remains a current world-X-aligned corridor, not yaw-aligned walking intention or future trajectory.',
            'Lateral tracks use0.12m sinusoidal amplitude and0.01m/s forward drift; zero translation rotation and low translation rotation are explicit controls.',
            'Camera position/velocity and analytic motion formulas are source/evaluator-only. Existing raw camera_in_body_m remains[0,0,1.7]; no metric translation is exported.',
            'Pitch remains-3deg and roll0 because the unchanged observation adapter has fixed pitch and delta_pitch0. IMU yaw retains its original hypothetical noise/drift.',
            'All obstacle/far-wall geometry and texture seeds are fresh. Unchanged global background/floor are excluded context only after every-frame source-AABB nonintersection checks.',
            'Analytic first-hit tests and reducer lineage establish source sampling opportunities, not rendered feature correspondence or parallax estimator efficacy.',
            'Far bias uses slant-range component means; current corridor labels use forward native AABB geometry separately.',
            'Sparse-ray misses, absent packets, moving surfaces, texture ambiguities, pure rotation and low parallax are retained; no guaranteed metric scale or depth observability.',
            'Constructed Development only; no calibrated optical/RF hardware, natural-domain, deployment or safety evidence.'], frames=frames)
    spec['analytic_mixture_precheck'] = analytic.analytic_precheck(spec)
    spec['analytic_source_helper_sha256'] = hashlib.sha256(Path(analytic.__file__).read_bytes()).hexdigest()
    check_source(spec)
    return spec


def check_source(spec):
    frames = spec['frames']; audits = spec['source_audit']
    assert len(frames) == len({f['id'] for f in frames}) == 240 and len(audits) == 20
    assert Counter(f['episode'] for f in frames) == {a['episode']:12 for a in audits}
    assert Counter(a['family'] for a in audits) == FAMILIES
    assert spec['camera_motion_episode_counts'] == CAMERA_COHORTS
    assert spec['object_motion_episode_counts'] == dict(static=18, moving=2)
    assert len({f['sensor_seed'] for f in frames}) == 20
    assert spec['source_design_aabb_frame_counts'] == dict(positive=155, negative=85)
    assert spec['source_design_transition_counts'] == dict(enter=2, exit=2)
    mixtures = spec['analytic_mixture_precheck']; assert len(mixtures['frames']) == 240
    assert mixtures['counts']['returned_real_mixture_frames'] >= 12
    assert mixtures['counts']['near_rod_unsampled_frames'] > 0
    assert mixtures['counts']['weak_near_far_biased_mean_targets'] > 0
    for ep in spec['designated_mixture_episodes']: assert mixtures['episode_counts'][ep]['returned_real_mixture_frames'] > 0, ep
    for frame in frames:
        cam = frame['camera']; assert cam['pitch'] == -3. and cam['roll'] == 0.
        assert frame['body_origin_m'] == [cam['x'], cam['y'], 0.]
        for context in (spec['background'], spec['floor']): assert not intersects(frame, context)
        for o in frame['objects']:
            if o['source_role'] == 'farwall_reference':
                assert 3.6 < o['center_m'][0]-o['size_m'][0]/2-cam['x'] <= 3.8 and not intersects(frame, o)
            if o['source_role'] == 'substantial_offroute':
                assert abs(o['center_m'][1]-cam['y'])-o['size_m'][1]/2 > .4 and not intersects(frame, o)
    for audit in audits:
        rows = [f for f in frames if f['episode'] == audit['episode']]
        assert [r['time_s'] for r in rows] == [k*DT for k in range(STEPS)]
        assert all([o['name'] for o in r['objects']] == [o['name'] for o in audit['objects']] for r in rows)
        assert audit['source_aabb_labels'] == [any(intersects(r, o) for o in r['objects']) for r in rows]
        if audit['camera_motion_mode'] == 'lateral': assert max(abs(r['camera']['y']) for r in rows) > .115
        if audit['camera_motion_mode'] in ('rotation_only', 'stationary'): assert all(r['camera']['x'] == r['camera']['y'] == 0 for r in rows)
        if audit['camera_motion_mode'] == 'low_translation_rotation': assert max(math.hypot(r['camera']['x'], r['camera']['y']) for r in rows) < .005
    return dict(status='PASS', frames=240, episodes=20, source_aabb_counts=spec['source_design_aabb_frame_counts'],
        transitions=spec['source_design_transition_counts'], camera_motion_counts=spec['camera_motion_episode_counts'],
        object_motion_counts=spec['object_motion_episode_counts'], analytic_mixture_counts=mixtures['counts'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--capture-source', type=Path, action='append', default=[])
    args = parser.parse_args(); spec = source()
    if args.self_test: print(json.dumps(check_source(spec), indent=2)); return
    if args.output is None or not args.capture_source: parser.error('--output and --capture-source required; freeze only after methods are ready')
    root = Path(__file__).resolve().parents[4]; output = args.output.resolve(); freeze = output.with_name('freeze.json')
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output == freeze or output.exists() or freeze.exists(): raise ValueError('Fresh canonical source/freeze files required')
    helpers = [p.resolve() for p in args.capture_source]+[Path(analytic.__file__).resolve()]
    if not all(p.is_file() for p in helpers): raise ValueError('Capture helpers must be ready before freeze')
    payload = (json.dumps(spec, indent=2, allow_nan=False)+'\n').encode('utf-8')
    receipt = dict(spec_sha256=hashlib.sha256(payload).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}, **check_source(spec),
        selection='FIXED_SEED119013_ALL20_EPISODES_ALL240_FRAMES_NO_MODEL_OUTCOME_SELECTION')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle: handle.write(payload)
    with freeze.open('x', encoding='utf-8') as handle: handle.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
