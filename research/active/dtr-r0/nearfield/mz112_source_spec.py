"""Fixed fresh 20-episode, 240-frame four-sensor Development source.

Reuse mz107_single_rgb_capture.py and mz107_sensors.py unchanged. No renderer,
outcomes, protected split or adaptive selection is used to define this panel.
Objects are static: the existing Radar velocity proxy models ego x translation
only. Camera pitch/roll/calibration stay fixed; yaw is camera rotation, not a
future-path label. Source AABB assertions are design checks, not UE measurements.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


SEED = 112013
EPISODES = 20
FRAMES_PER_EPISODE = 12
FAMILY_EPISODES = dict(pole=4, head=4, offroute=4, multitarget=4, clear=2,
                       boundary_1cm_stress=2)


def shape(center, size):
    return dict(center_m=list(center), size_m=list(size))


def design_intersects(frame, obj):
    """World-axis native-AABB corridor convention, before engine verification."""
    lo = [c - s / 2 - b for c, s, b in
          zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    hi = [c + s / 2 - b for c, s, b in
          zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    return (hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3
            and hi[2] >= .4 and lo[2] <= 2.05)


def source():
    rng = random.Random(SEED)
    scenes = []

    def add(episode, family, objects, positive, yaw=0., ghost=None):
        scenes.append(dict(episode='mz112_' + episode, family=family,
                           objects=objects, design_positive=positive,
                           yaw_amplitude=yaw, ghost=ghost))

    # Thin and side-offset poles. The shallowest ordinary overlap is 5.4cm,
    # intentionally separated from the two 1cm stress scenes below.
    add('pole_narrow_near', 'pole', [shape((1.91, .013, 1.08), (.034, .034, 2.16))], True)
    add('pole_left', 'pole', [shape((2.43, -.242, 1.13), (.046, .062, 2.26))], True, -5.)
    add('pole_right', 'pole', [shape((3.43, .285, 1.04), (.078, .078, 2.08))], True)
    add('pole_broad_turn', 'pole', [shape((2.97, -.173, 1.17), (.094, .118, 2.34))], True, 10.)

    # Suspended slabs/bars with no ground contact and actual corridor intrusion.
    add('head_center', 'head', [shape((3.51, -.045, 1.84), (.17, .42, .22))], True)
    add('head_right', 'head', [shape((2.67, .31, 1.91), (.21, .30, .18))], True, -6.)
    add('head_left', 'head', [shape((3.17, -.35, 1.76), (.14, .36, .26))], True, 8.)
    add('head_wide', 'head', [shape((2.29, .16, 1.96), (.26, .68, .14))], True)

    # Substantial wall/barrier extents, with inner edges 10--28cm outside.
    add('offroute_wall_right', 'offroute', [shape((3.37, .98, 1.24), (.31, 1.00, 2.48))], False)
    add('offroute_wall_left', 'offroute', [shape((2.73, -1.12, 1.31), (.27, 1.08, 2.62))], False, -9.)
    add('offroute_barrier_right', 'offroute', [shape((3.23, .83, .91), (.38, .86, 1.02))], False, 7.)
    add('offroute_barrier_left', 'offroute', [shape((2.61, -.92, 1.02), (.29, .92, 1.24))], False)

    # Persistent ghost competes with real targets in all four episodes. Two
    # episodes have actual on-route hazards; two contain only off-route objects.
    add('multi_body_head', 'multitarget', [
        shape((3.09, .075, 1.18), (.23, .32, .92)),
        shape((3.49, -.58, 1.89), (.16, .24, .20)),
    ], True, 11., dict(x=-.095, z=2.57))
    add('multi_head_pole', 'multitarget', [
        shape((2.89, -.29, 1.82), (.19, .38, .24)),
        shape((3.33, .57, 1.10), (.082, .082, 2.20)),
    ], True, -12., dict(x=.105, z=2.71))
    add('multi_outside_walls', 'multitarget', [
        shape((2.99, -.84, 1.20), (.24, .82, 2.40)),
        shape((3.57, .73, 1.78), (.22, .50, .30)),
    ], False, 6., dict(x=-.075, z=2.63))
    add('multi_outside_pole_barrier', 'multitarget', [
        shape((2.79, .49, 1.16), (.072, .096, 2.32)),
        shape((3.41, -.91, .98), (.34, .88, 1.16)),
    ], False, -8., dict(x=.085, z=2.81))

    # Clear means no near-corridor obstacle. Distinct distant off-route context
    # makes these geometry-disjoint from the empty MZ107--109 clear scenes.
    add('clear_distant_context', 'clear', [
        shape((5.83, 2.43, 1.11), (.23, .48, 2.22)),
    ], False, 4.)
    add('clear_ghost_distant_context', 'clear', [
        shape((6.17, -2.62, 1.26), (.19, .54, 2.52)),
    ], False, -4., dict(x=-.065, z=2.93))

    # Standalone sensitivity stratum: exactly 1cm inside versus 1cm outside.
    add('stress_inside_1cm', 'boundary_1cm_stress', [
        shape((3.29, .43, 1.28), (.20, .28, 1.36)),
    ], True)
    add('stress_outside_1cm', 'boundary_1cm_stress', [
        shape((3.39, -.45, 1.32), (.22, .28, 1.44)),
    ], False)

    frames = []
    design_counts = Counter()
    for i, scene in enumerate(scenes):
        for j, obj in enumerate(scene['objects']):
            obj.update(name=f'shape{j}', texture_seed=rng.randrange(2**30), texture_grid=[3, 6])
        sensor_seed = rng.randrange(2**30)
        speed = round(.56 + .006 * i, 3)
        for j in range(FRAMES_PER_EPISODE):
            t = j * .25
            x = speed * t
            yaw = scene['yaw_amplitude'] * math.sin(.9 * t)
            frame = dict(id=f"{scene['episode']}_{j:02d}", episode=scene['episode'],
                         family=scene['family'], time_s=t,
                         camera=dict(x=x, y=0., z=1.7, yaw=yaw, pitch=-3., roll=0.),
                         body_origin_m=[x, 0., 0.], objects=scene['objects'],
                         sensor_seed=sensor_seed, wearer_speed=speed, radar_ghost=scene['ghost'])
            positive = any(design_intersects(frame, obj) for obj in frame['objects'])
            assert positive == scene['design_positive'], frame['id']
            design_counts['positive' if positive else 'negative'] += 1
            frames.append(frame)

    assert len(scenes) == EPISODES
    assert Counter(s['family'] for s in scenes) == FAMILY_EPISODES
    assert len(frames) == EPISODES * FRAMES_PER_EPISODE == 240
    assert len({f['id'] for f in frames}) == 240
    assert Counter(f['episode'] for f in frames) == {s['episode']: 12 for s in scenes}
    assert design_counts == dict(positive=132, negative=108)
    assert len({f['sensor_seed'] for f in frames}) == EPISODES
    geometry = [json.dumps([(o['center_m'], o['size_m']) for o in s['objects']]) for s in scenes]
    assert len(set(geometry)) == EPISODES
    for obj, edge in ((scenes[-2]['objects'][0], .29), (scenes[-1]['objects'][0], .31)):
        assert math.isclose(abs(obj['center_m'][1]) - obj['size_m'][1] / 2, edge, abs_tol=1e-12)

    return dict(
        schema='mz112-single-rgb-fixed-diverse-development-v1', seed=SEED,
        authority='FRESH_CONTROLLED_UE_RGB_NATIVE_COLLISION_TOF_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640, height=360, hfov_deg=70., tof_hfov_deg=45., rgb_camera_count=1),
        background=dict(center_m=[11.7, 0., 1.65], size_m=[.13, 17.6, 8.2], texture=False),
        source_classes='20 scene-disjoint episodes x12 samples at 0.25s; all retained; disclosed Development',
        family_episode_counts=FAMILY_EPISODES,
        source_design_aabb_frame_counts=dict(design_counts),
        source_design_truth_authority='SOURCE_GEOMETRY_PREFLIGHT_ONLY_NOT_ENGINE_NATIVE_MEASUREMENT',
        corridor_m=dict(forward=[.2, 3.6], lateral=[-.3, .3], height=[.4, 2.05]),
        fixed_before_capture=True,
        limitations=[
            'Static objects only; hypothetical Radar velocity ignores object motion and assumes ego x translation.',
            'Yaw rotation only; camera pitch=-3deg, roll=0deg and calibration unchanged; no pitch/roll robustness claim.',
            'Current corridor is world-axis AABB relative to body origin; camera yaw is not future path intention.',
            'Clear episodes contain distant off-route context; one has a persistent hypothetical Radar ghost.',
            'Shared procedural capture environment; distinct episode geometries, textures, sensor seeds and trajectories are not natural-domain independence.',
            'No protected splits, adaptive outcome selection, hardware/RF, safety or deployment evidence.',
        ], frames=frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    artifacts = (root / 'artifacts.local').resolve()
    output = args.output.resolve()
    freeze_path = output.with_name('freeze.json')
    if not output.is_relative_to(artifacts):
        raise ValueError('Canonical artifacts only')
    if output == freeze_path or output.exists() or freeze_path.exists():
        raise ValueError('Preserve frozen source and freeze receipt; choose a fresh output directory')
    spec = source()
    spec_bytes = (json.dumps(spec, indent=2, allow_nan=False) + '\n').encode('utf-8')
    capture_files = ('mz107_single_rgb_capture.py', 'mz107_sensors.py',
                     'mz99_angle_information_capture.py', 'ue_capture_readiness.py')
    freeze = dict(
        spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_capture_source_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in capture_files},
        frames=240, episodes=20, family_episode_counts=FAMILY_EPISODES,
        source_design_aabb_frame_counts=spec['source_design_aabb_frame_counts'],
        selection='FIXED_SEED112013_ALL_20_SCENES_ALL_240_FRAMES_NO_OUTCOME_SELECTION',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(spec_bytes)
    with freeze_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(freeze, indent=2, allow_nan=False) + '\n')
    print(json.dumps(freeze))


if __name__ == '__main__':
    main()
