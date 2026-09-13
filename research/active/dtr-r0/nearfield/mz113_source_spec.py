"""Fixed 240-frame dynamic four-sensor Development source, seed 113013.

Source-only analytic positions/velocities and corridor labels are audit metadata,
never raw observations. Doppler comes from past native bounds in the separate
sensor generator. No capture outcome, protected split or adaptive selection is
used here. All 20 episodes and all 12 time steps are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


SEED = 113013
DT = .25
STEPS = 12
FAMILY_EPISODES = dict(crossing_pole=4, crossing_head=4, receding=2,
                       crossing_multitarget=4, offroute=2, clear=2,
                       boundary_1cm_stress=2)


def moving(start, size, velocity=(0., 0., 0.)):
    return dict(start_m=list(start), size_m=list(size), velocity_mps=list(velocity))


def intersects(frame, obj):
    lo = [c-s/2-b for c, s, b in zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    hi = [c+s/2-b for c, s, b in zip(obj['center_m'], obj['size_m'], frame['body_origin_m'])]
    return (hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3
            and hi[2] >= .4 and lo[2] <= 2.05)


def source():
    rng = random.Random(SEED)
    scenes = []

    def add(name, family, objects, yaw=0., ghost=None):
        scenes.append(dict(episode='mz113_'+name, family=family, objects=objects,
                           yaw_amplitude=yaw, ghost=ghost))

    # Opposite-side crossing and exits, with narrow and broader poles.
    for k, (side, width, forward) in enumerate(((1, .038, 2.81), (-1, .058, 3.07),
                                              (1, .086, 3.31), (-1, .112, 2.53))):
        add(f'pole_{k}', 'crossing_pole', [
            moving((forward, side*.77, 1.12), (.064+.009*k, width, 2.24),
                   (.11+.025*k, -side*.56, 0.)),
        ], yaw=(0., -5., 7., 0.)[k], ghost=dict(x=-.09, z=2.79) if k == 3 else None)

    # Suspended bars traverse completely across the corridor, then leave it.
    for k, (side, width, forward) in enumerate(((-1, .28, 3.19), (1, .44, 2.91),
                                              (-1, .62, 3.47), (1, .36, 2.69))):
        add(f'head_{k}', 'crossing_head', [
            moving((forward, side*1.02, 1.79+.04*k), (.155+.025*k, width, .19+.02*k),
                   (.14, -side*.75, 0.)),
        ], yaw=(4., 0., -6., 0.)[k], ghost=dict(x=.08, z=2.97) if k == 2 else None)

    # Positive range hazards recede through 3.6m despite forward wearer motion.
    add('receding_body', 'receding', [
        moving((1.67, .047, 1.17), (.245, .39, .98), (1.58, .025, 0.)),
    ])
    add('receding_head', 'receding', [
        moving((2.09, -.073, 1.87), (.185, .46, .21), (1.67, -.018, 0.)),
    ], -4., dict(x=-.11, z=3.13))

    # A closer body slab can occlude a farther HEAD actor around the crossing;
    # both target lists and names remain stable when hidden. Other episodes mix
    # a crossing pole/bar with sizeable off-route competitors and ghost returns.
    add('multi_occlusion_right', 'crossing_multitarget', [
        moving((2.39, -1.03, 1.18), (.26, .34, 1.88), (.12, .75, 0.)),
        moving((3.43, 1.07, 1.83), (.18, .42, .25), (.09, -.78, 0.)),
    ], 5., dict(x=.06, z=2.83))
    add('multi_occlusion_left', 'crossing_multitarget', [
        moving((2.57, 1.12, 1.20), (.28, .38, 1.94), (.17, -.82, 0.)),
        moving((3.51, -1.06, 1.90), (.20, .48, .23), (.13, .77, 0.)),
    ], -7., dict(x=-.07, z=3.03))
    add('multi_pole_wall', 'crossing_multitarget', [
        moving((2.93, -.82, 1.15), (.071, .068, 2.30), (.08, .61, 0.)),
        moving((3.37, 1.01, 1.27), (.33, .94, 2.54)),
    ], 0., dict(x=.10, z=2.91))
    add('multi_head_barrier', 'crossing_multitarget', [
        moving((3.11, .97, 1.85), (.225, .41, .27), (.19, -.72, 0.)),
        moving((3.59, -.96, .96), (.36, .90, 1.14)),
    ], -5., dict(x=-.08, z=3.19))

    # Ordinary true-negative extents have 13cm and 18cm lateral clearance.
    add('offroute_wall', 'offroute', [moving((3.23, 1.02, 1.29), (.35, 1.18, 2.58))], 4.)
    add('offroute_barrier', 'offroute', [moving((2.87, -.93, 1.03), (.32, .90, 1.22))], -4.)
    add('clear_context', 'clear', [moving((6.43, 2.71, 1.23), (.21, .56, 2.46))])
    add('clear_context_ghost', 'clear', [moving((6.69, -2.89, 1.16), (.25, .52, 2.32))],
        3., dict(x=.075, z=3.27))

    # Only these two episodes have a total +/-1cm lateral boundary excursion.
    add('stress_enter_1cm', 'boundary_1cm_stress', [
        moving((3.41, .34, 1.31), (.195, .06, 1.42), (.15, -.02/(11*DT), 0.)),
    ])
    add('stress_exit_1cm', 'boundary_1cm_stress', [
        moving((3.53, -.32, 1.34), (.215, .06, 1.48), (.18, -.02/(11*DT), 0.)),
    ])

    frames = []
    audits = []
    counts = Counter()
    transition_counts = Counter()
    for i, scene in enumerate(scenes):
        speed = round(.38+.008*i, 3)
        sensor_seed = rng.randrange(2**30)
        definitions = []
        for j, obj in enumerate(scene['objects']):
            definitions.append(dict(obj, name=f'shape{j}', texture_seed=rng.randrange(2**30)))
        labels = []
        for j in range(STEPS):
            t = j*DT
            x = speed*t
            objects = [dict(name=o['name'], center_m=[c+v*t for c,v in zip(o['start_m'],o['velocity_mps'])],
                            size_m=o['size_m'], texture_seed=o['texture_seed'], texture_grid=[3,6])
                       for o in definitions]
            frame = dict(id=f"{scene['episode']}_{j:02d}", episode=scene['episode'], family=scene['family'],
                         time_s=t, camera=dict(x=x,y=0.,z=1.7,yaw=scene['yaw_amplitude']*math.sin(.77*t),pitch=-3.,roll=0.),
                         body_origin_m=[x,0.,0.], objects=objects, sensor_seed=sensor_seed,
                         wearer_speed=speed, radar_ghost=scene['ghost'])
            positive = any(intersects(frame,o) for o in objects)
            labels.append(positive)
            counts['positive' if positive else 'negative'] += 1
            frames.append(frame)
        transitions = sum(a != b for a,b in zip(labels,labels[1:]))
        enters = sum(not a and b for a,b in zip(labels,labels[1:]))
        exits = sum(a and not b for a,b in zip(labels,labels[1:]))
        transition_counts.update(enter=enters, exit=exits)
        if scene['family'] in ('crossing_pole','crossing_head','crossing_multitarget'):
            assert not labels[0] and not labels[-1] and enters == exits == 1, scene['episode']
        elif scene['family'] == 'receding':
            assert labels[0] and not labels[-1] and exits == 1 and enters == 0
        elif scene['family'] in ('offroute','clear'):
            assert not any(labels)
        else:
            assert transitions == 1
        audits.append(dict(episode=scene['episode'], family=scene['family'],
                           camera_velocity_mps=[speed,0.,0.], objects=definitions,
                           source_aabb_labels=labels, transitions=transitions, enters=enters, exits=exits))

    assert len(scenes) == 20 and len(frames) == 240
    assert Counter(s['family'] for s in scenes) == FAMILY_EPISODES
    assert len({f['id'] for f in frames}) == 240
    assert Counter(f['episode'] for f in frames) == {s['episode']:12 for s in scenes}
    assert len({f['sensor_seed'] for f in frames}) == 20
    assert counts == dict(positive=88,negative=152)
    assert transition_counts == dict(enter=13,exit=15)
    assert sum(any(any(v != 0 for v in o['velocity_mps']) for o in a['objects']) for a in audits) == 16
    assert all(f['camera']['pitch']==-3. and f['camera']['roll']==0. for f in frames)
    for audit in audits:
        episode_frames=[f for f in frames if f['episode']==audit['episode']]
        assert all([o['name'] for o in f['objects']]==[o['name'] for o in audit['objects']] for f in episode_frames)

    return dict(schema='mz113-dynamic-single-rgb-native-past-doppler-v1',seed=SEED,
                authority='FRESH_CONTROLLED_UE_RGB_NATIVE_COLLISION_TOF_PAST_NATIVE_DOPPLER_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
                rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,rgb_camera_count=1),
                background=dict(center_m=[12.3,0.,1.72],size_m=[.14,18.2,8.4],texture=False),
                fixed_before_capture=True,dt_s=DT,frames_per_episode=STEPS,
                family_episode_counts=FAMILY_EPISODES,
                source_design_aabb_frame_counts=dict(counts),source_design_transition_counts=dict(transition_counts),
                source_audit_authority='SOURCE_DESIGN_ONLY_NOT_OBSERVATIONS_OR_ENGINE_NATIVE_MEASUREMENTS',
                source_audit=audits,
                corridor_m=dict(forward=[.2,3.6],lateral=[-.3,.3],height=[.4,2.05]),
                limitations=[
                    'Source-defined constant box velocities are audit-only; raw Doppler derives from current/past engine centers and camera positions.',
                    'Doppler uses horizontal center LOS while range retains the original native ray-hit proxy; neither is RF simulation.',
                    'First-sample Doppler missing; no future native transforms or source velocity fields are read by sensors.',
                    'World-axis current corridor, fixed pitch -3deg and roll0; camera yaw is not future walking intention.',
                    'Constructed scene-disjoint Development in a shared procedural environment; no protected splits or outcome selection.',
                    'No hardware, safety, deployment, natural-domain generalization or continuous-time simulation claim.',
                ],frames=frames)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[4]
    output=args.output.resolve()
    freeze_path=output.with_name('freeze.json')
    if not output.is_relative_to((root/'artifacts.local').resolve()):
        raise ValueError('Canonical artifacts only')
    if output==freeze_path or output.exists() or freeze_path.exists():
        raise ValueError('Preserve frozen source and receipt')
    spec=source()
    payload=(json.dumps(spec,indent=2,allow_nan=False)+'\n').encode('utf-8')
    capture_files=('mz113_dynamic_capture.py','mz113_dynamic_sensors.py',
                   'mz99_angle_information_capture.py','ue_capture_readiness.py')
    freeze=dict(spec_sha256=hashlib.sha256(payload).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                capture_source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in capture_files},
                frames=240,episodes=20,source_design_aabb_frame_counts=spec['source_design_aabb_frame_counts'],
                source_design_transition_counts=spec['source_design_transition_counts'],
                selection='FIXED_SEED113013_ALL20_EPISODES_ALL240_FRAMES_NO_OUTCOME_SELECTION')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:handle.write(payload)
    with freeze_path.open('x',encoding='utf-8') as handle:handle.write(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze))


if __name__=='__main__':main()
