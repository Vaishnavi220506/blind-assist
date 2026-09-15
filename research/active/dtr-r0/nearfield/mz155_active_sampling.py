"""Prospective stationary-world angular-sampling contrast; no predictor outcomes."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import mz136_paired_source as inherited

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work/mz155-active-sampling-20260916'
SEED = 155016
CELL_DEG = 45./8
YAW_DEG = tuple(0. if i in (0, 5) else CELL_DEG/2*math.sin(2*math.pi*i/5) for i in range(6))
AUTHORITY = 'PROSPECTIVE_CONTROLLED_DEVELOPMENT_ANGULAR_SAMPLING_NOT_TRANSLATION_PARALLAX_OR_HARDWARE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def source():
    old_seed = inherited.SEED
    try:
        inherited.SEED = SEED
        generated = inherited.source()
    finally:
        inherited.SEED = old_seed
    result = {key: copy.deepcopy(generated[key]) for key in ('rig', 'background', 'floor', 'corridor_m')}
    result.update(schema='mz155-static-world-yaw-sampling-v1', seed=SEED, authority=AUTHORITY,
        dt_s=.25, frames_per_episode=6, yaw_pattern_deg=list(YAW_DEG), frames=[], scene_groups=[], pairs=[],
        observation_pairs=[], source_audit=[],
        limitations=['Pure rotation changes angular sampling, not translation parallax.',
            'Entire co-located RGB/ToF/Radar rig rotates; this is not a body-fixed Radar.',
            'Stationary rig and world are experimental conditions, not inferred by a motion detector.',
            'Native trace footprint is hypothetical; physical full-zone ToF response is not validated.',
            'Shared RNG start does not imply identical later draws after geometry-dependent sensor branches.',
            'Frozen learned readout transfers from moving sources to stationary scenes without refitting.'])
    lookup = {f['episode']: f for f in reversed(generated['frames'])}
    for group in generated['scene_groups']:
        if int(group['scene_group'].rsplit('scene', 1)[1]) >= 3:
            continue
        name = group['scene_group'].replace('mz136', 'mz155')
        family = group['category']
        new_group = dict(scene_group=name, category=family, split='development',
            background_geometry_signature=group['background_geometry_signature'], episodes=[])
        result['scene_groups'].append(new_group)
        members = {}
        for old_episode in group['episodes']:
            original = lookup[old_episode]
            frame = copy.deepcopy(original)
            initial_y = frame['camera']['y']
            frame['objects'][0]['center_m'][1] -= initial_y
            frame['camera'] = dict(x=.023, y=0., z=1.7, yaw=0., pitch=-3., roll=0.)
            frame['body_origin_m'] = [.023, 0., 0.]
            inside = any(inherited.intersects(frame, obj) for obj in frame['objects'])
            member = 'in' if inside else 'out'
            assert member not in members
            members[member] = frame
        for arm in ('passive', 'scan'):
            pair_id = name+'_'+arm
            pair_episodes = []
            for member in ('in', 'out'):
                episode = name+'_'+member+'_'+arm
                pair_episodes.append(episode); new_group['episodes'].append(episode)
                for step in range(6):
                    frame = copy.deepcopy(members[member])
                    frame.update(id=episode+f'_{step:02d}', episode=episode, scene_group=name,
                        pair_id=pair_id, pair_member=member, pair_variant=member, split='development',
                        observation_arm=arm, matched_case=name+'_'+member,
                        time_s=.25*step, wearer_speed=0., radar_ghost=None)
                    frame['camera']['yaw'] = YAW_DEG[step] if arm == 'scan' else 0.
                    result['frames'].append(frame)
                result['source_audit'].append(dict(episode=episode, scene_group=name,
                    split='development', family=family, source_aabb_labels=[member == 'in']*6))
            result['pairs'].append(dict(pair_id=pair_id, scene_group=name, split='development',
                category=family, episodes=pair_episodes, intervention='TARGET_LATERAL_POSITION_ONLY'))
        for member in ('in', 'out'):
            result['observation_pairs'].append(dict(matched_case=name+'_'+member,
                passive=name+'_'+member+'_passive', scan=name+'_'+member+'_scan',
                intervention='PREDETERMINED_YAW_ONLY_SAME_SIX_FRAMES_AND_DURATION'))
    check_source(result)
    return result


def check_source(spec):
    assert spec['seed'] == SEED and spec['yaw_pattern_deg'] == list(YAW_DEG)
    assert spec['authority'] == AUTHORITY
    assert len(spec['frames']) == len({f['id'] for f in spec['frames']}) == 288
    assert len(spec['scene_groups']) == 12 and len(spec['pairs']) == len(spec['observation_pairs']) == 24
    assert Counter(f['observation_arm'] for f in spec['frames']) == {'passive': 144, 'scan': 144}
    episodes = {}
    for frame in spec['frames']:
        episodes.setdefault(frame['episode'], []).append(frame)
        assert frame['body_origin_m'] == [.023, 0., 0.]
        assert all(frame['camera'][k] == v for k, v in dict(x=.023,y=0.,z=1.7,pitch=-3.,roll=0.).items())
        assert frame['split'] == 'development' and frame['wearer_speed'] == 0.
        assert all(not inherited.intersects(frame, o) for o in frame['objects'][1:])
        assert inherited.intersects(frame, frame['objects'][0]) == (frame['pair_member'] == 'in')
    for rows in episodes.values():
        assert [f['time_s'] for f in rows] == [i*.25 for i in range(6)]
        assert all(f['objects'] == rows[0]['objects'] for f in rows)
        assert len({f['sensor_seed'] for f in rows}) == len({f['tof_sensor_seed'] for f in rows}) == 1
    for pair in spec['observation_pairs']:
        for passive, scan in zip(episodes[pair['passive']], episodes[pair['scan']]):
            a, b = copy.deepcopy(passive), copy.deepcopy(scan)
            for row in (a, b):
                for key in ('id', 'episode', 'pair_id', 'observation_arm'):
                    row.pop(key)
                row['camera']['yaw'] = 0.
            assert a == b, 'Motion arms differ beyond identity and camera yaw'
    for pair in spec['pairs']:
        for inside, outside in zip(*(episodes[e] for e in pair['episodes'])):
            a, b = copy.deepcopy(inside), copy.deepcopy(outside)
            for row in (a, b):
                for key in ('id','episode','pair_member','pair_variant','matched_case'):
                    row.pop(key)
                row['objects'][0]['center_m'][1] = 0.
            assert a == b, 'Corridor pair differs beyond lateral position and identity'
    return dict(status='PASS', frames=288, episodes=len(episodes), scene_groups=12,
        cases_per_arm=24, positive_frames_per_arm=72, negative_frames_per_arm=72,
        families=dict(Counter(f['family'] for f in spec['frames'])), world_static=True,
        matched_rig_positions=True, matched_motion_arms=True, yaw_pattern_deg=list(YAW_DEG),
        sampled_yaw_span_deg=max(YAW_DEG)-min(YAW_DEG), elapsed_observation_s=1.25,
        yaw_selection='PUBLIC_CELL_WIDTH_ONLY_NO_PRIVATE_SUBRAY_LAYOUT_OR_OUTCOMES')


def freeze(output):
    output = Path(output).resolve()
    assert output.is_relative_to(WORK.resolve()) and not output.exists()
    output.mkdir(parents=True)
    spec = source()
    write(output/'spec.json', spec); write(output/'source-audit.json', check_source(spec))
    near = Path(__file__).parent
    paths = [Path(__file__), near/'mz136_paired_source.py', near/'MZ155_PROTOCOL_20260916.md']
    model = ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1/fused_hgb.pkl'
    onset = ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1/onset-seal.json'
    assert sha(model) == 'a8df5dc6c06f80057100042af48eda9b1719276e5f83140f4acca94b9cef2fb6'
    write(output/'freeze.json', dict(files={name: sha(output/name) for name in ('spec.json','source-audit.json')},
        source_inputs={str(p): sha(p) for p in paths}, model_sha256=sha(model), onset_sha256=sha(onset),
        source_revision='6ddb850809b6436acc913d17a3ae3c301e5739a8',
        scoped_wip_input=True, capture_attempts=1, timeout_seconds=1200,
        outcome_access_before_freeze=False, authority=AUTHORITY))
    print(json.dumps(check_source(spec), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
