"""New paired geometries and appearance interventions; source truth stays private."""
from collections import Counter, defaultdict
import argparse
import copy
from datetime import datetime, timezone
import random
from pathlib import Path

from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify
from data_coverage_spec import RANGES, MATERIALS, BACKGROUNDS
from spatial_bce_spec import FAMILIES, RELATIONS
from query_occupancy_spec import FRONT_TRAJECTORY, relative_signature
from query_occupancy_data import sha, write

SEED = 202609225
FRAMES = 576
CLIPS = 48
DT_S = .2
APPEARANCES = ('base', 'changed')


def specification():
    groups, clips, cases = [], [], []
    for fi, (kind, layer, *_rest) in enumerate(FAMILIES):
        draws = [[], []]
        for axis, (low, high) in enumerate(RANGES[kind]):
            rng = random.Random(SEED + fi*100003 + axis*71)
            strata = [0, 1]
            rng.shuffle(strata)
            for k, stratum in enumerate(strata):
                draws[k].append(round(low+(high-low)*(stratum+rng.random())/2, 6))
        for k, draw in enumerate(draws):
            rng = random.Random(SEED + fi*1009 + k*67)
            group = f'local_transfer_{kind}_g{k:02d}'
            size, height = draw[:3], draw[3]
            side = -1 if (fi+k) % 2 else 1
            intrusion = round(rng.uniform(.006, .017), 6)
            inside, outside = rng.uniform(.105, .18), rng.uniform(.065, .15)
            world_x = round(31.7+rng.uniform(0, .8), 6)
            camera_y = round(rng.uniform(-.12, .12), 6)
            jitter = round(rng.uniform(-.03, .03), 6)
            background = dict(name='background', kind='cube',
                center_m=[round(world_x+rng.uniform(3.7,4.9),6),
                          round(camera_y+rng.uniform(-.33,.33),6),round(rng.uniform(2.08,2.18),6)],
                size_m=[round(rng.uniform(.22,.31),6),round(rng.uniform(6.2,7.3),6),round(rng.uniform(4.3,4.7),6)])
            common = dict(base_group_id=group, split='evaluation', type_id=kind,
                layer=layer, background='background_'+group, boundary_intrusion_m=intrusion,
                trajectory_jitter_m=jitter)
            groups.append(dict(**common, clips=[]))
            for ai, appearance in enumerate(APPEARANCES):
                target_material = MATERIALS[(fi*2+k) % len(MATERIALS)]
                back_material = BACKGROUNDS[(fi+k+2*ai) % len(BACKGROUNDS)]
                bg = dict(background, material='/Game/StreetLab/Materials/'+back_material)
                for relation in RELATIONS:
                    clip = f'{group}_{appearance}_{relation.lower()}'
                    groups[-1]['clips'].append(clip)
                    lateral = side*(.3+size[1]/2+(-inside if relation=='INSIDE' else
                                                -intrusion if relation=='BOUNDARY' else outside))
                    target = dict(name='target',kind='cube',center_m=[world_x,round(camera_y+lateral,9),height],
                        size_m=list(size),material='/Game/StreetLab/Materials/'+target_material)
                    meta = dict(**common, appearance=appearance, clip_id=clip,
                        layout_relation=relation,arrangement_id=clip,frames=12)
                    clips.append(meta)
                    for i, z in enumerate(FRONT_TRAJECTORY):
                        cases.append(dict(**{a:b for a,b in meta.items() if a!='frames'},
                            name=f'{clip}_{i:02d}', pair_id=group,
                            appearance_pair_id=f'{group}_{relation.lower()}_{i:02d}',
                            geometry_pair_id=f'{group}_{appearance}_{i:02d}',
                            frame_in_clip=i,time_s=round(i*DT_S,6),target_name='target',
                            phase='approach' if i<=7 else 'depart',
                            sensor_noise_key=f'{group}_t{i:02d}',
                            camera=dict(x=round(world_x-size[0]/2-z-jitter,9),y=camera_y,z=1.82,
                                        pitch=0.,yaw=0.,roll=0.),
                            objects=[copy.deepcopy(target),copy.deepcopy(bg)]))
    return dict(schema='local-transfer-source-v1',seed=SEED,frames=FRAMES,frames_per_clip=12,
        dt_s=DT_S,groups=groups,clips=clips,cases=cases,profile=copy.deepcopy(PROFILE),
        expected_map_sha256=MAP_SHA,map='/Game/StreetLab/WillowSampleV1',
        calibration=dict(width=640,height=360,hfov_deg=100.,camera_world_height_m=1.82,
                         pitch_deg=0.,yaw_deg=0.,roll_deg=0.,depth_axis='OPTICAL_Z_METRES'),
        sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORY_NOT_REAL_TIME_SENSOR_OR_HUMAN_MOTION',
        visibility_policy='RETAIN_ALL_FRAMES_REPORT_MISSING_OR_OCCLUDED_SUPPORT_SEPARATELY',
        appearance_intervention='Existing opaque backdrop material only; identical target material, geometry, camera and simulated noise seed',
        independence='Eight new procedural base geometries, crossed with two appearances and three lateral placements. '
            'No train/dev or fitting; paired frames are not independent layouts. Same map renderer and sensor assumptions.')


def check_spec(spec, old_specs=()):
    assert spec['schema']=='local-transfer-source-v1' and spec['seed']==SEED
    assert spec['expected_map_sha256']==MAP_SHA and spec['profile']==PROFILE
    assert spec['frames']==FRAMES and len(spec['cases'])==FRAMES
    assert len(spec['groups'])==8 and len(spec['clips'])==CLIPS
    assert len({c['name'] for c in spec['cases']})==FRAMES
    assert {c['split'] for c in spec['cases']}=={'evaluation'}
    signatures={relative_signature(c) for c in spec['cases']}
    assert len(signatures)==8*3*len(set(FRONT_TRAJECTORY))
    for old in old_specs:
        assert not signatures & {relative_signature(c) for c in old['cases']}, 'Old geometry reused'
    by_clip, by_appearance, by_geometry = (defaultdict(list) for _ in range(3))
    for c in spec['cases']:
        by_clip[c['clip_id']].append(c)
        by_appearance[c['appearance_pair_id']].append(c)
        by_geometry[c['geometry_pair_id']].append(c)
        assert c['camera']['z']==1.82 and all(c['camera'][a]==0 for a in ('pitch','yaw','roll'))
        assert c['time_s']==round(c['frame_in_clip']*DT_S,6)
        assert classify(*bounds(c))['truth']==(c['layout_relation']!='OUTSIDE' and c['frame_in_clip'] in range(2,10))
        for o in c['objects']:
            assert o['kind']=='cube' and 'rotation' not in o
    for seq in by_clip.values():
        assert len(seq)==12 and [c['frame_in_clip'] for c in seq]==list(range(12))
        assert all(c['objects']==seq[0]['objects'] for c in seq)
        for c in seq:
            assert abs(float(bounds(c)[0][2])-FRONT_TRAJECTORY[c['frame_in_clip']]-c['trajectory_jitter_m'])<2e-6
    for pair in by_appearance.values():
        assert len(pair)==2 and {c['appearance'] for c in pair}==set(APPEARANCES)
        assert pair[0]['camera']==pair[1]['camera'] and pair[0]['sensor_noise_key']==pair[1]['sensor_noise_key']
        objects=copy.deepcopy([c['objects'] for c in pair])
        assert objects[0][0]['material']==objects[1][0]['material']
        assert objects[0][1]['material']!=objects[1][1]['material']
        for obs in objects:
            for obj in obs: obj.pop('material')
        assert objects[0]==objects[1], 'Appearance changed geometry'
    for triple in by_geometry.values():
        assert len(triple)==3 and {c['layout_relation'] for c in triple}==set(RELATIONS)
        assert all(c['camera']==triple[0]['camera'] and c['sensor_noise_key']==triple[0]['sensor_noise_key'] for c in triple)
        objects=copy.deepcopy([c['objects'] for c in triple])
        for obs in objects: obs[0]['center_m'][1]=0
        assert objects[0]==objects[1]==objects[2], 'Lateral intervention changed another factor'
    counts={a:dict(frames=sum(c['appearance']==a for c in spec['cases']),
        positives=sum(c['appearance']==a and classify(*bounds(c))['truth'] for c in spec['cases'])) for a in APPEARANCES}
    assert all(c==dict(frames=288,positives=128) for c in counts.values())
    assert Counter(g['type_id'] for g in spec['groups'])=={f[0]:2 for f in FAMILIES}
    return dict(status='PASS',frames=FRAMES,clips=CLIPS,base_geometries=8,
        appearance_pairs=len(by_appearance),geometry_triples=len(by_geometry),counts=counts,
        no_old_relative_geometry=True,old_cohorts_compared=len(old_specs),
        scope='Source-only full-extent check; not rendered visibility or model outcome')


def freeze(output):
    from data_coverage_spec import specification as old_coverage
    from query_occupancy_spec import specification as old_query
    from launch_local_transfer import MANDATORY_CODE, MANDATORY_INPUTS, verify_protocol
    repo=Path(__file__).resolve().parents[4]
    output=Path(output).resolve()
    assert output.is_relative_to((repo/'artifacts.local').resolve()) and not output.exists()
    spec=specification()
    check=check_spec(spec,[old_query(),old_coverage()])
    output.mkdir(parents=True)
    write(output/'spec.json',spec);write(output/'source-check.json',check)
    write(output/'protocol.json',dict(schema='local-transfer-capture-protocol-v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),frames=FRAMES,clips=CLIPS,
        capture_timeout_s=900,spec_sha256=sha(output/'spec.json'),source_check_sha256=sha(output/'source-check.json'),
        code_hashes={n:sha(Path(__file__).parent/n) for n in sorted(MANDATORY_CODE)},
        input_hashes={n:sha(repo/n) for n in sorted(MANDATORY_INPUTS)}))
    verify_protocol(output/'protocol.json',output/'spec.json',repo)
    return check


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    print(freeze(p.parse_args().output))
