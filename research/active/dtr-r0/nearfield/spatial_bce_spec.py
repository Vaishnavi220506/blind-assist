"""Evaluator-only grouped source for the one spatial BCE comparison.

No labels, group/split identifiers, geometry or noise keys are model features.
The shared noise key specifies a common random start, not equal sensor returns.
"""
from collections import Counter, defaultdict
import copy
import random

from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify

SEED = 920241
FRAMES = 2880
RELATIONS = ('INSIDE', 'BOUNDARY', 'OUTSIDE')
FAMILIES = (
    ('head_horizontal', 'HEAD', (.19, .79, .16), 1.755),
    ('head_hanging_plane', 'HEAD', (.17, .61, .30), 1.765),
    ('body_protruding_plane', 'BODY', (.25, .65, .53), 1.335),
    ('body_suspended_solid', 'BODY', (.40, .55, .44), 1.285),
)


def relative_signature(case):
    return tuple((tuple(o['size_m']), tuple(round(o['center_m'][j]-case['camera'][k], 6)
        for j, k in enumerate(('x', 'y', 'z')))) for o in case['objects'])


def specification():
    groups, clips, cases, near_far_pairs = [], [], [], []
    materials = ('Sage', 'Cream', 'Terracotta', 'Charcoal', 'Paint', 'Plaster')
    backgrounds = ('Brick', 'Wood', 'Plaster', 'Cream')
    for fi, (kind, layer, base_size, base_height) in enumerate(FAMILIES):
        order = list(range(10))
        random.Random(SEED+fi).shuffle(order)
        partitions = {k: 'train' if i < 6 else 'dev' if i < 8 else 'test'
                      for i, k in enumerate(order)}
        for k in range(10):
            rng = random.Random(SEED+1009*fi+67*k)
            group = f'spatial_bce_{kind}_g{k:02d}'
            split = partitions[k]
            size = [round(v*rng.uniform(.86, 1.14), 6) for v in base_size]
            height = round(base_height+rng.uniform(-.035, .035), 6)
            side = (-1, 1)[rng.randrange(2)]
            intrusion = round(rng.uniform(.006, .017), 6)
            world_x = round(31.7+rng.uniform(0, .8), 6)
            camera_y = round(rng.uniform(-.12, .12), 6)
            near = round(rng.uniform(2.27, 2.61), 6)
            approach = round(rng.uniform(.172, .193), 6)
            depart = round(rng.uniform(.191, .214), 6)
            front = [round(near+approach*(8-i) if i <= 8 else near if i <= 13
                           else near+depart*(i-13), 6) for i in range(24)]
            target_material = '/Game/StreetLab/Materials/'+materials[rng.randrange(len(materials))]
            background = dict(name='background', kind='cube',
                center_m=[round(world_x+rng.uniform(3.7, 4.9), 6),
                          round(camera_y+rng.uniform(-.33, .33), 6), round(rng.uniform(2.08, 2.18), 6)],
                size_m=[round(rng.uniform(.22, .31), 6), round(rng.uniform(6.2, 7.3), 6),
                        round(rng.uniform(4.3, 4.7), 6)],
                material='/Game/StreetLab/Materials/'+backgrounds[rng.randrange(len(backgrounds))])
            group_meta = dict(base_group_id=group, split=split, type_id=kind, layer=layer,
                              background='background_'+group, boundary_intrusion_m=intrusion)
            groups.append(dict(**group_meta, clips=[]))
            inside = rng.uniform(.105, .18)
            outside = rng.uniform(.065, .15)
            for relation in RELATIONS:
                lateral = side*(.3+size[1]/2+(-inside if relation == 'INSIDE' else
                                            -intrusion if relation == 'BOUNDARY' else outside))
                clip_id = group+'_'+relation.lower()
                groups[-1]['clips'].append(clip_id)
                target = dict(name='target', kind='cube',
                    center_m=[world_x, round(camera_y+lateral, 9), height],
                    size_m=list(size), material=target_material)
                meta = dict(**group_meta, clip_id=clip_id, layout_relation=relation,
                            arrangement_id=clip_id, frames=24)
                clips.append(meta)
                for i in range(24):
                    cases.append(dict(**{key: value for key, value in meta.items() if key != 'frames'},
                        name=f'{clip_id}_{i:02d}', pair_id=group, frame_in_clip=i,
                        time_s=round(.2*i, 6), target_name='target',
                        phase='approach' if i <= 8 else 'dwell' if i <= 13 else 'depart',
                        sensor_noise_key=f'{group}_t{i:02d}',
                        camera=dict(x=round(world_x-size[0]/2-front[i], 9), y=camera_y,
                                    z=1.82, pitch=0., yaw=0., roll=0.),
                        objects=[copy.deepcopy(target), copy.deepcopy(background)]))
                near_far_pairs.append(dict(base_group_id=group, split=split, clip_id=clip_id,
                    near_frame_indices=[8, 13], far_frame_indices=[0, 23],
                    authority='SOURCE_GEOMETRY_DIAGNOSTIC_ONLY_NOT_PAIR_TRAINING'))
    return dict(schema='spatial-bce-source-v1', seed=SEED, frames=FRAMES, groups=groups,
        clips=clips, cases=cases, near_far_pairs=near_far_pairs, profile=copy.deepcopy(PROFILE),
        expected_map_sha256=MAP_SHA, map='/Game/StreetLab/WillowSampleV1',
        sampling='POSED_QUASI_STATIC_APPROACH_DWELL_RETREAT_NOT_REAL_TIME_SENSOR_OR_HUMAN_MOTION',
        independence='40 new procedural base groups; whole groups split 24/8/8. Lateral variants share target dimensions/material, background, camera trajectory and noise keys. New relative geometry versus four prior cohorts; same primitive families, Willow renderer and hypothetical sensor law. No natural-source or hardware claim.')


def check_spec(spec, old_specs):
    assert spec['frames'] == FRAMES and len(spec['cases']) == FRAMES
    assert len(spec['groups']) == 40 and len(spec['clips']) == 120
    assert spec['profile'] == PROFILE
    assert len({c['name'] for c in spec['cases']}) == FRAMES
    now = {relative_signature(c) for c in spec['cases']}
    for old in old_specs:
        assert not now & {relative_signature(c) for c in old['cases']}, 'Old relative geometry reused'
        assert not {c['name'] for c in spec['cases']} & {c['name'] for c in old['cases']}
    by_clip, by_group, signatures = defaultdict(list), defaultdict(list), defaultdict(set)
    for case in spec['cases']:
        by_clip[case['clip_id']].append(case)
        by_group[case['base_group_id']].append(case)
        signatures[case['split']].add(relative_signature(case))
    assert set(signatures) == {'train', 'dev', 'test'}
    assert all(not signatures[a] & signatures[b] for a, b in
               (('train', 'dev'), ('train', 'test'), ('dev', 'test')))
    assert Counter(g['split'] for g in spec['groups']) == dict(train=24, dev=8, test=8)
    for family, *_ in FAMILIES:
        assert Counter(g['split'] for g in spec['groups'] if g['type_id'] == family) == dict(train=6, dev=2, test=2)
    for group in spec['groups']:
        rows = by_group[group['base_group_id']]
        assert len(rows) == 72 and {c['split'] for c in rows} == {group['split']}
        assert {c['clip_id'] for c in rows} == set(group['clips'])
        for i in range(24):
            paired = [c for c in rows if c['frame_in_clip'] == i]
            assert len(paired) == 3 and {c['layout_relation'] for c in paired} == set(RELATIONS)
            assert all(c['camera'] == paired[0]['camera'] and c['objects'][1] == paired[0]['objects'][1]
                       and c['sensor_noise_key'] == paired[0]['sensor_noise_key'] for c in paired)
            targets = [copy.deepcopy(c['objects'][0]) for c in paired]
            for target in targets:
                target['center_m'][1] = 0
            assert targets[0] == targets[1] == targets[2], 'Lateral pair changed target appearance/shape/depth'
    counts = {}
    for clip in spec['clips']:
        seq = by_clip[clip['clip_id']]
        assert len(seq) == 24 and [c['frame_in_clip'] for c in seq] == list(range(24))
        assert all(c['objects'] == seq[0]['objects'] for c in seq)
        assert all(c['time_s'] == round(.2*i, 6) for i, c in enumerate(seq))
        labels = [classify(*bounds(c)) for c in seq]
        assert not labels[0]['truth'] and not labels[-1]['truth']
        positive = [i for i, label in enumerate(labels) if label['truth']]
        if clip['layout_relation'] == 'OUTSIDE':
            assert not positive
        else:
            assert positive == list(range(min(positive), max(positive)+1))
            assert min(positive) >= 3 and max(positive) <= 18
            assert all(labels[i]['truth'] for i in range(8, 14))
            if clip['layout_relation'] == 'BOUNDARY':
                assert all(0 <= labels[i]['signed_boundary_margin_m'] <= .02 for i in positive)
                assert all(labels[i]['boundary'] for i in positive)
    assert len(spec['near_far_pairs']) == 120
    for pair in spec['near_far_pairs']:
        seq = by_clip[pair['clip_id']]
        assert pair['split'] == seq[0]['split'] and pair['base_group_id'] == seq[0]['base_group_id']
        assert all(bounds(seq[i])[0][2] < 3 for i in pair['near_frame_indices'])
        assert all(bounds(seq[i])[0][2] > 3 for i in pair['far_frame_indices'])
    for split in ('train', 'dev', 'test'):
        counts[split] = {}
        for relation in RELATIONS:
            labels = [classify(*bounds(c)) for c in spec['cases']
                      if c['split'] == split and c['layout_relation'] == relation]
            counts[split][relation] = dict(frames=len(labels), positive=sum(l['truth'] for l in labels))
    return dict(status='PASS', groups=40, clips=120, frames=FRAMES, counts=counts,
        no_relative_overlap=True, compared_old_cohorts=len(old_specs), split_unit='BASE_GROUP',
        lateral_pair_audit='Only target lateral position changes; common sensor random starts',
        motion_scope='Back away along optical axis; not passing through/past obstacle')
