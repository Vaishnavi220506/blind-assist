"""24 new matched layouts with complete approach/dwell/depart trajectories."""
from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify


def specification():
    types = [
        ('head_horizontal', 'HEAD', [.183, .807, .153], 1.751),
        ('head_hanging_plane', 'HEAD', [.167, .613, .313], 1.763),
        ('body_protruding_plane', 'BODY', [.247, .653, .527], 1.337),
        ('body_suspended_solid', 'BODY', [.397, .547, .437], 1.287),
    ]
    materials = ['Sage', 'Cream', 'Terracotta', 'Charcoal']
    clips, cases = [], []
    for bg in range(2):
        for ti, (kind, layer, size, height) in enumerate(types):
            for ri, relation in enumerate(('INSIDE', 'BOUNDARY', 'OUTSIDE')):
                n = len(clips)
                clip = f'full_event_b{bg}_{kind}_{relation.lower()}'
                side = -1 if ti % 2 else 1
                lateral = (side * (.3 + size[1]/2 - .145) if ti % 2 == 0 else side*.037) if ri == 0 else side*(.3 + size[1]/2 + (0 if ri == 1 else .105))
                world_x, camera_y = 31.4 + .029*n, -.045 + .003*n
                target = dict(name='target', kind='cube', center_m=[world_x, camera_y+lateral, height],
                              size_m=size, material='/Game/StreetLab/Materials/'+materials[ti])
                background = dict(name='background', kind='cube',
                    center_m=[world_x+3.53+.81*bg, camera_y+(.19 if bg == 0 else -.23), 2.12],
                    size_m=[.237, 6.13+.41*bg, 4.37],
                    material='/Game/StreetLab/Materials/'+('Brick' if bg == 0 else 'Wood'))
                meta = dict(clip_id=clip, type_id=kind, layer=layer, background=f'background_{bg}',
                            layout_relation=relation, arrangement_id=f'full_event_new_{n:02d}', frames=24)
                clips.append(meta)
                near = 2.39 + .011*ti
                for i in range(24):
                    phase = 'approach' if i <= 8 else 'dwell' if i <= 13 else 'depart'
                    front = near + .165*(8-i) if i <= 8 else near if i <= 13 else near + .19*(i-13)
                    cases.append(dict(**{k:v for k,v in meta.items() if k != 'frames'},
                        name=f'{clip}_{i:02d}', pair_id=clip, phase=phase,
                        frame_in_clip=i, time_s=round(.2*i, 6), target_name='target',
                        camera=dict(x=round(world_x-size[0]/2-front, 9), y=camera_y, z=1.82,
                                    pitch=0., yaw=0., roll=0.), objects=[target, background]))
    return dict(schema='full-event-transfer-v1', frames=576, clips=clips, cases=cases,
        expected_map_sha256=MAP_SHA, map='/Game/StreetLab/WillowSampleV1', profile=PROFILE,
        sampling='POSED_QUASI_STATIC_APPROACH_DWELL_RETREAT_NOT_REAL_TIME_SENSOR_OR_HUMAN_MOTION',
        independence='New relative dimensions/poses versus all three previous Core cohorts; four shared cuboid families, same Willow renderer and hypothetical sensor law. Lateral comparisons keep object size/material/trajectory; background pairs change background only in camera-relative geometry. Not natural scenes or physical detector validation.')


def relative_signature(case):
    return tuple((tuple(o['size_m']), tuple(round(o['center_m'][j]-case['camera'][k], 6)
        for j,k in enumerate(('x','y','z')))) for o in case['objects'])


def check_spec(spec, old_specs):
    assert len(spec['cases']) == 576 and len(spec['clips']) == 24
    now = {relative_signature(c) for c in spec['cases']}
    for old in old_specs:
        assert not now & {relative_signature(c) for c in old['cases']}, 'Old relative geometry reused'
        assert not {c['name'] for c in spec['cases']} & {c['name'] for c in old['cases']}
    counts = {}
    for clip in spec['clips']:
        seq = [c for c in spec['cases'] if c['clip_id'] == clip['clip_id']]
        labels = [classify(*bounds(c)) for c in seq]
        assert len(seq) == 24 and [c['frame_in_clip'] for c in seq] == list(range(24))
        assert not labels[0]['truth'] and not labels[-1]['truth']
        positive = [i for i,l in enumerate(labels) if l['truth']]
        if clip['layout_relation'] == 'OUTSIDE':
            assert not positive
        else:
            assert positive and positive == list(range(min(positive), max(positive)+1))
            assert max(positive) <= 18 and min(positive) >= 3
            assert all(labels[i]['truth'] for i in range(9,14))
            if clip['layout_relation'] == 'BOUNDARY':
                assert all(labels[i]['boundary'] for i in positive)
    for relation in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
        labels = [classify(*bounds(c)) for c in spec['cases'] if c['layout_relation'] == relation]
        counts[relation] = dict(frames=len(labels), positive=sum(l['truth'] for l in labels))
    return dict(status='PASS', clips=24, frames=576, no_relative_overlap=True,
                compared_old_cohorts=len(old_specs), counts=counts,
                motion_scope='Back away along optical axis; not passing through/past obstacle')
