"""One new complete layout cohort for the frozen Core-only working point."""
from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify


def specification():
    types = [
        ('head_horizontal', 'HEAD', [.18, .68, .15], 1.76),
        ('head_hanging_plane', 'HEAD', [.16, .59, .24], 1.79),
        ('head_protruding_edge', 'HEAD', [.29, .51, .19], 1.75),
        ('body_protruding_plane', 'BODY', [.18, .62, .51], 1.34),
        ('body_suspended_solid', 'BODY', [.33, .53, .43], 1.28),
        ('body_large_solid', 'BODY', [.47, .65, .59], 1.39),
    ]
    materials = ['Paint', 'Sage', 'Cream', 'Terracotta', 'Charcoal', 'Plaster']
    clips, cases = [], []
    for bg in range(2):
        for ti, (name, layer, base_size, height) in enumerate(types):
            for ri, relation in enumerate(('INSIDE', 'BOUNDARY', 'OUTSIDE')):
                n = len(clips)
                clip = f'workpoint_b{bg}_{name}_{relation.lower()}'
                size = [round(s*(1+.017*ri+.043*bg), 6) for s in base_size]
                side = 1 if (ti+bg) % 2 else -1
                lateral = (side*.06 if ti % 2 else side*(.3+size[1]/2-(.11+.04*bg))) if ri == 0 else side*(.3+size[1]/2+(0 if ri == 1 else .065+.015*bg))
                world_x, camera_y = 31.2+.037*n, .045-.003*n
                target = dict(name='target', kind='cube', center_m=[world_x, camera_y+lateral, height-.006*ri],
                              size_m=size, material='/Game/StreetLab/Materials/'+materials[(ti+2*ri+bg+2)%6])
                background = dict(name='background', kind='cube', center_m=[world_x+3.65+.9*bg+.03*ti,camera_y-.18*bg,2.08],
                                  size_m=[.24,5.4+.5*bg,4.1], material='/Game/StreetLab/Materials/'+('Wood' if bg == 0 else 'Brick'))
                meta = dict(clip_id=clip, type_id=name, layer=layer, background=f'background_{bg}',
                            layout_relation=relation, arrangement_id=f'workpoint_new_{n:02d}', frames=12)
                clips.append(meta)
                for i in range(12):
                    front = round(3.55+.017*ti-.117*i, 6)
                    cases.append(dict(**{k:v for k,v in meta.items() if k != 'frames'}, name=f'{clip}_{i:02d}',
                        pair_id=clip, frame_in_clip=i, time_s=round(.2*i, 6), target_name='target',
                        camera=dict(x=world_x-size[0]/2-front, y=camera_y, z=1.82, pitch=0.,yaw=0.,roll=0.),
                        objects=[target, background]))
    return dict(schema='core-workpoint-transfer-v1',frames=432,clips=clips,cases=cases,
        expected_map_sha256=MAP_SHA,map='/Game/StreetLab/WillowSampleV1',profile=PROFILE,
        sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORIES_NOT_REAL_TIME_DYNAMIC_CAPTURE',
        independence='New dimensions, object heights, signed layouts, full camera paths and backdrop geometry; shared six cuboid types, map, materials, renderer and sensor proxy. Not a new-world, hardware or natural-distribution test.')


def check_spec(spec, old):
    assert len(spec['cases']) == 432 and len(spec['clips']) == 36
    assert not {r['name'] for r in spec['cases']} & {r['name'] for r in old['cases']}
    signatures = lambda s: {(tuple(r['objects'][0]['size_m']), tuple(r['objects'][0]['center_m']),
                            tuple(r['camera'][k] for k in ('x','y','z'))) for r in s['cases']}
    assert not signatures(spec) & signatures(old)
    for clip in spec['clips']:
        rows = [r for r in spec['cases'] if r['clip_id'] == clip['clip_id']]
        labels = [classify(*bounds(r)) for r in rows]
        assert len(rows) == 12 and not labels[0]['truth']
        assert labels[-1]['truth'] == (clip['layout_relation'] != 'OUTSIDE')
        if clip['layout_relation'] == 'BOUNDARY':
            assert all(x['boundary'] for x in labels if x['truth'])
    return {'frames':432,'clips':36,'no_exact_source_signature_overlap':True,
            'complete_preentry_and_positive_or_outside_clips':True}
