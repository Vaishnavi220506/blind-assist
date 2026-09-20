"""One new complete-layout cohort for the frozen strong-plus-hold policy."""
from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify


def specification():
    types = [
        ('head_horizontal', 'HEAD', [.21, .73, .17], 1.74),
        ('head_hanging_plane', 'HEAD', [.19, .56, .27], 1.77),
        ('head_protruding_edge', 'HEAD', [.26, .57, .21], 1.72),
        ('body_protruding_plane', 'BODY', [.22, .69, .48], 1.31),
        ('body_suspended_solid', 'BODY', [.36, .58, .46], 1.25),
        ('body_large_solid', 'BODY', [.43, .71, .63], 1.36),
    ]
    materials = ['Cream', 'Terracotta', 'Paint', 'Plaster', 'Sage', 'Charcoal']
    clips, cases = [], []
    for bg in range(2):
        for ti, (name, layer, base_size, height) in enumerate(types):
            for ri, relation in enumerate(('INSIDE', 'BOUNDARY', 'OUTSIDE')):
                n = len(clips)
                clip = f'holdcheck_b{bg}_{name}_{relation.lower()}'
                size = [round(s*(1+.021*ri+.037*bg), 6) for s in base_size]
                side = -1 if (ti+bg) % 2 else 1
                lateral = (side*.045 if ti % 2 else side*(.3+size[1]/2-(.125+.02*bg))) if ri == 0 else side*(.3+size[1]/2+(0 if ri == 1 else .075+.01*bg))
                world_x, camera_y = 32.8+.031*n, -.035+.002*n
                target = dict(name='target', kind='cube', center_m=[world_x, camera_y+lateral, height+.005*ri],
                              size_m=size, material='/Game/StreetLab/Materials/'+materials[(ti+ri+2*bg)%6])
                background = dict(name='background', kind='cube', center_m=[world_x+3.9+.55*bg+.027*ti,camera_y+.12-.2*bg,2.04],
                                  size_m=[.27,5.7+.3*bg,4.2], material='/Game/StreetLab/Materials/'+('Brick' if bg == 0 else 'Wood'))
                meta = dict(clip_id=clip, type_id=name, layer=layer, background=f'background_{bg}',
                            layout_relation=relation, arrangement_id=f'holdcheck_new_{n:02d}', frames=12)
                clips.append(meta)
                for i in range(12):
                    front = round(3.57+.013*ti-(.112+.004*(ti%3))*i, 6)
                    cases.append(dict(**{k:v for k,v in meta.items() if k != 'frames'}, name=f'{clip}_{i:02d}',
                        pair_id=clip, frame_in_clip=i, time_s=round(.2*i, 6), target_name='target',
                        camera=dict(x=world_x-size[0]/2-front, y=camera_y, z=1.82, pitch=0.,yaw=0.,roll=0.),
                        objects=[target, background]))
    return dict(schema='core-hold-validation-v1', frames=432, clips=clips, cases=cases,
        expected_map_sha256=MAP_SHA, map='/Game/StreetLab/WillowSampleV1', profile=PROFILE,
        sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORIES_NOT_REAL_TIME_DYNAMIC_CAPTURE',
        independence='New dimensions, heights, placements, approach trajectories, material assignments and background geometries versus both consumed Core cohorts. Shared cuboid families, map, assets, renderer and original single-return proxy. No natural, hardware, new-world or end-of-event release claim.')


def check_spec(spec, old_specs):
    assert len(spec['cases']) == 432 and len(spec['clips']) == 36
    signatures = lambda s: {(tuple(r['objects'][0]['size_m']), tuple(r['objects'][0]['center_m']),
                            tuple(r['camera'][k] for k in ('x','y','z'))) for r in s['cases']}
    relative = lambda s: {tuple((tuple(o['size_m']), tuple(round(o['center_m'][j]-r['camera'][k],6)
        for j,k in enumerate(('x','y','z')))) for o in r['objects']) for r in s['cases']}
    for old in old_specs:
        assert not {r['name'] for r in spec['cases']} & {r['name'] for r in old['cases']}
        assert not signatures(spec) & signatures(old)
        assert not relative(spec) & relative(old)
    for clip in spec['clips']:
        rows = [r for r in spec['cases'] if r['clip_id'] == clip['clip_id']]
        labels = [classify(*bounds(r)) for r in rows]
        assert len(rows)==12 and not labels[0]['truth']
        assert labels[-1]['truth']==(clip['layout_relation']!='OUTSIDE')
        if clip['layout_relation']=='BOUNDARY':
            assert all(x['boundary'] for x in labels if x['truth'])
    counts={}
    for relation in ('INSIDE','BOUNDARY','OUTSIDE'):
        labels=[classify(*bounds(r)) for r in spec['cases'] if r['layout_relation']==relation]
        counts[relation]=dict(frames=len(labels),positive=sum(r['truth'] for r in labels))
    return dict(frames=432, clips=36, no_exact_source_overlap=True, no_relative_geometry_overlap=True, counts=counts)
