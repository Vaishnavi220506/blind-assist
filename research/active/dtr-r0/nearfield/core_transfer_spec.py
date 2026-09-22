"""Fixed new controlled arrangements; no outcome-dependent scene selection."""
import numpy as np
from ba_camera_corridor_spec import MAP_SHA

TOLERANCE = .02
PROFILE = dict(volume=dict(x=[-.3, .3], y=[-.2, .9], z=[.3, 3.]), boundary_band_m=TOLERANCE)


def bounds(case):
    obj = next(o for o in case['objects'] if o['name'] == 'target')
    p, c, s = case['camera'], obj['center_m'], obj['size_m']
    center = np.array([c[1]-p['y'], p['z']-c[2], c[0]-p['x']])
    half = np.array([s[1], s[2], s[0]])/2
    return center-half, center+half


def classify(lower, upper):
    volume = np.array([PROFILE['volume'][a] for a in ('x', 'y', 'z')])
    margins = np.minimum(np.asarray(upper)-volume[:, 0], volume[:, 1]-np.asarray(lower))
    margin = float(margins.min())
    relation = 'BOUNDARY' if abs(margin) <= TOLERANCE+1e-9 else 'INSIDE' if margin > 0 else 'OUTSIDE'
    return dict(truth=margin >= -1e-9, boundary=relation == 'BOUNDARY', relation=relation,
                signed_boundary_margin_m=margin, axis_penetration_m=margins.tolist(),
                target_camera_bounds_m=dict(lower=list(lower), upper=list(upper)))


def specification():
    # World dimensions are depth, lateral width, height. All are opaque cubes:
    # geometric representatives, not semantic object recognition or realism.
    types = [
        ('head_horizontal', 'HEAD', [.14, .74, .12], 1.78),
        ('head_hanging_plane', 'HEAD', [.12, .54, .28], 1.77),
        ('head_protruding_edge', 'HEAD', [.34, .46, .16], 1.79),
        ('body_protruding_plane', 'BODY', [.14, .56, .56], 1.35),
        ('body_suspended_solid', 'BODY', [.38, .48, .38], 1.30),
        ('body_large_solid', 'BODY', [.42, .60, .64], 1.37),
    ]
    materials = ['Paint', 'Sage', 'Cream', 'Terracotta', 'Charcoal', 'Plaster']
    clips, cases = [], []
    for bg in range(2):
        for ti, (name, layer, base_size, height) in enumerate(types):
            for ri, relation in enumerate(('INSIDE', 'BOUNDARY', 'OUTSIDE')):
                n = len(clips)
                clip = f'core_b{bg}_{name}_{relation.lower()}'
                # Each cell is newly spawned with a distinct size, target pose,
                # material assignment and camera path, including every OUTSIDE.
                size = [round(s*(1+.025*ri+.035*bg), 6) for s in base_size]
                side = -1 if (ti+bg) % 2 else 1
                lateral = (0.025*side if ti % 2 else side*(.3+size[1]/2-.14)) if ri == 0 else side*(.3+size[1]/2+(0 if ri == 1 else .08))
                world_x, camera_y = 30.+.04*n, -.08+.004*n
                obj = dict(name='target', kind='cube', center_m=[world_x, camera_y+lateral, height+.008*ri],
                           size_m=size, material='/Game/StreetLab/Materials/'+materials[(ti+ri+bg)%6])
                background = dict(name='background', kind='cube', center_m=[world_x+4.0+bg*.7,camera_y+.25*bg,2.12],
                                  size_m=[.2, 5.+bg, 4.], material='/Game/StreetLab/Materials/'+('Brick' if bg == 0 else 'Wood'))
                meta = dict(clip_id=clip, type_id=name, layer=layer, background=f'background_{bg}',
                            layout_relation=relation, arrangement_id=f'new_arrangement_{n:02d}', frames=12)
                clips.append(meta)
                for i in range(12):
                    front = round(3.62+.01*(ti%3)-.12*i, 6)
                    cases.append(dict(**{k:v for k,v in meta.items() if k != 'frames'}, name=f'{clip}_{i:02d}',
                        pair_id=clip, frame_in_clip=i, time_s=round(.2*i, 6), target_name='target',
                        camera=dict(x=world_x-size[0]/2-front, y=camera_y, z=1.82, pitch=0.,yaw=0.,roll=0.),
                        objects=[obj, background]))
    return dict(schema='core-transfer-v1', frames=432, clips=clips, cases=cases,
                expected_map_sha256=MAP_SHA, map='/Game/StreetLab/WillowSampleV1', profile=PROFILE,
                sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORIES_NOT_REAL_TIME_DYNAMIC_CAPTURE',
                independence='New opaque target instances, sizes, materials, relative paths and full-extent layouts; new Brick/Wood backdrop configurations. Same Willow map, primitive mesh library, renderer and sensor proxy as Development; not new-world or natural-source independence.')
