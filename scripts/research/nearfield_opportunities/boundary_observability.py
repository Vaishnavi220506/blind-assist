"""Fixed angular/precision information probe and lateral-coordinate witnesses."""
from dataclasses import asdict, replace
import math

import active_view as av
import bent_path_observation as bent

PATH = bent.paths()['x_then_z']
SPACING = math.radians(45/8)
STEPS = {'coarse': .10, 'fine': .001}
RAW_TOL = 1e-10


def schedules():
    return dict(fixed=tuple(0. for _ in PATH),
                sweep_plus=tuple(i/12*SPACING for i in range(13)),
                sweep_minus=tuple(-i/12*SPACING for i in range(13)))


def scene(row):
    return av.Scene(tuple(av.Box(**b) for b in row['boxes']), row['wall_z'])


def capture(value, schedule, pose_x=0.):
    if schedule not in schedules() or not math.isfinite(pose_x):
        raise ValueError('Expected fixed schedule and finite camera offset')
    views = []
    for nominal, yaw in zip(PATH, schedules()[schedule], strict=True):
        actual = (round(nominal[0]+pose_x, 12), round(nominal[1], 12))
        raw, hits = [], []
        for local in av.ANGLES:
            angle = local+yaw
            wall = (value.wall_z-actual[1])/math.cos(angle)
            obstacle = min((av.hit_distance(b, actual, angle) for b in value.boxes), default=math.inf)
            radial = min(wall, obstacle)
            if not math.isfinite(radial) or radial <= 0:
                raise ValueError('Invalid forward radial observation')
            raw.append(radial)
            hits.append(obstacle < wall)
        bins = {name: [math.floor(r/step+.5) for r in raw] for name, step in STEPS.items()}
        views.append(dict(camera=list(nominal), actual_camera=list(actual), yaw_rad=yaw,
                          raw_ranges=raw, target_hits=hits, bins=bins))
    return views


def signature(views, precision):
    return tuple(b for view in views for b in view['bins'][precision])


def compare(a, b):
    raw = [abs(x-y) for va, vb in zip(a, b, strict=True)
           for x, y in zip(va['raw_ranges'], vb['raw_ranges'], strict=True)]
    hits = sum(x != y for va, vb in zip(a, b, strict=True)
               for x, y in zip(va['target_hits'], vb['target_hits'], strict=True))
    return dict(max_raw_difference_m=max(raw), raw_distinct=max(raw) > RAW_TOL,
                hit_mismatch_rays=hits, changed_bins={p: sum(x != y for x, y in
                zip(signature(a, p), signature(b, p), strict=True)) for p in STEPS})


def lateral_counterexample(row):
    """Evaluator-only alternative, using fixed maximum legal X translation."""
    if row['stratum'] not in ('boundary_left', 'boundary_right'):
        raise ValueError('Lateral argument does not cover other strata')
    original = scene(row)
    truth = av.intersects_query(original)
    if row['stratum'] == 'boundary_right':
        delta = .001 if truth else -.001
    else:
        delta = -.001 if truth else .001
    alternate = av.Scene(tuple(replace(b, x=round(b.x+delta, 12)) for b in original.boxes), original.wall_z)
    domain = all(-.9 <= b.x <= .9 and .6 <= b.z <= 3.4 and .03 <= b.width <= .6 and b.thickness == .04
                 for b in alternate.boxes)
    return dict(id=row['id'], original_truth=truth, alternative_truth=av.intersects_query(alternate),
                alternative=dict(boxes=[asdict(b) for b in alternate.boxes], wall_z=alternate.wall_z),
                bias=dict(range_m=0., pose_x_m=delta, pose_z_m=0.), domain_valid=domain,
                opposite_label=truth != av.intersects_query(alternate),
                authority='EVALUATOR_CONSTRUCTED_NONIDENTIFIABILITY_WITNESS_NOT_INFERENCE')
