"""Fixed new-instance generator and evaluator-only slow-drift observation law."""
from dataclasses import asdict
import math
import random

import active_view as av
import bent_path_observation as bent

SEED = 2026092201
FAMILIES = ('range', 'pose', 'combined')
SHAPES = ('constant_plus', 'constant_minus', 'drift_up', 'drift_down')
CONDITIONS = ('nominal',) + tuple(f'{f}_{s}' for f in FAMILIES for s in SHAPES)
PATH = bent.paths()['x_then_z']


def scene(row):
    return av.Scene(tuple(av.Box(**b) for b in row['boxes']), row['wall_z'])


def geometry_key(value):
    return av.canonical(dict(boxes=value['boxes'], wall_z=value['wall_z'])).decode()


def generate(excluded=(), statistics=None):
    """Select by geometry/truth only; never consult inference or sensor returns."""
    rng = random.Random(SEED)
    seen = set(excluded)
    rows = []
    stats = dict(seed=SEED, rng='Python random.Random / MT19937', general_draws=0,
                 general_quota_rejections=0, duplicate_rejections=0, boundary_draws=0)

    def candidate(x, z, width):
        s = av.Scene((av.Box(round(x, 9), round(z, 9), round(width, 9)),))
        return dict(boxes=[asdict(b) for b in s.boxes], wall_z=s.wall_z,
                    truth=av.intersects_query(s))

    def add(row, stratum, pair=None):
        key = geometry_key(row)
        if key in seen:
            stats['duplicate_rejections'] += 1
            return False
        seen.add(key)
        rows.append(dict(id=f'drift_{len(rows):03d}', stratum=stratum, pair=pair, **row))
        return True

    counts = {True: 0, False: 0}
    for _ in range(100000):
        if all(n == 60 for n in counts.values()):
            break
        row = candidate(rng.uniform(-.9, .9), rng.uniform(.6, 3.4), rng.uniform(.03, .60))
        stats['general_draws'] += 1
        if counts[row['truth']] < 60 and add(row, 'general'):
            counts[row['truth']] += 1
        elif counts[row['truth']] == 60:
            stats['general_quota_rejections'] += 1
    else:
        raise RuntimeError('Fixed generation attempt cap exhausted')
    for face in ('left', 'right', 'far'):
        for i in range(10):
            for _ in range(1000):
                stats['boundary_draws'] += 1
                width = round(rng.uniform(.06, .50), 9)
                margin = rng.uniform(.0001, .001)
                fixed = rng.uniform(-.2, .2) if face == 'far' else rng.uniform(.8, 2.8)
                if face == 'left':
                    pair = [candidate(-.3-width/2+sign*margin, fixed, width) for sign in (1, -1)]
                elif face == 'right':
                    pair = [candidate(.3+width/2-sign*margin, fixed, width) for sign in (1, -1)]
                else:
                    pair = [candidate(fixed, 3.02-sign*margin, width) for sign in (1, -1)]
                assert [r['truth'] for r in pair] == [True, False]
                if all(geometry_key(r) not in seen for r in pair):
                    for row in pair:
                        assert add(row, f'boundary_{face}', f'{face}_{i:02d}')
                    break
                stats['duplicate_rejections'] += 1
            else:
                raise RuntimeError('Fixed boundary generation cap exhausted')
    assert len(rows) == 180 and sum(r['truth'] for r in rows) == 90
    if statistics is not None:
        statistics.update(stats)
    return rows


def injection(condition, index):
    if condition not in CONDITIONS or type(index) is not int or not 0 <= index <= 12:
        raise ValueError('Expected frozen condition and view index0..12')
    if condition == 'nominal':
        return dict(range_m=0., pose_x_m=0., pose_z_m=0.)
    family, shape = condition.split('_', 1)
    a = dict(constant_plus=1., constant_minus=-1., drift_up=-1+index/6,
             drift_down=1-index/6)[shape]
    return dict(range_m=a*.002 if family in ('range', 'combined') else 0.,
                pose_x_m=a*.001 if family in ('pose', 'combined') else 0.,
                pose_z_m=a*.001 if family in ('pose', 'combined') else 0.)


def observe(value, condition):
    views = []
    for i, nominal in enumerate(PATH):
        bias = injection(condition, i)
        actual = (round(nominal[0]+bias['pose_x_m'], 12),
                  round(nominal[1]+bias['pose_z_m'], 12))
        raw = [min((value.wall_z-actual[1])/math.cos(a),
                   min((av.hit_distance(b, actual, a) for b in value.boxes), default=math.inf))
               for a in av.ANGLES]
        measured = [r+bias['range_m'] for r in raw]
        if any(not math.isfinite(r) or r <= 0 for r in measured):
            raise ValueError('Invalid simulated radial range')
        bins = [math.floor(r/av.RANGE_STEP+.5) for r in measured]
        if condition == 'nominal':
            assert bins == list(av.observe(value, nominal))
        views.append(dict(camera=list(nominal), bins=bins, actual_camera=list(actual),
                          raw_ranges=raw, biased_ranges=measured, bias=bias))
    return views


def public(views):
    return [dict(camera=list(v['camera']), bins=list(v['bins'])) for v in views]
