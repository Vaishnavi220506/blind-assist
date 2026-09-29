"""Unseen-structure corridor scenes: same target/margins as the four known families.

Known families come from the unchanged make_scenes(unit). Novel families use a
separate RNG stream, so known-family scenes are byte-identical to the original
generator. Context structures never change final central labels (asserted).
"""
import numpy as np
import cnh_proposal_attribution_scenes as S
from cnh_corridor_labels import labels_for_all

NOVEL = ('pole', 'railing', 'low_beam', 'opposite_wall', 'bollards', 'vehicle')
FLOOR = dict(lo=[-8., 1.65, -8.], hi=[8., 1.80, 9.], rho=.45)
BACK = dict(lo=[-8., -3., 4.5], hi=[8., 2., 4.7], rho=.35)
_KNOWN = S.make_scenes  # bound at import so callers may patch S.make_scenes


def _x(a, b, side):
    return (a, b) if side > 0 else (-b, -a)


def context(family, side, target_z, rng):
    boxes = []
    if family == 'pole':
        gap = float(rng.uniform(.08, .18)); width = float(rng.uniform(.08, .12))
        a, b = _x(.30 + gap, .30 + gap + width, side)
        z = target_z + float(rng.uniform(-.3, .3))
        boxes.append(dict(lo=[a, -.30, z], hi=[b, 1.65, z + .10], rho=.50))
    elif family == 'railing':
        a, b = _x(.42, .46, side)
        for z in np.arange(.8, 3.81, .5):
            boxes.append(dict(lo=[a, .50, float(z)], hi=[b, 1.65, float(z) + .04], rho=.55))
        boxes.append(dict(lo=[a, .50, .65], hi=[b, .56, 3.8], rho=.55))
    elif family == 'low_beam':
        z = float(rng.uniform(1.3, 2.2))
        boxes.append(dict(lo=[-1., -.40, z], hi=[1., -.26, z + .15], rho=.55))
    elif family == 'opposite_wall':
        a, b = _x(.44, 1.0, -side)
        boxes.append(dict(lo=[a, -.30, .65], hi=[b, 1.5, 3.8], rho=.60))
    elif family == 'bollards':
        a, b = _x(.40, .50, side)
        for z in np.arange(1.0, 3.01, 1.0):
            boxes.append(dict(lo=[a, .95, float(z)], hi=[b, 1.65, float(z) + .10], rho=.50))
    elif family == 'vehicle':
        a, b = _x(.62, 2.40, side)
        z = float(rng.uniform(.9, 1.4))
        boxes.append(dict(lo=[a, .10, z], hi=[b, 1.65, z + 4.], rho=.50))
    else:
        raise ValueError(family)
    return boxes


def make_novel_scenes(unit, start_config=22):
    base = _KNOWN(unit)
    ref = base[0]
    rng = np.random.default_rng([2026092901, int(unit)])
    scenes = []
    for fi, family in enumerate(NOVEL):
        group = (unit + fi) % 2
        for m in S.MARGINS:
            margin = float(m + rng.uniform(-.006, .006))
            width = float(rng.uniform(.08, .12)); side = int(rng.choice([-1, 1]))
            z = float(rng.uniform(1.15, 1.85)); thick = float(rng.uniform(.06, .18))
            yy = (-.10, .26) if group == 0 else (.50, .84)
            inside = .30 + margin
            xx = (inside, inside + width) if side == 1 else (-inside - width, -inside)
            target = dict(lo=[xx[0], yy[0], z], hi=[xx[1], yy[1], z + thick], rho=float(rng.uniform(.22, .65)))
            ctx = context(family, side, z, rng)
            # Context alone must never touch either final central query.
            alone = labels_for_all(ctx + [FLOOR, BACK], ref['travel'][-1:], boundary='closed')[0, [2, 3]]
            assert not alone.any(), (unit, family, margin)
            boxes = [target] + ctx + [FLOOR, BACK]
            scenes.append(dict(unit=int(unit), config=start_config + len(scenes), family=family, margin=margin,
                               group=int(group), boxes=boxes, poses=ref['poses'].copy(), travel=ref['travel'].copy(),
                               head=ref['head'].copy(), labels=S.labels_for(boxes, ref['travel']),
                               mode=ref['mode'], dt=.2, speed=.8))
    return scenes


def make_all_scenes(unit):
    return _KNOWN(unit) + make_novel_scenes(unit)
