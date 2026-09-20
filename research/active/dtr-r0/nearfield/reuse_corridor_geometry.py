"""Evaluator-only UE Cube OBB versus the fixed camera-forward query.

UE local-to-world basis matches mz101_stereo_capture.basis (forward/right/up).
Positive pitch raises forward; positive yaw rotates forward toward world +Y;
positive roll rotates local +Y toward -Z, as in contextual_geometry._solid.
city_pcg_capture applies these with u.Rotator(**rotation) and records actor origin,
scale and mesh-local bounds. No asset bounds are treated as exact non-cube solids.
Camera coordinates are X-right/Y-down/Z-forward. Compound targets are unions of
their separately verified cube components, never their enclosing bounding box.
"""
from itertools import product
from collections.abc import Mapping

import numpy as np

QUERY_LO = np.array([-.3, -.2, .3], dtype=np.float64)
QUERY_HI = np.array([.3, .9, 3.], dtype=np.float64)
CUBE_ASSET = '/Engine/BasicShapes/Cube.Cube'
CONTACT_TOLERANCE_M = 1e-10


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + ' must have three finite coordinates')
    return result


def ue_rotation(rotation_deg):
    """3x3 column basis mapping UE local XYZ into world XYZ (degrees)."""
    if not isinstance(rotation_deg, Mapping) or set(rotation_deg) != {'pitch', 'yaw', 'roll'}:
        raise ValueError('Explicit pitch/yaw/roll mapping required')
    p, y, r = np.deg2rad(_vector([rotation_deg[k] for k in ('pitch', 'yaw', 'roll')], 'rotation'))
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    forward = (cp*cy, cp*sy, sp)
    right = (sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp)
    up = (-cr*sp*cy-sr*sy, -cr*sp*sy+sr*cy, cr*cp)
    return np.asarray([forward, right, up], dtype=np.float64).T


def camera_obb(camera, receipt):
    """Verified centered one-metre Cube receipt -> camera-frame OBB and 8 corners.

    Returns arrays center[3], axes[3,3] (columns), half[3], corners[8,3].
    Positive nonuniform scale is supported. The caller authenticates receipt
    provenance; this function checks exact cube identity and local geometry.
    """
    if receipt.get('mesh_asset') != CUBE_ASSET:
        raise ValueError('Exact Engine Cube.Cube geometry receipt required')
    local = receipt['mesh_local_bounds_cm']
    lo, hi = _vector(local['min'], 'local min'), _vector(local['max'], 'local max')
    if not (np.array_equal(lo, [-50., -50., -50.]) and np.array_equal(hi, [50., 50., 50.])):
        raise ValueError('Cube local bounds must be centered +/-50 cm')
    scale = _vector(receipt['scale'], 'scale')
    if not (scale > 0).all():
        raise ValueError('Strictly positive cube scale required')
    actor = _vector(receipt['actor_origin_m'], 'actor origin')
    origin = _vector([camera[k] for k in ('x', 'y', 'z')], 'camera origin')
    camera_basis = ue_rotation({k: camera[k] for k in ('pitch', 'yaw', 'roll')})
    world_to_camera = np.stack([camera_basis[:, 1], -camera_basis[:, 2], camera_basis[:, 0]])
    center = world_to_camera @ (actor-origin)
    axes = world_to_camera @ ue_rotation(receipt['rotation_deg'])
    half = scale * .5
    signs = np.asarray(list(product((-1., 1.), repeat=3)), dtype=np.float64)
    corners = center + (signs * half) @ axes.T
    return dict(center=center, axes=axes, half=half, corners=corners)


def obb_intersects_aabb(center, axes, half, lo=QUERY_LO, hi=QUERY_HI):
    """Closed-volume 15-axis SAT; tolerance is 1e-10 metres on unit axes.

    Test three AABB axes, three OBB axes and all nine cross axes. Parallel
    cross-product axes below 1e-12 length are redundant and omitted. AABB-only
    overlap is insufficient for oblique rods and is never used as a positive.
    """
    center, half = _vector(center, 'OBB center'), _vector(half, 'OBB half extents')
    lo, hi = _vector(lo, 'query lo'), _vector(hi, 'query hi')
    axes = np.asarray(axes, dtype=np.float64)
    if axes.shape != (3, 3) or not np.isfinite(axes).all() or not np.allclose(axes.T@axes, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError('OBB axes must be finite orthonormal columns')
    if not (half > 0).all() or not (hi > lo).all():
        raise ValueError('OBB and query extents must be positive')
    query_center, query_half = (lo+hi)*.5, (hi-lo)*.5
    delta = center-query_center
    world_axes = np.eye(3)
    candidates = [*world_axes, *axes.T]
    candidates.extend(np.cross(a, b) for a in world_axes for b in axes.T)
    for axis in candidates:
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis/length
        distance = abs(float(delta@axis))
        radius_a = float(query_half @ np.abs(axis))
        radius_b = float(half @ np.abs(axes.T@axis))
        if distance > radius_a + radius_b + CONTACT_TOLERANCE_M:
            return False
    return True


def classify_cube(camera, receipt):
    """JSON-ready exact intersection plus conservative near-lateral exclusion.

    near_lateral_outside requires positive full-volume lateral separation and
    overlapping Y/Z projected extents. It certifies lateral exclusion, not exact
    intersection with a finite YZ slab or physical visibility. Root aggregates
    component union; an enclosing AABB must never fill a compound target's gaps.
    """
    obb = camera_obb(camera, receipt)
    lo, hi = obb['corners'].min(0), obb['corners'].max(0)
    yz_overlap = bool(np.all(hi[1:] >= QUERY_LO[1:]-CONTACT_TOLERANCE_M)
                      and np.all(lo[1:] <= QUERY_HI[1:]+CONTACT_TOLERANCE_M))
    separation = max(float(QUERY_LO[0]-hi[0]), float(lo[0]-QUERY_HI[0]), 0.)
    intersects = obb_intersects_aabb(obb['center'], obb['axes'], obb['half'])
    outside = yz_overlap and separation > CONTACT_TOLERANCE_M
    assert not (intersects and outside)
    return dict(camera_obb={k: v.tolist() for k, v in obb.items()},
                aabb_lo=lo.tolist(), aabb_hi=hi.tolist(), intersects=intersects,
                near_yz_envelope_overlap=yz_overlap, near_lateral_outside=outside,
                lateral_separation_m=separation,
                authority='VERIFIED_ENGINE_CUBE_OBB_CLOSED_15_AXIS_SAT',
                contact_tolerance_m=CONTACT_TOLERANCE_M)
