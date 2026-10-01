"""Evaluator-only signed distances to physical surfaces; no sensor features.

For each query, minimise its L-infinity signed-distance function over the
union of every physical box surface in the final travel frame. Negative means
a surface enters the query interior, zero is contact, positive is separation.
This is a surface definition, not overlap of solid bounding volumes.

The vertical-sides x/z control projects only the four vertical faces of every
box, without height selection. This is an explicit vertical-surface prior, not
a complete all-surface upper bound. Projecting horizontal faces too would make
the broad floor cover the query and turn the control into a constant.
"""
import numpy as np


QUERY_LOW = np.array([[-.30, -.20, .30], [-.30, .42, .30]])
QUERY_HIGH = np.array([[.30, .42, 3.00], [.30, .90, 3.00]])


def _local_boxes(scene):
    pose = np.asarray(scene['travel'], dtype=float)[-1]
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('finite final 4x4 travel pose required')
    if not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-12, rtol=0):
        raise ValueError('invalid homogeneous travel pose')
    rotation = pose[:3, :3]
    # Signed axis permutations preserve both axis-aligned surfaces and L-inf.
    rounded = np.round(rotation)
    if (not np.allclose(rotation, rounded, atol=1e-12, rtol=0)
            or not np.array_equal(np.abs(rounded).sum(0), np.ones(3))
            or not np.array_equal(np.abs(rounded).sum(1), np.ones(3))
            or not np.isclose(np.linalg.det(rounded), 1., atol=1e-12)):
        raise ValueError('non-axis-aligned final travel rotation is unsupported')
    out = []
    for box in scene['boxes']:
        lo, hi = np.asarray(box['lo'], float), np.asarray(box['hi'], float)
        if (lo.shape != (3,) or hi.shape != (3,)
                or not np.isfinite([lo, hi]).all() or not np.all(hi > lo)):
            raise ValueError('finite nondegenerate axis-aligned box required')
        endpoint_a = (lo - pose[:3, 3]) @ rounded
        endpoint_b = (hi - pose[:3, 3]) @ rounded
        out.append((np.minimum(endpoint_a, endpoint_b), np.maximum(endpoint_a, endpoint_b)))
    return out


def _surface_distance(boxes, query_lo, query_hi, axes, face_axes=(0, 1, 2)):
    """Exact min of query SDF on selected rectangular faces of local boxes.

    On each face the coordinates vary independently. The minimum of
    max(lo_i-p_i, p_i-hi_i) on a coordinate interval is achieved by clipping
    the query midpoint into that interval. Therefore the face minimum of the
    maximum across axes is the maximum of those coordinate minima.
    """
    axes = np.asarray(axes, dtype=int)
    centre = (query_lo[axes] + query_hi[axes]) / 2
    closest = np.inf
    for lo, hi in boxes:
        for face_axis in face_axes:
            for face_value in (lo[face_axis], hi[face_axis]):
                face_lo, face_hi = lo.copy(), hi.copy()
                face_lo[face_axis] = face_hi[face_axis] = face_value
                point = np.clip(centre, face_lo[axes], face_hi[axes])
                signed = np.maximum(query_lo[axes] - point, point - query_hi[axes])
                closest = min(closest, float(np.max(signed)))
    return closest


def scene_distances(scene):
    """Return metres, shape [full3d/vertical_sides_xz, HEAD/BODY].

    Uses every box and the final travel pose; target group, target margin,
    sensor pose and visibility never select surfaces. An empty scene has +inf
    separation. General rotations fail explicitly rather than approximate OBBs.
    """
    boxes = _local_boxes(scene)
    return np.array([[_surface_distance(boxes, lo, hi, axes, face_axes)
                      for lo, hi in zip(QUERY_LOW, QUERY_HIGH)]
                     for axes, face_axes in (((0, 1, 2), (0, 1, 2)),
                                             ((0, 2), (0, 2)))], dtype=float)


def scene_full3d_with_x_margin(scene, margin_m):
    """All-surface distances to queries expanded laterally, shape [HEAD,BODY].

    Height and longitudinal bounds are fixed. This is evaluator-only decision
    margin, not relabelling or training a readout with expanded labels.
    """
    margin = float(margin_m)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('lateral margin must be finite and nonnegative')
    lo, hi = QUERY_LOW.copy(), QUERY_HIGH.copy()
    lo[:, 0] -= margin
    hi[:, 0] += margin
    boxes = _local_boxes(scene)
    return np.array([_surface_distance(boxes, low, high, (0, 1, 2))
                     for low, high in zip(lo, hi)], dtype=float)


def self_check():
    """Small exact-geometry checks, returning a serialisable receipt."""
    def scene(lo, hi, pose=None):
        return dict(boxes=[dict(lo=lo, hi=hi)],
                    travel=np.array([np.eye(4) if pose is None else pose]))

    cases = {
        'head_intrudes_1cm': (scene([.29, -.10, 1.], [.39, .26, 1.1]),
                             [[-.01, .16], [-.01, -.01]]),
        'body_intrudes_1cm': (scene([.29, .50, 1.], [.39, .84, 1.1]),
                             [[.08, -.01], [-.01, -.01]]),
        'lateral_tangent': (scene([.30, -.10, 1.], [.40, .26, 1.1]),
                            [[0., .16], [0., 0.]]),
        'outside_3cm': (scene([.33, -.10, 1.], [.43, .26, 1.1]),
                        [[.03, .16], [.03, .03]]),
        'beyond_query_20cm': (scene([-.10, -.10, 3.2], [.10, .26, 3.3]),
                              [[.20, .20], [.20, .20]]),
        # Solid overlap would incorrectly return a negative full3d value.
        'query_enclosed_surfaces_outside': (scene([-1., -1., -.5], [1., 2., 4.]),
                                            [[.70, .70], [.70, .70]]),
        'floor_vertical_sides': (scene([-8., 1.65, -8.], [8., 1.8, 9.]),
                                [[1.23, .75], [6., 6.]]),
    }
    receipt = {}
    for name, (sc, expected) in cases.items():
        got = scene_distances(sc)
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0,
                                   err_msg=name)
        receipt[name] = got.tolist()
    floor = cases['floor_vertical_sides'][0]
    all_surface_xz_floor = [_surface_distance(_local_boxes(floor), lo, hi, (0, 2))
                            for lo, hi in zip(QUERY_LOW, QUERY_HIGH)]
    np.testing.assert_allclose(all_surface_xz_floor, [-.30, -.30], atol=1e-12, rtol=0)
    x_margin_checks = {}
    for margin in (0., .05, .10):
        head = scene_full3d_with_x_margin(cases['head_intrudes_1cm'][0], margin)
        body = scene_full3d_with_x_margin(cases['body_intrudes_1cm'][0], margin)
        assert head[0] < 0 and head[1] > 0 and body[0] > 0 and body[1] < 0
        np.testing.assert_allclose([head[1], body[0]], [.16, .08], atol=1e-12, rtol=0)
        x_margin_checks[str(margin)] = dict(head_target=head.tolist(), body_target=body.tolist())
    np.testing.assert_allclose(scene_full3d_with_x_margin(cases['outside_3cm'][0], .05),
                               [-.02, .16], atol=1e-12, rtol=0)
    # Coordinates are transformed into travel space, including translation.
    translated = scene([1.29, -.10, 1.], [1.39, .26, 1.1])
    translated['travel'][0, 0, 3] = 1.
    np.testing.assert_allclose(scene_distances(translated),
                               receipt['head_intrudes_1cm'], atol=1e-12, rtol=0)
    # An unsupported rotation must not silently turn into an enclosing AABB.
    angle = .1
    rotated = np.eye(4)
    rotated[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                      [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    try:
        scene_distances(scene([.29, -.10, 1.], [.39, .26, 1.1], rotated))
    except ValueError:
        rotation_rejected = True
    else:
        raise AssertionError('non-axis-aligned rotation was accepted')
    return dict(status='PASS', metric='minimum query Linf SDF on union of box surfaces',
                cases=receipt, translation='PASS',
                non_axis_rotation_rejected=rotation_rejected,
                all_surface_xz_floor_degeneracy_m=all_surface_xz_floor,
                x_only_margin_checks=x_margin_checks,
                second_arm='vertical sides only, height ignored; geometry-prior control')
