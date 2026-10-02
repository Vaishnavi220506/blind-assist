"""Observation-count-matched geometry control for RGB region proposals.

No reference depth, semantic labels, objects, or query membership is consumed.
The control receives the observed region count per zone, including an uncovered
(-1) region, but never receives its shape or position.
"""
import argparse
import json

import numpy as np


def matched_geometry_partition(observed_regions, zone_id, fx, fy):
    """Partition each zone into exactly its observed number of region IDs.

    All arrays have shape HxW. Integer zone IDs >=0 are included; -1 is outside.
    Observed IDs >=0 plus an observed -1 (uncovered) each count once per zone.
    fx/fy are public X/Z and Y/Z rays; arctan gives the angular split axes.

    Recursively split the widest angular axis at a pixel-count median, with
    flat native-pixel index breaking coordinate ties (X wins equal spans).
    Left/right leaf quotas are ceil(K/2), floor(K/2). Clamp the median only
    when needed to leave >=1 pixel for every requested leaf. Labels are unique
    across zones. All in-zone pixels are covered, even where SAM was uncovered.

    This is a count-matched geometry control, not an equal-coverage control:
    its full coverage is intentionally more favorable than SAM's UNKNOWNs.
    It matches region-zone patch counts, not global candidate count/adjacency.
    With N pixels and Z zones, mask enumeration costs O(Z*N); stable recursive
    sorting is at most O(N*log(N)*log(Kmax)), with O(N+Ktotal) working storage.
    """
    observed = np.asarray(observed_regions)
    zones = np.asarray(zone_id)
    rx, ry = np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    if observed.ndim != 2 or not (observed.shape == zones.shape == rx.shape == ry.shape):
        raise ValueError('all inputs must have the same HxW shape')
    if not np.issubdtype(observed.dtype, np.integer) or not np.issubdtype(zones.dtype, np.integer):
        raise ValueError('observed region and zone IDs must be integer arrays')
    if np.any(observed < -1) or np.any(zones < -1):
        raise ValueError('only -1 may denote uncovered/outside')
    inside = zones >= 0
    if not (np.isfinite(rx[inside]).all() and np.isfinite(ry[inside]).all()):
        raise ValueError('public rays must be finite on every included pixel')
    angles = np.stack((np.arctan(rx).ravel(), np.arctan(ry).ravel()), axis=1)
    flat_zones, flat_observed = zones.ravel(), observed.ravel()
    result = np.full(zones.size, -1, dtype=np.int32)
    next_label = 0

    def divide(indices, leaves):
        nonlocal next_label
        if leaves == 1:
            result[indices] = next_label
            next_label += 1
            return
        left_leaves = (leaves+1)//2
        right_leaves = leaves//2
        axis = int(np.argmax(np.ptp(angles[indices], axis=0)))
        ordered = indices[np.lexsort((indices, angles[indices, axis]))]
        cut = min(max((len(ordered)+1)//2, left_leaves), len(ordered)-right_leaves)
        assert left_leaves <= cut and right_leaves <= len(ordered)-cut
        divide(ordered[:cut], left_leaves)
        divide(ordered[cut:], right_leaves)

    for zone in np.unique(flat_zones[flat_zones >= 0]):
        indices = np.flatnonzero(flat_zones == zone)
        leaves = len(np.unique(flat_observed[indices]))
        assert 1 <= leaves <= len(indices)
        divide(indices, leaves)
    return result.reshape(zones.shape)


def selftest():
    """Synthetic public geometry and observed IDs only; no scientific images."""
    zones = np.array([[0, 0, 0, 1, 1, 1, -1],
                      [0, 0, 0, 1, 1, 1, -1],
                      [2, 2, 2, 3, 3, 3, -1]], dtype=np.int32)
    observed = np.array([[-1, 7, 8, 9, 9, 9, -1],
                         [-1, 7, 8, 9, 9, 9, -1],
                         [10, 11, 12, -1, -1, -1, -1]], dtype=np.int32)
    yy, xx = np.indices(zones.shape)
    fx, fy = (xx-3)/8., (yy-1)/8.
    output = matched_geometry_partition(observed, zones, fx, fy)
    assert np.all(output[zones >= 0] >= 0) and np.all(output[zones < 0] == -1)
    for zone in range(4):
        take = zones == zone
        assert len(np.unique(output[take])) == len(np.unique(observed[take]))
    for label in np.unique(output[output >= 0]):
        assert len(np.unique(zones[output == label])) == 1
    np.testing.assert_array_equal(output, matched_geometry_partition(observed, zones, fx, fy))
    # Odd K=N and coordinate ties must still make exactly K nonempty leaves.
    odd = np.arange(5, dtype=np.int32)[None, :]
    tied = matched_geometry_partition(odd, np.zeros_like(odd), np.zeros_like(odd), np.zeros_like(odd))
    np.testing.assert_array_equal(tied, odd)
    empty = np.full((2, 2), -1, dtype=np.int32)
    np.testing.assert_array_equal(matched_geometry_partition(empty, empty, empty, empty), empty)
    checks = dict(exact_per_zone_K=True, uncovered_counts_as_one=True,
                  full_in_zone_coverage=True, no_cross_zone_labels=True,
                  deterministic_ties=True, odd_singleton_leaves=True,
                  geometry_only_empty_fov=True, scientific_images_read=False)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(selftest()))
