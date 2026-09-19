"""Evaluator-only exact lineage replay for the frozen scalar sensor proxy."""
import hashlib

import numpy as np


def zone_map(boxes, shape):
    result = np.full(shape,-1,np.int32)
    for index,(y0,x0,y1,x1) in enumerate(boxes):
        if np.any(result[y0:y1,x0:x1] >= 0):
            raise ValueError('Overlapping frozen zones')
        result[y0:y1,x0:x1] = index
    return result


def trace_sensor(depth, identity, boxes, saved_values):
    """Reproduce original bin/RNG/noise exactly before revealing contributor indices.

    Reference depth only reconstructs provenance. Never output new observed ranges
    or fill dropout. The winning-bin contributors can belong to multiple surfaces.
    """
    h,w = depth.shape
    ys = np.linspace(round(.1*h),round(.9*h),9).astype(int)
    xs = np.linspace(round(.1*w),round(.9*w),9).astype(int)
    expected = np.array([[ys[y],xs[x],ys[y+1],xs[x+1]] for y in range(8) for x in range(8)])
    if not np.array_equal(boxes,expected) or np.asarray(saved_values).shape != (64,):
        raise ValueError('Frozen observation geometry changed')
    rng = np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:8],16))
    traced, reconstructed = [], []
    for zi,(y0,x0,y1,x1) in enumerate(boxes):
        patch = depth[y0:y1,x0:x1]
        eligible = np.isfinite(patch) & (patch > .001) & (patch >= .1) & (patch < 8)
        hits = patch[eligible]
        value, peak, indices, weight = np.nan,None,np.array([],np.int64),np.array([],float)
        reason = 'INSUFFICIENT_HITS'
        if hits.size >= 4:
            if rng.random() < .05:
                reason = 'SIMULATED_DROPOUT'
            else:
                bins = np.minimum((hits/.1).astype(int),79)
                energy = np.bincount(bins,weights=1/np.maximum(hits,.3)**2,minlength=80)
                selected = int(np.argmax(energy))
                center = np.mean(hits[bins == selected])
                noisy = center+rng.normal(0,.01+.02*center)
                reason = 'NOISY_RANGE_OUTSIDE_LIMIT'
                if .1 <= noisy < 8:
                    value,peak,reason = noisy,selected,'OBSERVED'
                    py,px = np.nonzero(eligible)
                    use = bins == selected
                    indices = ((py[use]+y0)*w+px[use]+x0).astype(np.int64)
                    weight = (1/np.maximum(hits[use],.3)**2).astype(float)
        reconstructed.append(value)
        traced.append(dict(zone_id=zi,observed=bool(np.isfinite(value)),
            distance_m=float(np.float32(value)) if np.isfinite(value) else None,
            winner_bin=peak,pixel_indices=indices,weights=weight,reason=reason))
    actual,expected_values = np.asarray(saved_values,np.float32),np.asarray(reconstructed,np.float32)
    if not np.array_equal(actual,expected_values,equal_nan=True):
        raise ValueError('Exact saved-return replay mismatch')
    return traced


def component_anchors(labels, traces, threshold):
    """Possible means any contribution; pure means every winning sample same label."""
    possible,pure = {},{}
    for trace in traces:
        if not trace['observed'] or not trace['distance_m'] < threshold:
            continue
        values = np.unique(labels.ravel()[trace['pixel_indices']])
        for label in values[values>0]:
            possible.setdefault(int(label),set()).add(trace['zone_id'])
        if len(values)==1 and values[0]>0:
            pure.setdefault(int(values[0]),set()).add(trace['zone_id'])
    return possible,pure
