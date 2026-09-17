"""Relative DA-V2 structure only; no labels, metric calibration or sensor veto."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mz115_spatial_allocation import zone_box
from mz136_boundary_geometry import camera_to_body, proposals

STAT_NAMES = ('present', 'q10', 'q25', 'median', 'q75', 'q90', 'spread',
              'grad_x_mean', 'grad_x_q90', 'grad_y_mean', 'ring_present', 'inside_minus_ring')


def normalize(depth):
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError('Finite 2D relative depth required')
    lo, hi = np.quantile(depth, [.05, .95])
    return np.clip((depth - lo) / max(float(hi-lo), 1e-6), -2, 3).astype(np.float32)


def region(depth, gx, gy, box):
    h, w = depth.shape
    if box is None:
        return np.zeros(len(STAT_NAMES), np.float32)
    l, t, r, b = box
    l, r = max(0, int(np.floor(l))), min(w, int(np.ceil(r)))
    t, b = max(0, int(np.floor(t))), min(h, int(np.ceil(b)))
    if r <= l or b <= t:
        return np.zeros(len(STAT_NAMES), np.float32)
    roi = depth[t:b, l:r]
    q = np.quantile(roi, [.1, .25, .5, .75, .9])
    pad = max(3, int(min(r-l, b-t)*.15))
    ll, rr, tt, bb = max(0,l-pad), min(w,r+pad), max(0,t-pad), min(h,b+pad)
    ring = np.concatenate([depth[tt:t,ll:rr].ravel(), depth[b:bb,ll:rr].ravel(),
                           depth[t:b,ll:l].ravel(), depth[t:b,r:rr].ravel()])
    return np.array([1, *q, q[-1]-q[0], gx[t:b,l:r].mean(),
                     np.quantile(gx[t:b,l:r], .9), gy[t:b,l:r].mean(),
                     float(len(ring)>0), q[2]-np.median(ring) if len(ring) else 0], np.float32)


def query_box(row, yaw, distance, low, high):
    """Project declared body-forward plane corners, not estimated obstacle points."""
    rotation = camera_to_body(row, yaw)
    origin = np.asarray(row['camera_in_body_m'])
    points = np.array([[distance, side, height] for side in (-.3,.3)
                       for height in (low,high)])
    cam = (points-origin) @ rotation
    if np.any(cam[:,0] <= .01):
        return None
    intr = row['rgb_intrinsics']
    u = intr['cx']+intr['fx']*cam[:,1]/cam[:,0]
    v = intr['cy']-intr['fy']*cam[:,2]/cam[:,0]
    return [u.min(),v.min(),u.max(),v.max()]


def extract(row, image, yaw, depth):
    """Fixed 924-D vector: 64 zones, 8 query windows and pooled RGB proposals.

    Zone projection inherits the simulated co-located angular camera/ToF model;
    this is not an adapter for a displaced physical ToF camera. Both raw returns
    remain unmodified in the separate 2485-D base features.
    """
    if depth.shape != image.shape[:2]:
        raise ValueError('Depth must match native RGB dimensions')
    d = normalize(depth)
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))/8
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))/8
    values, names = [], []
    def add(prefix, v):
        values.extend(v)
        names.extend(prefix+'.'+name for name in STAT_NAMES)
    for z in sorted(row['tof_zones'], key=lambda z:z['zone_id']):
        add('zone%02d'%z['zone_id'], region(d,gx,gy,zone_box(z,row['rgb_intrinsics'])))
    for distance in (.5,1.,2.,3.6):
        for band, low, high in [('body',.4,1.5),('head',1.5,2.05)]:
            add(f'query_{band}_{distance}',region(d,gx,gy,query_box(row,yaw,distance,low,high)))
    boxes, _ = proposals(row,image,yaw)
    desc = np.stack([region(d,gx,gy,p['box']) for p in boxes]) if boxes else np.zeros((1,12))
    for label, value in [('min',desc.min(0)),('mean',desc.mean(0)),('max',desc.max(0))]:
        add('proposal_'+label,value)
    add('whole',region(d,gx,gy,[0,0,d.shape[1],d.shape[0]]))
    add('central',region(d,gx,gy,[d.shape[1]*.3,0,d.shape[1]*.7,d.shape[0]]))
    result=np.asarray(values,np.float32)
    assert result.shape == (924,) and np.isfinite(result).all()
    return result,names
