"""Observation-only interval-conditioned RGB statistics; no surface ownership."""
import numpy as np
from PIL import Image
import torch
from torch import nn

from ba_camera_corridor import tof_readout

FEATURES = 60
RECIPE = dict(seed=20260921, steps=1200, batch_size=64, learning_rate=.001, weight_decay=.0001)
FOCAL = 640 / (2 * np.tan(np.deg2rad(50)))


def bands(box, interval):
    """Three disjoint pixel-centre query bands; clipped depth is NOT new support."""
    y0, x0, y1, x1 = np.asarray(box, float) * [360/192, 640/256, 360/192, 640/256]
    yy = np.arange(max(0, int(np.ceil(y0-.5))), min(360, int(np.ceil(y1-.5))))
    xx = np.arange(max(0, int(np.ceil(x0-.5))), min(640, int(np.ceil(x1-.5))))
    ax = (xx[None, :] + .5 - 320) / FOCAL
    ay = (yy[:, None] + .5 - 180) / FOCAL
    lo, hi = interval
    near, far = max(lo, .3), min(hi, 3.)
    shape = (len(yy), len(xx))
    if near >= far:
        inner = outer = np.zeros(shape, bool)
        relative = np.zeros(8)
        fraction = 0.
    else:
        inner = (abs(ax) <= .3/far) & (ay >= -.2/far) & (ay <= .9/far)
        outer = (abs(ax) <= .3/near) & (ay >= -.2/near) & (ay <= .9/near)
        relative = []
        for z in (near, far):
            relative.extend([(320 + FOCAL * v/z - x0)/(x1-x0) for v in (-.3, .3)])
            relative.extend([(180 + FOCAL * v/z - y0)/(y1-y0) for v in (-.2, .9)])
        relative = np.clip(relative, -4, 4)/4
        fraction = (far-near)/(hi-lo)
    return yy, xx, (inner, outer & ~inner, ~outer), np.r_[relative, lo/8, hi/8, fraction, 1.]


def extract(rgb_path, ranges, boxes):
    """Return [64,60] finite features + validity, using only original observations."""
    values = np.asarray(ranges)
    if values.shape != (64,) or np.asarray(boxes).shape != (64, 4):
        raise ValueError('Expected 64 ranges and zone boxes')
    with Image.open(rgb_path) as image:
        if image.size != (640, 360):
            raise ValueError('Native 640x360 RGB required')
        rgb = np.asarray(image.convert('RGB'), np.float32)/255
    gray = rgb @ np.array([.299, .587, .114], np.float32)
    gy, gx = np.gradient(gray)
    gradients = np.stack([abs(gx), abs(gy)], -1)
    answer, valid = np.zeros((64, FEATURES), np.float32), np.zeros(64, bool)
    _, anchors = tof_readout(boxes, values)
    for anchor in anchors:
        zone = anchor['zone']
        yy, xx, masks, geometry = bands(boxes[zone], anchor['interval_m'])
        patch = rgb[np.ix_(yy, xx)]
        intensity, gradient = gray[np.ix_(yy, xx)], gradients[np.ix_(yy, xx)]
        stats, means = [], []
        for mask in masks:
            if mask.any():
                pixels = patch[mask]
                mean = pixels.mean(0)
                stat = np.r_[mean, pixels.std(0), np.quantile(intensity[mask], [.1,.25,.5,.75,.9]),
                             gradient[mask].mean(0), mask.mean()]
            else:
                mean, stat = np.zeros(3), np.zeros(14)
            means.append(mean)
            stats.extend(stat)
        answer[zone] = np.r_[stats, means[0]-means[2], means[1]-means[2], geometry]
        valid[zone] = True
    assert np.isfinite(answer).all()
    return answer, valid


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES,32), nn.ReLU(), nn.Linear(32,32),
                                 nn.ReLU(), nn.Linear(32,1))

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (64, FEATURES):
            raise ValueError('Expected [N,64,60] features')
        scores = self.net(inputs).squeeze(-1)
        # Last geometry feature is original validity; no invalid-zone score votes.
        valid = inputs[:, :, -1] != 0
        maximum = scores.masked_fill(~valid, -torch.inf).max(1).values
        return torch.where(valid.any(1), maximum, torch.full_like(maximum, -20.))
