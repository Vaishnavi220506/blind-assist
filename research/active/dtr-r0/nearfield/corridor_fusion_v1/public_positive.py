"""Public ToF spatial-evidence head. Training labels are never accepted here."""
import numpy as np
import torch
from torch import nn


def spatial_features(tokens):
    """Explicit public envelope/center relations; no true point recovery claim."""
    x = np.asarray(tokens, np.float32)[..., :128, :]
    if x.shape[-2:] != (128, 21):
        raise ValueError('Expected public 132x21 or ToF128x21 tokens')
    scale = np.array([4., 2., 3.], np.float32)
    center = x[..., 5:8] * scale
    support = x[..., 8:14].reshape(*x.shape[:-1], 3, 2) * scale[:, None]
    low = np.array([.2, -.3, .4], np.float32)
    high = np.array([3.6, .3, 2.05], np.float32)
    center_margin = np.minimum(center-low, high-center)
    possible = np.minimum(support[..., 1]-low, high-support[..., 0])
    contained = np.minimum(support[..., 0]-low, high-support[..., 1])
    return np.concatenate([x, center_margin, possible, contained], -1)


class PositiveHead(nn.Module):
    def __init__(self, mean=None, scale=None):
        super().__init__()
        self.register_buffer('mean', torch.zeros(30) if mean is None else torch.as_tensor(mean).clone())
        self.register_buffer('scale', torch.ones(30) if scale is None else torch.as_tensor(scale).clone())
        self.net = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Linear(32, 16),
                                 nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features, valid):
        if features.shape[-2:] != (128, 30) or valid.shape != features.shape[:-1]:
            raise ValueError('Expected Bx128x30 and Bx128')
        clean = torch.where(valid[..., None], features, self.mean)
        logits = self.net((clean-self.mean)/self.scale).squeeze(-1)
        return logits.masked_fill(~valid, -30.)


def add_positive(a, logits, valid, threshold):
    valid = np.asarray(valid, bool)
    scores = np.where(valid, logits, -30.).max(1).astype(np.float64)
    branch = valid.any(1) & (scores >= threshold)
    return np.asarray(a, bool) | branch, scores, branch


def select_threshold(a, logits, valid, truth, clear):
    """Calibrate final OR F1, including a fully inactive branch candidate."""
    scores = np.where(valid, logits, -30.).max(1).astype(np.float64)
    eligible = valid.any(1)
    values = scores[eligible & clear].astype(np.float64)
    candidates = np.r_[np.nextafter(float(scores.max()), np.inf), np.unique(values)]
    curve = []
    for threshold in candidates:
        p = (a | (eligible & (scores >= threshold)))[clear]
        y = truth[clear]
        tp, fp, fn = int(sum(p & y)), int(sum(p & ~y)), int(sum(~p & y))
        curve.append(dict(threshold=float(threshold), TP=tp, FP=fp, FN=fn,
                          f1=2*tp/max(1, 2*tp+fp+fn)))
    return max(curve, key=lambda v: (v['f1'], -v['FP'], v['threshold'])), curve
