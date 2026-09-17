"""Grouped split and fit-only negative-peak loss for the public ToF head."""
import numpy as np
import torch
from torch.nn import functional as F


def nested_folds(meta):
    """Anchor is fit-only; scene indices are excluded jointly across cohorts."""
    result = []
    for outer in range(6):
        dev_scenes = [s for s in range(6) if s != outer]
        report = [i for i, m in enumerate(meta)
                  if m['cohort'] != 'anchor' and m['scene_index'] == outer]
        fit = [i for i, m in enumerate(meta)
               if m['cohort'] == 'anchor' or m['scene_index'] != outer]
        inner = []
        for j in range(3):
            held = dev_scenes[j::3]
            cal = [i for i in fit if meta[i]['cohort'] != 'anchor'
                   and meta[i]['scene_index'] in held]
            train = [i for i in fit if meta[i]['cohort'] == 'anchor'
                     or meta[i]['scene_index'] not in held]
            inner.append(dict(inner=j, fit=train, calibration=cal, scene_indices=held))
        result.append(dict(fold=outer, fit=fit, report=report, inner=inner))
    return result


def fitting_weights(meta, fit, valid, target, known):
    """Return and peak masses each sum to one over eligible fitting data only."""
    fit = np.asarray(fit, int)
    weight = np.zeros(target.shape, np.float32)
    pos = known[fit] & (target[fit] > 0)
    neg = known[fit] & (target[fit] == 0)
    if not pos.any() or not neg.any():
        raise ValueError('Both known return classes required')
    weight[fit] = .5*pos/pos.sum() + .5*neg/neg.sum()
    peak = np.zeros(len(meta), np.float32)
    eligible = [i for i in fit if meta[i]['A'] is False and not meta[i]['truth']
                and meta[i]['stratum'] == 'negative' and valid[i].any()]
    if eligible:
        peak[eligible] = 1./len(eligible)
    return weight, peak


def negative_peak_loss(logits, valid, frame_weights):
    """One-return evidence is retained; no frame-positive supervision enters."""
    maxima = logits.masked_fill(~valid, -torch.inf).max(1).values
    # Avoid undefined inf derivatives for empty frames. They carry zero mass.
    maxima = torch.where(valid.any(1), maxima, torch.zeros_like(maxima))
    return (F.softplus(maxima) * frame_weights).sum()
