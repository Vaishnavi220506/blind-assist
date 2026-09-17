"""Small differentiable residual on frozen HGB odds; public features only."""
import copy
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CorridorResidual(nn.Module):
    def __init__(self, mean=None, scale=None):
        super().__init__()
        self.register_buffer('mean', torch.zeros(2485) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer('scale', torch.ones(2485) if scale is None else torch.as_tensor(scale).float())
        self.net = nn.Sequential(nn.Linear(2485, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base_logit):
        if features.ndim != 2 or features.shape[1] != 2485:
            raise ValueError('Expected N x 2485 public features')
        return base_logit + self.net(((features-self.mean)/self.scale).clamp(-8, 8)).squeeze(-1)


def log_odds(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).astype(np.float32)


def pairing(meta, frames):
    """Native labels orient pairs; latent fields only authenticate supervision."""
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m['group'], m['time_s'])].append(i)
    rank, invalid = [], []
    for key, ii in groups.items():
        if len(ii) != 2:
            raise ValueError('Incomplete pair: '+str(key))
        a, b = ii
        fa, fb = (copy.deepcopy(frames[meta[i]['id']]) for i in ii)
        for f in (fa, fb):
            for k in ('id', 'episode', 'pair_member', 'pair_variant'):
                f.pop(k, None)
            f['objects'][0]['center_m'][1] = 0.
        if fa != fb:
            invalid.append(dict(ids=[meta[i]['id'] for i in ii], reason='BEYOND_LATERAL_DIFFERENCE'))
            continue
        if meta[a]['truth'] == meta[b]['truth']:
            invalid.append(dict(ids=[meta[i]['id'] for i in ii], reason='SAME_NATIVE_LABEL'))
            continue
        rank.append((a, b) if meta[a]['truth'] else (b, a))
    # Exact camera, body, target (geometry AND appearance), time and sensor seeds.
    # Only context is free to change. Same-label convenience matching is forbidden.
    fixed = defaultdict(list)
    for i, m in enumerate(meta):
        f = frames[m['id']]
        key = {k: f.get(k) for k in ('camera', 'body_origin_m', 'time_s', 'sensor_seed',
                                    'tof_sensor_seed', 'wearer_speed', 'radar_ghost')}
        key['target'] = f['objects'][0]
        key['truth'] = m['truth']
        fixed[json.dumps(key, sort_keys=True)].append(i)
    invariant = []
    for ii in fixed.values():
        for j, a in enumerate(ii):
            for b in ii[j+1:]:
                if frames[meta[a]['id']]['objects'][1:] != frames[meta[b]['id']]['objects'][1:]:
                    invariant.append((a, b))
    return np.asarray(rank, dtype=np.int64).reshape(-1, 2), np.asarray(invariant, dtype=np.int64).reshape(-1, 2), invalid


def objective(z, y, rank, invariant, use_rank=False, use_invariance=False):
    bce = F.binary_cross_entropy_with_logits(z, y)
    rank_loss = F.relu(1.-z[rank[:, 0]]+z[rank[:, 1]]).mean() if len(rank) else z.sum()*0
    inv_loss = ((z[invariant[:, 0]]-z[invariant[:, 1]])**2).mean() if len(invariant) else z.sum()*0
    return bce + .25*rank_loss*use_rank + .1*inv_loss*use_invariance


def local_pairs(pairs, indices):
    index = {int(i): j for j, i in enumerate(indices)}
    return np.asarray([(index[int(a)], index[int(b)]) for a, b in pairs
                       if int(a) in index and int(b) in index], dtype=np.int64).reshape(-1, 2)
