"""Matched raw-feature residual objectives; no claim of latent disentanglement."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RawResidual(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float())
        self.register_buffer('scale', torch.as_tensor(scale).float())
        self.net = nn.Sequential(nn.Linear(2224, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, base):
        return base + self.net(((x-self.mean)/self.scale).clamp(-8, 8)).squeeze(-1)


def loss(z, y, intrusion, background, arm):
    bce = F.binary_cross_entropy_with_logits(z, y)
    rank = F.relu(1.-z[intrusion[:, 0]]+z[intrusion[:, 1]]).mean() if len(intrusion) else z.sum()*0
    inv = (z[background[:, 0]]-z[background[:, 1]]).abs().mean() if len(background) else z.sum()*0
    return bce + .25*rank*(arm != 'B0') + .1*inv*(arm == 'B2')


def pair_report(logits, probabilities, flags, intrusion, background):
    return dict(intrusion_pairs=len(intrusion), background_pairs=len(background),
        intrusion_order_accuracy=float(np.mean(logits[intrusion[:, 0]] > logits[intrusion[:, 1]])),
        intrusion_logit_margin=float(np.mean(logits[intrusion[:, 0]]-logits[intrusion[:, 1]])),
        background_logit_drift=float(np.mean(np.abs(logits[background[:, 0]]-logits[background[:, 1]]))) if len(background) else None,
        background_probability_drift=float(np.mean(np.abs(probabilities[background[:, 0]]-probabilities[background[:, 1]]))) if len(background) else None,
        background_alert_flips=int(sum(flags[background[:, 0]] != flags[background[:, 1]])) if len(background) else None)
