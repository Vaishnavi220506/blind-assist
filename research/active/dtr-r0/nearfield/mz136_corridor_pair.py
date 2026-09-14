"""Task-level classification/ranking on unchanged MZ120 observation network.

Cell outputs are latent features here, NOT supervised occupancy or localization.
No evaluator identity, pair ID, background, label or incumbent alarm enters forward.
"""
import math
import torch
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet, CORE


class CorridorNet(OccupancyNet):
    def forward(self, batch):
        logits = super().forward(batch)[:, CORE]
        return torch.logsumexp(logits, dim=1) - math.log(int(CORE.sum()))


def corridor_rank(scores, target, margin=1.):
    a, b = scores[::2], scores[1::2]
    ya, yb = target[::2], target[1::2]
    different = ya != yb
    return (F.relu(margin-(a-b)*(ya-yb))*different).sum()/different.sum().clamp_min(1)
