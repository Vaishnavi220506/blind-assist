"""Frozen MZ136 features, standardized TRAIN statistics, direct scalar residual.

Cell identity remains explicit until the corridor decision. No evaluation-only
fields, pair identity, or per-example parameters enter inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet, CELLS, CORE


class DirectReadoutNet(OccupancyNet):
    def __init__(self, ordered=False):
        super().__init__()
        self.ordered = ordered
        dimension = 24*8*49 if ordered else 48*8
        self.register_buffer('feature_mean', torch.zeros(dimension))
        self.register_buffer('feature_scale', torch.ones(dimension))
        self.readout = nn.Linear(dimension, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def components(self, batch):
        features = self.rgb(batch['rgb'])
        sampled = F.grid_sample(features, batch['grid'], align_corners=True)
        rgb = torch.cat([sampled.mean(-1), sampled.amax(-1)], 1).transpose(1, 2)
        cells = self.cells[None].expand(len(rgb), -1, -1)
        imu = batch['imu'][:, None].expand(-1, len(CELLS), -1)
        q = self.query(torch.cat([rgb, cells, imu], -1))
        tof, radar = self.tof(batch['tof']), self.radar(batch['radar'])
        weights = (q @ tof.transpose(1, 2)/math.sqrt(24)).masked_fill(~batch['local'], -1e4).softmax(-1)
        rw = (q @ radar.transpose(1, 2)/math.sqrt(24)).softmax(-1)
        logits = self.head(torch.cat([rgb, weights@tof, rw@radar, cells, imu], -1)).squeeze(-1)[:, CORE]
        base = torch.logsumexp(logits, dim=1)-math.log(int(CORE.sum()))
        visual = sampled[:, :, CORE].flatten(1) if self.ordered else rgb[:, CORE].flatten(1)
        return base, visual

    def forward(self, batch):
        base, visual = self.components(batch)
        return base+self.readout((visual-self.feature_mean)/self.feature_scale).squeeze(-1)
