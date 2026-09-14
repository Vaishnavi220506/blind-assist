"""TRAIN fit repair: retain seven ordered horizontal RGB sample columns.

The zero-initialized residual preserves the original checkpoint exactly before
training. Public sampling geometry only; no pair IDs or evaluator features.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet, CELLS, CORE


class PositionCorridorNet(OccupancyNet):
    def __init__(self):
        super().__init__()
        self.horizontal = nn.Linear(24*7, 48, bias=False)
        nn.init.zeros_(self.horizontal.weight)

    def forward(self, batch):
        features = self.rgb(batch['rgb'])
        sampled = F.grid_sample(features, batch['grid'], align_corners=True)
        rgb = torch.cat([sampled.mean(-1), sampled.amax(-1)], 1).transpose(1, 2)
        # Encoder grid is row-major: 7 vertical positions x 7 horizontal ones.
        # Average vertically, retaining left-to-right positions within each cell.
        columns = sampled.unflatten(-1, (7, 7)).mean(-2).permute(0, 2, 1, 3).flatten(-2)
        rgb = rgb + self.horizontal(columns)
        cells = self.cells[None].expand(len(rgb), -1, -1)
        imu = batch['imu'][:, None].expand(-1, len(CELLS), -1)
        q = self.query(torch.cat([rgb, cells, imu], -1))
        tof = self.tof(batch['tof'])
        radar = self.radar(batch['radar'])
        weights = (q @ tof.transpose(1, 2)/math.sqrt(24)).masked_fill(~batch['local'], -1e4).softmax(-1)
        rw = (q @ radar.transpose(1, 2)/math.sqrt(24)).softmax(-1)
        logits = self.head(torch.cat([rgb, weights@tof, rw@radar, cells, imu], -1)).squeeze(-1)[:, CORE]
        return torch.logsumexp(logits, dim=1)-math.log(int(CORE.sum()))
