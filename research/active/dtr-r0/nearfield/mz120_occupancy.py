"""Single-frame RGB/return fusion. No evaluator, identities, or alert inputs.

The only temporal preprocessing is the incumbent's deterministic IMU yaw sum;
there are no historical pixels, ranges, tracks, or learned recurrent states.
Cells denote possible object-volume occupancy, not certified free space.
"""
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DISTANCE_EDGES = (.2, 1.4, 2.6, 3.2, 3.6, 4.2)
SIDE_EDGES = (-1.5, -.3, .3, 1.5)
HEIGHT_EDGES = (.4, 1.5, 2.05, 2.7)
CELLS = np.array([[DISTANCE_EDGES[d], SIDE_EDGES[s], HEIGHT_EDGES[h],
                   DISTANCE_EDGES[d+1], SIDE_EDGES[s+1], HEIGHT_EDGES[h+1]]
                  for h, s, d in itertools.product(range(3), range(3), range(5))], np.float32)
CORE = np.array([h < 2 and s == 1 and d < 4
                 for h, s, d in itertools.product(range(3), range(3), range(5))])
HEAD = CORE & (CELLS[:, 2] == np.float32(1.5))


def encode(row, yaw):
    """Explicit observation allowlist; identity/time/family never become features."""
    intr = row['rgb_intrinsics']; pitch = math.radians(row['camera_pitch_deg'])
    yaw = math.radians(yaw); cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    basis = np.array([[cp*cy, cp*sy, sp], [-sy, cy, 0], [-sp*cy, -sp*sy, cp]])
    grids = []; rectangles = []
    for cell in CELLS:
        corners = np.array(list(itertools.product(*zip(cell[:3], cell[3:]))))
        camera = (corners-np.array(row['camera_in_body_m'])) @ basis.T
        forward = np.maximum(camera[:, 0], .02)
        uv = np.stack([intr['cx']+intr['fx']*camera[:, 1]/forward,
                       intr['cy']-intr['fy']*camera[:, 2]/forward], -1)
        lo, hi = uv.min(0), uv.max(0)
        rectangles.append([*lo, *hi])
        # Keep a high-resolution pixel branch; sample both extent and center.
        grids.append([[2*u/(intr['width']-1)-1, 2*v/(intr['height']-1)-1]
                      for v in np.linspace(lo[1], hi[1], 7) for u in np.linspace(lo[0], hi[0], 7)])
    rectangles = np.array(rectangles)
    tof = []; boxes = []
    for zone in row['tof_zones']:
        a, b = zone['theta_bounds_deg']; c, d = zone['phi_bounds_deg']
        box = [intr['cx']+intr['fx']*math.tan(math.radians(a)), intr['cy']-intr['fy']*math.tan(math.radians(d)),
               intr['cx']+intr['fx']*math.tan(math.radians(b)), intr['cy']-intr['fy']*math.tan(math.radians(c))]
        for slot in range(2):
            t = zone['targets'][slot] if slot < len(zone['targets']) and row['tof_packet_received'] else None
            valid = bool(t and t['status'] in ('SIM_VALID', 'SIM_MERGED'))
            tof.append([t['distance_m']/4.2 if valid else 0, t['range_noise_sigma_m'] if valid else 0,
                        t['signal_strength_proxy'] if valid else 0, float(valid),
                        float(valid and t['status'] == 'SIM_MERGED'), (a+b)/45, (c+d)/45,
                        float(row['tof_packet_received'])])
            boxes.append(box)
    # A null token is always available; missing returns are explicitly encoded.
    tof.append([0]*8); boxes = np.array(boxes)
    local = ((rectangles[:, None, 0] <= boxes[None, :, 2]+8) &
             (rectangles[:, None, 2] >= boxes[None, :, 0]-8) &
             (rectangles[:, None, 1] <= boxes[None, :, 3]+8) &
             (rectangles[:, None, 3] >= boxes[None, :, 1]-8))
    local = np.concatenate([local, np.ones((len(CELLS), 1), bool)], 1)
    radar = []
    for r, a, v, valid in zip(row['radar_range_m'], row['radar_angle'], row['radar_velocity'], row['radar_valid']):
        valid = bool(valid and row['radar_packet_received'] and r is not None and a is not None)
        angle = math.radians(a or 0)+yaw
        radar.append([r/4.2 if valid else 0, math.sin(angle) if valid else 0,
                      math.cos(angle) if valid else 0, v/3 if valid and v is not None else 0,
                      float(valid), float(valid and v is not None), float(row['radar_packet_received'])])
    radar.append([0]*7)
    return dict(grid=np.array(grids, np.float32), tof=np.array(tof, np.float32),
                local=local, radar=np.array(radar, np.float32),
                imu=np.array([math.sin(yaw), math.cos(yaw), sp, cp, float(row['imu_valid'])], np.float32))


class OccupancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.SiLU(),
                                 nn.Conv2d(16, 24, 1), nn.SiLU())
        self.tof = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 24))
        self.radar = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Linear(24, 24))
        self.query = nn.Linear(48+6+5, 24)
        self.head = nn.Sequential(nn.Linear(48+24+24+6+5, 64), nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer('cells', torch.from_numpy(CELLS.copy())/torch.tensor([4.2, 1.5, 2.7, 4.2, 1.5, 2.7]))

    def forward(self, batch):
        features = self.rgb(batch['rgb'])
        sampled = F.grid_sample(features, batch['grid'], align_corners=True)
        rgb = torch.cat([sampled.mean(-1), sampled.amax(-1)], 1).transpose(1, 2)
        cells = self.cells[None].expand(len(rgb), -1, -1)
        imu = batch['imu'][:, None].expand(-1, len(CELLS), -1)
        q = self.query(torch.cat([rgb, cells, imu], -1))
        tof = self.tof(batch['tof']); radar = self.radar(batch['radar'])
        weights = (q @ tof.transpose(1, 2)/math.sqrt(24)).masked_fill(~batch['local'], -1e4).softmax(-1)
        rw = (q @ radar.transpose(1, 2)/math.sqrt(24)).softmax(-1)
        return self.head(torch.cat([rgb, weights@tof, rw@radar, cells, imu], -1)).squeeze(-1)
