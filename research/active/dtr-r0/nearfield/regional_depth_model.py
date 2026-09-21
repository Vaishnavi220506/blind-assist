"""Predict local depth distributions; regional returns supervise a soft winner.

Neither depth labels, target identities nor native contributor lists are accepted
by inference. Returned values remain regional optical-Z simulation observations.
"""
import copy
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation

RECIPE = dict(seed=20260921, steps=1200, batch_size=16, learning_rate=.001,
              weight_decay=.0001, depth_classes=81, measurement_weight=.25,
              alert_weight=.25, winner_temperature=.1, top_pixels=16,
              checkpoint='last', scope='CONSUMED_SIMULATION_DEVELOPMENT')
ARMS = ('POINT', 'REGION')


def depths(device=None, dtype=torch.float32):
    return torch.cat((torch.arange(80, device=device, dtype=dtype)*.1+.05,
                      torch.tensor([8.5], device=device, dtype=dtype)))


class Head(nn.Module):
    """One identical small encoder/decoder for both measurement operators."""
    def __init__(self):
        super().__init__()
        self.full = nn.Sequential(nn.Conv2d(7, 24, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU())
        self.low = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(48, 48, 3, padding=2, dilation=2), nn.ReLU())
        self.decode = nn.Sequential(nn.Conv2d(24+32+48, 48, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(48, 81, 1))

    def distribution_logits(self, x):
        if x.ndim != 4 or x.shape[1:] != (7, 64, 64):
            raise ValueError('Expected public [N,7,64,64] RGB/range/valid/ray input')
        f = self.full(x); h = self.middle(f); low = self.low(h)
        return self.decode(torch.cat([f, F.interpolate(h, size=(64,64), mode='bilinear', align_corners=False),
                                       F.interpolate(low, size=(64,64), mode='bilinear', align_corners=False)], 1))

    def forward(self, x):
        return alert_score(self.distribution_logits(x), x)[0]


def corridor_probability(logits, x):
    z = depths(logits.device, logits.dtype)[None, :, None, None]
    inside = ((z >= .3) & (z <= 3.) & (torch.abs(x[:, 5:6]*z) <= .3)
              & (x[:, 6:7]*z >= -.2) & (x[:, 6:7]*z <= .9))
    return (logits.softmax(1)*inside).sum(1)


def alert_score(logits, x):
    local = corridor_probability(logits, x)
    probability = local.flatten(1).topk(RECIPE['top_pixels'], dim=1).values.mean(1)
    return torch.logit(probability.clamp(1e-6, 1-1e-6)), local


def zone_distribution(logits, arm):
    if arm not in ARMS:
        raise ValueError('Expected POINT or REGION')
    p = logits.softmax(1).reshape(-1, 81, 8, 8, 8, 8)
    # Axes: batch,depth,zone-row,pixel-row,zone-col,pixel-col.
    return p.mean((3,5)) if arm == 'REGION' else p[:, :, :, 4, :, 4]


def measurement_loss(logits, x, arm):
    """Soft dominant inverse-square-energy bin, not a measured histogram.

    REGION marginalizes predicted surface distributions over the entire sampled
    zone; POINT substitutes one central sample. This is the only arm difference.
    Missing zones supply zero loss, not empty-space supervision.
    """
    p = zone_distribution(logits, arm)[:, 1:80]
    z = depths(logits.device, logits.dtype)[1:80][None, :, None, None]
    energy = p/z.clamp_min(.3).square()
    energy = energy/energy.sum(1, keepdim=True).clamp_min(1e-8)
    winner = (energy/RECIPE['winner_temperature']).softmax(1)
    observed = x[:, 3, ::8, ::8]*8
    valid = x[:, 4, ::8, ::8] > .5
    radius = .1+3*(.01+.02*observed)
    # Probability mass on an inherited public interval, with no private winner.
    compatible = (torch.abs(z-observed[:, None]) <= radius[:, None]) & (z >= .1)
    mass = (winner*compatible).sum(1).clamp_min(1e-8)
    return (-(mass.log())*valid).sum()/valid.sum().clamp_min(1)


def loss(head, x, classes, truth, arm):
    logits = head.distribution_logits(x)
    pixels = F.cross_entropy(logits, classes, ignore_index=-100, reduction='sum')
    pixels = pixels/(classes != -100).sum().clamp_min(1)
    measurement = measurement_loss(logits, x, arm)
    score, _ = alert_score(logits, x)
    alert = F.binary_cross_entropy_with_logits(score, truth)
    total = pixels+RECIPE['measurement_weight']*measurement+RECIPE['alert_weight']*alert
    return total, dict(depth=pixels, measurement=measurement, alert=alert)


def select_device(prototype, x, path, classes=None, truth=None):
    """Benchmark actual inference or full REGION training loss on disposable copies."""
    candidates = {}
    for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
        h = copy.deepcopy(prototype).to(device)
        inputs = x.to(device)
        if classes is None:
            h.eval()
            def probe(h=h, inputs=inputs):
                with torch.inference_mode():
                    return h(inputs)
        else:
            h.train(); c = classes.to(device); y = truth.to(device)
            opt = torch.optim.AdamW(h.parameters(), lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
            def probe(h=h, inputs=inputs, c=c, y=y, opt=opt):
                opt.zero_grad(set_to_none=True)
                value, _ = loss(h, inputs, c, y, 'REGION')
                value.backward(); opt.step()
                return value
        candidates[device] = BackendCandidate(device, device, probe,
            lambda output, h=h: torch_observation(model=h, output=output),
            torch.cuda.synchronize if device == 'cuda' else lambda: None)
    record = select_backend(Workload.MODEL_INFERENCE if classes is None else Workload.BATCH_TENSOR,
        cpu=candidates['cpu'], gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=Path(path), warmups=1, repeats=3)
    return record['selected_device_type'], record
