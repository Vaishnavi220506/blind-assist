"""Matched single-frame RGB/ToF query classifier and first-surface prototype.

Inputs are ImageNet-normalized 320x180 RGB, 64 regional sensor observations
``[range_m / 8, valid, y0, x0, y1, x1]``, and metric camera-frame queries
``[xmin, xmax, ymin, ymax]``. Boxes use normalized image-edge coordinates.
Range is an observation of a region, never assigned as exact pixel depth.

Both modes have identical encoders, regional fusion, query decoding and state
dictionary keys. Only the selected output heads differ. The last distance
class means no occupied surface in the specified distance interval, NOT
UNKNOWN. Sensor/evaluator evidence validity remains a separate caller-owned
axis; a model prediction cannot establish observed free space.

Distance bin boundaries and label validity belong to the data contract. The
model assumes neither an ordering among the 64 regions nor a fixed query count.
There is no history, baseline-alert input, auxiliary alert head for occupancy,
download, data loading, training loop or threshold selection in this module.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


RGB_HW = (180, 320)
MASK_HW = (45, 80)
CAMERA_HFOV_DEG = 100.0
RANGE_SCALE_M = 8.0


def canonicalize_tof(tof: Tensor) -> Tensor:
    """Retain region geometry and missingness; invalid range payloads become 0.

    A valid flag cannot rescue a nonfinite/out-of-range measurement. Missing
    readings retain valid=0 rather than receiving a far-range/clear sentinel.
    Bad flags or box geometry are malformed inputs and raise ValueError.
    This returns a new tensor and does not modify the supplied observations.
    """
    if not isinstance(tof, Tensor) or not tof.is_floating_point():
        raise ValueError("ToF must be a floating tensor [B,64,6]")
    if tof.ndim != 3 or tuple(tof.shape[1:]) != (64, 6) or tof.shape[0] == 0:
        raise ValueError("ToF must have shape [B,64,6] with B > 0")
    flags, boxes = tof[..., 1], tof[..., 2:]
    if not bool(((flags == 0) | (flags == 1)).all()):
        raise ValueError("ToF valid flags must be exactly 0 or 1")
    good_boxes = (torch.isfinite(boxes).all()
                  & ((boxes >= 0) & (boxes <= 1)).all()
                  & (boxes[..., 2:] > boxes[..., :2]).all())
    if not bool(good_boxes):
        raise ValueError("ToF boxes must be finite, nonempty normalized y0,x0,y1,x1")
    ranges = tof[..., 0]
    valid = (flags == 1) & torch.isfinite(ranges) & (ranges > 0) & (ranges < 1)
    observed = torch.where(valid, ranges, torch.zeros_like(ranges))
    return torch.cat((observed[..., None], valid.to(tof.dtype)[..., None], boxes), dim=-1)


def sample_region_features(features: Tensor, boxes: Tensor) -> Tensor:
    """Pool nine subcell feature samples for each region; no depth rasterization.

    ``[B,C,H,W]`` plus normalized ``[B,Z,4]`` boxes gives ``[B,Z,C]``.
    Validation of sensor geometry is done once by ``canonicalize_tof``.
    """
    if features.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("Expected features [B,C,H,W] and boxes [B,Z,4]")
    if features.shape[0] != boxes.shape[0] or features.device != boxes.device:
        raise ValueError("Region features and boxes must have matching batches and devices")
    boxes = boxes.to(dtype=features.dtype)
    fractions = (torch.arange(3, device=features.device, dtype=features.dtype) + .5) / 3
    yy = boxes[..., 0, None, None] + (boxes[..., 2] - boxes[..., 0])[..., None, None] * fractions[None, None, :, None]
    xx = boxes[..., 1, None, None] + (boxes[..., 3] - boxes[..., 1])[..., None, None] * fractions[None, None, None, :]
    grid = torch.stack((xx.expand(-1, -1, 3, 3), yy.expand(-1, -1, 3, 3)), dim=-1)
    grid = (2 * grid - 1).reshape(boxes.shape[0], boxes.shape[1], 9, 2)
    samples = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return samples.mean(dim=-1).transpose(1, 2)


def distance_cdf(distance_logits: Tensor) -> Tensor:
    """P(first occupied surface <= each bin upper edge), excluding no-surface.

    A categorical softmax gives a monotone CDF by construction. For six
    occupied-distance classes plus no-surface, ``[..., -1]`` of the returned
    CDF is the interval-wide occupancy probability. UNKNOWN is not a class.
    """
    if (not isinstance(distance_logits, Tensor) or not distance_logits.is_floating_point()
            or distance_logits.ndim < 1 or distance_logits.shape[-1] < 2):
        raise ValueError("Distance logits must end in occupied bins plus one no-surface class")
    if not bool(torch.isfinite(distance_logits).all()):
        raise ValueError("Distance logits must be finite")
    # Accumulate low-precision inference outputs in float32 for the probability contract.
    dtype = torch.float32 if distance_logits.dtype in (torch.float16, torch.bfloat16) else distance_logits.dtype
    return F.softmax(distance_logits, dim=-1, dtype=dtype)[..., :-1].cumsum(dim=-1).clamp(max=1.0)


class QueryOccupancyNet(nn.Module):
    """Small paired network; use one initial state_dict for both experiment arms.

    ``checkpoint_path`` is the local torchvision ImageNet MobileNetV3-small
    state_dict. ``None`` explicitly requests random initialization for synthetic
    engineering tests; research runs must supply their recorded local checkpoint.
    No network access occurs. All retained encoder blocks are trainable.

    ``forward_features`` exposes the shared representation for pairing audits.
    Both output heads always exist in state_dict, even when unused by a mode.
    """

    def __init__(self, checkpoint_path: str | Path | None, mode: str = "occupancy", n_bins: int = 6):
        super().__init__()
        if mode not in {"classifier", "occupancy"}:
            raise ValueError("mode must be classifier or occupancy")
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        self.mode = mode
        self.n_bins = n_bins
        backbone = mobilenet_v3_small(weights=None)
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Local ImageNet checkpoint not found: {checkpoint_path}")
            backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
        self.encoder = backbone.features[:7]
        channels = 32
        self.low_project = nn.Conv2d(16, 24, kernel_size=1)
        self.high_project = nn.Conv2d(40, 24, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(50, channels, kernel_size=1), nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.GroupNorm(8, channels), nn.SiLU(),
        )
        self.region_encoder = nn.Sequential(
            nn.Linear(channels + 6, channels), nn.SiLU(), nn.Linear(channels, channels), nn.SiLU(),
        )
        self.query_encoder = nn.Sequential(nn.Linear(4, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.region_key = nn.Linear(channels, channels, bias=False)
        self.condition = nn.Sequential(nn.Linear(2 * channels, 2 * channels), nn.SiLU(),
                                       nn.Linear(2 * channels, 2 * channels))
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1), nn.GroupNorm(8, channels), nn.SiLU(),
        )
        # Creation order and shapes are deliberately independent of mode.
        self.classifier_head = nn.Linear(2 * channels, 1)
        self.distance_head = nn.Linear(2 * channels, n_bins + 1)
        self.mask_head = nn.Conv2d(channels, 1, kernel_size=1)
        yy, xx = torch.meshgrid((torch.arange(MASK_HW[0], dtype=torch.float32) + .5) / MASK_HW[0] - .5,
                                (torch.arange(MASK_HW[1], dtype=torch.float32) + .5) / MASK_HW[1] - .5,
                                indexing="ij")
        # Camera X-right/Y-down/Z-forward ray ratios; these are not depth values.
        scale = 2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2))
        rays = torch.stack((xx * scale, yy * scale * RGB_HW[0] / RGB_HW[1]))[None]
        self.register_buffer("image_rays", rays, persistent=True)

    def forward_features(self, rgb: Tensor, tof: Tensor, queries: Tensor) -> dict[str, Tensor]:
        """Return shared spatial/query features and canonical sensor validity."""
        if (not isinstance(rgb, Tensor) or not rgb.is_floating_point()
                or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, *RGB_HW) or rgb.shape[0] == 0):
            raise ValueError("RGB must be a floating tensor [B,3,180,320] with B > 0")
        if not bool(torch.isfinite(rgb).all()):
            raise ValueError("RGB must contain finite ImageNet-normalized values")
        if (not isinstance(queries, Tensor) or not queries.is_floating_point()
                or queries.ndim != 2 or queries.shape[1] != 4 or queries.shape[0] == 0):
            raise ValueError("Queries must be a floating tensor [Q,4] with Q > 0")
        if not bool(torch.isfinite(queries).all() & (queries[:, 1] > queries[:, 0]).all()
                    & (queries[:, 3] > queries[:, 2]).all()):
            raise ValueError("Queries require finite ordered xmin,xmax,ymin,ymax bounds")
        sensor = canonicalize_tof(tof)
        if sensor.shape[0] != rgb.shape[0] or sensor.device != rgb.device or queries.device != rgb.device:
            raise ValueError("RGB, ToF and queries must have matching devices and RGB/ToF batches")
        x = rgb
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index == 1:
                low = x
        high = F.interpolate(self.high_project(x), size=MASK_HW, mode="bilinear", align_corners=False)
        rays = self.image_rays.to(dtype=low.dtype).expand(rgb.shape[0], -1, -1, -1)
        visual = self.fusion(torch.cat((self.low_project(low), high, rays), dim=1))
        region_rgb = sample_region_features(visual, sensor[..., 2:])
        tokens = self.region_encoder(torch.cat((region_rgb, sensor.to(dtype=region_rgb.dtype)), dim=-1))
        query = self.query_encoder(queries.to(dtype=visual.dtype) / 3.0)
        scores = torch.einsum("bzc,qc->bqz", self.region_key(tokens), query) / math.sqrt(tokens.shape[-1])
        # Invalid regions still carry explicit missingness and their RGB/box context;
        # they are neither far surfaces nor silently discarded all-masked softmax rows.
        regional = torch.einsum("bqz,bzc->bqc", scores.softmax(dim=-1), tokens)
        query_batch = query[None].expand(rgb.shape[0], -1, -1)
        gain, offset = self.condition(torch.cat((query_batch, regional), dim=-1)).chunk(2, dim=-1)
        spatial = visual[:, None] * (1 + gain.tanh()[..., None, None]) + offset[..., None, None]
        batch, count, channels, height, width = spatial.shape
        spatial = self.decoder(spatial.reshape(batch * count, channels, height, width))
        spatial = spatial.reshape(batch, count, channels, height, width)
        pooled = torch.cat((spatial.mean(dim=(-2, -1)), spatial.amax(dim=(-2, -1))), dim=-1)
        return {"spatial_features": spatial, "query_features": pooled, "tof_valid": sensor[..., 1]}

    def forward(self, rgb: Tensor, tof: Tensor, queries: Tensor) -> dict[str, Tensor]:
        features = self.forward_features(rgb, tof, queries)
        if self.mode == "classifier":
            return {"query_logits": self.classifier_head(features["query_features"]).squeeze(-1)}
        spatial = features["spatial_features"]
        batch, count, channels, height, width = spatial.shape
        masks = self.mask_head(spatial.reshape(batch * count, channels, height, width))
        return {"distance_logits": self.distance_head(features["query_features"]),
                "mask_logits": masks.reshape(batch, count, height, width)}
