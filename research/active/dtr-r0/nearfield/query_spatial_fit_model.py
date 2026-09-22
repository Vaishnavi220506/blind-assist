"""Matched global-FiLM and spatial-query occupancy arms for a bounded fit test.

The spatial arm adds a zero-initialized 24-to-32 pointwise projection before
the original decoder. Its 768 weights are additional effective capacity; the
global control retains the same parameters/state layout but bypasses them.
Thus parameter storage is paired, while effective capacity is not identical.

Geometry uses the original analytic camera rays at six FIXED axial hypotheses,
not observed pixel depth, ToF-derived pixel depth, or inferred metric surfaces.
For each hypothesis the four channels are inward signed distances to query
xmin/xmax/ymin/ymax, divided by query width/height and clipped to [-1, 1].
Channel order is depth-major, then [X-xmin, xmax-X, Y-ymin, ymax-Y].

The frozen base implementation is imported without modification. Outputs remain
occupancy distance logits and visible-mask logits; UNKNOWN and invalid ToF
remain the base model's separate evidence axis. No data access or training is
performed here. Supply a local ImageNet checkpoint for research runs; None is
reserved for synthetic engineering tests, as in the base model.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from query_occupancy_model import (MASK_HW, RGB_HW, QueryOccupancyNet,
                                   canonicalize_tof, sample_region_features)


DEPTH_HYPOTHESES_M = (.525, 1., 1.5, 2., 2.5, 2.875)


def _validate_queries(queries: Tensor) -> None:
    if (not isinstance(queries, Tensor) or not queries.is_floating_point()
            or queries.ndim != 2 or queries.shape[1] != 4 or queries.shape[0] == 0):
        raise ValueError("Queries must be a floating tensor [Q,4] with Q > 0")
    if not bool(torch.isfinite(queries).all() & (queries[:, 1] > queries[:, 0]).all()
                & (queries[:, 3] > queries[:, 2]).all()):
        raise ValueError("Queries require finite ordered xmin,xmax,ymin,ymax bounds")


class QuerySpatialFitNet(QueryOccupancyNet):
    """Occupancy-only base plus a paired, optional spatial query residual.

    Construct both carriers from the same seed or copy the complete state_dict.
    Both have identical state_dict keys and parameter counts. Zero projection
    weights give exactly the original base predictions with shared base weights.
    ``encoder`` and occupancy output keys retain the original training interface.
    """

    def __init__(self, checkpoint_path: str | Path | None,
                 carrier: str = "global", n_bins: int = 6):
        if carrier not in {"global", "spatial"}:
            raise ValueError("carrier must be global or spatial")
        super().__init__(checkpoint_path, mode="occupancy", n_bins=n_bins)
        self.carrier = carrier
        self.register_buffer("query_depth_hypotheses_m", torch.tensor(DEPTH_HYPOTHESES_M), persistent=True)
        self.spatial_query_projection = nn.Conv2d(24, 32, kernel_size=1, bias=False)
        nn.init.zeros_(self.spatial_query_projection.weight)

    def query_geometry(self, queries: Tensor) -> Tensor:
        """Return [Q,24,45,80] geometry independent of RGB and sensor returns."""
        _validate_queries(queries)
        if queries.device != self.image_rays.device:
            raise ValueError("Queries and model geometry must have matching devices")
        rays = self.image_rays[0].to(dtype=queries.dtype)
        depth = self.query_depth_hypotheses_m.to(dtype=queries.dtype)[:, None, None]
        x, y = (rays[0] * depth)[None], (rays[1] * depth)[None]
        xmin, xmax, ymin, ymax = (queries[:, i, None, None, None] for i in range(4))
        width, height = xmax - xmin, ymax - ymin
        geometry = torch.stack(((x - xmin) / width, (xmax - x) / width,
                                (y - ymin) / height, (ymax - y) / height), dim=2)
        return geometry.flatten(1, 2).clamp(-1., 1.)

    def forward_features(self, rgb: Tensor, tof: Tensor, queries: Tensor) -> dict[str, Tensor]:
        if self.carrier == "global":
            # A real bypass: the stored branch cannot affect the control, even
            # if a loaded state contains nonzero spatial projection weights.
            return super().forward_features(rgb, tof, queries)

        # The base has no pre-decoder extension point. Preserve its validation
        # and computation verbatim up to the single residual addition below;
        # no hooks, shared mutable query state, or repeated encoder pass.
        if (not isinstance(rgb, Tensor) or not rgb.is_floating_point()
                or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, *RGB_HW) or rgb.shape[0] == 0):
            raise ValueError("RGB must be a floating tensor [B,3,180,320] with B > 0")
        if not bool(torch.isfinite(rgb).all()):
            raise ValueError("RGB must contain finite ImageNet-normalized values")
        _validate_queries(queries)
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
        regional = torch.einsum("bqz,bzc->bqc", scores.softmax(dim=-1), tokens)
        query_batch = query[None].expand(rgb.shape[0], -1, -1)
        gain, offset = self.condition(torch.cat((query_batch, regional), dim=-1)).chunk(2, dim=-1)
        spatial = visual[:, None] * (1 + gain.tanh()[..., None, None]) + offset[..., None, None]
        geometry = self.query_geometry(queries).to(dtype=spatial.dtype)
        spatial = spatial + self.spatial_query_projection(geometry)[None]
        batch, count, channels, height, width = spatial.shape
        spatial = self.decoder(spatial.reshape(batch * count, channels, height, width))
        spatial = spatial.reshape(batch, count, channels, height, width)
        pooled = torch.cat((spatial.mean(dim=(-2, -1)), spatial.amax(dim=(-2, -1))), dim=-1)
        return {"spatial_features": spatial, "query_features": pooled, "tof_valid": sensor[..., 1]}
