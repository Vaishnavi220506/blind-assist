"""One frozen-encoder spatial RGB/ToF BCE model; no evaluator data in inference.

The 64 supplied zones are row-major normalized y0,x0,y1,x1 image-edge boxes.
Each has nine ordered samples of frozen layer-3 features. No global RGB average,
depth reconstruction, semantic mask, previous frame, or baseline alert is used.
The fixed nominal projection is a simulation assumption, not hardware calibration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.research_backend import (BackendCandidate, Workload, select_backend,
                                    torch_observation)
from tof_fov45_core import boxes45

RECIPE = dict(
    id="SPATIAL_BCE_20260920", seed=20260920, steps=1200, batch_size=64,
    optimizer="AdamW", learning_rate=0.001, weight_decay=0.0001,
    loss="BCEWithLogitsLoss; unweighted mean", sampling="replacement; uniform train rows",
    checkpoint_selection="last step only", augmentation="none", ranking_loss="none",
    encoder="torchvision MobileNetV3-small ImageNet1K V1 features[:4]; frozen eval",
    rgb_size_wh=[256, 144], image_normalization="ImageNet mean/std; PIL bilinear",
    zone_sampling="3x3 subcell centers per normalized ToF box; bilinear align_corners=False",
    rgb_channels=216, sensor_channels=["range_m/8", "valid", "horizontal_angle/45deg", "vertical_angle/45deg"],
    sensor_validity="finite and 0.1 <= axial proxy range < 8 m; invalid range channel=0",
    head="Conv1x1(220,32)-ReLU-Conv3x3(32,32,pad1)-ReLU-Flatten-Linear(2048,32)-ReLU-Linear(32,1)",
    raw_output="one independent current-frame logit; not a measured distance",
    precision="float32; TF32 disabled", test_labels="not accepted by training or prediction API",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalized_boxes45():
    """Convert actual rounded 192x256 lattice bounds into full-image coordinates."""
    return boxes45().astype(np.float32) / np.array([192, 256, 192, 256], np.float32)


def _boxes(boxes):
    boxes = np.asarray(normalized_boxes45() if boxes is None else boxes, np.float32)
    if boxes.shape != (64, 4) or not np.isfinite(boxes).all():
        raise ValueError("Expected finite normalized [64,4] y0,x0,y1,x1 boxes")
    if not ((boxes >= 0).all() and (boxes <= 1).all()
            and (boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("Empty or out-of-image normalized boxes")
    grid = boxes.reshape(8, 8, 4)
    if not (np.allclose(grid[:, :, 0], grid[:, :1, 0])
            and np.allclose(grid[:, :, 2], grid[:, :1, 2])
            and np.allclose(grid[:, :, 1], grid[:1, :, 1])
            and np.allclose(grid[:, :, 3], grid[:1, :, 3])
            and np.allclose(grid[:, :-1, 3], grid[:, 1:, 1])
            and np.allclose(grid[:-1, :, 2], grid[1:, :, 0])):
        raise ValueError("Zones must form a contiguous row-major 8x8 grid")
    return boxes


def sample_zones(feature_map, boxes=None):
    """[N,24,H,W] -> [N,216,8,8]; channels are feature, subrow, subcol."""
    if feature_map.ndim != 4 or feature_map.shape[1] != 24:
        raise ValueError("Expected layer-3 feature map [N,24,H,W]")
    b = torch.as_tensor(_boxes(boxes), dtype=feature_map.dtype, device=feature_map.device)
    frac = (torch.arange(3, device=b.device, dtype=b.dtype) + 0.5) / 3
    yy = b[:, 0, None, None] + (b[:, 2] - b[:, 0])[:, None, None] * frac[None, :, None]
    xx = b[:, 1, None, None] + (b[:, 3] - b[:, 1])[:, None, None] * frac[None, None, :]
    xy = torch.stack([xx.expand(-1, 3, 3), yy.expand(-1, 3, 3)], -1)
    grid = (2 * xy - 1).reshape(1, 64, 9, 2).expand(len(feature_map), -1, -1, -1)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.permute(0, 1, 3, 2).reshape(len(feature_map), 216, 8, 8)


def build_inputs(rgb_features, ranges, boxes=None):
    """Observation-only construction. Invalid values stay masked, not far negatives."""
    features = np.asarray(rgb_features, np.float32)
    ranges = np.asarray(ranges, np.float32)
    if features.ndim != 4 or features.shape[1:] != (216, 8, 8):
        raise ValueError("Expected cached RGB features [N,216,8,8]")
    if ranges.shape != (len(features), 64) or not np.isfinite(features).all():
        raise ValueError("Expected [N,64] ranges and finite RGB features")
    b = _boxes(boxes)
    valid = np.isfinite(ranges) & (ranges >= 0.1) & (ranges < 8)
    observed = np.where(valid, ranges, 0) / 8
    x = (b[:, 1] + b[:, 3]) / 2
    y = (b[:, 0] + b[:, 2]) / 2
    # Original 640x360 pinhole, 100-degree HFOV; vertical focal equals horizontal.
    focal = 640 / (2 * np.tan(np.deg2rad(50)))
    ax = np.arctan((x * 640 - 320) / focal) / np.deg2rad(45)
    ay = np.arctan((y * 360 - 180) / focal) / np.deg2rad(45)
    sensor = np.stack([observed, valid.astype(np.float32),
                       np.broadcast_to(ax, ranges.shape), np.broadcast_to(ay, ranges.shape)], 1)
    return np.concatenate([features, sensor.reshape(-1, 4, 8, 8)], 1).astype(np.float32)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(220, 32, 1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(2048, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (220, 8, 8):
            raise ValueError("Expected fused observations [N,220,8,8]")
        return self.net(inputs).squeeze(-1)


def _precision():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select(model, batch, output_path, labels=None):
    """Same-batch CPU/GPU probe; disposable model copies never alter actual fit."""
    candidates = {}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        probe_model = copy.deepcopy(model).to(device)
        probe_x = batch.to(device)
        probe_y = None if labels is None else labels.to(device)
        if labels is None:
            probe_model.eval()
            def run(m=probe_model, x=probe_x):
                with torch.inference_mode():
                    return m(x)
        else:
            probe_model.train()
            optimizer = torch.optim.AdamW(probe_model.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
            def run(m=probe_model, x=probe_x, y=probe_y, opt=optimizer):
                opt.zero_grad(set_to_none=True)
                out = m(x)
                F.binary_cross_entropy_with_logits(out, y).backward()
                opt.step()
                return out.detach()
        candidates[device] = BackendCandidate(device, device, run,
            lambda out, m=probe_model: torch_observation(model=m, output=out),
            torch.cuda.synchronize if device == "cuda" else lambda: None)
    record = select_backend(Workload.BATCH_TENSOR if labels is not None else Workload.MODEL_INFERENCE,
        cpu=candidates["cpu"], gpu=candidates.get("cuda"),
        cpu_reason=None if "cuda" in candidates else "ACCELERATOR_UNAVAILABLE",
        record_path=Path(output_path), warmups=1, repeats=3)
    return record["selected_device_type"], record


def _load_encoder(checkpoint_path):
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    encoder = model.features[:4].eval()
    encoder.requires_grad_(False)
    return encoder


def _read_rgb(path):
    with Image.open(path) as image:
        if image.width * 9 != image.height * 16:
            raise ValueError("Frozen 256x144 resize requires 16:9 RGB; no silent aspect distortion")
        array = np.asarray(image.convert("RGB").resize((256, 144), Image.Resampling.BILINEAR), np.float32).copy()
    x = torch.from_numpy(array).permute(2, 0, 1) / 255
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def encode_rgb(rgb_paths, checkpoint_path, output_dir, boxes=None, batch_size=64):
    """Return [N,216,8,8] cache with source/checkpoint hashes; no labels accepted.

    Reuse requires an exact fingerprint match. Caller may supply only train/dev
    images first; prediction is a separate operation with no implicit test pass.
    """
    _precision()
    paths = [Path(p).resolve() for p in rgb_paths]
    if not paths or batch_size != RECIPE["batch_size"]:
        raise ValueError("Need RGB paths and frozen feature batch_size=64")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    b = _boxes(boxes)
    manifest = dict(recipe=RECIPE, boxes=b.tolist(), checkpoint_sha256=sha256(checkpoint_path),
                    sources=[dict(path=str(p), sha256=sha256(p)) for p in paths])
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache, receipt = dest / "rgb_features.npy", dest / "feature_receipt.json"
    if cache.exists() or receipt.exists():
        if not (cache.exists() and receipt.exists()):
            raise FileExistsError("Incomplete feature cache; inspect before any retry")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint or previous["cache_sha256"] != sha256(cache):
            raise ValueError("Feature cache source/recipe/content mismatch")
        return np.load(cache, mmap_mode="r")
    encoder = _load_encoder(checkpoint_path)
    probe = torch.stack([_read_rgb(p) for p in paths[:64]])
    device, backend = _select(encoder, probe, dest / "encoder_backend.json")
    encoder = encoder.to(device).eval()
    features = np.empty((len(paths), 216, 8, 8), np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            rgb = torch.stack([_read_rgb(p) for p in paths[start:start + batch_size]]).to(device)
            features[start:start + len(rgb)] = sample_zones(encoder(rgb), b).cpu().numpy()
            if start % 512 == 0:
                print(f"SPATIAL_FEATURES {start + len(rgb)}/{len(paths)}", flush=True)
    np.save(cache, features)
    _write_json(receipt, dict(**manifest, fingerprint=fingerprint, cache_sha256=sha256(cache),
        shape=list(features.shape), elapsed_seconds=time.perf_counter() - started,
        encoder_parameters=sum(p.numel() for p in encoder.parameters()), backend=backend))
    return features


def train_fit(inputs, train_indices, train_labels, output_dir):
    """Fixed 1200-step fit. Returns Head. Labels align ONLY with indices.

    No dev/test labels or threshold selection enter this API. The output directory
    cannot contain a prior fit; no implicit retraining or checkpoint selection.
    """
    _precision()
    x = np.asarray(inputs, np.float32)
    idx = np.asarray(train_indices, np.int64)
    y = np.asarray(train_labels, np.float32)
    if x.ndim != 4 or x.shape[1:] != (220, 8, 8) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,220,8,8] input array")
    if idx.ndim != 1 or y.shape != idx.shape or len(idx) == 0 or len(set(idx.tolist())) != len(idx):
        raise ValueError("Train indices must be unique; labels align only with these indices")
    if idx.min() < 0 or idx.max() >= len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid train indices or nonbinary train labels")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / name).exists() for name in ("head_last.pt", "training.jsonl", "train_receipt.json")):
        raise FileExistsError("Prior fit or partial fit exists; no automatic retraining")
    torch.manual_seed(RECIPE["seed"])
    model = Head()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    train_x, train_y = torch.from_numpy(x[idx].copy()), torch.from_numpy(y.copy())
    probe_idx = torch.arange(RECIPE["batch_size"]) % len(idx)
    device, backend = _select(model, train_x[probe_idx], dest / "train_backend.json", train_y[probe_idx])
    model.load_state_dict(initial)
    model.to(device).train()
    train_x, train_y = train_x.to(device), train_y.to(device)
    rng = np.random.default_rng(RECIPE["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    started = time.perf_counter()
    with (dest / "training.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, RECIPE["steps"] + 1):
            batch = torch.as_tensor(rng.integers(0, len(idx), RECIPE["batch_size"]), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0:
                item = dict(step=step, train_batch_bce=float(loss.detach().cpu()), elapsed_seconds=time.perf_counter() - started)
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                print("SPATIAL_TRAIN " + json.dumps(item), flush=True)
    model.eval()
    final = dest / "head_last.pt"
    torch.save(dict(recipe=RECIPE, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), final)
    receipt = dict(recipe=RECIPE, train_rows=len(idx), train_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
                   train_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                   train_inputs_sha256=hashlib.sha256(x[idx].tobytes()).hexdigest(),
                   trainable_parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint_sha256=sha256(final), elapsed_seconds=time.perf_counter() - started, backend=backend)
    _write_json(dest / "train_receipt.json", receipt)
    return model


def load_head(checkpoint_path, device="cpu"):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["recipe"] != RECIPE:
        raise ValueError("Checkpoint recipe mismatch")
    model = Head()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(head, inputs, indices=None, batch_size=64):
    """Return logits for explicit selected observations; no labels or hidden evaluation."""
    x = np.asarray(inputs, np.float32)
    selected = np.arange(len(x)) if indices is None else np.asarray(indices, np.int64)
    if selected.ndim != 1 or (len(selected) and (selected.min() < 0 or selected.max() >= len(x))):
        raise ValueError("Prediction indices out of bounds")
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    head.eval()
    device = next(head.parameters()).device
    result = []
    for start in range(0, len(selected), batch_size):
        value = torch.from_numpy(x[selected[start:start + batch_size]].copy()).to(device)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite prediction inputs")
        result.append(head(value).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, np.float32)
