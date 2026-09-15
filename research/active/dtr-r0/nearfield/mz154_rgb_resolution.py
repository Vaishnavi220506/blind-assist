"""Frozen DVSR RGB-resolution contrast with exactly the old ToF/support.

Direct 256 RGB and bilinearly enlarged MZ153 128 RGB share one 8x8 depth
array. Both run the same scale=32 architecture and original checkpoint.
"""
import hashlib
import time

import cv2
import numpy as np
import torch

import mz153_dvsr_runtime as runtime153
import mz153_temporal_depth as adapter153

SIZE = 256
DEPTH_SCALE_M = adapter153.DEPTH_SCALE_M
METHOD = dict(
    contrast='DIRECT_ANGULAR_256_VS_EXACT_128_BILINEAR_TO_256',
    model='STRICT_ORIGINAL_DVSR_WEIGHTS_SCALE32_BOTH_ARMS',
    depth='EXACT_MZ153_ORIGINAL_8X8_SLOT_SELECTION_AND_FIXED4M',
    support='EXACT_MZ153_SHARED_RGB_MASK',
    validity='ALL_POSITIVE_WEIGHT_INTERPOLATION_CONTRIBUTORS_FINITE_POSITIVE',
)


def state_dict_digest(model):
    """Hash names, dtypes, shapes and exact frozen tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(str(value.dtype).encode('ascii') + b'\0')
        digest.update(str(value.shape).encode('ascii') + b'\0')
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_model(runtime, device='cuda'):
    """Return (model, info); only the nonparameter scale changes to 32."""
    model, info = runtime153.load_model(runtime, device)
    before = state_dict_digest(model)
    previous_scale = model.generator.scale
    if previous_scale != 16:
        raise ValueError('Expected original MZ153 scale=16')
    model.generator.scale = 32
    after = state_dict_digest(model)
    if before != after:
        raise RuntimeError('Scale change unexpectedly mutated frozen weights')
    info.update(rgb_size=SIZE, original_scale=previous_scale, scale=32,
                state_dict_sha256=after, state_dict_unchanged=True,
                method=METHOD)
    model.runtime_metadata = dict(info)
    return model, info


def make_inputs(row, bgr):
    """Return direct256, enlarged128, identical original depth, and audit."""
    old_rgb, depth, old_audit = adapter153.make_input(row, bgr)
    _, _, _, (a, b, c, d) = adapter153.angular_grid(row)
    intr = row['rgb_intrinsics']
    theta = a + (np.arange(SIZE) + .5) * (b-a) / SIZE
    phi = d - (np.arange(SIZE) + .5) * (d-c) / SIZE
    u = intr['cx'] + intr['fx'] * np.tan(np.deg2rad(theta))
    v = intr['cy'] - intr['fy'] * np.tan(np.deg2rad(phi))
    xx, yy = np.meshgrid(u, v)
    visible = (xx >= 0) & (xx <= intr['width']-1) & (yy >= 0) & (yy <= intr['height']-1)
    direct = cv2.remap(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                       xx.astype(np.float32), yy.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    direct[~visible] = 0
    direct = direct.transpose(2, 0, 1).astype(np.float32) / 255.
    enlarged = cv2.resize(old_rgb.transpose(1, 2, 0), (SIZE, SIZE),
                          interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).copy()
    audit = dict(mz153=old_audit, method=METHOD,
                 depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                 old_rgb_sha256=hashlib.sha256(old_rgb.tobytes()).hexdigest(),
                 direct_angular_rgb_coverage=float(visible.mean()),
                 identical_depth_for_both_arms=True)
    return direct, enlarged, depth, audit


def full_rgb_depth(row, raw256):
    """Map to source RGB, preserving exactly MZ153's geometric support mask."""
    raw = np.asarray(raw256, np.float32)
    if raw.shape != (SIZE, SIZE):
        raise ValueError('raw depth must be 256x256')
    intr = row['rgb_intrinsics']
    _, _, _, (a, b, c, d) = adapter153.angular_grid(row)
    yy, xx = np.mgrid[:intr['height'], :intr['width']]
    theta = np.rad2deg(np.arctan((xx-intr['cx'])/intr['fx']))
    phi = np.rad2deg(np.arctan((intr['cy']-yy)/intr['fy']))
    x = ((theta-a)/(b-a)*SIZE-.5).astype(np.float32)
    y = ((d-phi)/(d-c)*SIZE-.5).astype(np.float32)
    # Call the frozen adapter instead of approximating its boundary rounding.
    _, shared = adapter153.full_rgb_depth(
        row, np.ones((adapter153.SIZE, adapter153.SIZE), np.float32))
    valid = np.isfinite(raw) & (raw > 0)
    values = np.where(valid, raw*DEPTH_SCALE_M, 0.)
    result = cv2.remap(values, x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    weight = cv2.remap(valid.astype(np.float32), x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    result[~shared | (weight < 1.-1e-6) | ~np.isfinite(result) | (result <= 0)] = np.nan
    return result, shared


def infer_latest(model, rgb, depth):
    """Run all official passes inside T=2..6; return raw latest 256 square."""
    start = time.perf_counter()
    device = next(model.parameters()).device
    rgb = torch.as_tensor(rgb, dtype=torch.float32, device=device).contiguous()
    depth = torch.as_tensor(depth, dtype=torch.float32, device=device).contiguous()
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, SIZE, SIZE):
        raise ValueError('rgb must be T,3,256,256')
    if tuple(depth.shape) != (rgb.shape[0], 1, 8, 8) or not 2 <= rgb.shape[0] <= 6:
        raise ValueError('depth must be T,1,8,8 and T in 2..6')
    if model.generator.scale != 32:
        raise ValueError('MZ154 requires scale=32 for both comparison arms')
    for name, value in [('rgb', rgb), ('depth', depth)]:
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(name+' must be finite in 0..1')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    compute_start = time.perf_counter()
    with torch.inference_mode():
        dense, _ = model(depth.unsqueeze(0), rgb.unsqueeze(0))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    compute_s = time.perf_counter()-compute_start
    output = dense[0, -1, 0].detach().float().cpu().numpy().copy()
    if output.shape != (SIZE, SIZE) or not np.isfinite(output).all():
        raise RuntimeError('Nonfinite or wrong-shaped dense depth')
    model.last_inference_runtime = dict(
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch=torch.__version__, cuda=torch.version.cuda, T=rgb.shape[0], rgb_size=SIZE,
        compute_seconds=compute_s, total_seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        output_min=float(output.min()), output_max=float(output.max()), finite=True, clamped=False)
    return output
