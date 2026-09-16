"""Frozen UniDepthV2 RGB-only metric prior; public calibration is optional.

No source identity, ToF, evaluator geometry, or depth calibration is accepted.
Native project integer pixel centres are converted to upstream half centres.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM_REVISION = '8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
CHECKPOINT_SHA256 = '93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648'
NATIVE_SHAPE = (360, 640)
MODEL_SHAPE = (364, 644)
ARMS = ('known_camera', 'estimated_camera')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def camera_matrix(intr):
    """Public native calibration -> resized upstream half-centre matrix."""
    values = {k: float(intr[k]) for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy')}
    if (values['height'], values['width']) != NATIVE_SHAPE:
        raise ValueError('This fixed adapter requires native 360x640 RGB')
    if not all(np.isfinite(v) for v in values.values()) or min(values['fx'], values['fy']) <= 0:
        raise ValueError('Invalid public intrinsics')
    sy, sx = np.asarray(MODEL_SHAPE) / np.asarray(NATIVE_SHAPE)
    return np.array([[values['fx']*sx, 0., (values['cx']+.5)*sx],
                     [0., values['fy']*sy, (values['cy']+.5)*sy],
                     [0., 0., 1.]], dtype=np.float32)


def native_intrinsics(matrix):
    """Upstream resized half-centre matrix -> native integer-centre matrix."""
    result = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    sy, sx = np.asarray(MODEL_SHAPE) / np.asarray(NATIVE_SHAPE)
    result[0] /= sx
    result[1] /= sy
    result[0, 2] -= .5
    result[1, 2] -= .5
    return result


def prepare_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (*NATIVE_SHAPE, 3):
        raise ValueError('Expected RGB uint8 array [360,640,3]')
    # Keep fractional RGB intensities; upstream normalize=True divides by255.
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    return F.interpolate(tensor, size=MODEL_SHAPE, mode='bilinear', align_corners=False)


def load_model(work, device='cuda'):
    """Strictly load official ViT-S weights; skip unused training losses only."""
    work = Path(work).resolve()
    upstream = work/'upstream'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError('Unexpected UniDepth source revision')
    subprocess.run(['git', '-C', str(upstream), 'diff', '--exit-code', 'HEAD', '--'], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    checkpoint = work/'model.safetensors'
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint hash differs from frozen official weight')
    config = json.loads((work/'config.json').read_text(encoding='utf-8'))
    if config['model']['pixel_encoder']['name'] != 'dinov2_vits14' or config['model']['pixel_encoder']['pretrained']:
        raise ValueError('Expected local ViT-S construction without backbone download')
    runtime_config = deepcopy(config)
    skipped_losses = sorted(runtime_config['training']['losses'])
    runtime_config['training']['losses'] = {}
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import unidepth.models.unidepthv2.unidepthv2 as implementation
    from safetensors.torch import load_file
    if not Path(implementation.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError('An unrelated UniDepth module was already imported')
    model = implementation.UniDepthV2(runtime_config)
    state = load_file(str(checkpoint), device='cpu')
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('Strict checkpoint loading did not match')
    if not all(torch.equal(v, state[k]) for k, v in model.state_dict().items()):
        raise RuntimeError('Loaded parameter bytes differ')
    state_count = len(state)
    del state
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = model.to(target).eval()
    model.requires_grad_(False)
    model.resolution_level = 0
    constraints = model.shape_constraints
    bounds = (constraints['pixels_min'], constraints['pixels_min']+
              (constraints['pixels_max']-constraints['pixels_min'])/10)
    padding, padded = implementation.get_paddings(MODEL_SHAPE, constraints['ratio_bounds'])
    factor, resized = implementation.get_resize_factor(padded, bounds)
    if padding != (0,0,0,0) or factor != 1. or tuple(resized) != MODEL_SHAPE:
        raise RuntimeError('Upstream internal resize is not identity')
    source_hashes = {str(p):sha(p) for p in sorted(upstream.rglob('*.py'))}
    import torchvision
    info = dict(upstream_revision=revision, checkpoint_sha256=CHECKPOINT_SHA256,
        config_sha256=sha(work/'config.json'), license_sha256=sha(upstream/'LICENSE'),
        strict_load=True, missing_keys=[], unexpected_keys=[], state_entries=state_count,
        all_loaded_tensors_exact=True, parameters=sum(p.numel() for p in model.parameters()),
        device=str(target), device_name=torch.cuda.get_device_name(target) if target.type=='cuda' else 'CPU',
        torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
        python=sys.executable, source_hashes=source_hashes, resolution_level=0,
        native_shape=list(NATIVE_SHAPE), model_shape=list(MODEL_SHAPE),
        internal_resize_factor=factor, internal_padding=list(padding),
        adaptations={'training.losses':{}, 'skipped_unused_training_losses':skipped_losses},
        inference_precision='Upstream infer CUDA float16 autocast; output converted to float32',
        returned_intrinsics_authority='Learned camera-head estimate in BOTH arms; supplied rays override only ray path',
        depth_authority='Official camera-axis z metric prediction, no ToF or evaluator scale alignment')
    model.mz164_info = info
    return model, info


@torch.inference_mode()
def predict(model, rgb_uint8_RGB, intr, arm):
    """Return raw float32 native axis-z depth and JSON public-input audit."""
    if arm not in ARMS:
        raise ValueError('Unknown camera arm')
    if model.training or model.resolution_level != 0:
        raise ValueError('Frozen model must be eval at resolution level zero')
    matrix = camera_matrix(intr)
    rgb = prepare_rgb(rgb_uint8_RGB)
    device = torch.device(model.device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs = model.infer(rgb, camera=torch.from_numpy(matrix.copy()).unsqueeze(0) if arm=='known_camera' else None,
                          normalize=True)
    if tuple(outputs['depth'].shape) != (1,1,*MODEL_SHAPE):
        raise RuntimeError('Unexpected depth output shape')
    depth = F.interpolate(outputs['depth'].float(), size=NATIVE_SHAPE,
                          mode='bilinear', align_corners=False)[0,0].cpu().numpy().copy()
    predicted_k = outputs['intrinsics'][0].float().cpu().numpy()
    supplied_ray_error = None
    if arm == 'known_camera':
        # This verifies the decoder actually used supplied rays, not just K metadata.
        yy, xx = torch.meshgrid(torch.arange(MODEL_SHAPE[0], device=device, dtype=torch.float32)+.5,
                               torch.arange(MODEL_SHAPE[1], device=device, dtype=torch.float32)+.5, indexing='ij')
        expected = torch.stack(((xx-float(matrix[0,2]))/float(matrix[0,0]),
                                (yy-float(matrix[1,2]))/float(matrix[1,1]), torch.ones_like(xx)), dim=0)
        expected = F.normalize(expected, dim=0).unsqueeze(0)
        supplied_ray_error = float((outputs['rays'].float()-expected).abs().max().item())
        if supplied_ray_error > 2e-5:
            raise RuntimeError('Supplied camera rays were not preserved')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    audit = dict(arm=arm, device=str(device), seconds=time.perf_counter()-started,
        native_shape=list(NATIVE_SHAPE), model_shape=list(MODEL_SHAPE),
        resize_sx=MODEL_SHAPE[1]/NATIVE_SHAPE[1], resize_sy=MODEL_SHAPE[0]/NATIVE_SHAPE[0],
        rgb_interpolation='bilinear_align_corners_false_float_0_255_then_upstream_imagenet_normalize',
        depth_interpolation='bilinear_align_corners_false_axis_z_no_range_or_scale_conversion',
        supplied_intrinsics_resized_half_centres=matrix.tolist() if arm=='known_camera' else None,
        supplied_ray_max_abs_error=supplied_ray_error,
        predicted_intrinsics_resized_half_centres=predicted_k.tolist(),
        predicted_intrinsics_native_integer_centres=native_intrinsics(predicted_k).tolist(),
        returned_intrinsics_authority='LEARNED_ESTIMATE_NOT_SUPPLIED_CAMERA',
        finite_pixels=int(np.isfinite(depth).sum()), positive_finite_pixels=int((np.isfinite(depth)&(depth>0)).sum()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    # No clipping or invalid-value repair; caller retains UNKNOWN semantics.
    return depth.astype(np.float32, copy=False), audit
