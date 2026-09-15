"""MZ141 training mechanics; predictor accepts only MZ140 public sensor inputs.

TRAIN native references are labels only. Neither labels nor surface masks are
arguments to the model. Frozen MZ140 inference implementation stays unchanged.
"""
import importlib
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from mz140_depthor import load_model, local_conv


def load_trainable(upstream, checkpoint, device='cuda'):
    model, info = load_model(upstream, checkpoint, device)
    # BpConvLocal's custom backward requires an unavailable compiled BpOps op.
    # Bypass that wrapper with the same forward algebra and native autograd.
    importlib.import_module('src.models.refine').bpconvlocal = local_conv
    selected = list(model.get_lr_params())
    for parameter in selected:
        parameter.requires_grad_(True)
    assert {id(p) for p in selected} == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in model.depth_anything.parameters())
    model.train()
    model.depth_anything.eval()
    info.update(trainable_parameters=sum(p.numel() for p in selected),
                local_convolution='SAME_UNFOLD_FORWARD_NATIVE_AUTOGRAD', backbone_frozen=True)
    return model, info


def training_prediction(model, data):
    """Unclipped final depth, differentiably resampled to native RGB pixels."""
    raw = model(data)[1]
    if not torch.isfinite(raw).all():
        raise FloatingPointError('Nonfinite model output')
    return F.interpolate(raw, size=(360, 640), mode='bilinear', align_corners=False)[0, 0]


def supervision(reference):
    """Disjoint partition of the SAME known pixels for both loss reductions.

    Each visible saved object is a surface unit (connected visible cuboid
    surface, not isolated ToF hits). Separate referenced 8px silhouette ring;
    all other visible objects, including the far wall, remain supervised.
    """
    depth = reference['depth'].astype(np.float32)
    known = np.isfinite(depth) & (depth > 0)
    target = reference['target_visible'] & known
    ring = np.zeros_like(known)
    if target.any():
        distances = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
        ring = known & ~target & (distances <= 8.)
    surfaces = [known & ~ring & (reference['owner'] == i) for i in range(len(reference['names']))]
    surfaces = [m for m in surfaces if m.any()]
    assert surfaces and np.array_equal(np.logical_or.reduce(surfaces) | ring, known)
    weight = np.zeros_like(depth)
    surface_mass = .5 if ring.any() else 1.
    for mask in surfaces:
        weight[mask] = surface_mass / len(surfaces) / mask.sum()
    if ring.any():
        weight[ring] = .5 / ring.sum()
    assert np.isclose(weight.sum(), 1., atol=1e-6) and not weight[~known].any()
    return dict(depth=np.nan_to_num(depth, nan=0.), known=known, balanced_weight=weight,
                audit=dict(known_pixels=int(known.sum()), unknown_pixels=int((~known).sum()),
                           target_pixels=int(target.sum()), ring_pixels=int(ring.sum()),
                           surface_pixels=[int(m.sum()) for m in surfaces]))


def loss_contribution(prediction, labels, arm, batch_known_pixels, batch_frames):
    truth = torch.as_tensor(labels['depth'], device=prediction.device)
    error = (prediction - truth).abs()
    if arm == 'pixel':
        known = torch.as_tensor(labels['known'], device=prediction.device)
        return error[known].sum() / batch_known_pixels
    if arm == 'surface':
        weight = torch.as_tensor(labels['balanced_weight'], device=prediction.device)
        return (error * weight).sum() / batch_frames
    raise ValueError(arm)
