"""MZ142: preserve pretrained BatchNorm running statistics during fine-tuning.

Only normalization mode changes relative to MZ141. BatchNorm affine parameters
remain trainable, and training-mode DropPath stays active outside the frozen
Depth Anything backbone. No calibration or current-image BN substitution.
"""
import hashlib

import torch

from mz141_depthor_training import load_trainable


def _batch_norms(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]


def _require_running_buffers(name, module):
    if not module.track_running_stats:
        raise ValueError(f'BatchNorm {name!r} must track running statistics')
    for key in ('running_mean', 'running_var', 'num_batches_tracked'):
        if getattr(module, key, None) is None:
            raise ValueError(f'BatchNorm {name!r} is missing {key}')


def training_mode(model):
    """Restore training after evaluation while keeping all BN buffers fixed.

    Call again after every evaluation pass. eval() changes module behavior,
    not requires_grad, so the existing BN affine parameter selection survives.
    Track-running-stats flags and momentum are validated/preserved, not reset.
    """
    model.train()
    model.depth_anything.eval()
    for name, module in _batch_norms(model):
        _require_running_buffers(name, module)
        module.eval()
    return model


def bn_digest(model):
    """Hash only named BN running buffers; affine parameters are excluded."""
    digest = hashlib.sha256()
    for name, module in _batch_norms(model):
        _require_running_buffers(name, module)
        for key in ('running_mean', 'running_var', 'num_batches_tracked'):
            tensor = getattr(module, key).detach().cpu().contiguous()
            digest.update(f'{name}.{key}|{tensor.dtype}|{tuple(tensor.shape)}\n'.encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_fixed_bn(upstream, checkpoint, device='cuda'):
    """Load the same original strict pretrained state, then fix BN mode."""
    model, info = load_trainable(upstream, checkpoint, device)
    training_mode(model)
    norms = _batch_norms(model)
    affine = [parameter for _, module in norms for parameter in module.parameters(recurse=False)]
    info.update(
        normalization='BATCHNORM_EVAL_PRETRAINED_RUNNING_STATISTICS_FIXED_FROM_START',
        batchnorm_modules=len(norms),
        batchnorm_running_buffers_sha256=bn_digest(model),
        batchnorm_affine_parameters=sum(parameter.numel() for parameter in affine),
        batchnorm_trainable_affine_parameters=sum(parameter.numel() for parameter in affine if parameter.requires_grad),
        batchnorm_momentum_and_requires_grad_unchanged=True,
        other_training_modes='MODEL_TRAIN_WITH_DROPPATH_ACTIVE_EXCEPT_DEPTH_ANYTHING_EVAL_AND_ALL_BATCHNORM_EVAL',
    )
    return model, info
