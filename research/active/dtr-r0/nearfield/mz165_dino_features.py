"""Frozen official DINOv2 ViT-S/14 patch features; public RGB pixels only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
WEIGHT_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
SEED = 165016
INPUT_SHAPE = (360,640)
RESIZED_SHAPE = (518,924)
FEATURE_SHAPE = (384,37,66)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def prepare_rgb(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (*INPUT_SHAPE,3):
        raise ValueError('Expected native RGB uint8 [360,640,3]')
    x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).unsqueeze(0).float()/255.
    x=F.interpolate(x,size=RESIZED_SHAPE,mode='bilinear',align_corners=False)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (x-mean)/std


def load_backbones(work,device='cuda'):
    """No online loading: authenticate local upstream and strictly load tensors."""
    work=Path(work).resolve();upstream=work/'upstream'
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if revision != REVISION:raise ValueError('Unpinned DINOv2 source')
    subprocess.run(['git','-C',str(upstream),'diff','--exit-code','HEAD','--'],check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    checkpoint=work/'dinov2_vits14_pretrain.pth'
    if sha(checkpoint) != WEIGHT_SHA256:raise ValueError('Official checkpoint hash mismatch')
    if str(upstream) not in sys.path:sys.path.insert(0,str(upstream))
    from dinov2.hub import backbones
    from dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name
    if not Path(backbones.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError('DINOv2 module imported from another source')
    official_name=_make_dinov2_model_name('vit_small',14,0)
    if f'{_DINOV2_BASE_URL}/{official_name}/{official_name}_pretrain.pth' != WEIGHT_URL:
        raise RuntimeError('Official hub checkpoint URL differs')
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    # CPU generator scope protects the caller's random stream; all initialization is CPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        random_model=backbones.dinov2_vits14(pretrained=False)
        pretrained_model=backbones.dinov2_vits14(pretrained=False)
    initial=random_model.state_dict()
    random_digest=state_digest(initial)
    random_path=work/f'random_seed{SEED}.pth'
    if random_path.exists():
        restored=torch.load(random_path,map_location='cpu',weights_only=True)
        if state_digest(restored)!=random_digest:raise ValueError('Saved random state differs from seed/source reconstruction')
        random_model.load_state_dict(restored,strict=True)
        del restored
    else:
        with random_path.open('xb') as handle:torch.save(initial,handle)
    learned=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(learned,dict) or not all(isinstance(v,torch.Tensor) for v in learned.values()):
        raise ValueError('Expected tensor-only official state dictionary')
    loaded=pretrained_model.load_state_dict(learned,strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:raise RuntimeError('Strict checkpoint mismatch')
    if not all(torch.equal(v,learned[k]) for k,v in pretrained_model.state_dict().items()):
        raise RuntimeError('Checkpoint tensors did not load exactly')
    pretrained_digest=state_digest(learned)
    assert initial.keys()==learned.keys() and all(initial[k].shape==learned[k].shape for k in learned)
    state_entries=len(learned);del initial,learned
    models={'random':random_model,'pretrained':pretrained_model}
    for arm,model in models.items():
        if model.num_register_tokens!=0 or model.embed_dim!=384 or model.patch_size!=14:
            raise ValueError('Unexpected backbone architecture')
        model.to(target).eval().requires_grad_(False)
        model.mz165_arm=arm
    source_hashes={str(p):sha(p) for p in sorted(upstream.rglob('*.py'))}
    receipt=dict(upstream_revision=revision,official_url=WEIGHT_URL,checkpoint_sha256=WEIGHT_SHA256,
        checkpoint_bytes=checkpoint.stat().st_size,random_seed=SEED,random_state_path=str(random_path),
        random_file_sha256=sha(random_path),state_sha256={'random':random_digest,'pretrained':pretrained_digest},
        source_hashes=source_hashes,license_sha256=sha(upstream/'LICENSE'),
        torch=torch.__version__,numpy=np.__version__,python=sys.executable,device=str(target),
        device_name=torch.cuda.get_device_name(target) if target.type=='cuda' else 'CPU',
        parameters=sum(p.numel() for p in random_model.parameters()),state_entries=state_entries,
        strict_load=True,weights_only=True,learned_tensors_exact=True,upstream_modified=False,
        pretrained_source='Official facebookresearch/dinov2 original ViT-S/14, no registers, no UniDepth weights',
        input_shape=list(INPUT_SHAPE),resized_shape=list(RESIZED_SHAPE),feature_shape=list(FEATURE_SHAPE),
        feature='x_norm_patchtokens: final model LayerNorm, no extra L2 normalization or class token',
        precision='float32 input/parameters/inference/output; autocast explicitly disabled',
        attention_backend='Unmodified official implementation; optional xFormers else PyTorch scaled-dot-product attention')
    return models,receipt


@torch.inference_mode()
def extract(model,rgb):
    """One RGB image -> float32 CxHxW patch grid and JSON execution audit."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Feature backbone must be frozen and in eval mode')
    device=next(model.parameters()).device
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    image=prepare_rgb(rgb).to(device)
    with torch.autocast(device_type=device.type,enabled=False):
        tokens=model.forward_features(image)['x_norm_patchtokens']
    if tuple(tokens.shape)!=(1,37*66,384):raise RuntimeError('Unexpected patch feature shape')
    grid=tokens[0].reshape(37,66,384).permute(2,0,1).float().contiguous()
    features=grid.cpu().numpy().copy()
    if not np.isfinite(features).all():raise RuntimeError('Nonfinite frozen backbone features')
    if device.type=='cuda':torch.cuda.synchronize(device)
    audit=dict(arm=getattr(model,'mz165_arm','unspecified'),device=str(device),seconds=time.perf_counter()-started,
        input_shape=list(rgb.shape),resized_shape=list(image.shape),feature_shape=list(features.shape),
        feature_dtype=str(features.dtype),finite_elements=int(np.isfinite(features).sum()),
        preprocessing='RGB/255, bilinear518x924 align_corners=False, ImageNet mean/std',
        feature_semantics='Official final-LayerNorm patch tokens in row-major spatial grid; CLS/registers excluded',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    return features,audit
