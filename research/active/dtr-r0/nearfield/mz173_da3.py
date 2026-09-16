"""Unmodified frozen DA3 core with public causal inputs and strict weight binding."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

SOURCE_REVISION = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
MODEL_REVISION = 'e08cab65ca0ec38e7826075418411ab90cab4da3'
MODEL_SHA256 = '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf'


def load_model(assets):
    assets = Path(assets)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tools'))
    from dataclasses import asdict
    from research_backend import torch_observation
    sys.path.insert(0, str(assets/'da3-source/src'))
    from depth_anything_3.cfg import create_object
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.utils.io.input_processor import InputProcessor
    config = json.loads((assets/'da3-small/config.json').read_text())
    model = create_object(OmegaConf.create(config['config'])).eval()
    saved = load_file(str(assets/'da3-small/model.safetensors'))
    assert all(k.startswith('model.') for k in saved)
    state = {k[6:]: v for k, v in saved.items()}
    expected, aliases, stored = model.state_dict(), {}, set(state)
    for key in sorted(set(expected)-stored):
        value = expected[key]
        matches = [k for k, v in expected.items() if k in stored and
                   v.data_ptr() == value.data_ptr() and v.shape == value.shape and
                   v.stride() == value.stride()]
        assert len(matches) == 1, (key, matches)
        aliases[key] = matches[0]
        state[key] = state[matches[0]]
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert len(stored) == 437 and len(state) == 443 and len(aliases) == 6
    model.requires_grad_(False)
    # Do not migrate an already-used CPU instance: upstream position cache is
    # keyed by size only. This new instance performs all forwards on CUDA.
    model = model.cuda()
    return model, InputProcessor(), dict(source_revision=SOURCE_REVISION,
        model_revision=MODEL_REVISION, checkpoint_sha256=MODEL_SHA256,
        stored_entries=len(stored), strict_entries=len(state), aliases=aliases,
        parameters=sum(p.numel() for p in model.parameters()),
        device_observation=asdict(torch_observation(model=model)),
        all_weights_exact=True, learned_source_modified=False,
        backend='CUDA_FLOAT32_NO_AUTOCAST_TF32_DISABLED',
        forward='OFFICIAL_CORE_NO_EXPORT_POSE_ALIGNMENT_OR_FULL_API')


def prefix_indices(rows, current):
    row = rows[current]
    indices = [i for i in range(current+1) if rows[i]['episode_id'] == row['episode_id']][-5:]
    assert indices[-1] == current
    assert all(rows[i]['time_s'] <= row['time_s'] for i in indices)
    assert all(rows[a]['time_s'] < rows[b]['time_s'] for a,b in zip(indices,indices[1:]))
    return indices


def predict(model, normalized_images):
    images = normalized_images.unsqueeze(0).cuda()
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        out = model(images, extrinsics=None, intrinsics=None,
                    infer_gs=False, use_ray_pose=False, ref_view_strategy='first')
    raw = {k: v.float().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
    assert raw['depth'].shape[0] == 1 and np.isfinite(raw['depth']).all()
    assert (raw['depth'] > 0).all()
    # Upstream boundary resize covers the complete image, without a crop.
    # Half-pixel bilinear output resize, explicit rather than a geometric claim.
    maps = F.interpolate(out['depth'][0].float()[:,None], (360,640),
                         mode='bilinear', align_corners=False)[:,0]
    return maps.cpu().numpy(), raw
