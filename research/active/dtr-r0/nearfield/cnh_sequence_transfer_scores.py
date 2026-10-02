"""Frozen NEAR/M3 scores on existing 94000..94095 sequence voxels only.

NEAR is copied from its retained graded cache. M3 uses five frozen models on
the existing early/late chunks. No rendering, materialization or training.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import cnh_graded_alert as GA
import cnh_near_range as NR
import cnh_structure_space as SS
import cnh_three_level_sequence as SE

OUT = SS.WORK / 'cnh-sequence-transfer-20261002'
MARGIN = SS.WORK / 'cnh-margin-labels-20261002'
UNITS = np.arange(94000, 94096)
FRAMES = np.arange(3, 16)
PARTS = ('evaluation_early', 'evaluation')
SHAPE = (96, 40, 13, 2)


def file_record(path, digest=False):
    path = Path(path)
    stat = path.stat()
    value = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if digest:
        value['sha256'] = SS.sha(path)
    return value


def models():
    return [MARGIN / 'models/M3' / f'model_seed{seed}.pt' for seed in range(5)]


def input_snapshot():
    files = {}
    for name in PARTS:
        chunks = sorted((NR.OUT / 'data' / name).glob('features_c*.npy'))
        if not chunks:
            raise FileNotFoundError(f'Missing retained voxel chunks: {name}')
        files[name] = []
        for path in chunks:
            metadata = path.with_name(path.name.replace('features_', 'metadata_').replace('.npy', '.npz'))
            files[name].append(dict(voxel=file_record(path), metadata=file_record(metadata, True)))
    return dict(voxel_parts=files,
        near_cache=file_record(NR.OUT / 'graded_frame_scores.npz', True),
        final_scores={arm: file_record(path, True) for arm, path in
                      [('NEAR', NR.OUT / 'scores_NEAR.npz'), ('M3', MARGIN / 'scores_M3.npz')]},
        m3_models={str(path): SS.sha(path) for path in models()},
        source_sha256={Path(module.__file__).name: SS.sha(module.__file__) for module in (GA, NR, SS, SE)},
        plan=file_record(OUT / 'PLAN.json', True),
        voxel_integrity='Existing multi-GB voxels identified by path/size/mtime, shape/dtype and metadata SHA; no new whole-voxel hashing')


def indices(meta, expected_frames):
    """Map each metadata row to one output slot; reject duplicates or omissions."""
    arrays = {}
    for key in ('unit', 'config', 'frame'):
        value = np.asarray(meta[key])
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f'Metadata {key} must be a one-dimensional integer array')
        arrays[key] = value.astype(np.int64, copy=False)
    u, c, f = (arrays[k] for k in ('unit', 'config', 'frame'))
    if len({len(u), len(c), len(f)}) != 1:
        raise ValueError('Metadata columns have inconsistent lengths')
    if not np.isin(u, UNITS).all() or not ((c >= 0) & (c < 40)).all() or not np.isin(f, expected_frames).all():
        raise ValueError('Unexpected unit/config/frame identity')
    flat = (u - UNITS[0]) * 40 * len(FRAMES) + c * len(FRAMES) + f - FRAMES[0]
    expected = np.asarray([(int(unit) - UNITS[0]) * 40 * len(FRAMES) + config * len(FRAMES) + frame - FRAMES[0]
                           for unit in UNITS for config in range(40) for frame in expected_frames])
    if len(flat) != len(expected) or not np.array_equal(np.sort(flat), np.sort(expected)):
        raise ValueError('Each (unit,config,frame) must occur exactly once')
    return flat


def close_parts(parts):
    for arrays, _, _ in parts.values():
        for array in arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()
        arrays.clear()
    parts.clear()


def load_parts(snapshot):
    parts, receipts = {}, {}
    try:
        all_indices = []
        for name in PARTS:
            arrays, meta = GA._parts(name)
            # Register immediately so errors still close every memory mapping.
            parts[name] = (arrays, meta, None)
            frame_ids = np.arange(3, 11) if name == 'evaluation_early' else np.arange(11, 16)
            flat = indices(meta, frame_ids)
            paths = snapshot['voxel_parts'][name]
            if len(arrays) != len(paths):
                raise ValueError('GA._parts chunk count differs from retained paths')
            offset, chunks = 0, []
            for array, file in zip(arrays, paths):
                with np.load(file['metadata']['path'], allow_pickle=False) as original:
                    n = len(array)
                    if any(len(original[key]) != n for key in ('unit', 'config', 'frame')):
                        raise ValueError('Voxel chunk and metadata row count differ')
                    for key in ('unit', 'config', 'frame'):
                        if not np.array_equal(meta[key][offset:offset+n], original[key]):
                            raise ValueError('Concatenated metadata and voxel chunk order differ')
                if array.ndim != 5 or array.shape[1] != 3:
                    raise ValueError(f'Unexpected voxel tensor shape: {array.shape}')
                chunks.append(dict(rows=n, shape=list(array.shape), dtype=str(array.dtype),
                                   voxel=file['voxel'], metadata=file['metadata']))
                offset += n
            if offset != len(flat):
                raise ValueError('Voxel row sum differs from metadata length')
            parts[name] = (arrays, meta, flat)
            all_indices.append(flat)
            receipts[name] = dict(rows=offset, frames=frame_ids.tolist(), chunks=chunks)
        joined = np.concatenate(all_indices)
        if not np.array_equal(np.sort(joined), np.arange(np.prod(SHAPE[:-1]))):
            raise ValueError('Early and late parts do not cover exactly all sequence frames')
        return parts, receipts
    except BaseException:
        close_parts(parts)
        raise


def final_parity(arm, logit):
    if logit.shape != SHAPE or not np.isfinite(logit).all():
        raise ValueError(f'{arm}: incomplete/nonfinite raw frame scores')
    smooth = SE.smooth(logit)
    path = NR.OUT / 'scores_NEAR.npz' if arm == 'NEAR' else MARGIN / 'scores_M3.npz'
    with np.load(path, allow_pickle=False) as frozen:
        error = max(float(np.abs(smooth[i, :, -1] - frozen[str(u) if arm == 'NEAR' else f'near|{u}']).max())
                    for i, u in enumerate(UNITS))
    if not error < 1e-4:
        raise ValueError(f'{arm}: final score parity failed: {error}')
    return error


def completed_receipt():
    path = OUT / 'scores_receipt.json'
    outputs = [OUT / f'frame_scores_{arm}.npz' for arm in ('NEAR', 'M3')]
    if not path.exists():
        if any(p.exists() for p in outputs):
            raise RuntimeError('Partial score outputs require inspection; do not overwrite')
        return None
    receipt = json.loads(path.read_text(encoding='utf8'))
    if receipt.get('status') != 'COMPLETE':
        raise RuntimeError('Incomplete score receipt requires inspection')
    for arm, output in zip(('NEAR', 'M3'), outputs):
        if SS.sha(output) != receipt['output_sha256'][arm]:
            raise ValueError('Completed score cache checksum changed: ' + arm)
    if input_snapshot() != receipt['inputs']:
        raise ValueError('Completed cache inputs differ; no implicit rerun')
    print('Existing COMPLETE scores verified; inference not repeated', flush=True)
    return receipt


def infer():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Parent must freeze PLAN and RUNS before inference')
    previous = completed_receipt()
    if previous is not None:
        return previous
    started = time.monotonic()
    snapshot = input_snapshot()
    parts, part_receipts = load_parts(snapshot)
    nets, masks, batch, prediction, net, tensor = [], None, None, None, None, None
    torch = None
    runtime = {}
    try:
        with np.load(NR.OUT / 'graded_frame_scores.npz', allow_pickle=False) as cache:
            near = np.asarray(cache['logit']).copy()
        near_error = final_parity('NEAR', near)
        import torch as torch_runtime
        from cnh_cvr_pilot import CVR
        from cnh_cvr_projection import query_masks
        torch = torch_runtime
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen five-seed inference requires actual CUDA')
        device = torch.cuda.current_device()
        runtime = dict(device=torch.cuda.get_device_name(device), device_index=device,
            cuda_available=True, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
            threads=torch.get_num_threads(), batch_size=128, seeds=5,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        for path in models():
            net = CVR().cuda()
            net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
            nets.append(net.eval())
        net = None
        m3 = np.full(SHAPE, np.nan, dtype=np.float32)
        flat_m3 = m3.reshape(-1, 2)
        with torch.no_grad():
            for name in PARTS:
                arrays, _, flat = parts[name]
                offset = 0
                for chunk, array in enumerate(arrays):
                    for start in range(0, len(array), 128):
                        raw = array[start:start+128]
                        if not np.isfinite(raw).all():
                            raise ValueError(f'Nonfinite retained voxel batch: {name}/{chunk}/{start}')
                        batch = SS.prep(torch, raw, masks)
                        tensor = torch.stack([model(batch) for model in nets]).mean(0)
                        prediction = tensor.cpu().numpy()
                        if prediction.shape != (len(raw), 2) or not np.isfinite(prediction).all():
                            raise ValueError('Invalid/nonfinite five-seed inference output')
                        slots = flat[offset+start:offset+start+len(raw)]
                        flat_m3[slots] = prediction
                    offset += len(array)
                    print('M3 existing voxels', name, 'chunk', chunk, 'rows', offset,
                          'elapsed_s', round(time.monotonic()-started, 1), flush=True)
        torch.cuda.synchronize()
        m3_error = final_parity('M3', m3)
        if input_snapshot() != snapshot:
            raise ValueError('Inputs changed during inference; outputs not published')
        parity = {'NEAR': near_error, 'M3': m3_error}
    finally:
        net, masks, batch, prediction, tensor = None, None, None, None, None
        nets.clear()
        close_parts(parts)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            runtime['cuda_allocated_after_release'] = int(torch.cuda.memory_allocated())
            runtime['cuda_reserved_after_release'] = int(torch.cuda.memory_reserved())
    # Exclusive creation prevents accidental overwrite of partial/complete evidence.
    for arm, logit in [('NEAR', near), ('M3', m3)]:
        with (OUT / f'frame_scores_{arm}.npz').open('xb') as file:
            np.savez_compressed(file, logit=logit, units=UNITS, frames=FRAMES)
    receipt = dict(status='COMPLETE', shape=list(SHAPE), units=UNITS.tolist(), frames=FRAMES.tolist(),
        inputs=snapshot, validated_parts=part_receipts, final_score_max_abs_error=parity,
        output_sha256={arm: SS.sha(OUT / f'frame_scores_{arm}.npz') for arm in ('NEAR', 'M3')},
        runtime=runtime, elapsed_s=time.monotonic()-started, evaluator_sha256=SS.sha(__file__),
        operations=dict(near_cache_reused=True, m3_frozen_inference=True, rendering=False, materialization=False, training=False))
    with (OUT / 'scores_receipt.json').open('x', encoding='utf8') as file:
        json.dump(receipt, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    print('COMPLETE scores', json.dumps(dict(parity=parity, runtime=runtime)), flush=True)
    return receipt


def check():
    fields = np.asarray([(u, c, f) for u in UNITS for c in range(40) for f in range(3, 11)])
    metadata = {key: fields[:, i] for i, key in enumerate(('unit', 'config', 'frame'))}
    ids = indices(metadata, np.arange(3, 11))
    assert len(ids) == 96*40*8 and len(np.unique(ids)) == len(ids)
    broken = {key: value.copy() for key, value in metadata.items()}
    for key in broken:
        broken[key][-1] = broken[key][0]
    try:
        indices(broken, np.arange(3, 11))
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate frame identity was not rejected')
    raw = np.broadcast_to(np.arange(3., 16.)[None, None, :, None], (1, 1, 13, 2)).copy()
    smooth = SE.smooth(raw)
    assert smooth[0, 0, 0, 0] == 3
    assert np.isclose(smooth[0, 0, -1, 0], np.dot(np.arange(11., 16.), [1, 2, 4, 8, 16])/31)
    assert np.array_equal(UNITS, np.asarray(NR.SPLITS['evaluation'])) and np.array_equal(FRAMES, GA.ALL)
    print('PASS: complete frame-slot identities, duplicate rejection, retained units/frames and causal smoothing; no actual inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'infer'))
    args = parser.parse_args()
    check() if args.stage == 'check' else infer()
