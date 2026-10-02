"""Observation-only fixed SAM2 automatic regions, with reusable frame receipts.

Region IDs index selected_indices (score-ordered top 32); -1 is uncovered.
All post-filter AMG masks survive in packed form, including unselected masks.
No depth, object identity, query, target, pose, labels or GT enters inference.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REVISION = 'ee5bba1d82bb8749febdf90f45e84b687142ba03'
MODEL = ROOT/'artifacts.local/models/facebook--sam2.1-hiera-small/snapshots'/REVISION
SHAPE = (768, 1024)
FILES = {
    'model.safetensors': (184305280, '0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60'),
    'config.json': (5698, '97ff9f65b76d107acda4247885f0a5555d0048850ae3c5f97183df289aaecde9'),
    'preprocessor_config.json': (683, '6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d'),
}
AMG = dict(points_per_crop=16, points_per_batch=32, crops_n_layers=0,
           pred_iou_thresh=.88, stability_score_thresh=.95,
           stability_score_offset=1, mask_threshold=0, crops_nms_thresh=.7)
POLICY = dict(max_masks=32, selection='stable score descending, original index tie',
              overlap='smallest native mask area, then score descending, original index',
              label_ids='position in selected_indices', uncovered=-1,
              mask_packbits_axis=1, mask_packbits_bitorder='little',
              shape=list(SHAPE), dtype='torch.float32', torch_threads=4,
              image_decode='Pillow RGB; no EXIF transpose or ICC conversion')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.part')
    if tmp.exists():
        raise RuntimeError(f'orphan partial requires inspection: {tmp}')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def partition(masks, scores, shape=SHAPE):
    """Pure synthetic-testable deterministic allocation of AMG masks."""
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    if masks.size == 0:
        masks = np.empty((0, *shape), dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != tuple(shape):
        raise ValueError('all masks must have native H,W shape')
    if scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError('one finite score per mask required')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    selected = np.argsort(-scores, kind='stable')[:32].astype(np.int64)
    labels = np.full(shape, -1, dtype=np.int16)
    priority = sorted(range(len(selected)),
                      key=lambda rank: (int(areas[selected[rank]]),
                                        -float(scores[selected[rank]]), int(selected[rank])))
    for rank in priority:
        labels[masks[selected[rank]] & (labels < 0)] = rank
    flat = masks.reshape(len(masks), int(np.prod(shape)))
    return dict(labels=labels, labelmap=labels,
                masks_packbits=np.packbits(flat, axis=1, bitorder='little'),
                scores=scores, selected_indices=selected, mask_areas=areas,
                mask_shape=np.asarray(shape, dtype=np.int32))


def fixtures():
    masks = np.zeros((5, 5, 6), bool)
    masks[0, :4, :4] = True
    masks[1, 1:3, 1:3] = True
    masks[2, 1:3, 2:4] = True
    masks[3] = masks[1]
    masks[4, 4, 5] = True
    scores = np.array([.99, .94, .96, .94, .90], np.float32)
    p = partition(masks, scores, (5, 6))
    assert p['selected_indices'].tolist() == [0, 2, 1, 3, 4]
    assert p['labelmap'][0, 0] == 0
    assert p['labelmap'][1, 1] == 2  # smaller area; equal score prefers index 1 over 3
    assert p['labelmap'][1, 2] == 1  # equal area prefers higher score, index 2
    assert p['labelmap'][4, 0] == -1
    unpacked = np.unpackbits(p['masks_packbits'], axis=1, count=30,
                             bitorder='little').reshape(masks.shape).astype(bool)
    assert np.array_equal(unpacked, masks)
    many = partition(np.ones((34, 1, 1), bool), np.ones(34), (1, 1))
    assert many['selected_indices'].tolist() == list(range(32))
    assert many['labelmap'][0, 0] == 0 and len(many['scores']) == 34
    empty = partition([], [], (2, 3))
    assert np.all(empty['labelmap'] == -1) and empty['masks_packbits'].shape == (0, 1)
    try:
        partition(masks, [float('nan')]*5, (5, 6))
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite output was accepted')
    return dict(status='PASS', checks=['area priority', 'score and index ties',
                'stable top32 retains all34', 'packbits roundtrip', 'valid empty', 'reject nonfinite'])


@contextlib.contextmanager
def owner_lock(out):
    import msvcrt
    with (out/'inference.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('another inference owner holds output lock') from exc
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def cache_read(pred, identity, binding):
    npz, receipt = pred/(identity['id']+'.npz'), pred/(identity['id']+'.json')
    for p in (npz, receipt):
        if p.with_name(p.name+'.part').exists():
            raise RuntimeError(f'orphan partial requires inspection: {p}')
    if not npz.exists() and not receipt.exists():
        return None
    if not npz.exists() or not receipt.exists():
        raise RuntimeError(f'orphan frame output requires inspection: {identity["id"]}')
    r = json.loads(receipt.read_text(encoding='utf-8'))
    if r.get('identity') != identity or r.get('binding') != binding or r.get('status') != 'PASS':
        raise RuntimeError(f'cache identity/model/code/parameters mismatch: {identity["id"]}')
    if sha(npz) != r['output_sha256']:
        raise RuntimeError(f'cache NPZ hash mismatch: {npz}')
    with np.load(npz, allow_pickle=False) as a:
        count = r['candidate_count']; selected = a['selected_indices']
        assert a['labelmap'].shape == SHAPE and a['labelmap'].dtype == np.int16
        assert np.array_equal(a['labels'], a['labelmap'])
        assert a['masks_packbits'].shape == (count, np.prod(SHAPE)//8)
        assert a['scores'].shape == (count,) and np.isfinite(a['scores']).all()
        assert np.array_equal(a['mask_shape'], SHAPE)
        assert np.array_equal(selected, np.argsort(-a['scores'], kind='stable')[:32])
        assert np.all((a['labelmap'] >= -1) & (a['labelmap'] < len(selected)))
    return r


def infer(observations, out, limit=None):
    start = time.perf_counter()
    observations, out = Path(observations).resolve(), Path(out).resolve()
    rows = json.loads(observations.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty observation JSON list required')
    seen = set(); identities = []
    for row in rows:
        if set(row) != {'id', 'rgb_path', 'rgb_sha256'}:
            raise ValueError('only id, rgb_path, rgb_sha256 observation fields permitted')
        identifier = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in seen:
            raise ValueError('unsafe or duplicate frame id')
        seen.add(identifier)
        path = Path(row['rgb_path'])
        path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        identities.append(dict(id=identifier, rgb_path=str(path), rgb_sha256=row['rgb_sha256']))
    if limit is not None:
        if limit < 1:
            raise ValueError('positive limit required')
        identities = identities[:limit]
    for row in identities:
        if sha(row['rgb_path']) != row['rgb_sha256']:
            raise ValueError(f'RGB hash mismatch: {row["id"]}')
    for name, (size, digest) in FILES.items():
        if (MODEL/name).stat().st_size != size or sha(MODEL/name) != digest:
            raise ValueError(f'frozen model file mismatch: {name}')
    binding = dict(adapter_sha256=sha(__file__), model_revision=REVISION,
                   model_files={k: v[1] for k, v in FILES.items()}, amg=AMG, policy=POLICY)
    out.mkdir(parents=True, exist_ok=True)
    pred = out/'predictions'; pred.mkdir(exist_ok=True)
    summary = dict(status='RUNNING', observations=str(observations),
                   observations_sha256=sha(observations), list_count=len(rows),
                   requested_count=len(identities), limit=limit, binding=binding,
                   frames=[], inferred=0, reused=0)
    model = generator = processor = result = image = None
    torch = None
    with owner_lock(out):
        try:
            cached = [cache_read(pred, row, binding) for row in identities]
            if any(r is None for r in cached):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
                import torch
                import transformers
                from PIL import Image
                from transformers import Sam2Model, Sam2ImageProcessor, pipeline
                torch.set_num_threads(4)
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA required; CPU fallback is forbidden')
                torch.cuda.set_device(0)
                load_start = time.perf_counter()
                processor = Sam2ImageProcessor.from_pretrained(MODEL, local_files_only=True)
                model = Sam2Model.from_pretrained(MODEL, local_files_only=True,
                         use_safetensors=True, dtype=torch.float32).eval().to('cuda:0')
                generator = pipeline('mask-generation', model=model, image_processor=processor,
                                     device=0, dtype=torch.float32)
                torch.cuda.synchronize()
                runtime = dict(python=sys.executable, torch=torch.__version__,
                    transformers=transformers.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), model_device=str(next(model.parameters()).device),
                    model_dtype=str(next(model.parameters()).dtype),
                    load_seconds=time.perf_counter()-load_start)
                summary['runtime'] = runtime
            for identity, existing in zip(identities, cached):
                if existing is not None:
                    summary['frames'].append(dict(id=identity['id'], disposition='reused',
                                                  output_sha256=existing['output_sha256']))
                    summary['reused'] += 1
                    continue
                t = time.perf_counter()
                if sha(identity['rgb_path']) != identity['rgb_sha256']:
                    raise ValueError('RGB changed during inference')
                with Image.open(identity['rgb_path']) as opened:
                    image = opened.convert('RGB')
                if image.size != (SHAPE[1], SHAPE[0]):
                    raise ValueError(f'native 768x1024 required: {identity["id"]}')
                decode_seconds = time.perf_counter()-t
                torch.cuda.reset_peak_memory_stats()
                ti = time.perf_counter()
                with torch.inference_mode():
                    result = generator(image, **AMG)
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter()-ti
                tp = time.perf_counter()
                scores = result['scores'].detach().cpu().numpy()
                arrays = partition(result['masks'], scores)
                postprocess_seconds = time.perf_counter()-tp
                npz = pred/(identity['id']+'.npz'); temp = npz.with_name(npz.name+'.part')
                with temp.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temp, npz)
                receipt = dict(status='PASS', identity=identity, binding=binding,
                    rgb_sha256=identity['rgb_sha256'],
                    output_sha256=sha(npz), output_bytes=npz.stat().st_size,
                    native_shape=list(SHAPE), candidate_count=len(scores),
                    selected_count=len(arrays['selected_indices']),
                    uncovered_pixels=int((arrays['labelmap'] < 0).sum()),
                    runtime=runtime, scores_output_device=str(result['scores'].device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    timing=dict(decode_seconds=decode_seconds, amg_seconds=inference_seconds,
                                partition_seconds=postprocess_seconds,
                                frame_total_seconds=time.perf_counter()-t))
                json_write(pred/(identity['id']+'.json'), receipt)
                summary['frames'].append(dict(id=identity['id'], disposition='inferred',
                                              output_sha256=receipt['output_sha256']))
                summary['inferred'] += 1
                print(json.dumps(dict(id=identity['id'], masks=len(scores),
                                      seconds=receipt['timing']['frame_total_seconds'])), flush=True)
                image.close(); image = result = None
            summary['status'] = 'PASS'
        except BaseException as exc:
            summary['status'] = 'FAILED'; summary['error'] = repr(exc)
            raise
        finally:
            if image is not None:
                image.close()
            result = generator = model = processor = None
            gc.collect()
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                summary['cuda_allocated_after_release_bytes'] = torch.cuda.memory_allocated()
                summary['cuda_reserved_after_release_bytes'] = torch.cuda.memory_reserved()
            summary['elapsed_seconds'] = time.perf_counter()-start
            json_write(out/'inference-result.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--fixtures', action='store_true')
    a = p.parse_args()
    if a.fixtures:
        print(json.dumps(fixtures()))
        return
    if a.observations is None or a.out is None:
        p.error('--observations and --out required for inference')
    result = infer(a.observations, a.out, a.limit)
    print(json.dumps({k: result[k] for k in ('status', 'inferred', 'reused', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
