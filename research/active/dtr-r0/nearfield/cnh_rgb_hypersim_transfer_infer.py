"""One frozen native DepthPro observation-only transfer, exactly 96 frames.

Consumes only RGB identity and public camera matrices. No reference depth,
labels, poses, floor, source admission or model/parameter selection occurs here.
Resumes only complete hash-verified predictions. Orphans require inspection.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

import ba_nfo_depthpro as B

ROOT = B.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-hypersim-transfer-20261002'
SHAPE = (768, 1024)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'repository-relative path required: {value}')
    return ROOT/path


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    if temporary.exists() or (path.exists() and not replace):
        raise RuntimeError(f'existing destination/partial requires inspection: {path}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def gpu_owner():
    """OS-held exclusive lock releases on exit/crash; no stale-PID override."""
    import msvcrt
    handle = (OUT/'inference.lock').open('a+b')
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('another owner holds this inference lock; do not launch a duplicate') from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def validate_inputs():
    plan_path = OUT/'inference-plan.json'
    observations_path = OUT/'observations.json'
    plan = read(plan_path)
    plan_sha = B.sha(plan_path)
    assert B.sha(observations_path) == plan['observations_sha256'], 'observations changed'
    assert plan['weight_sha256'] == B.WEIGHT_SHA
    assert B.sha(B.OUT/'depth_pro.pt') == B.WEIGHT_SHA, 'frozen model weight changed'
    sources = {relative_path(p): digest for p, digest in plan['source_sha256'].items()}
    assert Path(__file__).resolve() in {p.resolve() for p in sources}, 'runner must be bound before execution'
    assert Path(B.__file__).resolve() in {p.resolve() for p in sources}, 'legacy adapter must be bound'
    for path, digest in sources.items():
        assert B.sha(path) == digest, f'bound source changed: {path}'
    upstream = B.OUT/'upstream'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == B.UPSTREAM, 'upstream revision changed'
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), 'upstream tracked files dirty'
    backend_ref = plan['backend_reference']
    backend_path = relative_path(backend_ref['path'])
    assert B.sha(backend_path) == backend_ref['sha256'], 'backend reference changed'
    backend = read(backend_path)
    capabilities = backend['runtime_capabilities']
    assert backend['selected_device_type'] == 'cuda' and not backend['cpu_fallback']
    assert capabilities['torch']['version'] == torch.__version__, 'benchmark torch version differs'
    assert capabilities['torch']['cuda_version'] == torch.version.cuda, 'benchmark CUDA build differs'
    assert Path(capabilities['python_executable']).resolve() == Path(sys.executable).resolve(), 'benchmark Python environment differs'
    assert capabilities['python'] == platform.python_version(), 'benchmark Python version differs'
    rows = read(observations_path)
    assert isinstance(rows, list) and len(rows) == 96, 'exactly 96 preselected observations required'
    ids = []
    for row in rows:
        identifier = row['id']
        assert isinstance(identifier, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier), 'unsafe prediction id'
        ids.append(identifier)
        matrix = np.asarray(row['camera_matrix'], dtype=float)
        assert matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.linalg.det(matrix) != 0
        assert B.sha(relative_path(row['rgb_path'])) == row['rgb_sha256'], f'RGB changed: {identifier}'
    assert len(set(ids)) == 96, 'duplicate observation id'
    return plan, plan_sha, rows, dict(upstream_revision=revision,
        backend_reference=backend_ref, torch=torch.__version__, cuda_build=torch.version.cuda,
        python=platform.python_version(), python_executable=sys.executable,
        numpy=np.__version__, opencv=cv2.__version__, source_sha256=plan['source_sha256'])


def validate_prediction(path):
    with np.load(path, allow_pickle=False) as payload:
        assert payload.files == ['native_depth'], f'unexpected prediction fields: {path}'
        depth = payload['native_depth']
        assert depth.shape == SHAPE and depth.dtype == np.float32
        assert np.isfinite(depth).all() and np.all(depth > 0), f'invalid prediction: {path}'


def existing_receipt(row, plan_sha):
    path = OUT/'predictions'/f'{row["id"]}.npz'
    receipt_path = path.with_suffix('.json')
    for item in (path, receipt_path):
        if item.with_name(item.name+'.partial').exists():
            raise RuntimeError(f'orphan partial requires inspection; no inference retry: {item}')
    if path.exists() != receipt_path.exists():
        raise RuntimeError(f'orphan prediction/receipt requires inspection: {path}')
    if not path.exists():
        return None
    receipt = read(receipt_path)
    assert receipt['id'] == row['id'] and receipt['plan_sha256'] == plan_sha
    assert receipt['rgb_sha256'] == row['rgb_sha256'] and receipt['camera_matrix'] == row['camera_matrix']
    assert receipt['weight_sha256'] == B.WEIGHT_SHA and receipt['upstream_revision'] == B.UPSTREAM
    assert receipt['output_sha256'] == B.sha(path)
    assert receipt['model_device_type'] == receipt['output_device_type'] == 'cuda'
    validate_prediction(path)
    return receipt


def run():
    # The root owns preselection and plan writing. This runner never creates them.
    assert OUT.is_dir() and (OUT/'inference-plan.json').is_file()
    with gpu_owner():
        plan, plan_sha, rows, environment = validate_inputs()
        destination = OUT/'predictions'
        destination.mkdir(exist_ok=True)
        expected_files = {f'{row["id"]}.{ext}' for row in rows for ext in ('npz', 'json')}
        unexpected = [p.name for p in destination.iterdir() if p.name not in expected_files]
        if unexpected:
            raise RuntimeError(f'unexpected/orphan prediction files require inspection: {unexpected}')
        outputs = {}
        pending = []
        for row in rows:
            receipt = existing_receipt(row, plan_sha)
            if receipt is None:
                pending.append(row)
            else:
                outputs[row['id']] = receipt
        result_path = OUT/'inference-result.json'
        if result_path.exists():
            result = read(result_path)
            assert not pending and result['status'] == 'COMPLETE' and result['plan_sha256'] == plan_sha
            assert result['outputs'] == outputs, 'completed result differs from verified receipts'
            print('COMPLETE: 96 cached native predictions independently verified; no model loaded', flush=True)
            return result
        assert not result_path.with_name(result_path.name+'.partial').exists(), 'orphan final-result partial requires inspection'
        model = transform = predicted = tensor = None
        gpu_used = False
        started = time.perf_counter()
        timings = []
        owner = dict(pid=os.getpid(), plan_sha256=plan_sha, status='ACTIVE', pending=len(pending),
                     scientific_batch_size=1, torch_threads=4)
        atomic_json(OUT/'inference-owner.json', owner, replace=True)
        try:
            if pending:
                assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
                torch.set_num_threads(4)
                gpu_used = True
                model, transform = B.model_and_transform('cuda')
                devices = {p.device.type for p in model.parameters()}
                assert devices == {'cuda'}, f'model not wholly on CUDA: {devices}'
                assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.float16}
                environment['actual_gpu'] = torch.cuda.get_device_name()
                environment['model_load_seconds'] = time.perf_counter()-started
                for row in pending:
                    rgb_path = relative_path(row['rgb_path'])
                    assert B.sha(rgb_path) == row['rgb_sha256'], 'RGB changed after input check'
                    native_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    assert native_bgr is not None and native_bgr.shape == (*SHAPE, 3)
                    native = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
                    image, focal, bx, by = B.rectify(native, row['camera_matrix'])
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    began = time.perf_counter()
                    with torch.inference_mode():
                        tensor = transform(image)
                        assert tensor.device.type == 'cuda'
                        predicted = model.infer(tensor, f_px=torch.tensor(focal, device='cuda'))
                        assert predicted['depth'].device.type == 'cuda'
                        rectified = predicted['depth'].float().cpu().numpy()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-began
                    assert rectified.shape == SHAPE and np.isfinite(rectified).all() and np.all(rectified > 0)
                    # Exact old native method: bilinear remap inverse optical Z.
                    native_depth = 1/cv2.remap(1/rectified, bx, by, cv2.INTER_LINEAR)
                    native_depth = np.asarray(native_depth, dtype=np.float32)
                    assert np.isfinite(native_depth).all() and np.all(native_depth > 0)
                    path = destination/f'{row["id"]}.npz'
                    temporary = path.with_name(path.name+'.partial')
                    assert not path.exists() and not path.with_suffix('.json').exists()
                    with temporary.open('xb') as handle:
                        np.savez_compressed(handle, native_depth=native_depth)
                        handle.flush(); os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    receipt = dict(id=row['id'], output_sha256=B.sha(path), plan_sha256=plan_sha,
                        rgb_sha256=row['rgb_sha256'], camera_matrix=row['camera_matrix'],
                        weight_sha256=B.WEIGHT_SHA, upstream_revision=B.UPSTREAM, input_shape=list(native.shape),
                        output_shape=list(native_depth.shape), focal_px=float(focal), seconds=seconds,
                        model_device_type='cuda', input_device_type=tensor.device.type,
                        output_device_type=predicted['depth'].device.type, output_device=str(predicted['depth'].device),
                        gpu_name=torch.cuda.get_device_name(), precision='float16 model; float32 saved native optical Z',
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), environment=environment)
                    atomic_json(path.with_suffix('.json'), receipt)
                    outputs[row['id']] = receipt
                    timings.append(seconds)
                    predicted = tensor = None
                    print(f'inferred {row["id"]}: {len(outputs)}/96, {seconds:.3f}s', flush=True)
            assert len(outputs) == 96
            result = dict(status='COMPLETE', plan_sha256=plan_sha, observations_sha256=plan['observations_sha256'],
                count=96, new_inferences=len(timings), reused_predictions=96-len(timings),
                elapsed_seconds=time.perf_counter()-started, inference_seconds=timings,
                environment=environment, outputs=outputs, readout='native optical Z; no GT or source-quality filtering')
            atomic_json(result_path, result)
            owner['status'] = 'COMPLETE'
            return result
        except BaseException as error:
            owner.update(status='FAILED', error=repr(error))
            raise
        finally:
            predicted = tensor = model = transform = None
            gc.collect()
            if gpu_used:
                torch.cuda.empty_cache()
            owner.update(lock_release='OS handle released on leaving context', model_references_released=True)
            atomic_json(OUT/'inference-owner.json', owner, replace=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['infer'])
    parser.parse_args()
    run()
