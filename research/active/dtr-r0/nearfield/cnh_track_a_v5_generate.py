"""Frozen-source v5 primary geometry/sensor orchestration; never reads scores.

Writes source hashes, per-unit logs, progress and terminal receipts. Resume skips
completed unit artifacts only after matching the frozen request identity. A crash
during a unit loses at most that unit's work; never automatically retries failures.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--revision', default='9b4e68ae')
    p.add_argument('--workers', type=int, default=17)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--readouts-only', action='store_true')
    p.add_argument('--gpu-workers', type=int, default=4)
    a = p.parse_args()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    if not a.output.is_absolute():
        raise ValueError('Output must be absolute')
    if not a.output.resolve().is_relative_to((a.repo / 'artifacts.local').resolve()):
        raise ValueError('Output must resolve within canonical artifacts.local')
    a.output.mkdir(parents=True, exist_ok=True)
    source = a.output / 'source'
    source.mkdir(exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', a.revision], text=True).strip()
    family = 'cnh-track-a-scale-v5-' + ('smoke-' if a.smoke else '') + '20260928'
    splits = [0, 2, 2] if a.smoke else [0, 32, 64]
    request = dict(revision=revision, family=family, splits=splits, mount=-10,
                   rate_hz=5, analysis_snr=6,
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest())
    rp = a.output / 'request.json'
    if rp.exists() and json.loads(rp.read_text()) != request:
        raise RuntimeError('Request identity differs; do not resume')
    save(rp, request)
    prefix = 'research/active/dtr-r0/nearfield/'
    paths = subprocess.check_output(['git', '-C', str(a.repo), 'ls-tree', '-r', '--name-only', revision, '--', prefix], text=True).splitlines()
    hashes = {}
    for path in paths:
        relative = path[len(prefix):]
        if not relative.endswith('.py') or '/' in relative:
            continue
        content = subprocess.check_output(['git', '-C', str(a.repo), 'show', revision + ':' + path])
        target = source / relative
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError('Existing source snapshot differs: ' + relative)
        target.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    save(a.output / 'source_manifest.json', dict(revision=revision, sha256=hashes))
    (a.output / 'runner_frozen.py').write_bytes(Path(__file__).read_bytes())
    if a.prepare_only:
        return
    budget_path = a.output / 'budget.json'
    if not budget_path.exists():
        save(budget_path, dict(started_unix=time.time(), wall_limit_s=10800, storage_limit_bytes=10 * 1024**3))
    budget = json.loads(budget_path.read_text())
    def budget_check():
        elapsed = time.time() - budget['started_unix']
        size = sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())
        if elapsed > budget['wall_limit_s'] or size > budget['storage_limit_bytes']:
            raise RuntimeError(f'Frozen budget exceeded: elapsed={elapsed}, bytes={size}')
        return max(1., budget['wall_limit_s'] - elapsed)
    budget_check()
    if a.readouts_only:
        terminal = json.loads((a.output / 'terminal.json').read_text())
        if terminal['status'] != 'GENERATION_READY':
            raise RuntimeError('Generation gates not ready')
        sys.path.insert(0, str(source))
        import numpy as np
        import cnh_track_a_scale_evaluate as se
        from cnh_track_a_scale_gpu_run import score_unit_gpu
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no unapproved CPU fallback')
        se.sensor_module.FAMILY = family
        out = a.output / 'readouts-gpu' / 'primary-mount-10-snr6'
        out.mkdir(parents=True, exist_ok=True)
        geometry, sensor = a.output / 'geometry', a.output / 'sensor'
        started = time.monotonic()
        if not (out / 'bias.npy').exists():
            calib = []
            for u in range(splits[1]):
                if (geometry / f'unit{u:02d}' / f'unit{u:02d}.json').exists():
                    calib.extend(r['hist'] for r in se.unit_records(geometry, sensor, u, -10, 1)[1])
            np.save(out / 'bias.npy', np.median(np.concatenate(calib), axis=0))
        jobs = [(str(geometry), str(sensor), str(out), u, -10, 1, family)
                for u in range(sum(splits)) if (geometry / f'unit{u:02d}' / f'unit{u:02d}.json').exists()]
        for job in jobs:
            target = out / f'unit{job[3]:02d}.npz'
            if target.exists():
                with np.load(target) as saved:
                    if saved['S2__noisy@0.75'].shape != (384, 6) or saved['labels'].shape != (384, 6):
                        raise RuntimeError('Partial readout: ' + str(target))
        # Child process imports resolve to the immutable snapshot via inherited sys.path.
        lock = a.output / 'READOUTS_RUNNING.lock'
        with lock.open('x') as f:
            f.write(str(os.getpid()))
        try:
            with ProcessPoolExecutor(a.gpu_workers) as pool:
                futures = [pool.submit(score_unit_gpu, job) for job in jobs]
                for i, f in enumerate(as_completed(futures), 1):
                    f.result()
                    budget_check()
                    save(a.output / 'progress.json', dict(stage='gpu_readouts', completed=i, total=len(jobs), elapsed_s=time.monotonic()-started))
            save(a.output / 'readout_terminal.json', dict(status='GPU_READY_PENDING_CONCORDANCE', elapsed_s=time.monotonic()-started,
                 backend='CUDA', device=torch.cuda.get_device_name(), completed=len(jobs), request=request))
            # First valid calib only; compare frozen S2 rather than inspecting metrics.
            import cnh_track_a_scale_fast as sf
            import shutil
            cpu = a.output / 'readouts-cpu-check' / 'primary-mount-10-snr6'
            cpu.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / 'bias.npy', cpu / 'bias.npy')
            u = min(j[3] for j in jobs if j[3] < splits[1])
            budget_check()
            sf.score_unit_fast((str(geometry), str(sensor), str(cpu), u, -10, 1, family, True))
            key = 'S2__noisy@0.75'
            with np.load(out / f'unit{u:02d}.npz') as g, np.load(cpu / f'unit{u:02d}.npz') as c:
                relative = float(np.max(np.abs(g[key] - c[key])) / max(np.max(np.abs(c[key])), 1e-12))
            save(a.output / 'concordance.json', dict(unit=u, key=key, max_relative_error=relative, tolerance=5e-7, pass_gate=relative <= 5e-7))
            if relative > 5e-7:
                raise RuntimeError('S2 CPU/GPU concordance failed')
            budget_check()
            save(a.output / 'readout_terminal.json', dict(status='READOUTS_READY', elapsed_s=time.monotonic()-started,
                 backend='CUDA', device=torch.cuda.get_device_name(), completed=len(jobs), request=request,
                 concordance_relative_error=relative))
        except BaseException as e:
            save(a.output / 'readout_terminal.json', dict(status='STOP', error=repr(e), elapsed_s=time.monotonic()-started))
            raise
        finally:
            lock.unlink()
        return
    lock = a.output / 'RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    started = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    n = sum(splits)
    def run(script, arguments, log):
        with log.open('w', encoding='utf-8') as out:
            subprocess.run([sys.executable, str(source / script), *map(str, arguments)],
                           cwd=source, env=env, stdout=out, stderr=subprocess.STDOUT, check=True,
                           timeout=budget_check())
    def batch(stage, jobs):
        save(a.output / 'progress.json', dict(stage=stage, completed=0, total=len(jobs), eta='unknown'))
        with ThreadPoolExecutor(a.workers) as pool:
            futures = [pool.submit(run, *j) for j in jobs]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                budget_check()
                save(a.output / 'progress.json', dict(stage=stage, completed=i, total=len(jobs), elapsed_s=time.monotonic()-started, eta='unknown'))
    try:
        geometry, sensor = a.output / 'geometry', a.output / 'sensor'
        geometry.mkdir(exist_ok=True)
        sensor.mkdir(exist_ok=True)
        jobs = []
        for u in range(n):
            unit = geometry / f'unit{u:02d}'
            unit.mkdir(exist_ok=True)
            completed = unit / f'unit{u:02d}.json'
            if completed.exists():
                meta = json.loads(completed.read_text(encoding='utf-8-sig'))
                if meta['unit'] != u or len(meta['configs']) != 32 or not all(len(c['labels']) == 12 for c in meta['configs']):
                    raise RuntimeError(f'Partial geometry unit {u}: not automatically resumable')
            if not (unit / f'unit{u:02d}.json').exists() and not (unit / f'unit{u:02d}-INCOMPLETE.json').exists():
                jobs.append(('cnh_track_a_v13_generate.py', ['--output', unit, '--units', u, '--family', family, '--split-counts', *splits, '--fast-margin', '--no-10hz'], geometry / f'log-unit{u:02d}.txt'))
        batch('geometry', jobs)
        run('cnh_track_a_v13_repair.py', ['--root', geometry, '--n-units', n, '--family', family, '--split-counts', *splits, '--fast-margin'], geometry / 'repair.txt')
        run('cnh_track_a_v13_merge.py', ['--root', geometry, '--n-units', n, '--v2-gates', '--split-counts', *splits], geometry / 'merge.txt')
        gates = json.loads((geometry / 'result.json').read_text())
        if gates['status'] != 'GEOMETRY_READY':
            raise RuntimeError('Geometry hard gate failed; see geometry/result.json')
        jobs = []
        for u in gates['completed_units']:
            meta_path = sensor / f'unit{u:02d}-mount-10.json'
            if meta_path.exists():
                import numpy as np
                meta = json.loads(meta_path.read_text())
                with np.load(sensor / f'unit{u:02d}-mount-10-observations.npz') as obs:
                    if meta['unit'] != u or meta['rate'] != 5 or obs['hist'].shape[:2] != (3, 384) or len(obs['config']) != 384:
                        raise RuntimeError(f'Partial sensor unit {u}: not automatically resumable')
                if meta['oracle']:
                    with np.load(sensor / f'unit{u:02d}-mount-10-oracle.npz') as oracle:
                        if len(oracle['object_id']) != 384:
                            raise RuntimeError(f'Partial oracle unit {u}: not automatically resumable')
            else:
                jobs.append(('cnh_track_a_v13_sensor.py', ['--geometry', geometry, '--output', sensor, '--unit', u, '--mount', -10, '--family', family, '--rate', 5], sensor / f'log-unit{u:02d}.txt'))
        batch('sensor', jobs)
        run('cnh_track_a_scale_g3g4.py', [a.output, n, -10], sensor / 'g3g4.txt')
        checks = json.loads((sensor / 'g3g4.txt').read_text())
        if not checks['G3'] or not checks['G4']:
            raise RuntimeError('Sensor hard gate failed')
        save(a.output / 'terminal.json', dict(status='GENERATION_READY', elapsed_s=time.monotonic()-started, request=request, sensor_gates=checks))
    except BaseException as e:
        save(a.output / 'terminal.json', dict(status='STOP', error=repr(e), elapsed_s=time.monotonic()-started))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
