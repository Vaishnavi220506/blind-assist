"""Public-only A* entry; one `alert` drives saved results and reminder records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from astar_inference import AStarSystem


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_rows(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def image_path(root, row):
    path = (root/row['rgb_path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('RGB path escapes the supplied capture root')
    return path


def main(bundle, raw, rgb_root, output, warmup_raw=None, warmup_rgb_root=None):
    if output.exists():
        raise FileExistsError('Use a new replay output directory: '+str(output))
    if (warmup_raw is None) != (warmup_rgb_root is None):
        raise ValueError('Supply both warmup input paths or neither')
    rows = read_rows(raw)
    if not rows:
        raise ValueError('Empty replay')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frame IDs')
    output.mkdir(parents=True)
    try:
        config_path = bundle/'config.json'
        config = json.loads(config_path.read_text())
        inputs = {str(raw.resolve()): sha(raw), str(config_path.resolve()): sha(config_path),
            str((bundle/config['model_file']).resolve()): sha(bundle/config['model_file'])}
        sources = {str(p.resolve()): sha(p) for p in [Path(__file__), Path(__file__).with_name('astar_inference.py')]}
        tick = time.perf_counter()
        system = AStarSystem(bundle)
        load_s = time.perf_counter()-tick
        warmup_count = 0
        if warmup_raw is not None:
            warmup = read_rows(warmup_raw)[:3]
            if len(warmup) != 3:
                raise ValueError('Three consumed warmup frames required')
            yaw, episode = 0., None
            for row in warmup:
                if row['episode_id'] != episode:
                    yaw = 0.
                if row['imu_valid']:
                    yaw += row['delta_yaw']
                episode = row['episode_id']
                rgb = cv2.imread(str(image_path(warmup_rgb_root, row)))
                if rgb is None:
                    raise ValueError('Warmup RGB decode failed')
                system.predict(row, rgb, yaw)
            warmup_count = len(warmup)
        write(output/'input-seal.json', dict(inputs=inputs, sources=sources,
            warmup_frames=warmup_count, warmup_raw=str(warmup_raw) if warmup_raw else None,
            model_count=1, evaluator_read=False, backend='CPU_NUMPY_OPENCV_SKLEARN',
            backend_reason='GPU_BACKEND_UNAVAILABLE_FOR_UNCHANGED_FRONTEND_HGB'))
        all_results, reminders = [], []
        yaw, episode, previous_alert = 0., None, False
        with (output/'predictions.jsonl').open('x', encoding='utf-8') as saved, (output/'reminders.jsonl').open('x', encoding='utf-8') as alarm:
            for row in rows:
                new_episode = row['episode_id'] != episode
                if new_episode:
                    yaw, previous_alert = 0., False
                if row['imu_valid']:
                    yaw += row['delta_yaw']
                episode = row['episode_id']
                path = image_path(rgb_root, row)
                tick = time.perf_counter()
                rgb = cv2.imread(str(path))
                decode_ms = (time.perf_counter()-tick)*1000
                if rgb is None:
                    raise ValueError('RGB decode failed: '+row['id'])
                result = system.predict(row, rgb, yaw)
                complete_ms = (time.perf_counter()-tick)*1000
                result.update(id=row['id'], episode_id=episode, time_s=row['time_s'],
                    rgb_path=row['rgb_path'], tof_packet_received=row['tof_packet_received'],
                    yaw=yaw, rgb_decode_ms=decode_ms, complete_ms=complete_ms)
                # Consumers use this same Boolean; there is no second decision path.
                reminder = dict(id=row['id'], episode_id=episode, time_s=row['time_s'],
                    alert=result['alert'], reminder_onset=bool(result['alert'] and not previous_alert))
                previous_alert = result['alert']
                saved.write(json.dumps(result, allow_nan=False)+'\n')
                alarm.write(json.dumps(reminder)+'\n')
                all_results.append(result)
                reminders.append(reminder)
        def stats(key):
            values = np.array([r[key] for r in all_results])
            return dict(mean=float(values.mean()), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), observed_max=float(values.max()))
        forbidden = sorted(set(sys.modules) & {'torch', 'public_positive', 'single_positive_inference'})
        if forbidden:
            raise AssertionError('Unexpected comparison/depth runtime imported: '+str(forbidden))
        write(output/'summary.json', dict(frames=len(all_results), episodes=len({r['episode_id'] for r in rows}),
            alerts=sum(r['alert'] for r in all_results), reminder_onsets=sum(r['reminder_onset'] for r in reminders),
            model_load_seconds=load_s, warmup_frames=warmup_count,
            latency_ms={k: stats(k) for k in ['rgb_decode_ms', 'algorithm_ms', 'complete_ms']},
            timing_scope='Per-frame wall time RGB read/decode plus public frontend and only A* HGB; excludes JSON writes, display, audio, sensor capture and transport',
            model_count=1, old_A_executed=False, positive_branch_executed=False, evaluator_read=False,
            forbidden_modules=forbidden, backend='CPU_NUMPY_OPENCV_SKLEARN'))
        assert all(sha(p) == h for p, h in {**inputs, **sources}.items())
        write(output/'completion.json', dict(status='PASS', outputs={n: sha(output/n) for n in
            ['predictions.jsonl', 'reminders.jsonl', 'input-seal.json', 'summary.json']},
            resources='One process-local model released at exit'))
        print(json.dumps(dict(status='PASS', frames=len(rows), latency_ms=stats('complete_ms'))))
    except Exception as error:
        write(output/'failure.json', dict(status='FAILED', error=repr(error), preserve_partial_output=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['bundle', 'raw', 'rgb-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--warmup-raw', type=Path)
    parser.add_argument('--warmup-rgb-root', type=Path)
    args = parser.parse_args()
    with threadpool_limits(4):
        main(args.bundle, args.raw, args.rgb_root, args.output, args.warmup_raw, args.warmup_rgb_root)
