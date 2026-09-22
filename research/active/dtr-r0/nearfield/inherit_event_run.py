"""Governed consumed-Development run; seal public predictions before label join."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from inherit_event_core import ARMS, THRESHOLD, Support, SupportInheritance
from tof_corridor_calibration import score_frame, decide
from tof_fov45_core import boxes45
from ba_camera_corridor_metrics import evaluate_rows
from full_event_metrics_20260920 import _clip_metrics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_files():
    names = ('inherit_event_core.py', 'inherit_event_run.py', 'inherit_event_tests.py',
             'inherit_event_audit.py', 'INHERIT_EVENT_PROTOCOL_20260922.md',
             'tof_corridor_calibration.py', 'tof_fov45_core.py',
             'ba_camera_corridor.py', 'ba_camera_corridor_metrics.py',
             'full_event_metrics_20260920.py')
    return [HERE / n for n in names] + [REPO / 'tools/research_backend.py']


def supports_for(tokens):
    boxes = boxes45()
    if tokens.shape != (64, 6) or not np.isfinite(tokens).all():
        raise ValueError('Malformed prepared public ToF tokens')
    reconstructed = np.rint(tokens[:, 2:] * [192, 256, 192, 256]).astype(int)
    if not np.array_equal(reconstructed, boxes):
        raise ValueError('Changed public zone geometry')
    if not np.isin(tokens[:, 1], [0., 1.]).all():
        raise ValueError('Nonboolean valid mask')
    values = np.where(tokens[:, 1] == 1, tokens[:, 0] * 8, np.nan).astype(np.float32)
    if np.any((tokens[:, 1] == 1) & ((values < .1) | (values >= 8))):
        raise ValueError('Invalid observed range')
    scored = score_frame(boxes, values)
    decision = decide(scored, THRESHOLD)
    scores = {s['zone']: s['joint'] for s in scored['zone_scores']}
    supports = [Support(a['zone'], float(values[a['zone']]), *a['interval_m'],
                        a['possible'], a['definite'], scores[a['zone']])
                for a in scored['anchors']]
    return decision, supports


def predict(identities, tokens):
    model = SupportInheritance()
    output, timings = [], []
    for i, meta in enumerate(identities):
        assert i == meta['index']
        started = time.perf_counter()
        base, supports = supports_for(tokens[i])
        # Deliberately pass only public clock/recording identity and support.
        decision = model.step(meta['id'], meta['clip_id'], meta['frame_in_clip'],
                              round(meta['time_s'] * 1e9), base['alert'], supports)
        timings.append(time.perf_counter() - started)
        output.append(dict(index=i, id=meta['id'], clip_id=meta['clip_id'],
                           frame_in_clip=meta['frame_in_clip'], time_s=meta['time_s'],
                           captured_at_ns=round(meta['time_s'] * 1e9),
                           unknown=base['unknown'], base=base,
                           supports=[asdict(s) for s in supports], **decision))
    return output, timings


def compare(rows, left, right):
    out = {name: [] for name in ('TP_gained', 'TP_lost', 'FP_added', 'FP_removed')}
    for r in rows:
        if r['truth'] is None:
            continue
        a, b = (r['predictions'][arm]['alert'] for arm in (left, right))
        if a == b:
            continue
        key = ('TP_gained' if b else 'TP_lost') if r['truth'] else ('FP_added' if b else 'FP_removed')
        out[key].append(r['id'])
    return out


def details(rows):
    clips = defaultdict(list)
    for r in rows:
        if r['truth'] is None:
            raise ValueError('Unknown truth: rich event metrics require declared complete known clips')
        clips[r['clip_id']].append(dict(r, flags={a: r['predictions'][a]['alert'] for a in ARMS}))
    result = {}
    for arm in ARMS:
        result[arm] = [_clip_metrics(sorted(seq, key=lambda r: r['frame_in_clip']), arm, .2)
                       for _, seq in sorted(clips.items())]
    return result


def event_differences(event_details, left, right):
    out = []
    for a, b in zip(event_details[left], event_details[right]):
        assert a['clip_id'] == b['clip_id']
        if a != b:
            out.append(dict(clip_id=a['clip_id'], **{left: a, right: b}))
    return out


def check_retention(rows, metrics, event_details):
    ch = compare(rows, 'A_current', 'A_hold')
    ci = compare(rows, 'A_current', 'support_inherit')
    hi = compare(rows, 'A_hold', 'support_inherit')
    lookup = {r['id']: r for r in rows}
    group_count = len({lookup[f]['base_group_id'] for f in ci['TP_gained']})
    hold_tp, hold_fp = len(ch['TP_gained']), len(ch['FP_added'])
    rescued_fraction = len(ci['TP_gained']) / hold_tp if hold_tp else None
    removed_fp_fraction = len(hi['FP_removed']) / hold_fp if hold_fp else None
    event_losses, delayed = [], []
    for a, b in zip(event_details['A_current'], event_details['support_inherit']):
        if not a['event'] or not a['event']['detected']:
            continue
        if not b['event']['detected']:
            event_losses.append(a['clip_id'])
        elif b['event']['first_in_event_alert_time_s'] > a['event']['first_in_event_alert_time_s']:
            delayed.append(a['clip_id'])
    gates = dict(hold_TP_gain_nonzero=hold_tp > 0, hold_FP_cost_nonzero=hold_fp > 0,
                 retain_75_percent_hold_TP_gain=rescued_fraction is not None and rescued_fraction >= .75,
                 remove_50_percent_hold_added_FP=removed_fp_fraction is not None and removed_fp_fraction >= .5,
                 at_least_8_added_TP=len(ci['TP_gained']) >= 8,
                 at_least_4_gain_layouts=group_count >= 4,
                 preserve_A_current_TP=not ci['TP_lost'],
                 no_lost_current_events=not event_losses, no_delayed_current_events=not delayed,
                 no_new_false_segments_vs_hold=metrics['arms']['support_inherit']['false_alert_segment_count']
                 <= metrics['arms']['A_hold']['false_alert_segment_count'])
    return dict(gates=gates, retain=all(gates.values()),
                hold_extra_TP=hold_tp, hold_extra_FP=hold_fp,
                inherited_extra_TP=len(ci['TP_gained']), inherited_extra_FP=len(ci['FP_added']),
                retained_hold_TP_fraction=rescued_fraction, removed_hold_FP_fraction=removed_fp_fraction,
                gain_layouts=group_count, lost_events=event_losses, delayed_events=delayed,
                paired_current_to_hold=ch, paired_current_to_inherit=ci, paired_hold_to_inherit=hi)


def evaluate(identities, predictions, label_paths):
    roles, all_rows = {}, []
    opened = []
    for role, path in label_paths.items():
        labels = dict(np.load(path, allow_pickle=False))
        opened.append(dict(role=role, path=str(path), sha256=sha(path)))
        truth = (labels['classes'][:, [1, 4]] < 6).any(1)
        valid = labels['valid'][:, [1, 4]].all(1)
        rows = []
        for k, raw_i in enumerate(labels['indices']):
            i = int(raw_i)
            meta, p = identities[i], predictions[i]
            assert meta['index'] == i == p['index'] and meta['split'] == role
            row = {key: meta[key] for key in ('id', 'clip_id', 'frame_in_clip', 'time_s',
                   'base_group_id', 'type_id', 'layer', 'layout_relation')}
            row.update(index=i, role=role, truth=bool(truth[k]) if valid[k] else None,
                       boundary=meta['layout_relation'] == 'BOUNDARY',
                       predictions={a: dict(alert=p['flags'][a], unknown=p['unknown'],
                                            ambiguous=bool(p['flags'][a] and p['unknown'])) for a in ARMS})
            rows.append(row)
        assert len(rows) == dict(train=864, dev=288, evaluation=576)[role]
        all_rows.extend(rows)
        metrics = evaluate_rows(rows, arms=ARMS)
        event_details = details(rows)
        strata = {key: {value: evaluate_rows([r for r in rows if r[key] == value], arms=ARMS)
                        for value in sorted({r[key] for r in rows})}
                  for key in ('type_id', 'layer', 'layout_relation', 'base_group_id')}
        roles[role] = dict(metrics=metrics, event_details=event_details, strata=strata,
                           comparison=check_retention(rows, metrics, event_details),
                           event_differences=event_differences(event_details, 'A_hold', 'support_inherit'))
    assert len(all_rows) == 1728 and len({r['index'] for r in all_rows}) == 1728
    return roles, sorted(all_rows, key=lambda r: r['index']), opened


def run(args):
    out = args.result.parent
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Nonempty output; preserve prior attempt')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = {str(p.relative_to(REPO)): sha(p) for p in source_files()}
    materialization = read(args.materialization)
    observed_hashes = {str(p): sha(p) for p in (args.identities, args.tof, args.materialization)}
    assert observed_hashes[str(args.identities)] == materialization['hashes']['observations/identities.json']
    assert observed_hashes[str(args.tof)] == materialization['hashes']['observations/tof.npy']
    write(out / 'freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha(HERE / 'INHERIT_EVENT_PROTOCOL_20260922.md'),
          source_hashes=sources, public_input_hashes=observed_hashes,
          threshold=THRESHOLD, no_label_arrays_opened=True, candidate_count=1,
          scope='CONSUMED_CONTROLLED_SIMULATION_DEVELOPMENT'))
    (out / 'protocol-before-run.md').write_bytes((HERE / 'INHERIT_EVENT_PROTOCOL_20260922.md').read_bytes())
    backend = select_backend('scalar-scoring', cpu=BackendCandidate(
        'python-numpy-cpu', 'cpu', lambda: supports_for(np.array(
            [[0., 0., *list(b / [192, 256, 192, 256])] for b in boxes45()], np.float32)),
        lambda _: DeviceObservation('cpu', platform.processor() or 'CPU', 'python/numpy')),
        record_path=out / 'backend.json', capabilities={'task_class': 'scalar-scoring',
        'placement_reason': 'TASK_NOT_GPU_SUITABLE', 'model_inference': False, 'training': False})
    identities = read(args.identities)
    tokens = np.load(args.tof, allow_pickle=False, mmap_mode='r')
    assert tokens.shape == (1728, 64, 6) and len(identities) == 1728
    assert len({m['clip_id'] for m in identities}) == 144
    assert all(v == 12 for v in Counter(m['clip_id'] for m in identities).values())
    predictions, timings = predict(identities, tokens)
    write(out / 'predictions.json', predictions)
    write(out / 'prediction-seal.json', dict(status='PASS', frames=len(predictions),
          predictions_sha256=sha(out / 'predictions.json'), freeze_sha256=sha(out / 'freeze.json'),
          sealed_at_utc=datetime.now(timezone.utc).isoformat(), label_arrays_opened=False,
          runtime_inputs='public observed ranges, footprints, clip/frame identity and capture time only',
          detector_track_tristate='NOT_EVALUABLE_NO_REAL_DETECTOR_TRACK_INPUT',
          candidate_count=1, recursive=False, ttl_ns=200_000_000))
    # Only after prediction seal: historical parity and evaluator-only labels.
    for meta, p in zip(identities, predictions):
        for key in ('alert', 'unknown', 'possible_zones', 'definite_zones', 'valid_zones', 'score', 'threshold'):
            assert p['base'][key] == meta['baseline'][key], (p['id'], key)
    label_paths = {r: getattr(args, r + '_labels') for r in ('train', 'dev', 'evaluation')}
    for role, path in label_paths.items():
        assert sha(path) == materialization['hashes'][f'labels/{role}.npz']
    roles, rows, opened = evaluate(identities, predictions, label_paths)
    ev = roles['evaluation']['metrics']['arms']
    for a, expected in (('A_current', (193, 6, 63)), ('A_hold', (225, 26, 31))):
        f = ev[a]['frames']['all_known']
        assert tuple(f[k] for k in ('TP', 'FP', 'FN')) == expected, a
    primary = roles['evaluation']['comparison']
    terminal = 'RETAIN_SCOPED_CHALLENGER' if primary['retain'] else 'EXACT_RECIPE_NEGATIVE_CONTROL'
    counters = dict(reasons=dict(Counter(p['reason'] for p in predictions)),
                    support_check_reasons=dict(Counter(c['reason'] for p in predictions for c in p['support_checks'])),
                    inherited_unknown=sum(p['inherited'] and p['unknown'] for p in predictions),
                    missing_detector_tracks='NOT_EVALUABLE')
    report = dict(status='PASS', terminal=terminal, roles=roles, support_diagnostics=counters,
        source_boundary='All 1728 consumed frames, 48 layouts, one procedural generator; original 576-frame evaluation role primary',
        truth_labels='All declared controlled AABB intersections; BODY/HEAD centre query union; no natural actionability truth',
        unknown_preserved=True, prediction_seal_sha256=sha(out / 'prediction-seal.json'),
        evaluator_inputs=opened, backend=backend,
        runtime=dict(total_s=time.perf_counter()-started,
                     scoring_plus_decoder_ms_mean=float(np.mean(timings)*1000),
                     scoring_plus_decoder_ms_p50=float(np.quantile(timings,.5)*1000),
                     scoring_plus_decoder_ms_p95=float(np.quantile(timings,.95)*1000),
                     description='Host replay from cached public ranges; excludes capture/ToF generation; not sustained device latency'),
        current_parity_frames=1728, owned_processes_remaining=0)
    write(out / 'frame-results.json', rows)
    write(out / 'metrics.json', report)
    for name, old_sha in sources.items():
        assert sha(REPO / name) == old_sha, 'Execution source mutated: ' + name
    write(out / 'evaluation-seal.json', dict(status='PASS', source_hashes=sources,
          hashes={n: sha(out / n) for n in ('freeze.json','prediction-seal.json','predictions.json',
                  'frame-results.json','metrics.json','backend.json','protocol-before-run.md')},
          evaluator_inputs=opened, sealed_at_utc=datetime.now(timezone.utc).isoformat()))
    result = dict(status='PASS', terminal=terminal, source_frames=1728, primary_frames=576,
                  source_boundary=report['source_boundary'],
                  primary_arms={a: dict(ev[a]['frames']['all_known'],
                    events=f"{ev[a]['detected_events']}/{ev[a]['event_count']}",
                    false_segments=ev[a]['false_alert_segment_count']) for a in ARMS},
                  primary_gate=primary['gates'], metrics_sha256=sha(out / 'metrics.json'))
    write(args.result, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('identities', 'tof', 'materialization', 'train-labels', 'dev-labels', 'evaluation-labels', 'result'):
        parser.add_argument('--' + name, required=True, type=Path)
    run(parser.parse_args())
