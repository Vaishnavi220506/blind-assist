"""Frozen-score current-only audit; no training or protected test access."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import full_event_metrics_20260920 as metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-last-layer-20260921'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OLD = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
OUT = ROOT / 'artifacts.local/work/ba-current-only-20260921'
PROTOCOL = HERE / 'CURRENT_ONLY_PROTOCOL_20260921.md'
ROLES = ('selection', 'evaluation')
HEADS = ('original', 'uniform', 'balanced')
ARMS = tuple(a + '_' + s for a in ('A',) + HEADS for s in ('current', 'hold'))
metrics.ARMS = ARMS


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_source():
    for name in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'):
        assert not (OLD / name).exists(), name
    for seal_name in ('role-seal.json', 'prediction-seal.json'):
        seal = read(SOURCE / seal_name)
        assert seal['protocol_sha256'] == sha(SOURCE / 'protocol.json')
        for name, digest in seal['hashes'].items():
            assert sha(SOURCE / name) == digest, name


def verify():
    check_source()
    for name, digest in read(OUT / 'protocol.json')['inputs'].items():
        assert sha(ROOT / name) == digest, name


def seal(name, files):
    write(OUT / name, dict(protocol_sha256=sha(OUT / 'protocol.json'),
                          hashes={f: sha(OUT / f) for f in files}))


def check_seal(name):
    saved = read(OUT / name)
    assert saved['protocol_sha256'] == sha(OUT / 'protocol.json')
    for file, digest in saved['hashes'].items():
        assert sha(OUT / file) == digest, file


def cutoff(rows, predictions, labels, head):
    assert [r['id'] for r in rows] == [r['id'] for r in predictions] == [r['id'] for r in labels]
    forbidden = [(r, p['scores'][head]) for r, p, y in zip(rows, predictions, labels)
                 if not y['truth'] and not r['a']]
    assert forbidden and all(math.isfinite(s) for _, s in forbidden)
    maximum = max(s for _, s in forbidden)
    return dict(threshold=math.nextafter(maximum, math.inf), forbidden_max=maximum,
                forbidden_ids=[r['id'] for r, _ in forbidden],
                blockers=[dict(id=r['id'], group=r['base_group_id'],
                    relation=r['layout_relation'], time_s=r['time_s'], score=s)
                    for r, s in forbidden if s == maximum])


def predict():
    assert not OUT.exists(), 'Do not overwrite a completed or partial audit'
    check_source()
    files = [SOURCE / n for n in ('protocol.json', 'role-seal.json', 'prediction-seal.json', 'selection.json')]
    for role in ROLES:
        files += [SOURCE / (role + '-' + s + '.json') for s in ('rows', 'predictions')]
        files.append(SOURCE / 'labels' / (role + '.json'))
    files += [TRANSFER / 'source-admission.json', TRANSFER / 'spec.json',
              PROTOCOL, Path(__file__), HERE / 'full_event_metrics_20260920.py']
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', dict(id='ba-current-only-20260921',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), scope='CONSUMED_DEVELOPMENT',
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        backend='CPU', reason='TASK_NOT_GPU_SUITABLE', original_test_activated=False,
        automatic_successor=False, roles={'selection': 'g00/g01', 'evaluation': 'g02/g03'}))
    (OUT / 'protocol-before-run.md').write_bytes(PROTOCOL.read_bytes())
    rows = read(SOURCE / 'selection-rows.json')
    saved = read(SOURCE / 'selection-predictions.json')
    labels = read(SOURCE / 'labels/selection.json')
    assert all(type(y['truth']) is bool for y in labels)
    thresholds = {head: cutoff(rows, saved, labels, head) for head in HEADS}
    old = read(SOURCE / 'selection.json')
    for head in ('uniform', 'balanced'):
        thresholds[head]['prior_current_plus_hold_threshold'] = old[head]['threshold']
        assert thresholds[head]['threshold'] <= old[head]['threshold']
    write(OUT / 'selection.json', thresholds)
    outputs = ['selection.json']
    group_sets = []
    for role in ROLES:
        rows = read(SOURCE / (role + '-rows.json'))
        saved = read(SOURCE / (role + '-predictions.json'))
        assert [r['id'] for r in rows] == [r['id'] for r in saved]
        assert len(rows) == 576 and len({r['id'] for r in rows}) == 576
        groups = {r['base_group_id'] for r in rows}
        assert len(groups) == 8
        assert all(g.endswith(('g00', 'g01') if role == 'selection' else ('g02', 'g03')) for g in groups)
        group_sets.append(groups)
        clips = defaultdict(list)
        predictions = []
        for row, p in zip(rows, saved):
            assert row['source'] == 'transfer' and p['flags']['A_current'] == row['a']
            assert all(math.isfinite(p['scores'][h]) for h in HEADS)
            flags = {'A_current': row['a']}
            flags.update({h + '_current': bool(row['a'] or p['scores'][h] >= thresholds[h]['threshold']) for h in HEADS})
            record = dict(id=row['id'], scores=p['scores'], flags=flags,
                          current_unknown={a: row['unknown'] for a in ARMS})
            predictions.append(record)
            clips[row['clip_id']].append((row, record))
        for clip in clips.values():
            clip.sort(key=lambda pair: pair[0]['frame_in_clip'])
            assert [r['frame_in_clip'] for r, _ in clip] == list(range(24))
            for i, (r, p) in enumerate(clip):
                assert math.isclose(r['time_s'], i * .2, abs_tol=1e-8)
                for head in ('A',) + HEADS:
                    p['flags'][head + '_hold'] = bool(p['flags'][head + '_current'] or
                        (i > 0 and clip[i - 1][1]['flags'][head + '_current']))
        name = role + '-predictions.json'
        write(OUT / name, predictions)
        outputs.append(name)
    assert not group_sets[0] & group_sets[1]
    write(OUT / 'prediction-receipt.json', dict(evaluation_label_values_read=False,
        evaluation_label_file_hashed=True, native_values_read=False,
        source_scores_reused=True, training=False, model_inference=False,
        thresholds_selected_on='selection only', hold_used_for_selection=False))
    seal('prediction-seal.json', outputs + ['prediction-receipt.json'])
    verify()
    print(json.dumps({h: dict(threshold=thresholds[h]['threshold'],
        blockers=thresholds[h]['blockers']) for h in HEADS}))


def geometry(row, cases):
    case = cases[row['index']]
    camera = case['camera']
    assert all(camera[k] == 0 for k in ('pitch', 'yaw', 'roll'))
    target = next(o for o in case['objects'] if o['name'] == case['target_name'])
    near = target['center_m'][0] - target['size_m'][0] / 2 - camera['x']
    far = target['center_m'][0] + target['size_m'][0] / 2 - camera['x']
    return 'axial_depth_outside_contract' if near > 3. or far < .3 else 'within_depth_outside_corridor'


def assess(rows, report, head):
    name = head + '_current'
    boundary = [r for r in rows if r['layout_relation'] == 'BOUNDARY']
    gains = [r for r in boundary if r['truth'] and r['flags'][name] and not r['a']]
    retain = all(not r['a'] or r['flags'][name] for r in rows)
    no_extra = all(report[p]['arms'][name]['frames']['FP'] == report[p]['arms']['A_current']['frames']['FP']
        and report[p]['arms'][name]['false_alert_segment_count'] <= report[p]['arms']['A_current']['false_alert_segment_count']
        for p in ('Core', 'Boundary'))
    gain = report['Boundary']['arms'][name]['frames']['recall'] - report['Boundary']['arms']['A_current']['frames']['recall']
    groups = sorted({r['base_group_id'] for r in gains})
    return dict(all_A_current_retained=retain, no_added_FP_or_segments=no_extra,
        boundary_recall_gain=gain, boundary_rescued=len(gains), gaining_groups=groups,
        supported=bool(retain and no_extra and gain >= .10 and len(groups) >= 4))


def evaluate():
    started = time.perf_counter()
    verify()
    check_seal('prediction-seal.json')
    write(OUT / 'evaluation-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(OUT / 'prediction-seal.json')))
    native = {r['id']: r['returned_target_corridor_samples'] for r in read(TRANSFER / 'source-admission.json')['frames']}
    cases = read(TRANSFER / 'spec.json')['cases']
    summaries, gates, files = {}, {}, []
    for role in ROLES:
        metadata = read(SOURCE / (role + '-rows.json'))
        predictions = read(OUT / (role + '-predictions.json'))
        labels = read(SOURCE / 'labels' / (role + '.json'))
        assert [r['id'] for r in metadata] == [r['id'] for r in predictions] == [r['id'] for r in labels]
        rows = [dict(**r, truth=y['truth'], scores=p['scores'], flags=p['flags'],
                    current_unknown=p['current_unknown'], native_target_corridor_samples=native[r['source_id']])
                for r, p, y in zip(metadata, predictions, labels)]
        report = metrics.evaluate(rows)
        group_reports = {g: metrics.evaluate([r for r in rows if r['base_group_id'] == g])
                         for g in sorted({r['base_group_id'] for r in rows})}
        summaries[role] = {part: {arm: dict(counts=[v['frames'][k] for k in ('TP', 'FP', 'FN')],
            precision=v['frames']['precision'], FPR=v['frames']['FPR'],
            events=[v['detected_events'], v['event_count']], false_segments=v['false_alert_segment_count'],
            false_sampled_s=v['false_alert_sampled_s'], unknown=v['current_unknown'])
            for arm, v in report[part]['arms'].items()} for part in ('Core', 'Boundary')}
        gates[role] = {h: assess(rows, report, h) for h in HEADS}
        changes, fp_geometry = {}, {}
        for arm in ARMS:
            fp_geometry[arm] = dict(Counter(r['layout_relation'] + ':' + geometry(r, cases)
                for r in rows if not r['truth'] and r['flags'][arm]))
            if arm.startswith('A_'):
                continue
            base = 'A_' + arm.rsplit('_', 1)[1]
            changes[arm] = {}
            for label, predicate in (
                ('rescued_A_TP', lambda r: r['truth'] and r['flags'][arm] and not r['flags'][base]),
                ('added_A_FP', lambda r: not r['truth'] and r['flags'][arm] and not r['flags'][base]),
                ('lost_A_TP', lambda r: r['truth'] and r['flags'][base] and not r['flags'][arm]),
                ('A_hold_TP_missing', lambda r: r['truth'] and r['flags']['A_hold'] and not r['flags'][arm])):
                chosen = [r for r in rows if predicate(r)]
                changes[arm][label] = dict(ids=[r['id'] for r in chosen], count=len(chosen),
                    native_backed=sum(r['native_target_corridor_samples'] > 0 for r in chosen),
                    by_relation=dict(Counter(r['layout_relation'] for r in chosen)),
                    by_group=dict(Counter(r['base_group_id'] for r in chosen)))
        for suffix, value in (('frame-results', rows), ('metrics', report), ('group-metrics', group_reports),
                              ('changes', changes), ('fp-geometry', fp_geometry)):
            name = role + '-' + suffix + '.json'
            write(OUT / name, value)
            files.append(name)
    support = {h: all(gates[r][h]['supported'] for r in ROLES) for h in HEADS}
    result = dict(status='COMPLETE', scope='CONSUMED_DEVELOPMENT', metrics=summaries, gates=gates,
        current_only_support=support, original_test_activated=False, automatic_successor=False,
        elapsed_s=time.perf_counter() - started)
    write(OUT / 'result.json', result)
    seal('evaluation-seal.json', files + ['result.json'])
    verify()
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('predict', 'evaluate'))
    globals()[parser.parse_args().stage]()
