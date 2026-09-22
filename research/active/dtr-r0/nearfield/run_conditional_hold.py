"""One fixed-entry original-score conditional continuation experiment."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import full_event_metrics_20260920 as metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-last-layer-20260921'
CURRENT = ROOT / 'artifacts.local/work/ba-current-only-20260921'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT / 'artifacts.local/work/ba-conditional-hold-20260921'
PROTOCOL = HERE / 'CONDITIONAL_HOLD_PROTOCOL_20260921.md'
ROLES = ('selection', 'evaluation')
ARMS = ('A_current', 'A_hold', 'original_current', 'original_hold', 'conditional')
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


def check_seal(folder, name):
    seal = read(folder / name)
    assert seal['protocol_sha256'] == sha(folder / 'protocol.json')
    for file, digest in seal['hashes'].items():
        assert sha(folder / file) == digest, file


def seal(name, files):
    write(OUT / name, dict(protocol_sha256=sha(OUT / 'protocol.json'),
                          hashes={f: sha(OUT / f) for f in files}))


def verify_inputs():
    for file, digest in read(OUT / 'protocol.json')['inputs'].items():
        assert sha(ROOT / file) == digest, file


def clips(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['clip_id']].append(r)
    return [sorted(groups[k], key=lambda r: r['frame_in_clip']) for k in sorted(groups)]


def decode(trigger, scores, keep):
    assert len(trigger) == len(scores) and math.isfinite(keep)
    state, output = False, []
    for start, score in zip(trigger, scores):
        assert math.isfinite(score)
        state = bool(start or (state and score >= keep))
        output.append(state)
    return output


def predict_rows(rows, keep, on):
    output = {}
    for c in clips(rows):
        assert [r['frame_in_clip'] for r in c] == list(range(24))
        assert all(math.isclose(r['time_s'], i * .2, abs_tol=1e-8) for i, r in enumerate(c))
        a = [r['a'] for r in c]
        scores = [r['score'] for r in c]
        trigger = [bool(x or s >= on) for x, s in zip(a, scores)]
        condition = decode(trigger, scores, keep)
        for i, r in enumerate(c):
            flags = dict(A_current=a[i], A_hold=bool(a[i] or (i and a[i-1])),
                         original_current=trigger[i],
                         original_hold=bool(trigger[i] or (i and trigger[i-1])),
                         conditional=condition[i])
            output[r['id']] = dict(id=r['id'], flags=flags,
                                  current_unknown={arm: r['unknown'] for arm in ARMS})
    return [output[r['id']] for r in rows]


def load_rows(role):
    rows = read(SOURCE / (role + '-rows.json'))
    saved = read(SOURCE / (role + '-predictions.json'))
    assert [r['id'] for r in rows] == [p['id'] for p in saved]
    assert len(rows) == 576 and len(clips(rows)) == 24
    groups = {r['base_group_id'] for r in rows}
    assert len(groups) == 8
    assert all(g.endswith(('g00', 'g01') if role == 'selection' else ('g02', 'g03')) for g in groups)
    return [dict(r, score=p['scores']['original']) for r, p in zip(rows, saved)]


def selection_record(rows, labels, keep, on):
    ps = predict_rows(rows, keep, on)
    extra = [r['id'] for r, p, y in zip(rows, ps, labels)
             if not y['truth'] and p['flags']['conditional'] and not p['flags']['original_current']]
    gain = {part: [r['id'] for r, p, y in zip(rows, ps, labels)
                  if (r['layout_relation'] == 'BOUNDARY') == (part == 'Boundary')
                  and y['truth'] and p['flags']['conditional'] and not p['flags']['original_current']]
            for part in ('Core', 'Boundary')}
    return dict(tau_keep=keep, feasible=not extra, added_FP_ids=extra,
                gains=gain, boundary_gain=len(gain['Boundary']), core_gain=len(gain['Core']))


def predict():
    assert not OUT.exists(), 'Do not overwrite any completed or partial run'
    for folder in (SOURCE, CURRENT):
        check_seal(folder, 'prediction-seal.json')
    check_seal(SOURCE, 'role-seal.json')
    old = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((old / n).exists() for n in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'))
    inputs = [PROTOCOL, Path(__file__), HERE / 'full_event_metrics_20260920.py',
              CURRENT / 'selection.json', CURRENT / 'prediction-seal.json',
              SOURCE / 'protocol.json', SOURCE / 'role-seal.json', SOURCE / 'prediction-seal.json',
              TRANSFER / 'source-admission.json']
    for role in ROLES:
        inputs += [SOURCE / (role + '-rows.json'), SOURCE / (role + '-predictions.json'),
                   SOURCE / 'labels' / (role + '.json'), CURRENT / (role + '-predictions.json')]
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', dict(id='ba-conditional-hold-20260921',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), scope='CONSUMED_DEVELOPMENT',
        backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs}))
    (OUT / 'protocol-before-run.md').write_bytes(PROTOCOL.read_bytes())
    on = read(CURRENT / 'selection.json')['original']['threshold']
    assert on == 15.769264221191408
    rows = load_rows('selection')
    labels = read(SOURCE / 'labels/selection.json')
    assert [r['id'] for r in rows] == [y['id'] for y in labels]
    candidates = sorted({r['score'] for r in rows if r['score'] < on} | {on})
    curve = [selection_record(rows, labels, k, on) for k in candidates]
    chosen = max((r for r in curve if r['feasible']),
                 key=lambda r: (r['boundary_gain'], r['core_gain'], r['tau_keep']))
    write(OUT / 'selection-curve.json', curve)
    write(OUT / 'selection.json', dict(tau_on=on, selected=chosen, candidates=len(curve),
        selection_rule='zero added negative flags; maximize Boundary gain, Core gain, tau_keep'))
    outputs = ['selection.json', 'selection-curve.json']
    for role in ROLES:
        records = predict_rows(load_rows(role), chosen['tau_keep'], on)
        previous = read(CURRENT / (role + '-predictions.json'))
        assert [r['id'] for r in records] == [p['id'] for p in previous]
        assert all(r['flags'][a] == p['flags'][a] for r, p in zip(records, previous) for a in ARMS[:-1])
        name = role + '-predictions.json'
        write(OUT / name, records)
        outputs.append(name)
    write(OUT / 'prediction-receipt.json', dict(evaluation_label_values_read=False,
        evaluation_label_file_hashed=True, native_values_read=False, source_scores_reused=True,
        training=False, model_inference=False, entry_threshold_changed=False,
        threshold_selected_on='selection only', recursive_continuation=True))
    seal('prediction-seal.json', outputs + ['prediction-receipt.json'])
    verify_inputs()
    print(json.dumps(read(OUT / 'selection.json')))


def opportunity(rows):
    result = []
    for c in clips(rows):
        positive = [i for i, r in enumerate(c) if r['truth']]
        if not positive:
            continue
        seeds = [i for i, r in enumerate(c) if r['flags']['original_current']]
        first = seeds[0] if seeds else len(c)
        detected = [i for i in positive if c[i]['flags']['original_current']]
        added = [i for i in positive if c[i]['flags']['conditional'] and not c[i]['flags']['original_current']]
        kinds = Counter('previously_missed' if not detected else
                        'initial' if i < detected[0] else 'terminal' if i > detected[-1] else 'internal'
                        for i in added)
        result.append(dict(clip_id=c[0]['clip_id'], group=c[0]['base_group_id'], relation=c[0]['layout_relation'],
            positive_count=len(positive), first_trigger_id=c[first]['id'] if seeds else None,
            first_trigger_truth=c[first]['truth'] if seeds else None,
            positives_before_any_trigger=[c[i]['id'] for i in positive if i < first],
            reachable_true_suffix=[c[i]['id'] for i in positive if i >= first],
            no_release_additional_TP_ceiling=[c[i]['id'] for i in positive if i >= first and not c[i]['flags']['original_current']],
            current_detected=bool(detected), added_TP_ids=[c[i]['id'] for i in added],
            added_TP_kinds=dict(kinds)))
    return result


def evaluate():
    started = time.perf_counter()
    verify_inputs()
    check_seal(OUT, 'prediction-seal.json')
    write(OUT / 'evaluation-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(OUT / 'prediction-seal.json')))
    native = {r['id']: r['returned_target_corridor_samples'] for r in read(TRANSFER / 'source-admission.json')['frames']}
    summary, gates, outputs = {}, {}, []
    for role in ROLES:
        metadata = load_rows(role)
        predictions = read(OUT / (role + '-predictions.json'))
        labels = read(SOURCE / 'labels' / (role + '.json'))
        assert [r['id'] for r in metadata] == [p['id'] for p in predictions] == [y['id'] for y in labels]
        rows = [dict(r, truth=y['truth'], flags=p['flags'], current_unknown=p['current_unknown'],
                     native_target_corridor_samples=native[r['source_id']])
                for r, p, y in zip(metadata, predictions, labels)]
        report = metrics.evaluate(rows)
        opportunities = opportunity(rows)
        changes = {}
        for kind, predicate in (
                ('added_TP', lambda r: r['truth'] and r['flags']['conditional'] and not r['flags']['original_current']),
                ('added_FP', lambda r: not r['truth'] and r['flags']['conditional'] and not r['flags']['original_current']),
                ('lost_current', lambda r: r['flags']['original_current'] and not r['flags']['conditional']),
                ('held_A_TP_missing', lambda r: r['truth'] and r['flags']['A_hold'] and not r['flags']['conditional'])):
            changes[kind] = [dict(id=r['id'], group=r['base_group_id'], relation=r['layout_relation'],
                                 time_s=r['time_s'], score=r['score'], native=r['native_target_corridor_samples'])
                             for r in rows if predicate(r)]
        bg = [r for r in changes['added_TP'] if r['relation'] == 'BOUNDARY']
        groups = sorted({r['group'] for r in bg})
        gates[role] = dict(all_current_flags_retained=not changes['lost_current'],
            no_added_negative_frames=not changes['added_FP'], boundary_gain=len(bg), boundary_gaining_groups=groups,
            useful=bool(not changes['lost_current'] and not changes['added_FP'] and len(bg) >= 3 and len(groups) >= 2))
        summary[role] = {part: {arm: dict(counts=[v['frames'][k] for k in ('TP', 'FP', 'FN')],
            precision=v['frames']['precision'], FPR=v['frames']['FPR'],
            events=[v['detected_events'], v['event_count']], false_segments=v['false_alert_segment_count'],
            false_sampled_s=v['false_alert_sampled_s'], unknown=v['current_unknown'],
            internal_silent_frames=sum(e['internal_silent_frames'] for e in v['events']),
            terminal_silent_frames=sum(e['terminal_silent_frames'] for e in v['events']),
            exit_carryover_frames=sum(e['postexit_carryover_alert_frames'] or 0 for e in v['events']))
            for arm, v in report[part]['arms'].items()} for part in ('Core', 'Boundary')}
        for suffix, data in (('frame-results', rows), ('metrics', report), ('changes', changes), ('opportunity', opportunities)):
            name = role + '-' + suffix + '.json'
            write(OUT / name, data)
            outputs.append(name)
    write(OUT / 'result.json', dict(status='COMPLETE', scope='CONSUMED_DEVELOPMENT',
        selection=read(OUT / 'selection.json'), metrics=summary, gates=gates,
        candidate_supported=gates['evaluation']['useful'], original_test_activated=False,
        automatic_successor=False, elapsed_s=time.perf_counter()-started))
    seal('evaluation-seal.json', outputs + ['result.json'])
    verify_inputs()
    print(json.dumps(read(OUT / 'result.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('predict', 'evaluate'))
    globals()[parser.parse_args().stage]()
