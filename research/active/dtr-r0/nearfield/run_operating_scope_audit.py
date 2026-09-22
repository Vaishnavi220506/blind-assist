"""Fixed public-input replay and retrospective operating-scope audit."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np

from run_core_projection_stress import (
    HERE, ROOT, T, T0, DT, OFFSETS, read, sha, write, shifted_boxes,
    readout, metric_module, checks_for,
)
from tof_corridor_calibration import score_frame, decide

NAME = 'ba-operating-scope-20260922'
OUT = ROOT / 'artifacts.local/work' / NAME
SOURCES = {
    'original': ('ba-spatial-complement-transfer-20260921', 'transfer-labels.json'),
    'broader': ('ba-data-coverage-20260921', 'evaluation-labels.json'),
}
BRIEF = HERE / 'OPERATING_SCOPE_PROTOCOL_20260922.md'
CODE = ('run_operating_scope_audit.py', 'run_core_projection_stress.py',
        'tof_corridor_calibration.py', 'ba_camera_corridor.py',
        'tof_fov45_core.py', 'full_event_metrics_20260920.py')
LIMITS = dict(core_events=16, onset_delay_s=.2, coverage=5/6,
              max_silent_samples=1, fp_reduction_fraction=.5)


def freeze():
    assert not OUT.exists(), 'New output only; preserve existing runs'
    sources = {}
    for cohort, (name, labels) in SOURCES.items():
        source = ROOT / 'artifacts.local/work' / name
        selected = ('observations.npz', 'identities.json', 'baseline.json',
                    'evaluator/metadata.json', 'evaluator/' + labels)
        for seal_name, files in [('observation-seal.json', selected),
                                 ('prediction-seal.json', ('predictions.json',))]:
            receipt = read(source / seal_name)
            assert receipt['protocol_sha256'] == sha(source / 'protocol.json')
            for file in files:
                assert sha(source / file) == receipt['hashes'][file], (cohort, file)
        inputs = (*selected, 'predictions.json', 'spec.json', 'protocol.json',
                  'observation-seal.json', 'prediction-seal.json', 'metrics.json')
        sources[cohort] = dict(path=source.relative_to(ROOT).as_posix(), labels=labels,
                              hashes={f: sha(source / f) for f in inputs})
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', dict(id=NAME, time_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=sources, code_hashes={n: sha(HERE/n) for n in CODE},
        brief_sha256=sha(BRIEF), thresholds=dict(strong=T, calibration=T0),
        offsets=OFFSETS, limits=LIMITS, scope='CONSUMED_SIMULATION_RETROSPECTIVE_SCOPE_AUDIT',
        backend='CPU_SCALAR_GEOMETRY', placement='TASK_NOT_GPU_SUITABLE',
        original_test_activated=False, capture_count=0, trained_models=0))
    (OUT / 'protocol-before-run.md').write_bytes(BRIEF.read_bytes())


def verify():
    p = read(OUT/'protocol.json')
    assert sha(BRIEF) == p['brief_sha256'] == sha(OUT/'protocol-before-run.md')
    for name, digest in p['code_hashes'].items():
        assert sha(HERE/name) == digest, name
    for source in p['sources'].values():
        for name, digest in source['hashes'].items():
            assert sha(ROOT/source['path']/name) == digest, name
    return p


def replay():
    p = verify()
    records, parity = [], {}
    start = time.perf_counter()
    for cohort, info in p['sources'].items():
        source = ROOT/info['path']
        ids = {r['index']: r for r in read(source/'identities.json')}
        baseline = {r['index']: r for r in read(source/'baseline.json')}
        selected = read(source/'predictions.json')
        assert len(selected) == 1152 and len({r['index'] for r in selected}) == 1152
        with np.load(source/'observations.npz', allow_pickle=False) as data:
            ranges, boxes = data['ranges'].copy(), data['boxes'].copy()
        states = {}
        for old in selected:
            i = old['index']
            identity, b = ids[i], baseline[i]
            assert old['id'] == identity['id'] and old['clip_id'] == identity['clip_id']
            row = {k: identity[k] for k in ('id', 'index', 'clip_id', 'frame_in_clip', 'time_s')}
            row.update(cohort=cohort, flags={}, current_unknown={}, support={})
            for condition, offset in OFFSETS.items():
                scored = score_frame(shifted_boxes(boxes, offset), ranges[i])
                key = (condition, identity['clip_id'])
                flags, unknown, states[key] = readout(scored, states.get(key), identity['time_s'])
                row['flags'].update({condition+'_'+policy: bool(flag) for policy, flag in flags.items()})
                row['current_unknown'].update({condition+'_'+policy: bool(unknown) for policy in flags})
                base = scored['baseline']
                row['support'][condition] = dict(score=scored['score'],
                    valid_zones=base['valid_zones'], possible_zones=base['possible_zones'],
                    definite_zones=base['definite_zones'], state=decide(scored, T)['state'])
                if condition == 'nominal':
                    assert scored['score'] == b['score']
                    assert flags['strong'] == b['current'] == old['flags']['A_current']
                    assert flags['hold'] == old['flags']['A_hold']
                    assert unknown == b['unknown'] == old['current_unknown']['A_hold']
                    assert base['valid_zones'] == b['valid_zones']
                    assert base['definite_zones'] == b['definite_zones']
            records.append(row)
        parity[cohort] = len(selected)
        print('REPLAY', cohort, len(selected), flush=True)
    write(OUT/'predictions.json', records)
    write(OUT/'prediction-receipt.json', dict(nominal_parity=parity,
        frames=len(records), condition_frames=3*len(records), elapsed_s=time.perf_counter()-start,
        evaluator_parsed_in_replay=False, rgb_model_calls=0, native_depth_reads=0))
    write(OUT/'prediction-seal.json', dict(protocol_sha256=sha(OUT/'protocol.json'),
        time_utc=datetime.now(timezone.utc).isoformat(),
        hashes={n: sha(OUT/n) for n in ('predictions.json', 'prediction-receipt.json')}))


def phase_map(rows):
    clips = defaultdict(list)
    for r in rows:
        clips[r['clip_id']].append(r)
    mapping = {}
    for clip in clips.values():
        positive = [r['frame_in_clip'] for r in clip if r['truth']]
        for r in clip:
            mapping[r['id']] = ('OUTSIDE' if not positive else 'positive' if r['truth']
                else 'preentry' if r['frame_in_clip'] < min(positive) else 'postexit')
    return mapping


def summarize(rows, report, arm):
    condition = arm.split('_')[0]
    f = report['frames']
    events = report['events']
    phases = phase_map(rows)
    unknown_reasons, states, fp_phases, fn_reasons = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        a, u, s = r['flags'][arm], r['current_unknown'][arm], r['support'][condition]
        reason = ('NO_VALID_RETURN' if not s['valid_zones'] else
                  'NO_POSSIBLE_CORRIDOR_SUPPORT' if not s['possible_zones'] else
                  'AMBIGUOUS_CORRIDOR_SUPPORT' if not s['definite_zones'] else 'DEFINITE_SUPPORT')
        state = ('UNKNOWN_ALERT' if a else 'UNKNOWN_SILENT') if u else ('DEFINITE_ALERT' if a else 'DEFINITE_SILENT')
        states[state] += 1
        if u:
            unknown_reasons[reason] += 1
        if a and not r['truth']:
            fp_phases[phases[r['id']]] += 1
        if not a and r['truth']:
            fn_reasons[reason] += 1
    return dict(**f, detected_events=report['detected_events'], events=report['event_count'],
        false_segments=report['false_alert_segment_count'], false_sampled_s=report['false_alert_sampled_s'],
        max_detected_delay_s=max((e['first_in_event_alert_delay_s'] for e in events if e['detected']), default=None),
        min_event_coverage=min((e['positive_coverage'] for e in events), default=None),
        missed_event_ids=[e['clip_id'] for e in events if not e['detected']],
        imperfect_events=[e for e in events if e['positive_coverage'] < 1],
        fp_phases=dict(fp_phases), fn_readout_reasons=dict(fn_reasons),
        uncertainty_states=dict(states), unknown_reasons=dict(unknown_reasons))


def evaluate():
    p = verify()
    seal = read(OUT/'prediction-seal.json')
    assert seal['protocol_sha256'] == sha(OUT/'protocol.json')
    for name, digest in seal['hashes'].items():
        assert sha(OUT/name) == digest
    predictions = read(OUT/'predictions.json')
    metrics, summary, checks, joined = {}, {}, {}, []
    module = metric_module()
    for cohort, info in p['sources'].items():
        source = ROOT/info['path']
        meta = {r['index']: r for r in read(source/'evaluator/metadata.json')}
        labels = {r['index']: r for r in read(source/'evaluator'/info['labels'])}
        rows = []
        for pred in (r for r in predictions if r['cohort'] == cohort):
            m, label = meta[pred['index']], labels[pred['index']]
            assert all(pred[k] == m[k] for k in ('id', 'index', 'clip_id', 'frame_in_clip', 'time_s'))
            rows.append({**m, **pred, 'truth': label['truth']})
        assert len(rows) == 1152 and len({r['clip_id'] for r in rows}) == 48
        assert len({r['base_group_id'] for r in rows}) == 16
        assert all(sum(r['clip_id'] == c for r in rows) == 24 for c in {r['clip_id'] for r in rows})
        report = module.evaluate(rows, DT)
        # Compare independent replay against saved published A metrics.
        old_metrics = read(source/'metrics.json')
        for stratum in ('Core', 'Boundary'):
            assert report[stratum]['arms']['nominal_hold']['frames'] == old_metrics[stratum]['arms']['A_hold']['frames']
        metrics[cohort] = report
        summary[cohort] = {}
        for stratum in ('Core', 'Boundary'):
            subset = [r for r in rows if (r['layout_relation'] != 'BOUNDARY') == (stratum == 'Core')]
            summary[cohort][stratum] = {condition: summarize(subset, report[stratum]['arms'][condition+'_hold'], condition+'_hold')
                                      for condition in OFFSETS}
        summary[cohort]['families'] = {}
        for family in sorted({r['type_id'] for r in rows}):
            subset = [r for r in rows if r['type_id'] == family]
            family_report = module.evaluate(subset, DT)
            summary[cohort]['families'][family] = {
                stratum: {c: summarize([r for r in subset if (r['layout_relation'] != 'BOUNDARY') == (stratum == 'Core')],
                         family_report[stratum]['arms'][c+'_hold'], c+'_hold') for c in OFFSETS}
                for stratum in ('Core', 'Boundary')}
        checks[cohort] = {c: checks_for(rows, report, c, LIMITS) for c in OFFSETS}
        joined.extend(rows)
    write(OUT/'frame-results.json', joined)
    write(OUT/'metrics.json', metrics)
    write(OUT/'summary.json', summary)
    write(OUT/'checks.json', checks)
    errors = []
    for cohort in SOURCES:
        rows = [r for r in joined if r['cohort'] == cohort]
        phases = phase_map(rows)
        for r in rows:
            for c in OFFSETS:
                alert = r['flags'][c+'_hold']
                if alert != r['truth']:
                    errors.append(dict(cohort=cohort, condition=c, id=r['id'], clip_id=r['clip_id'],
                        type_id=r['type_id'], stratum='Boundary' if r['layout_relation'] == 'BOUNDARY' else 'Core',
                        time_s=r['time_s'], error='FP' if alert else 'FN', phase=phases[r['id']],
                        unknown=r['current_unknown'][c+'_hold'], **r['support'][c]))
    with (OUT/'errors.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(errors[0]))
        writer.writeheader(); writer.writerows(errors)
    write(OUT/'evaluation-seal.json', dict(protocol_sha256=sha(OUT/'protocol.json'),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'),
        hashes={n: sha(OUT/n) for n in ('frame-results.json', 'metrics.json', 'summary.json', 'checks.json', 'errors.csv')}))
    print('EVALUATED', {g: {c: all(v.values()) for c,v in checks[g].items()} for g in checks}, flush=True)


if __name__ == '__main__':
    freeze()
    replay()
    evaluate()
