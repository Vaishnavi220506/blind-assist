"""Independent saved-DEV complement recount; no model calls or test labels."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from audit_spatial_bce import digest, read, replay

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts.local/work/ba-spatial-complement-diagnostic-20260921'
NAMES = ('A', 'OR_zero', 'OR_high', 'OR_high_calibration', 'OR_group_score', 'OR_group_calibration')


def compare(report, recounted, name):
    for stratum in ('Core', 'Boundary'):
        for suffix in ('current', 'hold'):
            for local, primary in (('A', 'A'), ('B', name)):
                ours = recounted[stratum][local+'_'+suffix]
                theirs = report[stratum]['arms'][primary+'_'+suffix]
                for key in ('TP', 'FP', 'FN', 'TN'):
                    assert ours[key] == theirs['frames'][key], (name, stratum, suffix, key)
                assert ours['segments'] == theirs['false_alert_segment_count']
                assert ours['unknown'] == theirs['current_unknown']
                assert ours['events'] == theirs['event_count'] and ours['detected_events'] == theirs['detected_events']
                assert math.isclose(ours['FP']*.2, theirs['false_alert_sampled_s'], abs_tol=1e-9)
                for event in theirs['events']:
                    value = ours['first_delays'][event['clip_id']]
                    actual = event['first_in_event_alert_delay_s']
                    assert (actual is None) if value is None else math.isclose(value, actual, abs_tol=1e-8)


def run(out):
    out = out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    assert not (out/'independent-audit.json').exists(), 'Write-once audit exists'
    protocol = read(out/'protocol.json')
    source = Path(protocol['source']).resolve()
    assert source.is_relative_to((ROOT/'artifacts.local').resolve())
    assert digest(out/'protocol-before-run.md') == protocol['protocol_sha256']
    for name, expected in protocol['input_hashes'].items():
        assert digest(source/name) == expected, name
    for name, expected in protocol['code_hashes'].items():
        assert digest(HERE/name) == expected, name
    for name in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json', 'test-frame-results.json'):
        assert not (source/name).exists(), name
    assert read(source/'operating-point.json')['status'] == 'DEV_NO_ADMISSIBLE_OPERATING_POINT'
    assert digest(out/'public-fixed-predictions.json') == read(out/'public-fixed-seal.json')['sha256']
    for name, expected in read(out/'prediction-seal.json')['hashes'].items():
        assert digest(out/name) == expected
    for name, expected in read(out/'evaluation-seal.json').items():
        assert digest(out/name) == expected
    meta = [r for r in read(source/'evaluator/metadata.json') if r['split'] == 'dev']
    labels = read(source/'evaluator/dev-labels.json')
    public = read(out/'public-fixed-predictions.json')
    truth = [r['truth'] for r in labels]
    assert len(meta) == len(labels) == len(public) == 576
    assert [r['index'] for r in labels] == [r['index'] for r in meta] == [r['index'] for r in public]
    group_names = sorted({r['base_group_id'] for r in meta})
    assert len(group_names) == 8
    for group in group_names:
        rows = [r for r in meta if r['base_group_id'] == group]
        assert len(rows) == 72 and len({r['clip_id'] for r in rows}) == 3
        assert {r['layout_relation'] for r in rows} == {'INSIDE', 'BOUNDARY', 'OUTSIDE'}
    saved_logits = np.load(source/'dev-logits.npy', allow_pickle=False)
    logits = [float(r['logit']) for r in public]
    assert logits == [float(x) for x in saved_logits] and all(math.isfinite(x) for x in logits)
    a = [r['A_current'] for r in public]
    unknown = [r['unknown'] for r in public]
    eligible = [r['calibration_eligible'] for r in public]
    baseline = read(source/'baseline.json')
    from tof_corridor_calibration import score_frame, decide
    with np.load(source/'observations.npz', allow_pickle=False) as packet:
        for i, r in enumerate(meta):
            assert r['id'] == public[i]['id']
            prior = baseline[r['index']]
            scored = score_frame(packet['boxes'], packet['ranges'][r['index']])
            strong = decide(scored, protocol['geometric_T'])
            permissive = decide(scored, protocol['calibration_T0'])
            assert a[i] == prior['current'] == strong['alert']
            assert unknown[i] == prior['unknown'] == strong['unknown']
            assert eligible[i] == permissive['alert']
    high = protocol['fixed_high_logit']
    assert high == 7.6612162590026855
    raw = dict(A=a, OR_zero=[x or l >= 0 for x, l in zip(a, logits)],
        OR_high=[x or l >= high for x, l in zip(a, logits)],
        OR_high_calibration=[x or (e and l >= high) for x, e, l in zip(a, eligible, logits)])
    for i, row in enumerate(public):
        assert row['fixed_current_flags'] == {name: flags[i] for name, flags in raw.items()}
    recorded_folds = read(out/'fold-calibration.json')
    fold_receipts = {}
    for name, screen in (('OR_group_score', [True]*576), ('OR_group_calibration', eligible)):
        output, expected_records = list(a), []
        for group in group_names:
            held_positions = [i for i, r in enumerate(meta) if r['base_group_id'] == group]
            other = [i for i, r in enumerate(meta) if r['base_group_id'] != group]
            negatives = [i for i in other if not a[i] and not truth[i] and screen[i]]
            maximum = max((logits[i] for i in negatives), default=None)
            cutoff = max(0., math.nextafter(maximum, math.inf)) if maximum is not None else 0.
            assert not any(screen[i] and logits[i] >= cutoff for i in negatives)
            for i in held_positions:
                output[i] = a[i] or (screen[i] and logits[i] >= cutoff)
            expected_records.append(dict(held_group=group,
                calibration_groups=sorted({meta[i]['base_group_id'] for i in other}), cutoff=cutoff,
                eligible_A_negative_calibration_negatives=len(negatives),
                max_calibration_negative_logit=maximum, calibration_extra_raw_FP=0, held_indices=held_positions))
        assert expected_records == recorded_folds[name], name
        raw[name] = output
        fold_receipts[name] = expected_records
    assert raw['OR_high'] == raw['OR_high_calibration']
    assert raw['OR_group_score'] == raw['OR_group_calibration']
    predictions = read(out/'predictions.json')
    frame_results = read(out/'frame-results.json')
    primary_metrics = read(out/'metrics.json')
    primary_groups = read(out/'group-metrics.json')
    incremental = read(out/'incremental-ids.json')
    admission = {r['id']: r for r in read(source/'source-admission.json')['frames'] if r['id'] in {m['id'] for m in meta}}
    counts, group_counts, additions = {}, {}, {}
    for name in NAMES:
        _, g, flags = replay(raw[name], .5, truth, meta, a, unknown)
        compare(primary_metrics, g, name)
        counts[name], additions[name] = {}, {}
        for stratum in ('Core', 'Boundary'):
            for suffix in ('current', 'hold'):
                aa, bb = g[stratum]['A_'+suffix], g[stratum]['B_'+suffix]
                counts[name][stratum+'_'+suffix] = dict(
                    A=[aa[k] for k in ('TP', 'FP', 'FN')], candidate=[bb[k] for k in ('TP', 'FP', 'FN')],
                    extra_TP=bb['TP']-aa['TP'], extra_FP=bb['FP']-aa['FP'],
                    A_segments=aa['segments'], candidate_segments=bb['segments'])
                for clip_id, delay in aa['first_delays'].items():
                    if delay is not None:
                        assert bb['first_delays'][clip_id] is not None and bb['first_delays'][clip_id] <= delay+1e-8
        for suffix in ('current', 'hold'):
            added = []
            for i, row in enumerate(meta):
                assert not flags['A_'+suffix][i] or flags['B_'+suffix][i]
                assert predictions[i]['id'] == row['id'] == frame_results[i]['id']
                assert frame_results[i]['truth'] == truth[i]
                assert predictions[i]['flags'][name+'_'+suffix] == flags['B_'+suffix][i]
                assert frame_results[i]['flags'][name+'_'+suffix] == flags['B_'+suffix][i]
                assert predictions[i]['current_unknown'][name+'_'+suffix] == unknown[i]
                if flags['B_'+suffix][i] and not flags['A_'+suffix][i]:
                    previous = i-1 if i and meta[i-1]['clip_id'] == row['clip_id'] else None
                    added.append(dict(id=row['id'], base_group_id=row['base_group_id'], clip_id=row['clip_id'],
                        frame_in_clip=row['frame_in_clip'], time_s=row['time_s'], phase=row['phase'],
                        layout_relation=row['layout_relation'], truth=truth[i], logit=logits[i],
                        current_proposal=bool(raw[name][i] and not a[i]),
                        previous_id=meta[previous]['id'] if previous is not None else None,
                        previous_truth=truth[previous] if previous is not None else None,
                        previous_candidate_raw=bool(raw[name][previous]) if previous is not None else False,
                        previous_A_raw=bool(a[previous]) if previous is not None else False,
                        native_target_support_samples=admission[row['id']]['returned_target_corridor_samples']))
            additions[name][suffix] = added
            if name != 'A':
                saved = incremental[name][suffix]
                rescued = [r for r in added if r['truth']]
                assert [r['id'] for r in rescued] == saved['rescued_ids']
                assert [r['id'] for r in added if not r['truth']] == saved['new_fp_ids']
                assert sum(r['native_target_support_samples'] > 0 for r in rescued) == saved['recovered_with_returned_target_corridor_support']
                assert sum(r['native_target_support_samples'] == 0 for r in rescued) == saved['recovered_without_returned_target_corridor_support']
        group_counts[name] = {}
        for group in group_names:
            idx = [i for i, r in enumerate(meta) if r['base_group_id'] == group]
            _, rec, _ = replay([raw[name][i] for i in idx], .5, [truth[i] for i in idx],
                              [meta[i] for i in idx], [a[i] for i in idx], [unknown[i] for i in idx])
            compare(primary_groups[group], rec, name)
            group_counts[name][group] = {stratum+'_'+suffix:
                {key: rec[stratum]['B_'+suffix][key]-rec[stratum]['A_'+suffix][key] for key in ('TP', 'FP', 'segments')}
                for stratum in ('Core', 'Boundary') for suffix in ('current', 'hold')}
    for name, expected in protocol['input_hashes'].items():
        assert digest(source/name) == expected, 'Source changed during audit: '+name
    report = dict(status='PASS', audit_utc=datetime.now(timezone.utc).isoformat(),
        audit_code_sha256=digest(__file__), independent_counter_sha256=digest(HERE/'audit_spatial_bce.py'),
        protocol_sha256=digest(out/'protocol.json'), scope='CONSUMED_8_GROUP_DEV_ONLY', frames=576, groups=8,
        all_source_hashes_unchanged=True, all_fixed_and_fold_outputs_reproduced=True,
        float64_strict_negative_max_cutoffs_verified=True, calibration_outputs_identical=True,
        all_A_flags_events_onsets_retained=True, test_unactivated=True, test_label_or_model_output_access=False,
        counts=counts, group_increments=group_counts, folds=fold_receipts,
        incremental_rows=additions,
        limitation='Consumed development threshold diagnostic; shared renderer; no fresh confirmation, new model inference or selected deployment arm.')
    with (out/'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('counts', 'group_increments', 'folds', 'incremental_rows')}))
    for name in ('OR_high', 'OR_group_score'):
        print(name, json.dumps(counts[name]))
        for suffix in ('current', 'hold'):
            print(name, suffix, 'NEW_FP', json.dumps([r for r in additions[name][suffix] if not r['truth']]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)
