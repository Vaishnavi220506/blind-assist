"""Development-only joint threshold selection; never a test-set rescue.

CLI authenticates and reconstructs the consumed MZ120 development curve from
saved logits/labels. It performs no training or model inference and does not
open the MZ119 source, predictions or curve. Original MZ120 remains immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def recall(m):
    total = m['TP']+m['FN']
    return m['TP']/total if total else None


def precision(m):
    total = m['TP']+m['FP']
    return m['TP']/total if total else None


def joint_checks(row, baseline):
    r = recall(row['metrics']); br = recall(baseline['metrics'])
    head = recall(row['spatial']['HEAD']); rod = recall(row['spatial']['rod_true_cells'])
    delay = row['events']['max_detected_delay_s']; bd = baseline['events']['max_detected_delay_s']
    return dict(
        frame_recall=r is not None and br is not None and r >= br-.02,
        fp_reduction=baseline['metrics']['FP'] > 0 and row['metrics']['FP'] <= .7*baseline['metrics']['FP'],
        duration_reduction=baseline['events']['false_alert_bin_duration_s'] > 0 and
            row['events']['false_alert_bin_duration_s'] <= .7*baseline['events']['false_alert_bin_duration_s'],
        event_retention=bool(baseline['event_hits']) and all(
            not hit or row['event_hits'].get(ep, False) for ep, hit in baseline['event_hits'].items()),
        delay=delay is not None and bd is not None and delay <= bd+.25,
        head_recall=head is not None and head >= .9,
        rod_recall=rod is not None and rod >= .9)


def select_joint(curve, baseline):
    feasible = [r for r in curve if all(joint_checks(r, baseline).values())]
    selected = min(feasible, key=lambda r: (r['metrics']['FP'], -r['threshold'])) if feasible else None
    return selected, feasible


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(source, output):
    import numpy as np
    from run_mz120_occupancy import ROOT, SOURCES, readrows, read_evaluation, labels, evaluate

    started = time.perf_counter(); source = source.resolve(); output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    completion = json.loads((source/'completion.json').read_text())
    assert digest(source/'summary.json') == completion['summary_sha256']
    assert digest(source/'model.pt') == completion['model_sha256']
    # Summary contains old, already-known transfer results; only development baseline is used.
    summary = json.loads((source/'summary.json').read_text()); baseline = summary['development']['baseline']
    frozen = json.loads((source/'freeze.json').read_text())
    assert digest(source/'split.json') == frozen['split_sha256']
    split = json.loads((source/'split.json').read_text()); rows = []; es = []; source_hashes = {}
    for name in ('mz115', 'mz117'):
        capture = ROOT/'artifacts.local/work'/SOURCES[name]/'capture-v1'
        assert digest(capture/'raw.jsonl') == frozen['source'][name]['raw']
        assert digest(capture/'receipt.json') == frozen['source'][name]['receipt']
        receipt = json.loads((capture/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
        for key in ('raw.jsonl', 'evaluator.jsonl'):
            assert digest(capture/key) == receipt['hashes'][key], key
        assert digest(capture/'spec.json') == receipt['spec_sha256']
        rows.extend(readrows(capture/'raw.jsonl')); es.extend(read_evaluation(capture))
        source_hashes[name] = {key:digest(capture/key) for key in ('raw.jsonl','evaluator.jsonl','spec.json','receipt.json')}
    ids = [r['id'] for r in rows]
    assert ids == [e['id'] for e in es] and len(set(ids)) == len(ids)
    ids_to_index = {value:i for i,value in enumerate(ids)}
    tr = np.array([ids_to_index[value] for value in split['train']])
    dv = np.array([ids_to_index[value] for value in split['dev']])
    assert set(tr).isdisjoint(dv) and set(tr)|set(dv) == set(range(len(rows)))
    assert {rows[i]['episode_id'] for i in tr}.isdisjoint({rows[i]['episode_id'] for i in dv})
    cache = np.load(source/'development_predictions.npz')
    probabilities = cache['probabilities']; target = np.stack([labels(e) for e in es])
    assert np.array_equal(target, cache['labels']) and probabilities.shape == target.shape == (480,45)
    devrows = [rows[i] for i in dv]; deves = [es[i] for i in dv]
    oldcurve = json.loads((source/'development_curve.json').read_text()); reconstructed = []
    # Baseline vector is only used for gained/lost bookkeeping, not joint gates.
    # Authenticate original sealed MZ116 vectors separately rather than invent it.
    baselines = []
    for name in ('mz115','mz117'):
        if name == 'mz115':
            path = ROOT/'artifacts.local/work/mz116-merged-information-20260913/guard-v1/consumed_mz115/predictions.json'; arm='guard'
        else:
            path = ROOT/'artifacts.local/work'/SOURCES[name]/'analysis-v1/predictions.json'; arm='resolution_guard'
        assert digest(path) == frozen['source'][name]['baseline']
        baselines.extend(p['candidate'] for p in json.loads(path.read_text())['3.6'][arm])
    baselines = np.array(baselines,bool)
    for row in oldcurve:
        result = dict(threshold=row['threshold'], **evaluate(devrows,deves,target[dv],probabilities[dv],row['threshold'],baselines[dv]))
        assert result == row, row['threshold']
        reconstructed.append(result)
    assert len(reconstructed) == 101
    selected, feasible = select_joint(reconstructed, baseline)
    original = next(r for r in reconstructed if r['threshold'] == summary['threshold'])
    output.mkdir(parents=True)
    validation = dict(reconstructed_curve_points=101,reconstructed_label_cells=int(target.size),
        train_frames=len(tr),development_frames=len(dv),episode_overlap=0,
        scope='POSTHOC_CONSUMED_DEVELOPMENT_ONLY',training_runs=0,model_inference_runs=0,
        transfer_curve_opened=False,source_hashes=source_hashes)
    write(output/'validation.json',validation)
    annotated = [dict(r,joint_checks=joint_checks(r,baseline),spatial_precision=precision(r['spatial']['metrics'])) for r in reconstructed]
    write(output/'joint_curve.json',annotated)
    write(output/'readout-candidate.json',dict(
        authority='DEVELOPMENT_READOUT_CHALLENGER_NOT_BASELINE_PROMOTION',
        model_sha256=completion['model_sha256'],threshold=None if selected is None else selected['threshold'],
        selection='All seven original retention requirements before FP minimization; higher threshold breaks ties',
        original_threshold=summary['threshold'],original_terminal_unchanged=True))
    training = {}
    for tag,point in [('original',original),('joint',selected)]:
        if point is not None:
            training[tag] = evaluate([rows[i] for i in tr],[es[i] for i in tr],target[tr],probabilities[tr],point['threshold'],baselines[tr])
    result = dict(status='JOINT_DEVELOPMENT_FEASIBLE' if selected else 'NO_JOINT_DEVELOPMENT_POINT',
        baseline=baseline,original=original,selected=selected,
        feasible_thresholds=[r['threshold'] for r in feasible],
        spatial_nonincrease_feasible=[r['threshold'] for r in feasible
            if r['spatial']['metrics']['FP'] <= original['spatial']['metrics']['FP']],
        spatial_nonincrease_note='Additional diagnostic, not a retroactively declared acceptance criterion',
        per_constraint_thresholds={key:[r['threshold'] for r in reconstructed if joint_checks(r,baseline)[key]]
                                  for key in joint_checks(original,baseline)},
        training_diagnostic_at_dev_thresholds=training,
        backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',workload='JSON and saved scalar/occupancy scoring'),
        elapsed_seconds=time.perf_counter()-started)
    write(output/'summary.json',result)
    tracked_inputs = ['summary.json','completion.json','freeze.json','split.json','development_predictions.npz','development_curve.json']
    write(output/'receipt.json',dict(status='PASS',inputs={n:digest(source/n) for n in tracked_inputs},
        script_sha256=digest(Path(__file__)),outputs={p.name:digest(p) for p in output.glob('*.json')},resources_started=[]))
    print(json.dumps(dict(status=result['status'],feasible_points=len(feasible),
        thresholds=[r['threshold'] for r in feasible], selected_threshold=selected['threshold'] if selected else None,
        spatial_nonincrease_feasible=result['spatial_nonincrease_feasible']),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
