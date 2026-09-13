"""Consumed-only falsifiers for frozen MZ111 temporal attribution.

Zero velocity changes only finite observed radar_velocity values; all frozen
RGB correspondence, current-return competition and age gates remain unchanged.
Simple hold is nominal OR a nominal alert within .5s, with no held-alert refresh.
These are fixed diagnostics, not a retune or change to candidate selection.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import mz111_temporal_geometry as temporal
from run_mz107_four_sensor import ROOT, readrows, sha, truth, write
from run_mz111_spatial_temporal import score


MAX_HOLD_S = .5


def zero_velocities(rows):
    result = copy.deepcopy(rows)
    for row in result:
        row['radar_velocity'] = [0. if v is not None and math.isfinite(v) else v
                                 for v in row['radar_velocity']]
    return result


def simple_hold(rows, nominal):
    if len(rows) != len(nominal):
        raise ValueError('Rows and nominal predictions must have equal length')
    result = []
    episode = None
    previous_time = None
    measured_positive_time = None
    for row, pred in zip(rows, nominal):
        assert row['id'] == pred['id']
        if row['episode_id'] != episode:
            episode = row['episode_id']
            previous_time = None
            measured_positive_time = None
        now = row['time_s']
        assert math.isfinite(now) and (previous_time is None or now > previous_time)
        previous_time = now
        if pred['candidate']:
            measured_positive_time = now
        age = None if measured_positive_time is None else now - measured_positive_time
        candidate = bool(pred['candidate'] or (age is not None and 0 <= age <= MAX_HOLD_S))
        result.append(dict(id=row['id'], candidate=candidate,
                           candidate_state='ALERT' if candidate else 'UNKNOWN',
                           nominal_positive_time_s=measured_positive_time,
                           original_nominal_age_s=age))
    return result


def check_hold_contract():
    rows = [dict(id=str(i), episode_id='a' if i < 4 else 'b', time_s=i*.25 if i < 4 else 0.)
            for i in range(5)]
    preds = [dict(id=str(i), candidate=i == 0) for i in range(5)]
    assert [p['candidate'] for p in simple_hold(rows, preds)] == [True, True, True, False, False]
    values = [dict(radar_velocity=[None, -1., 0., .4])]
    changed = zero_velocities(values)
    assert changed == [dict(radar_velocity=[None, 0., 0., 0.])]
    assert values == [dict(radar_velocity=[None, -1., 0., .4])]


def run(analysis, output):
    analysis = analysis.resolve()
    output = output.resolve()
    work = (ROOT / 'artifacts.local/work').resolve()
    allowed_analysis = work / 'mz111-spatial-temporal-20260913/analysis-v1'
    if analysis != allowed_analysis:
        raise ValueError('This diagnostic is restricted to consumed MZ111 analysis-v1')
    if not output.is_relative_to(work) or output.exists():
        raise ValueError('Fresh canonical artifact output required')
    check_hold_contract()
    input_seal = json.loads((analysis / 'prediction-seal.json').read_text())
    assert input_seal['status'] == 'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE'
    assert set(input_seal['panels']) == {'consumed_mz107', 'consumed_mz108', 'consumed_mz109'}
    for name, digest in input_seal['code_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest, name
        assert sha(analysis / name) == digest, name
    output.mkdir(parents=True)
    packets = {}
    predictions = {}
    seals = {}
    differences = {}
    for panel, seal in input_seal['panels'].items():
        capture = Path(seal['capture']).resolve()
        assert capture.is_relative_to(work)
        assert sha(capture / 'receipt.json') == seal['receipt_sha256']
        assert sha(capture / 'raw.jsonl') == seal['raw_sha256']
        old_predictions = analysis / panel / 'predictions.json'
        assert sha(old_predictions) == seal['predictions_sha256']
        original = json.loads(old_predictions.read_text())
        rows = readrows(capture / 'raw.jsonl')
        nominal = original['nominal']
        assert len(rows) == 96
        assert [r['id'] for r in rows] == [p['id'] for p in nominal]
        values = dict(baseline=original['baseline'], nominal=nominal,
                      temporal=original['temporal'],
                      zero_velocity=temporal.predict(zero_velocities(rows), nominal),
                      simple_hold=simple_hold(rows, nominal))
        assert all(len(p) == len(rows) for p in values.values())
        # Authenticate the true-velocity arm by reproducing the sealed outputs.
        assert temporal.predict(rows, nominal) == values['temporal']
        differences[panel] = [dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'],
                                  temporal=bool(t['candidate']), zero_velocity=bool(z['candidate']))
                              for r, t, z in zip(rows, values['temporal'], values['zero_velocity'])
                              if bool(t['candidate']) != bool(z['candidate'])]
        folder = output / panel
        folder.mkdir()
        write(folder / 'predictions.json', values)
        packets[panel] = rows
        predictions[panel] = values
        seals[panel] = dict(capture=str(capture), raw_sha256=seal['raw_sha256'],
                           source_predictions_sha256=seal['predictions_sha256'],
                           predictions_sha256=sha(folder / 'predictions.json'))
    source_paths = [Path(__file__)] + [Path(__file__).with_name(n) for n in input_seal['code_sha256']]
    source_paths += [Path(__file__).with_name(n) for n in
                     ('run_mz107_four_sensor.py', 'mz107_rgb_association.py')]
    for path in source_paths:
        shutil.copyfile(path, output / path.name)
    write(output / 'prediction-differences.json', differences)
    write(output / 'prediction-seal.json', dict(
        status='ALL_CONTROLS_SEALED_BEFORE_EVALUATOR_PARSE',
        scope='CONSUMED_DEVELOPMENT_ATTRIBUTION_FALSIFIER_NO_RETUNE_OR_CANDIDATE_SELECTION_CHANGE',
        source_analysis=str(analysis), input_seal_sha256=sha(analysis / 'prediction-seal.json'),
        panels=seals, code_sha256={p.name: sha(p) for p in output.glob('*.py')},
        prediction_differences_sha256=sha(output / 'prediction-differences.json'),
        controls=dict(zero_velocity='Replace finite observed radar_velocity with zero; otherwise frozen temporal.predict',
                      simple_hold='Nominal OR original nominal positive age<=0.5s; no held refresh; episode reset'),
        parameters_fixed_before_evaluator=True))

    # First evaluator parsing happens only after every panel and control is sealed.
    results = {}
    all_rows = []
    all_evaluations = []
    all_values = {}
    scored_differences = {}
    evaluator_hashes = {}
    for panel, rows in packets.items():
        capture = Path(seals[panel]['capture'])
        receipt = json.loads((capture / 'receipt.json').read_text())
        evaluator_path = capture / 'evaluator.jsonl'
        evaluator_hashes[panel] = sha(evaluator_path)
        assert evaluator_hashes[panel] == receipt['hashes']['evaluator.jsonl']
        evaluations = readrows(evaluator_path)
        assert [r['id'] for r in rows] == [e['id'] for e in evaluations]
        results[panel] = score(rows, evaluations, predictions[panel])
        truths = {e['id']: bool(truth(e)) for e in evaluations}
        scored_differences[panel] = [dict(d, native_truth=truths[d['id']]) for d in differences[panel]]
        all_rows.extend(dict(r, episode_id=panel + '/' + r['episode_id']) for r in rows)
        all_evaluations.extend(evaluations)
        for arm, values in predictions[panel].items():
            all_values.setdefault(arm, []).extend(values)
    assert len(all_rows) == 288
    overall = score(all_rows, all_evaluations, all_values)
    result = dict(status='FIXED_TEMPORAL_CONTROLS_COMPLETE', scope='CONSUMED_DEVELOPMENT_FALSIFIER',
                  frames=288, panels=results, overall=overall,
                  temporal_vs_zero_velocity_candidate_differences=scored_differences,
                  temporal_vs_zero_velocity_candidate_difference_count=sum(map(len, differences.values())),
                  evaluator_sha256=evaluator_hashes,
                  interpretation_boundary='If all true/zero-velocity candidates are equal, this panel does not attribute alert gains to velocity; synthetic capability tests are separate evidence.')
    write(output / 'summary.json', result)
    write(output / 'completion.json', dict(status='PASS', resources_started=[],
          output_hashes={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))}))
    print(json.dumps(dict(overall={arm: value['metrics'] for arm, value in overall.items()},
                         temporal_vs_zero_velocity_differences=scored_differences), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.analysis, args.output)
