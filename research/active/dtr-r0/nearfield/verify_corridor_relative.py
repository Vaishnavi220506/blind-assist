"""Saved-output audit of one corridor-relative Development pilot; never fits.

Readout, all dev candidate costs, transfer gates and evaluator joins are rebuilt
here. Complete-event metrics reuse the prior independent audit implementation,
not the production runner or production metric module. No protected-test labels,
images or predictions are opened. Only the independent receipt may be written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT = ROOT / 'artifacts.local/work/ba-corridor-relative-20260921'
ORIGINAL = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
ARMS = ('A_current', 'B_current', 'R_current', 'A_hold', 'B_hold', 'R_hold')
OLD_THRESHOLD = 7.6612162590026855
METRIC_AUDIT = HERE / 'verify_core_projection_stress.py'
spec = importlib.util.spec_from_file_location('independent_projection_audit', METRIC_AUDIT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ARMS = ARMS
read, digest, require, compare, check_hash = audit.read, audit.digest, audit.require, audit.compare, audit.check_hash


def seal(directory, name):
    value = read(directory / name)
    check_hash(directory / 'protocol.json', value['protocol_sha256'])
    for path, expected in value['hashes'].items():
        check_hash(directory / path, expected)
    audit.COUNTS['seals'] += 1


def causal_hold(current, identities):
    """Previous genuine current alert only; no held-alert recursion."""
    result = []
    previous = None
    for flag, row in zip(current, identities):
        continuation = bool(previous is not None and previous[0] == row['clip_id']
                            and abs(row['time_s'] - previous[1] - .2) < 1e-8 and previous[2])
        result.append(bool(flag or continuation))
        previous = row['clip_id'], row['time_s'], bool(flag)
    return result


def flags_at(scores, threshold, a, identities):
    # Python binary64 comparisons preserve the disabled nextafter cutoff even
    # when the original stored tensor has float32 dtype.
    r = [bool(flag or float(score) >= float(threshold)) for flag, score in zip(a, scores)]
    return {'A_current': list(a), 'R_current': r,
            'A_hold': causal_hold(a, identities), 'R_hold': causal_hold(r, identities)}


def cost_table(truth, identities, flags):
    result = {}
    for region in ('Core', 'Boundary'):
        selected = [(row['layout_relation'] == 'BOUNDARY') == (region == 'Boundary') for row in identities]
        negative_count = sum(keep and not value for keep, value in zip(selected, truth))
        clip_count = len({row['clip_id'] for row, keep in zip(identities, selected) if keep})
        for suffix, fraction in (('current', .01), ('hold', .02)):
            fp_counts, segment_counts = {}, {}
            for name in ('A', 'R'):
                active = [keep and not value and flag for keep, value, flag in
                          zip(selected, truth, flags[name + '_' + suffix])]
                fp_counts[name] = sum(active)
                segment_counts[name] = sum(flag and (i == 0 or not active[i - 1]
                    or identities[i]['clip_id'] != identities[i - 1]['clip_id']) for i, flag in enumerate(active))
            extra_fp = fp_counts['R'] - fp_counts['A']
            extra_segments = segment_counts['R'] - segment_counts['A']
            fp_cap, segment_cap = math.floor(fraction * negative_count), math.floor(.125 * clip_count)
            result[region + '_' + suffix] = {
                'negative_frames': negative_count, 'clips': clip_count, 'added_FP': extra_fp,
                'FP_cap': fp_cap, 'added_segments': extra_segments, 'segment_cap': segment_cap,
                'pass_cost': extra_fp <= fp_cap and extra_segments <= segment_cap}
    return result


def rebuilt_metrics(rows):
    """Use the independent prior audit on arbitrary complete-layout subsets."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['clip_id']].append(row)
    clips = []
    for name in sorted(groups):
        clip = sorted(groups[name], key=lambda row: row['frame_in_clip'])
        require(len(clip) == 24, name + ': 24 complete samples')
        for i, row in enumerate(clip):
            require(row['frame_in_clip'] == i and abs(row['time_s'] - i * .2) < 1e-8, name + ': timeline')
            require(all(row[k] == clip[0][k] for k in ('layout_relation', 'base_group_id', 'layer', 'background')),
                    name + ': constant clip strata')
        clips.append(clip)
    longest = {arm: {} for arm in ARMS}
    group = lambda selected: audit.aggregate(selected, longest)
    report = {'dt_s': .2, 'duration_convention': 'sample count * dt_s; not wall-clock latency',
              'Core_definition': 'Complete INSIDE and OUTSIDE clips',
              'overall': group(clips),
              'Core': group([c for c in clips if c[0]['layout_relation'] != 'BOUNDARY']),
              'Boundary': group([c for c in clips if c[0]['layout_relation'] == 'BOUNDARY']),
              'subgroups': {}}
    for key in ('layout_relation', 'layer', 'background'):
        report['subgroups'][key] = {value: group([c for c in clips if c[0][key] == value])
                                  for value in sorted({c[0][key] for c in clips})}
    return report, longest


def compact(report):
    return {region: {arm: {
        **{key: value['frames'][key] for key in ('TP', 'FP', 'FN', 'precision', 'recall', 'FPR', 'current_unknown')},
        'detected_events': value['detected_events'], 'event_count': value['event_count'],
        'false_segments': value['false_alert_segment_count'], 'false_sampled_s': value['false_alert_sampled_s'],
        'max_detected_delay_s': max((e['first_in_event_alert_delay_s'] for e in value['events'] if e['detected']), default=None),
        'min_event_coverage': min((e['positive_coverage'] for e in value['events']), default=None),
        'preentry_FP': sum(e['preentry_false_alert_frames'] for e in value['events']),
        'postexit_FP': sum(e['postexit_false_alert_frames'] for e in value['events']),
    } for arm, value in report[region]['arms'].items()} for region in ('Core', 'Boundary')}


def verify(out):
    protocol = read(out / 'protocol.json')
    require(protocol['id'] == 'ba-corridor-relative-20260921', 'Run identity')
    require(protocol['recipe'] == {'seed': 20260921, 'steps': 1200, 'batch_size': 64,
                                   'learning_rate': .001, 'weight_decay': .0001}, 'One frozen training recipe')
    require(protocol['protected_test_activated'] is False and protocol['automatic_successor'] is False,
            'Protocol scope flags')
    check_hash(out / 'protocol-before-run.md', protocol['protocol_text_sha256'])
    check_hash(HERE / 'CORRIDOR_RELATIVE_PROTOCOL_20260921.md', protocol['protocol_text_sha256'])
    for entries in (protocol['code_hashes'], protocol['source_hashes']):
        for path, expected in entries.items():
            check_hash(ROOT / path, expected)
    for name in ('features-seal.json', 'fit-seal.json', 'selection-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
        seal(out, name)
    # Authenticate only the historical parent outputs used by this experiment.
    # No protected-test image, label or prediction is read by the verifier.
    for source, output_seal, output_name in ((ORIGINAL, 'fit-seal.json', 'dev-logits.npy'),
                                             (TRANSFER, 'prediction-seal.json', 'predictions.json')):
        observation, outputs = read(source / 'observation-seal.json'), read(source / output_seal)
        require(observation['protocol_sha256'] == outputs['protocol_sha256'] == digest(source / 'protocol.json'),
                'Parent protocol binding')
        check_hash(source / output_name, outputs['hashes'][output_name])
        for path, expected in protocol['source_hashes'].items():
            relative = Path(path)
            try:
                name = relative.relative_to(source.relative_to(ROOT)).as_posix()
            except ValueError:
                continue
            if name in observation['hashes']:
                require(expected == observation['hashes'][name], 'Original observation binding: ' + name)
    for marker in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'):
        require(not (ORIGINAL / marker).exists(), 'Protected test remains unactivated: ' + marker)

    original_meta = read(ORIGINAL / 'evaluator/metadata.json')
    transfer_meta = read(TRANSFER / 'evaluator/metadata.json')
    originals = read(ORIGINAL / 'baseline.json')
    source_base = read(TRANSFER / 'baseline.json')
    splits, group_ids = {}, {}
    for name, metadata, size, group_count in (('train', original_meta, 1728, 24),
                                             ('dev', original_meta, 576, 8),
                                             ('transfer', transfer_meta, 1152, 16)):
        indices = read(out / (name + '-indices.json'))
        expected = [row['index'] for row in metadata if row['split'] == name]
        require(indices == expected and len(indices) == size, name + ': selected complete indices')
        subset = [metadata[i] for i in indices]
        counts = Counter(row['base_group_id'] for row in subset)
        require(len(counts) == group_count and set(counts.values()) == {72}, name + ': group counts')
        require(len({row['clip_id'] for row in subset}) == group_count * 3, name + ': complete lateral clips')
        group_ids[name] = set(counts)
        splits[name] = indices, subset
        features = np.load(out / (name + '-features.npy'), allow_pickle=False, mmap_mode='r')
        require(features.shape == (size, 64, 60) and features.dtype == np.float32
                and bool(np.isfinite(features).all()), name + ': 60 finite features')
        valid = features[:, :, 59]
        require(bool(np.isin(valid, [0., 1.]).all()), name + ': explicit original validity')
        require(bool((features[valid == 0] == 0).all()), name + ': invalid zone fields stay zero')
        base = source_base if name == 'transfer' else originals
        require(np.array_equal(valid.sum(1).astype(int), [base[i]['valid_zones'] for i in indices]),
                name + ': original valid-zone count')
        # Area fractions are the missing-band masks; occupied bands partition a zone.
        fractions = features[:, :, [13, 27, 41]]
        require(bool(((fractions >= 0) & (fractions <= 1)).all()), name + ': band fractions')
        require(bool(np.allclose(fractions.sum(2), valid, rtol=0, atol=2e-7)), name + ': disjoint exhaustive bands')
        for index, offset in enumerate((0, 14, 28)):
            require(bool((features[:, :, offset:offset + 14][fractions[:, :, index] == 0] == 0).all()),
                    name + ': absent-band fields masked')
        require(bool(np.allclose(features[:, :, 42:45], features[:, :, :3] - features[:, :, 28:31], atol=1e-7, rtol=0)),
                name + ': inner/exterior contrast')
        require(bool(np.allclose(features[:, :, 45:48], features[:, :, 14:17] - features[:, :, 28:31], atol=1e-7, rtol=0)),
                name + ': uncertain/exterior contrast')
        audit.COUNTS['feature_frames_checked'] += size
    for first, second in (('train', 'dev'), ('train', 'transfer'), ('dev', 'transfer')):
        require(group_ids[first].isdisjoint(group_ids[second]), 'Group-disjoint roles')
    features_receipt = read(out / 'features-receipt.json')
    require(features_receipt['protected_test_images_read'] == 0 and features_receipt['native_depth_read'] is False
            and features_receipt['labels_read'] is False and features_receipt['features'] == 60, 'Feature access declaration')
    fit = read(out / 'fit-receipt.json')
    require(fit['parameters'] == 3041 and fit['train_rows'] == 1728, 'Shared MLP size and train rows')
    train_labels = read(ORIGINAL / 'evaluator/train-labels.json')
    require([row['index'] for row in train_labels] == splits['train'][0], 'Train label role')
    logs = [json.loads(line) for line in (out / 'training.jsonl').read_text(encoding='utf-8').splitlines() if line]
    require([row['step'] for row in logs] == [1] + list(range(100, 1201, 100)), 'Fixed final-step training receipt')

    dev_indices, dev_meta = splits['dev']
    dev_labels = read(ORIGINAL / 'evaluator/dev-labels.json')
    require([row['index'] for row in dev_labels] == dev_indices, 'Dev-only selection labels')
    dev_truth = [row['truth'] for row in dev_labels]
    require(all(type(value) is bool for value in dev_truth), 'Admitted dev Boolean truth')
    a_dev = [originals[i]['current'] for i in dev_indices]
    dev_scores = np.load(out / 'dev-logits.npy', allow_pickle=False).astype(np.float64)
    require(dev_scores.shape == (576,) and bool(np.isfinite(dev_scores).all()), 'Finite dev scores')
    thresholds = sorted(set(float(v) for v in dev_scores) | {float(np.nextafter(dev_scores.max(), np.inf))})
    candidates = []
    for threshold in thresholds:
        cost = cost_table(dev_truth, dev_meta, flags_at(dev_scores, threshold, a_dev, dev_meta))
        candidates.append({'threshold': threshold, 'cost': cost,
                           'admissible': all(item['pass_cost'] for item in cost.values())})
    eligible = [row for row in candidates if row['admissible']]
    require(bool(eligible), 'Disabled addition is a feasible control')
    selected = eligible[0]
    compare(read(out / 'selection-candidates.json'), candidates, 'all_dev_threshold_candidates')
    compare(read(out / 'selection.json'), {
        'threshold': selected['threshold'], 'rule': 'Lowest dev score satisfying fixed added-cost caps',
        'cost': selected['cost'], 'thresholds': len(thresholds)}, 'dev_only_selection')
    audit.COUNTS['dev_threshold_candidates'] = len(candidates)

    pred = read(out / 'predictions.json')
    scores = np.load(out / 'transfer-logits.npy', allow_pickle=False).astype(np.float64)
    identities = read(TRANSFER / 'identities.json')
    old = read(TRANSFER / 'predictions.json')
    labels = read(TRANSFER / 'evaluator/transfer-labels.json')
    native = read(TRANSFER / 'source-admission.json')['frames']
    require(len(pred) == len(identities) == len(old) == len(labels) == len(native) == len(transfer_meta) == 1152,
            'Full transfer denominators')
    require(scores.shape == (1152,) and bool(np.isfinite(scores).all()), 'Transfer scores')
    a = [row['current'] for row in source_base]
    flags = flags_at(scores, selected['threshold'], a, identities)
    b = [bool(value or float(row['logit']) >= OLD_THRESHOLD) for value, row in zip(a, old)]
    flags.update(B_current=b, B_hold=causal_hold(b, identities))
    rows = []
    for i, (p, identity, m, y, n, previous, base) in enumerate(zip(pred, identities, transfer_meta, labels, native, old, source_base)):
        require(p['index'] == identity['index'] == m['index'] == y['index'] == previous['index'] == base['index'] == i,
                'Transfer index join')
        require(p['id'] == identity['id'] == m['id'] == n['id'] == previous['id'], 'Transfer identity join')
        require(all(p[key] == identity[key] == m[key] for key in ('clip_id', 'frame_in_clip', 'time_s')), 'Transfer timeline join')
        require(type(y['truth']) is bool and type(n['returned_target_corridor_samples']) is int,
                'Evaluator-only source fields')
        require(p['logit'] == float(scores[i]), 'Sealed scalar output')
        expected_flags = {arm: flags[arm][i] for arm in ARMS}
        compare(p['flags'], expected_flags, 'public_readout/' + p['id'])
        compare(p['current_unknown'], {arm: base['unknown'] for arm in ARMS}, 'UNKNOWN/' + p['id'])
        require(expected_flags['A_current'] == previous['flags']['A_current']
                and expected_flags['A_hold'] == previous['flags']['A_hold'], 'Frozen A retained')
        require(expected_flags['B_current'] == previous['flags']['C_current']
                and expected_flags['B_hold'] == previous['flags']['C_hold'], 'Frozen old B operating point')
        require(all(not expected_flags['A_' + suffix] or expected_flags['R_' + suffix] for suffix in ('current', 'hold')),
                'OR preserves every incumbent flag, including negatives')
        rows.append({**m, 'truth': y['truth'], 'flags': expected_flags,
                     'current_unknown': {arm: base['unknown'] for arm in ARMS},
                     'native_target_corridor_samples': n['returned_target_corridor_samples']})
        audit.COUNTS['transfer_prediction_rows'] += 1
    compare(read(out / 'frame-results.json'), rows, 'independent_evaluator_join')
    metrics, longest = rebuilt_metrics(rows)
    compare(read(out / 'metrics.json'), metrics, 'all_transfer_metrics')
    groups = {group: rebuilt_metrics([row for row in rows if row['base_group_id'] == group])[0]
              for group in sorted(group_ids['transfer'])}
    compare(read(out / 'group-metrics.json'), groups, 'all_layout_metrics')
    cost = cost_table([row['truth'] for row in rows], transfer_meta, flags)
    gained = [group for group, report in groups.items()
              if report['Boundary']['arms']['R_hold']['frames']['TP'] > report['Boundary']['arms']['A_hold']['frames']['TP']]
    retention = all(not row['truth'] or all(not row['flags']['A_' + suffix] or row['flags']['R_' + suffix]
                                          for suffix in ('current', 'hold')) for row in rows)
    gates = {'cost': all(item['pass_cost'] for item in cost.values()), 'retain_A': retention,
             'boundary_recall': metrics['Boundary']['arms']['R_hold']['frames']['recall'] >= .5,
             'broad_gain': len(gained) >= 8}
    gained_rows = [row for row in rows if row['truth'] and row['flags']['R_current'] and not row['flags']['A_current']]
    terminal = {'status': 'PASS' if all(gates.values()) else 'NO_GO', 'gates': gates, 'cost': cost,
                'boundary_gain_groups': gained, 'new_current_TP': len(gained_rows),
                'new_current_TP_without_native_corridor_contributor': sum(row['native_target_corridor_samples'] == 0 for row in gained_rows),
                'protected_test_activated': False, 'automatic_successor': False, 'scope': 'CONSUMED_SIMULATION_DEVELOPMENT'}
    compare(read(out / 'result.json'), terminal, 'entire_terminal')
    inheritance = read(out / 'local-inheritance.json')
    require(inheritance['inheritance_role'] == ('COMPONENT_OR_CHALLENGER' if all(gates.values()) else 'NEGATIVE_CONTROL'),
            'Disposition follows frozen gates')
    require(inheritance['inheritance_mode'] == ('CHALLENGER' if all(gates.values()) else None), 'Disposition mode')
    events = {region: {arm: [{**event, 'max_silent_samples': longest[arm][event['clip_id']]}
                             for event in metrics[region]['arms'][arm]['events']] for arm in ARMS}
              for region in ('Core', 'Boundary')}
    return {'status': 'PASS', 'experiment_status': terminal['status'],
            'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'verifier_sha256': digest(Path(__file__)),
            'independent_metric_implementation': str(METRIC_AUDIT), 'independent_metric_sha256': digest(METRIC_AUDIT),
            'python_executable': sys.executable, 'backend': 'TASK_NOT_GPU_SUITABLE',
            'counts': dict(audit.COUNTS), 'checked_hashes': audit.HASHES,
            'selected_dev_threshold': selected['threshold'], 'dev_cost': selected['cost'],
            'summary': compact(metrics), 'groups': {key: compact(value) for key, value in groups.items()},
            'events': events, 'terminal': terminal,
            'limits': ['Saved model logits and RGB statistics are authenticated, not re-inferred or retrained.',
                       'Feature dimensions, validity, empty-band masks, partitions and contrasts are independently checked; per-pixel statistics are not re-extracted.',
                       'Metric logic is reused from the prior independent audit, not from the production scorer/runner.',
                       'This new bounded added-cost contract does not change any historical negative gate.',
                       'A OR R cannot remove incumbent A false alerts; native-unsupported true classifications are not measured ranges.',
                       'Consumed same-simulator Development only; no protected-test access, hardware claim or automatic successor.']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--stdout-only', action='store_true')
    args = parser.parse_args()
    report = verify(args.out)
    if not args.stdout_only:
        with (args.out / 'independent-verification.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': report['status'], 'experiment_status': report['experiment_status'],
                      'counts': report['counts'], 'threshold': report['selected_dev_threshold'],
                      'terminal': report['terminal'], 'summary': report['summary']}))


if __name__ == '__main__':
    main()
