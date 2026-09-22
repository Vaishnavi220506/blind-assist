"""Independent saved-output audit of the matched ordered-spatial U/G pilot.

Reconstructs cache ordering, original-interval geometry, dev-only selections,
readout, evaluator joins, complete-event metrics and the frozen terminal gates.
No production runner/model/metric helper is imported, no encoder or fitted head
is executed, and no training or protected-test evaluation is performed.
Only the requested audit receipt is written, exclusively by default.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT = ROOT / 'artifacts.local/work/ba-spatial-structure-20260921'
ORIGINAL = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
PRIOR = ROOT / 'artifacts.local/work/ba-corridor-relative-20260921'
ARMS = tuple(name + '_' + suffix for suffix in ('current', 'hold') for name in ('A', 'B', 'R', 'U', 'G'))
RECIPE = {'seed': 20260921, 'steps': 1200, 'batch_size': 64, 'learning_rate': .001, 'weight_decay': .0001}
METRIC_AUDIT = HERE / 'verify_corridor_relative.py'
spec = importlib.util.spec_from_file_location('independent_corridor_audit', METRIC_AUDIT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.ARMS = ARMS
audit = helper.audit
audit.ARMS = ARMS
read, digest, require, compare, check_hash = audit.read, audit.digest, audit.require, audit.compare, audit.check_hash


def source_bindings(protocol):
    receipts = {}
    for source, output_seal in ((ORIGINAL, 'fit-seal.json'), (TRANSFER, 'prediction-seal.json')):
        observation, outputs = read(source / 'observation-seal.json'), read(source / output_seal)
        require(observation['protocol_sha256'] == outputs['protocol_sha256'] == digest(source / 'protocol.json'),
                'Parent protocol binding')
        for relative, expected in protocol['source_hashes'].items():
            try:
                name = Path(relative).relative_to(source.relative_to(ROOT)).as_posix()
            except ValueError:
                continue
            for parent in (observation, outputs):
                if name in parent['hashes']:
                    require(expected == parent['hashes'][name], 'Parent observation/output binding: ' + name)
        receipt = read(source / 'features/feature_receipt.json')
        check_hash(source / 'features/rgb_features.npy', receipt['cache_sha256'])
        receipts[source] = receipt
    require(receipts[ORIGINAL]['recipe'] == receipts[TRANSFER]['recipe']
            and receipts[ORIGINAL]['checkpoint_sha256'] == receipts[TRANSFER]['checkpoint_sha256'],
            'One frozen RGB encoder and cache recipe')
    # Only the inherited outputs needed here; never open any test artifacts.
    old_seal = read(TRANSFER / 'prediction-seal.json')
    check_hash(TRANSFER / 'predictions.json', old_seal['hashes']['predictions.json'])
    for name in ('prediction-seal.json', 'selection-seal.json', 'evaluation-seal.json'):
        sealed = read(PRIOR / name)
        require(sealed['protocol_sha256'] == digest(PRIOR / 'protocol.json'), 'R parent protocol')
        for needed in ('predictions.json', 'selection.json', 'result.json'):
            if needed in sealed['hashes']:
                check_hash(PRIOR / needed, sealed['hashes'][needed])
    for marker in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'):
        require(not (ORIGINAL / marker).exists(), 'Protected test remains unactivated: ' + marker)
    return receipts


def numeric_channels(ranges, boxes):
    """Independent binary64 geometric equations at every ordered sample ray."""
    values = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(values) & (values >= .1) & (values < 8)
    result = np.empty((len(values), 12, 24, 24), dtype=np.float64)
    focal = 320 / math.tan(math.radians(50))
    for zone in range(64):
        zr, zc = divmod(zone, 8)
        y0, x0, y1, x1 = [float(v) for v in boxes[zone]]
        r = np.where(valid[:, zone], values[:, zone], 0.)
        radius = .1 + 3 * (.01 + .02 * r)
        ends = (np.maximum(.1, r - radius), r + radius)
        for sr in range(3):
            for sc in range(3):
                row, col = 3 * zr + sr, 3 * zc + sc
                x = (x0 + (x1 - x0) * ((sc + .5) / 3)) * 640 / 256
                y = (y0 + (y1 - y0) * ((sr + .5) / 3)) * 360 / 192
                ax, ay = (x - 320) / focal, (y - 180) / focal
                result[:, :4, row, col] = np.stack((r / 8, valid[:, zone],
                    np.full(len(r), math.atan(ax) / math.radians(45)),
                    np.full(len(r), math.atan(ay) / math.radians(45))), axis=1)
                distances = np.stack([value for z in ends for value in
                    (ax * z + .3, .3 - ax * z, ay * z + .2, .9 - ay * z)], axis=1)
                result[:, 4:, row, col] = np.where(valid[:, zone, None], np.clip(distances, -4, 4) / 4, 0)
    return result, valid


def input_audit(out, metadata, receipts):
    splits, group_ids, numeric_errors = {}, {}, {}
    for name, source, size, n_groups in (('train', ORIGINAL, 1728, 24), ('dev', ORIGINAL, 576, 8),
                                        ('transfer', TRANSFER, 1152, 16)):
        meta = metadata[source]
        indices = read(out / (name + '-indices.json'))
        require(indices == [row['index'] for row in meta if row['split'] == name] and len(indices) == size,
                name + ': only admitted indices')
        subset = [meta[i] for i in indices]
        counts = Counter(row['base_group_id'] for row in subset)
        require(len(counts) == n_groups and set(counts.values()) == {72}, name + ': complete layout groups')
        require(len({row['clip_id'] for row in subset}) == 3 * n_groups, name + ': complete three-relation clips')
        group_ids[name] = set(counts)
        splits[name] = indices, subset
        inputs = np.load(out / (name + '-inputs.npy'), allow_pickle=False, mmap_mode='r')
        cache = np.load(source / 'features/rgb_features.npy', allow_pickle=False, mmap_mode='r')
        require(inputs.shape == (size, 36, 24, 24) and inputs.dtype == np.float32, name + ': input shape/dtype')
        require(cache.shape[1:] == (216, 8, 8) and cache.dtype == np.float32, name + ': original cache shape/dtype')
        with np.load(source / 'observations.npz', allow_pickle=False) as observations:
            ranges, boxes = observations['ranges'][indices], observations['boxes']
        require(ranges.shape == (size, 64) and boxes.shape == (64, 4), name + ': original sensor support')
        require(np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all(), name + ': finite nonempty boxes')
        normalized = (boxes.astype(np.float64) / [192, 256, 192, 256]).astype(np.float32)
        require(np.array_equal(normalized, np.asarray(receipts[source]['boxes'], dtype=np.float32)),
                name + ': RGB cache and raw sensor use the same boxes')
        baseline = read(source / 'baseline.json')
        maximum_error = 0.
        for start in range(0, size, 64):
            stop = min(start + 64, size)
            stored = np.asarray(inputs[start:stop])
            require(bool(np.isfinite(stored).all()), name + ': finite selected inputs')
            # Direct interleaved slice assignment, not the production reshape/transpose.
            original = np.asarray(cache[indices[start:stop]])
            ordered = np.empty((stop - start, 24, 24, 24), dtype=np.float32)
            for sr in range(3):
                for sc in range(3):
                    ordered[:, :, sr::3, sc::3] = original[:, sr * 3 + sc::9, :, :]
            require(np.array_equal(stored[:, :24], ordered), name + ': every RGB feature retains value and position')
            expected, valid = numeric_channels(ranges[start:stop], boxes)
            maximum_error = max(maximum_error, float(np.max(np.abs(stored[:, 24:] - expected))))
            require(np.allclose(stored[:, 24:], expected, atol=5e-7, rtol=0),
                    name + ': full original intervals and signed corridor distances')
            require(np.array_equal(stored[:, 24], expected[:, 0].astype(np.float32)), name + ': unmodified normalized range')
            require(np.array_equal(stored[:, 25], expected[:, 1]), name + ': original valid mask')
            invalid = stored[:, 25] == 0
            require(bool((stored[:, 24][invalid] == 0).all())
                    and bool((stored[:, 28:].transpose(0, 2, 3, 1)[invalid] == 0).all()),
                    name + ': invalid metric inputs are zero; RGB/ray channels retained')
            require(np.array_equal(valid.sum(axis=1), [baseline[i]['valid_zones'] for i in indices[start:stop]]),
                    name + ': original valid-zone counts')
            audit.COUNTS['ordered_RGB_values_checked'] += int(ordered.size)
            audit.COUNTS['geometry_values_checked'] += int(expected[:, 4:].size)
            audit.COUNTS['input_frames_checked'] += stop - start
        numeric_errors[name] = maximum_error
    for left, right in (('train', 'dev'), ('train', 'transfer'), ('dev', 'transfer')):
        require(group_ids[left].isdisjoint(group_ids[right]), 'Group-disjoint roles')
    prepare = read(out / 'prepare-receipt.json')
    require(prepare['labels_read'] is False and prepare['protected_test_rows_indexed'] == 0
            and prepare['native_depth_read'] is False and prepare['inverse_order_exact'] is True,
            'Preparation access declaration')
    return splits, group_ids, numeric_errors


def training_audit(out, train_indices):
    """Recreate only untrained CPU initialization; no forward/backward/optimizer."""
    import torch
    from torch import nn

    train_labels = read(ORIGINAL / 'evaluator/train-labels.json')
    require([row['index'] for row in train_labels] == train_indices, 'Train labels match admitted indices')
    require(all(type(row['truth']) is bool for row in train_labels), 'Train Boolean labels')
    batches = np.load(out / 'batch-indices.npy', allow_pickle=False)
    require(np.array_equal(batches, np.random.default_rng(RECIPE['seed']).integers(0, 1728, (1200, 64))),
            'Common batch schedule independently reproduced from frozen seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RECIPE['seed'])
        initial = nn.Sequential(nn.Conv2d(36, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(1152, 32), nn.ReLU(), nn.Linear(32, 1))
    state = {'net.' + key: value for key, value in initial.state_dict().items()}
    hashed = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        hashed.update(key.encode())
        hashed.update(tensor.numpy().tobytes())
    initial_hash = hashed.hexdigest()
    parameters = sum(t.numel() for t in state.values())
    require(parameters == 65825, 'Fixed matched architecture parameter count')
    fit, backend = read(out / 'fit-receipt.json'), read(out / 'fit-backend.json')
    compare(fit['backend'], backend, 'common_training_backend')
    require(set(fit['arms']) == {'U', 'G'}, 'Exactly two fit receipts')
    common = ('initial_state_sha256', 'batch_indices_sha256', 'train_labels_sha256', 'parameters', 'device', 'steps')
    require(all(fit['arms']['U'][key] == fit['arms']['G'][key] for key in common), 'Matched training conditions')
    for arm in ('U', 'G'):
        receipt = fit['arms'][arm]
        require(receipt['initial_state_sha256'] == initial_hash, arm + ': independently reproduced initial state')
        require(receipt['steps'] == 1200 and receipt['parameters'] == parameters, arm + ': fixed budget and architecture')
        require(receipt['device'].split(':')[0] == backend['selected_device_type'], arm + ': common selected device')
        check_hash(out / 'batch-indices.npy', receipt['batch_indices_sha256'])
        check_hash(ORIGINAL / 'evaluator/train-labels.json', receipt['train_labels_sha256'])
        logs = [json.loads(line) for line in (out / (arm + '-training.jsonl')).read_text(encoding='utf-8').splitlines() if line]
        require([r['step'] for r in logs] == [1] + list(range(200, 1201, 200)), arm + ': frozen final-step receipt')
        require(all(math.isfinite(r['bce']) and r['bce'] >= 0 for r in logs), arm + ': finite loss receipts')
        checkpoint = torch.load(out / (arm + '-head_last.pt'), weights_only=True, map_location='cpu')
        require(checkpoint['recipe'] == RECIPE and checkpoint['arm'] == arm, arm + ': checkpoint identity')
        require(checkpoint['state_dict'].keys() == state.keys(), arm + ': checkpoint architecture keys')
        for key, tensor in checkpoint['state_dict'].items():
            require(tensor.shape == state[key].shape and tensor.dtype == torch.float32
                    and torch.isfinite(tensor).all().item(), arm + ': finite checkpoint parameter ' + key)
        for split, size in (('train', 1728), ('dev', 576), ('transfer', 1152)):
            scores = np.load(out / (arm + '-' + split + '-logits.npy'), allow_pickle=False)
            require(scores.shape == (size,) and bool(np.isfinite(scores).all()), arm + ': finite ' + split + ' scores')
    return {'initial_state_sha256': initial_hash, 'batch_indices_sha256': digest(out / 'batch-indices.npy'),
            'parameters_per_arm': parameters, 'steps_per_arm': 1200, 'training_device': fit['arms']['U']['device'],
            'initialization_reconstruction_device': 'cpu', 'initialization_framework': str(torch.__version__)}


def selection_audit(out, splits):
    indices, metadata = splits['dev']
    labels, baseline = read(ORIGINAL / 'evaluator/dev-labels.json'), read(ORIGINAL / 'baseline.json')
    require([row['index'] for row in labels] == indices and all(type(row['truth']) is bool for row in labels),
            'Only admitted dev labels used for selection')
    truth, a = [row['truth'] for row in labels], [baseline[i]['current'] for i in indices]
    selection = {}
    for arm in ('U', 'G'):
        scores = np.load(out / (arm + '-dev-logits.npy'), allow_pickle=False).astype(np.float64)
        thresholds = sorted(set(float(value) for value in scores) | {float(np.nextafter(scores.max(), np.inf))})
        candidates = []
        for cutoff in thresholds:
            cost = helper.cost_table(truth, metadata, helper.flags_at(scores, cutoff, a, metadata))
            candidates.append({'threshold': cutoff, 'cost': cost, 'admissible': all(v['pass_cost'] for v in cost.values())})
        acceptable = [row for row in candidates if row['admissible']]
        require(bool(acceptable), arm + ': disabled supplement must remain feasible')
        chosen = acceptable[0]
        selection[arm] = {'threshold': chosen['threshold'], 'cost': chosen['cost'], 'candidates': len(candidates)}
        compare(read(out / (arm + '-selection-candidates.json')), candidates, arm + '/all_dev_candidates')
        audit.COUNTS['dev_threshold_candidates'] += len(candidates)
    compare(read(out / 'selection.json'), selection, 'independent_dev_selection')
    return selection


def readout_audit(out, selection, metadata):
    predictions, old, prior = read(out / 'predictions.json'), read(TRANSFER / 'predictions.json'), read(PRIOR / 'predictions.json')
    identities, baseline = read(TRANSFER / 'identities.json'), read(TRANSFER / 'baseline.json')
    labels, native = read(TRANSFER / 'evaluator/transfer-labels.json'), read(TRANSFER / 'source-admission.json')['frames']
    require(all(len(value) == 1152 for value in (predictions, old, prior, identities, baseline, labels, native, metadata)),
            'Complete transfer denominators')
    a = [row['current'] for row in baseline]
    flags = {'A_current': a, 'A_hold': helper.causal_hold(a, identities)}
    b = [bool(flag or float(row['logit']) >= helper.OLD_THRESHOLD) for flag, row in zip(a, old)]
    flags.update(B_current=b, B_hold=helper.causal_hold(b, identities))
    r_cut = read(PRIOR / 'selection.json')['threshold']
    r = [bool(flag or float(row['logit']) >= r_cut) for flag, row in zip(a, prior)]
    flags.update(R_current=r, R_hold=helper.causal_hold(r, identities))
    scores = {arm: np.load(out / (arm + '-transfer-logits.npy'), allow_pickle=False).astype(np.float64) for arm in ('U', 'G')}
    for arm in ('U', 'G'):
        current = [bool(flag or float(score) >= selection[arm]['threshold']) for flag, score in zip(a, scores[arm])]
        flags[arm + '_current'], flags[arm + '_hold'] = current, helper.causal_hold(current, identities)
    rows = []
    for i, (pred, identity, meta, label, native_row, base, old_row, prior_row) in enumerate(zip(
            predictions, identities, metadata, labels, native, baseline, old, prior)):
        require(all(row['index'] == i for row in (pred, identity, meta, label, base, old_row, prior_row)), 'Transfer index join')
        require(all(row['id'] == identity['id'] for row in (pred, meta, native_row, old_row, prior_row)), 'Transfer identity join')
        require(all(pred[key] == identity[key] == meta[key] == prior_row[key]
                    for key in ('clip_id', 'frame_in_clip', 'time_s')), 'Transfer time join')
        require(type(label['truth']) is bool and type(base['unknown']) is bool, 'Boolean truth and public UNKNOWN')
        require(type(native_row['returned_target_corridor_samples']) is int, 'Evaluator-only native contribution')
        compare(pred['logits'], {arm: float(scores[arm][i]) for arm in ('U', 'G')}, 'saved_scalar_scores/' + str(i))
        expected = {arm: bool(flags[arm][i]) for arm in ARMS}
        compare(pred['flags'], expected, 'all_decisions/' + str(i))
        unknown = {arm: base['unknown'] for arm in ARMS}
        compare(pred['current_unknown'], unknown, 'public_UNKNOWN/' + str(i))
        compare(prior_row['flags'], {arm: expected[arm] for arm in prior_row['flags']}, 'inherited_controls/' + str(i))
        require(expected['B_current'] == old_row['flags']['C_current']
                and expected['B_hold'] == old_row['flags']['C_hold'], 'Frozen B cutoff')
        require(all(not expected['A_' + suffix] or expected[arm + '_' + suffix]
                    for arm in ('U', 'G') for suffix in ('current', 'hold')), 'All incumbent alerts retained')
        rows.append({**meta, 'truth': label['truth'], 'flags': expected, 'current_unknown': unknown,
                     'native_target_corridor_samples': native_row['returned_target_corridor_samples']})
        audit.COUNTS['transfer_prediction_rows'] += 1
        audit.COUNTS['decisions_checked'] += len(ARMS)
    compare(read(out / 'frame-results.json'), rows, 'independent_evaluator_join')
    return rows, flags


def terminal_audit(out, rows, flags, metrics, groups, horizontal):
    findings = {}
    for arm in ('U', 'G'):
        selected = {prefix + '_' + suffix: flags[('A' if prefix == 'A' else arm) + '_' + suffix]
                    for prefix in ('A', 'R') for suffix in ('current', 'hold')}
        costs = helper.cost_table([row['truth'] for row in rows], rows, selected)
        gains = [group for group, report in groups.items() if report['Boundary']['arms'][arm + '_hold']['frames']['TP']
                 > report['Boundary']['arms']['A_hold']['frames']['TP']]
        h = horizontal['Boundary']['arms'][arm + '_hold']
        timely = sum(event['detected'] and event['first_in_event_alert_delay_s'] <= .4 + 1e-8 for event in h['events'])
        gates = {'cost': all(c['pass_cost'] for c in costs.values()),
                 'retain_A': all(not row['flags']['A_' + suffix] or row['flags'][arm + '_' + suffix]
                                 for row in rows for suffix in ('current', 'hold')),
                 'boundary_recall': metrics['Boundary']['arms'][arm + '_hold']['frames']['recall'] >= .5,
                 'broad_gain': len(gains) >= 8, 'horizontal_recall': h['frames']['recall'] >= .5,
                 'horizontal_events': h['detected_events'] >= 3,
                 'horizontal_timing': timely >= 3 and all(not event['detected']
                     or event['first_in_event_alert_delay_s'] <= .4 + 1e-8 for event in h['events'])}
        new = [row for row in rows if row['truth'] and row['flags'][arm + '_current'] and not row['flags']['A_current']]
        findings[arm] = {'usable': all(gates.values()), 'gates': gates, 'costs': costs, 'gain_groups': gains,
                        'horizontal_timely_events': timely, 'new_current_TP': len(new),
                        'new_current_TP_without_native_corridor_contributor': sum(row['native_target_corridor_samples'] == 0 for row in new)}
    h_gains = [group for group, report in groups.items() if 'head_horizontal' in group
               and report['Boundary']['arms']['G_hold']['frames']['TP'] > report['Boundary']['arms']['U_hold']['frames']['TP']]
    contribution = {'G_usable': findings['G']['usable'],
        'horizontal_extra_TP': horizontal['Boundary']['arms']['G_hold']['frames']['TP'] - horizontal['Boundary']['arms']['U_hold']['frames']['TP'],
        'horizontal_gain_groups': h_gains,
        'boundary_TP_not_lower': metrics['Boundary']['arms']['G_hold']['frames']['TP'] >= metrics['Boundary']['arms']['U_hold']['frames']['TP']}
    supported = contribution['G_usable'] and contribution['horizontal_extra_TP'] >= 5 and len(h_gains) >= 2 and contribution['boundary_TP_not_lower']
    terminal = {'geometry_hypothesis': 'SUPPORTED_DEVELOPMENT' if supported else 'NOT_SUPPORTED', 'arms': findings,
                'contribution': contribution, 'protected_test_activated': False, 'automatic_successor': False,
                'scope': 'CONSUMED_SIMULATION_MATCHED_INPUT_ABLATION'}
    compare(read(out / 'result.json'), terminal, 'entire_terminal')
    disposition = read(out / 'local-inheritance.json')
    require(disposition['terminal_id'] == DEFAULT.name, 'Disposition identity')
    require(disposition['inheritance_role'] == ('COMPONENT_OR_CHALLENGER' if any(f['usable'] for f in findings.values())
                                               else 'NEGATIVE_CONTROL'), 'Disposition follows any-arm usability')
    compare(disposition['arm_roles'], {a: 'COMPONENT_OR_CHALLENGER' if f['usable'] else 'NEGATIVE_CONTROL'
                                      for a, f in findings.items()}, 'arm_dispositions')
    require(disposition['geometry_hypothesis'] == terminal['geometry_hypothesis'], 'Geometry disposition')
    return terminal


def verify(out):
    protocol = read(out / 'protocol.json')
    require(protocol['id'] == DEFAULT.name and protocol['recipe'] == RECIPE, 'One frozen matched recipe')
    require(protocol['arms'] == list(ARMS) and protocol['scope'] == 'CONSUMED_SIMULATION_MATCHED_INPUT_ABLATION', 'Arm/scope contract')
    require(protocol['protected_test_activated'] is False and protocol['automatic_successor'] is False, 'Protocol scope flags')
    check_hash(out / 'protocol-before-run.md', protocol['protocol_text_sha256'])
    check_hash(HERE / 'SPATIAL_STRUCTURE_PROTOCOL_20260921.md', protocol['protocol_text_sha256'])
    for key in ('code_hashes', 'source_hashes'):
        for path, expected in protocol[key].items():
            check_hash(ROOT / path, expected)
    for name in ('inputs-seal.json', 'fit-seal.json', 'selection-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
        helper.seal(out, name)
    receipts = source_bindings(protocol)
    metadata = {source: read(source / 'evaluator/metadata.json') for source in (ORIGINAL, TRANSFER)}
    splits, group_ids, numeric_errors = input_audit(out, metadata, receipts)
    training = training_audit(out, splits['train'][0])
    selection = selection_audit(out, splits)
    rows, flags = readout_audit(out, selection, metadata[TRANSFER])
    metrics, longest = helper.rebuilt_metrics(rows)
    compare(read(out / 'metrics.json'), metrics, 'all_transfer_metrics')
    groups = {group: helper.rebuilt_metrics([row for row in rows if row['base_group_id'] == group])[0]
              for group in sorted(group_ids['transfer'])}
    compare(read(out / 'group-metrics.json'), groups, 'all_layout_metrics')
    horizontal_rows = [row for row in rows if 'head_horizontal' in row['base_group_id'] and row['layout_relation'] == 'BOUNDARY']
    require(len(horizontal_rows) == 96 and sum(row['truth'] for row in horizontal_rows) == 43, 'Horizontal subset denominators')
    horizontal, _ = helper.rebuilt_metrics(horizontal_rows)
    compare(read(out / 'horizontal-metrics.json'), horizontal, 'all_horizontal_metrics')
    terminal = terminal_audit(out, rows, flags, metrics, groups, horizontal)
    events = {region: {arm: [{**event, 'max_silent_samples': longest[arm][event['clip_id']]}
                             for event in metrics[region]['arms'][arm]['events']] for arm in ARMS}
              for region in ('Core', 'Boundary')}
    return {'status': 'PASS', 'experiment_geometry_hypothesis': terminal['geometry_hypothesis'],
            'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'verifier_sha256': digest(Path(__file__)),
            'independent_helpers': {str(path): digest(path) for path in (METRIC_AUDIT, helper.METRIC_AUDIT)},
            'python_executable': sys.executable, 'backend': 'TASK_NOT_GPU_SUITABLE',
            'counts': dict(audit.COUNTS), 'checked_hashes': audit.HASHES,
            'geometry_max_absolute_errors_vs_binary64': numeric_errors, 'geometry_absolute_tolerance': 5e-7,
            'matched_training': training, 'selections': selection, 'summary': helper.compact(metrics),
            'horizontal_summary': helper.compact(horizontal), 'groups': {g: helper.compact(m) for g, m in groups.items()},
            'events': events, 'terminal': terminal,
            'limits': [
                'Saved logits and learned checkpoints are authenticated, not independently inferred or trained.',
                'Original RGB cache values and their complete spatial ordering are verified; the frozen encoder is not rerun.',
                'Original ranges, validity and all interval-conditioned geometry channels are reconstructed; RGB is not metric range evidence.',
                'Common batches and untrained CPU initialization are independently reproduced; fit execution is supported by sealed code and receipts, not retraining.',
                'U zero geometry slots and G retained slots are defined in authenticated fit/predict code; no fitted forward execution is repeated.',
                'Complete event metrics reuse independent audit code, never production metrics; separated negative samples are never concatenated.',
                'Per-arm cutoffs use the same fixed dev cost contract, not identical realized transfer false-positive cost.',
                'Single seed, fixed representation/head/budget and consumed same-simulator Development only; failure is not an information-theoretic impossibility result.',
                'No protected-test row is selected and no protected-test labels, RGB images or predictions are opened; opaque source files are hashed.',
                'No historical negative gate is reclassified and no automatic successor is authorized.']}


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
    print(json.dumps({'status': report['status'], 'experiment_geometry_hypothesis': report['experiment_geometry_hypothesis'],
                      'counts': report['counts'], 'matched_training': report['matched_training'],
                      'geometry_errors': report['geometry_max_absolute_errors_vs_binary64'],
                      'summary': report['summary'], 'horizontal_summary': report['horizontal_summary'],
                      'terminal': report['terminal']}))


if __name__ == '__main__':
    main()
