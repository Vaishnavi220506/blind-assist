"""Independent arithmetic and seal audit for frozen LOCAL transfer outputs.

Does not import inference/metrics, fit models, select cutoffs or make predictions.
Model provenance is checked by sealed hashes; sklearn tree arithmetic is not replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('A_current', 'raw', 'local', 'raw_standalone', 'local_standalone')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def scalar_counts(rows, arm):
    result = dict.fromkeys(('frames', 'TP', 'FP', 'FN', 'TN', 'prediction_unknown',
        'prediction_unknown_positive', 'prediction_unknown_negative', 'abstained_positive',
        'abstained_negative', 'ambiguous', 'ambiguous_alerts'), 0)
    for r in rows:
        if r['truth'] is None:
            continue
        y, p = r['truth'], r['predictions'][arm]
        result['frames'] += 1
        result['TP'] += bool(y and p['alert'])
        result['FP'] += bool(not y and p['alert'])
        result['FN'] += bool(y and not p['alert'])
        result['TN'] += bool(not y and not p['alert'] and not p['unknown'])
        result['prediction_unknown'] += p['unknown']
        result['prediction_unknown_positive' if y else 'prediction_unknown_negative'] += p['unknown']
        result['abstained_positive' if y else 'abstained_negative'] += p['unknown'] and not p['alert']
        result['ambiguous'] += p['ambiguous']
        result['ambiguous_alerts'] += p['ambiguous'] and p['alert']
    for key, numerator, denominator in (
        ('recall', result['TP'], result['TP']+result['FN']),
        ('precision', result['TP'], result['TP']+result['FP']),
        ('false_alert_rate_known_negative', result['FP'], result['FP']+result['TN']+result['abstained_negative'])):
        result[key] = numerator/denominator if denominator else None
    return result


def runs_and_events(rows, arm):
    events, false = [], []
    for clip in sorted({r['clip_id'] for r in rows}):
        subset = sorted([r for r in rows if r['clip_id'] == clip], key=lambda r: r['frame_in_clip'])
        for positive in (True, False):
            i = 0
            while i < len(subset):
                active = subset[i]['truth'] is positive and (positive or subset[i]['predictions'][arm]['alert'])
                if not active:
                    i += 1
                    continue
                start = i
                while i < len(subset) and subset[i]['truth'] is positive and (positive or subset[i]['predictions'][arm]['alert']):
                    i += 1
                current = subset[start:i]
                if positive:
                    first = next((r for r in current if r['predictions'][arm]['alert']), None)
                    events.append(dict(clip_id=clip, start_frame=current[0]['frame_in_clip'],
                        end_frame=current[-1]['frame_in_clip'], detected=first is not None,
                        first_alert_time_s=first['time_s'] if first else None,
                        entry_time_s=current[0]['time_s'], sampled_duration_s=len(current)*.2,
                        first_alert_relative_to_entry_s=first['time_s']-current[0]['time_s'] if first else None))
                else:
                    false.append(dict(clip_id=clip, start_frame=current[0]['frame_in_clip'],
                        end_frame=current[-1]['frame_in_clip'], sampled_duration_s=len(current)*.2))
    return events, false


def independently_decide(rows, g, source_ok, public_pairs, labels, stored_pairs, geometry):
    def status(value, available=True):
        return 'NOT_EVALUABLE' if not available else 'PASS' if value else 'FAIL'
    def summary(part, arm):
        c = scalar_counts(part, arm)
        c['segments'] = len(runs_and_events(part, arm)[1])
        return c
    groups = sorted({r['base_group_id'] for r in rows})
    by_app = {}
    for app in ('base', 'changed'):
        part = [r for r in rows if r['appearance'] == app]
        local, raw = summary(part, 'local'), summary(part, 'raw')
        improved = sum(sum(r['truth'] is True and r['predictions']['local']['alert'] for r in part if r['base_group_id'] == group) >
            sum(r['truth'] is True and r['predictions']['raw']['alert'] for r in part if r['base_group_id'] == group) for group in groups)
        layer_ok = True
        for layer in ('BODY', 'HEAD'):
            pos = [r for r in part if r['layer'] == layer and r['truth'] is True]
            layer_ok = layer_ok and bool(pos) and sum(int(r['predictions']['local']['alert'])-int(r['predictions']['raw']['alert']) for r in pos)/max(1, len(pos)) >= -.05-1e-12
        okay = local['recall'] is not None and raw['recall'] is not None and local['recall']-raw['recall'] >= .05-1e-12 and local['FP']-raw['FP'] <= 1 and local['segments']-raw['segments'] <= 1 and improved >= 4 and layer_ok
        by_app[app] = status(okay, source_ok)
    totals = []
    for group in groups:
        part = [r for r in rows if r['base_group_id'] == group and r['truth'] is True]
        totals.append([len(part), sum(r['predictions']['local']['alert'] for r in part), sum(r['predictions']['raw']['alert'] for r in part)])
    rng = np.random.default_rng(202609225)
    totals = np.asarray(totals)
    values = []
    for _ in range(1000):
        selected = totals[rng.integers(0, len(groups), len(groups))].sum(0)
        values.append((selected[1]-selected[2])/max(1, selected[0]))
    ci = np.quantile(values, [.025, .975]).tolist()
    component = status(all(s == 'PASS' for s in by_app.values()) and ci[0] > 0, source_ok)
    b, c = [summary([r for r in rows if r['appearance'] == a], 'local') for a in ('base', 'changed')]
    rescues = [p for p in stored_pairs if p['truth'] is True and p['arms']['local']['base_alert'] and not p['arms']['A_current']['base_alert']]
    kept = sum(p['arms']['local']['changed_alert'] for p in rescues)
    stable = b['recall'] is not None and c['recall'] is not None and c['recall']-b['recall'] >= -.05-1e-12 and c['FP']-b['FP'] <= 1 and c['segments']-b['segments'] <= 1 and kept/max(1, len(rescues)) >= .75
    appearance = status(stable, source_ok and len(rescues) >= 8)
    geo = {}
    for app in ('base', 'changed'):
        valid = [p for p in geometry if p['appearance'] == app and p['eligible']]
        correct = sum(p['arms']['local']['boundary_frame_score'] > p['arms']['local']['outside_frame_score'] for p in valid)
        geo[app] = status(correct/max(1, len(valid)) >= .75, source_ok and bool(valid))
    geo_status = 'NOT_EVALUABLE' if 'NOT_EVALUABLE' in geo.values() else status(all(s == 'PASS' for s in geo.values()))
    local, base = summary(rows, 'local'), summary(rows, 'A_current')
    retained = all(not (r['truth'] is True and r['predictions']['A_current']['alert'] and not r['predictions']['local']['alert']) for r in rows)
    old_events, _ = runs_and_events(rows, 'A_current')
    new_events, _ = runs_and_events(rows, 'local')
    timing = all(not old['detected'] or (new['detected'] and new['first_alert_time_s'] <= old['first_alert_time_s']+1e-9) for old, new in zip(old_events, new_events))
    strict = status(local['recall'] is not None and base['recall'] is not None and local['recall']-base['recall'] >= .10-1e-12 and local['FP']-base['FP'] <= 2 and local['segments']-base['segments'] <= 1 and retained and timing, source_ok)
    all_status = [component, appearance, geo_status]
    overall = ('FAIL' if 'FAIL' in all_status else 'NOT_EVALUABLE'
               if 'NOT_EVALUABLE' in all_status else 'PASS')
    return dict(component=component, component_by_appearance=by_app, bootstrap95=ci,
        appearance=appearance, base_rescues=len(rescues), retained_rescues=kept,
        geometry=geo_status, geometry_by_appearance=geo, strict_system=strict, retained=overall)


def audit(args):
    pred, ev = args.predictions, args.evaluated
    seal, freeze = read(pred/'prediction-seal.json'), read(pred/'freeze.json')
    result, m, rows = read(ev/'result.json'), read(ev/'metrics.json'), read(ev/'frame-results.json')
    checks = 0
    def check(ok, why):
        nonlocal checks
        if not ok:
            raise AssertionError(why)
        checks += 1
    def same(a, b, why):
        if isinstance(a, float) and isinstance(b, (float, int)):
            check(math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), why)
        else:
            check(a == b, why)
    check(not seal['evaluation_labels_opened'] and not freeze['evaluation_labels_opened'], 'sealed before labels')
    check(freeze['fits'] == freeze['cutoff_selections'] == result['fits'] == result['cutoff_selections'] == 0, 'zero fit/selection')
    check(digest(args.protocol) == freeze['protocol_sha256'] == result['protocol_sha256'], 'protocol')
    check(digest(args.materialization) == freeze['materialization_sha256'], 'materialization')
    manifest = read(args.materialization)
    for name, value in seal['hashes'].items():
        check(digest(pred/name) == value, 'prediction '+name)
    for name, value in freeze['source_hashes'].items():
        check(digest(pred/'source-snapshot'/name) == digest(args.repo/name) == value, 'source '+name)
    for key, value in freeze['input_hashes'].items():
        check(digest(args.observations/Path(key).name) == manifest['hashes'][key] == value, 'public input '+key)
    for name, key in [('metrics.json', 'metrics_sha256'), ('frame-results.json', 'frame_results_sha256')]:
        check(digest(ev/name) == result[key], name)
    check(digest(pred/'prediction-seal.json') == result['prediction_seal_sha256'], 'prediction seal')
    check(digest(args.labels) == freeze['expected_evaluation_labels_sha256'] == result['evaluator_labels_sha256'], 'labels')
    original = read(args.frozen_run/'model-seal.json')
    check(digest(args.frozen_run/'model-seal.json') == freeze['frozen_model_seal_sha256'], 'original model seal')
    check(digest(args.frozen_run/'selection.json') == digest(pred/'selection.json') == original['selection_sha256'] == freeze['frozen_selection_sha256'], 'frozen selection')
    selections = read(pred/'selection.json')
    for arm in ('raw', 'local'):
        check(selections[arm]['selection']['threshold'] == freeze['thresholds'][arm] == m['thresholds'][arm], 'cutoff '+arm)
        check(digest(args.frozen_run/(arm+'.pkl')) == original['models'][arm] == freeze['frozen_model_hashes'][arm], 'frozen model '+arm)
    labels = dict(np.load(args.labels, allow_pickle=False))
    probabilities = dict(np.load(pred/'probabilities.npz', allow_pickle=False))
    features = dict(np.load(pred/'features.npz', allow_pickle=False))
    check(np.array_equal(labels['indices'], probabilities['indices']) and np.array_equal(labels['indices'], np.arange(576)), 'all indices')
    check(features['raw'].shape == features['local'].shape == (576, 6, 961), 'features')
    check(np.array_equal(features['raw'][:, :, :910], features['local'][:, :, :910]) and not features['raw'][:, :, 910:].any(), 'unchanged representation contrast')
    ids, baseline = read(pred/'identities.json'), read(pred/'baseline.json')
    check(len(rows) == len(ids) == len(baseline) == 576, 'frame denominator')
    for i, (r, meta, b) in enumerate(zip(rows, ids, baseline)):
        check(r['id'] == meta['id'] and r['index'] == i, 'identity join')
        for field in ('appearance', 'appearance_pair_id', 'base_group_id', 'type_id', 'layer', 'layout_relation', 'clip_id', 'frame_in_clip', 'time_s'):
            same(r[field], meta[field], 'identity '+field)
        y = bool(np.any(labels['classes'][i, [1, 4]] < 6)) if np.all(labels['valid'][i, [1, 4]]) else None
        same(r['truth'], y, 'truth join')
        same(r['query_truth'], (labels['classes'][i] < 6).tolist(), 'query truth')
        same(r['query_valid'], labels['valid'][i].astype(bool).tolist(), 'query validity')
        flags = {'A_current': bool(b['alert'])}
        for arm in ('raw', 'local'):
            same(r['query_probabilities'][arm], probabilities[arm][i].tolist(), 'query probabilities')
            score = float(max(probabilities[arm][i, [1, 4]]))
            same(r['frame_scores'][arm], score, 'frame score')
            flags[arm+'_standalone'] = score >= freeze['thresholds'][arm]
            flags[arm] = bool(b['alert']) or flags[arm+'_standalone']
        for arm in ARMS:
            same(r['predictions'][arm], dict(alert=flags[arm], unknown=bool(b['unknown']), ambiguous=flags[arm] and bool(b['unknown'])), 'frozen readout '+arm)
    def inspect(part, saved):
        same(saved['frames'], len(part), 'stratum frame denominator')
        same(saved['known_truth_frames'], sum(r['truth'] is not None for r in part), 'known truth')
        for arm in ARMS:
            s = saved['arms'][arm]
            for name, sub in [('all_known', part), ('interior', [r for r in part if not r['boundary']]), ('boundary', [r for r in part if r['boundary']])]:
                for key, value in scalar_counts(sub, arm).items():
                    same(s['frames'][name][key], value, 'count '+arm+'/'+name+'/'+key)
            events, segments = runs_and_events(part, arm)
            same(s['event_count'], len(events), 'event count')
            same(s['detected_events'], sum(e['detected'] for e in events), 'detected events')
            same(s['false_alert_segment_count'], len(segments), 'segments')
            same(s['false_alert_sampled_duration_s'], sum(e['sampled_duration_s'] for e in segments), 'false duration')
            same(s['prediction_unknown_frames'], sum(r['predictions'][arm]['unknown'] for r in part), 'unknown count')
            for actual, stored in zip(events, s['events']):
                for key, value in actual.items():
                    same(stored[key], value, 'event '+key)
            for actual, stored in zip(segments, s['false_alert_segments']):
                for key, value in actual.items():
                    same(stored[key], value, 'false segment '+key)
    inspect(rows, m['metrics'])
    for axis, groups in m['strata'].items():
        same(set(groups), {r[axis] for r in rows}, 'stratum coverage '+axis)
        for group, saved in groups.items():
            inspect([r for r in rows if r[axis] == group], saved)
    def inspect_comparisons(part, comparisons):
        for key, stored in comparisons.items():
            arm, reference = key.split('_vs_')
            a, b = scalar_counts(part, arm), scalar_counts(part, reference)
            ea, sa = runs_and_events(part, arm)
            eb, sb = runs_and_events(part, reference)
            for field, value in dict(TP_delta=a['TP']-b['TP'], FP_delta=a['FP']-b['FP'],
                recall_delta=a['recall']-b['recall'] if a['recall'] is not None else None,
                false_segment_delta=len(sa)-len(sb)).items():
                same(stored[field], value, 'comparison '+field)
            for field, truth, old, new in [('lost_true_frames', True, True, False),
                ('gained_true_frames', True, False, True), ('added_false_frames', False, False, True),
                ('removed_false_frames', False, True, False)]:
                same(stored[field], [r['id'] for r in part if r['truth'] is truth and r['predictions'][reference]['alert'] is old
                    and r['predictions'][arm]['alert'] is new], 'comparison ids '+field)
            deltas = {group: sum(int(r['predictions'][arm]['alert'])-int(r['predictions'][reference]['alert'])
                for r in part if r['base_group_id'] == group and r['truth'] is True) for group in sorted({r['base_group_id'] for r in part})}
            same(stored['group_TP_delta'], deltas, 'group TP deltas')
            same(stored['improved_groups'], sum(value > 0 for value in deltas.values()), 'improved groups')
            same(len(stored['event_differences']), len(ea), 'all event differences')
            for current, previous, saved in zip(ea, eb, stored['event_differences']):
                t1, t0 = current['first_alert_time_s'], previous['first_alert_time_s']
                for field, value in dict(clip_id=current['clip_id'], start_frame=current['start_frame'],
                    candidate_first_s=t1, reference_first_s=t0, delay_s=None if t0 is None or t1 is None else t1-t0,
                    lost=previous['detected'] and not current['detected'], gained=current['detected'] and not previous['detected']).items():
                    same(saved[field], value, 'onset comparison '+field)
    inspect_comparisons(rows, m['comparisons'])
    for app in ('base', 'changed'):
        inspect_comparisons([r for r in rows if r['appearance'] == app], m['comparisons_by_appearance'][app])
    for arm in ('raw', 'local'):
        for name, qs in [('all', range(6))] + [(str(i), [i]) for i in range(6)]:
            tp = fp = fn = tn = 0
            for i in range(576):
                for q in qs:
                    if not labels['valid'][i, q]:
                        continue
                    y, p = labels['classes'][i, q] < 6, probabilities[arm][i, q] >= freeze['thresholds'][arm]
                    tp += bool(y and p); fp += bool(not y and p)
                    fn += bool(y and not p); tn += bool(not y and not p)
            saved = m['query_metrics'][arm]['all'] if name == 'all' else m['query_metrics'][arm]['by_query'][name]
            for key, value in dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=tp+fp+fn+tn).items():
                same(saved[key], value, 'query confusion')
    rgb, tof = [np.load(args.observations/n, mmap_mode='r', allow_pickle=False) for n in ('rgb.npy', 'tof.npy')]
    pub = read(pred/'public-pair-checks.json')
    row_by_id = {r['id']: r for r in rows}
    source_ok = bool(np.all(labels['valid'])) and len(pub['pairs']) == len(m['appearance_pairs']['pairs']) == 288
    l1_values = []
    for p in pub['pairs']:
        b, c = p['base_index'], p['changed_index']
        equal = np.array_equal(tof[b], tof[c], equal_nan=True)
        changed = not np.array_equal(rgb[b], rgb[c])
        l1 = float(np.mean(np.abs(rgb[b].astype(np.int16)-rgb[c].astype(np.int16)))/255)
        l1_values.append(l1)
        same(p['tof_exactly_equal'], bool(equal), 'paired ToF')
        same(p['rgb_changed'], bool(changed), 'paired RGB')
        same(p['normalized_rgb_L1'], l1, 'paired normalized RGB L1')
        source_ok = source_ok and equal and changed
        for key in ('classes', 'distances', 'valid'):
            source_ok = source_ok and np.array_equal(labels[key][b], labels[key][c], equal_nan=True)
    mean_l1 = sum(l1_values)/len(l1_values) if l1_values else None
    same(pub['mean_normalized_rgb_L1'], mean_l1, 'mean normalized RGB L1')
    source_ok = source_ok and mean_l1 is not None and mean_l1 >= 1/255
    for pair in m['appearance_pairs']['pairs']:
        b, c = row_by_id[pair['base_id']], row_by_id[pair['changed_id']]
        same(pair['pair_id'], b['appearance_pair_id'], 'pair identity')
        same(pair['pair_id'], c['appearance_pair_id'], 'pair identity')
        source_ok = source_ok and all(b[k] == c[k] for k in ('base_group_id', 'type_id', 'layer', 'layout_relation', 'frame_in_clip', 'time_s', 'truth', 'query_truth', 'query_valid'))
        for arm in ARMS:
            same(pair['arms'][arm], dict(base_alert=b['predictions'][arm]['alert'], changed_alert=c['predictions'][arm]['alert']), 'appearance alert pair')
        for arm in ('raw', 'local'):
            same(pair['score_delta'][arm], c['frame_scores'][arm]-b['frame_scores'][arm], 'appearance score delta')
            same(pair['query_score_delta'][arm], (np.asarray(c['query_probabilities'][arm])-b['query_probabilities'][arm]).tolist(), 'appearance query delta')
    pairs = m['appearance_pairs']['pairs']
    for arm in ARMS:
        saved = m['appearance_pairs']['summary'][arm]
        values = dict(pairs=len(pairs),
            alert_flips=sum(p['arms'][arm]['base_alert'] != p['arms'][arm]['changed_alert'] for p in pairs),
            positive_lost=sum(p['truth'] is True and p['arms'][arm]['base_alert'] and not p['arms'][arm]['changed_alert'] for p in pairs),
            positive_gained=sum(p['truth'] is True and not p['arms'][arm]['base_alert'] and p['arms'][arm]['changed_alert'] for p in pairs),
            negative_added=sum(p['truth'] is False and not p['arms'][arm]['base_alert'] and p['arms'][arm]['changed_alert'] for p in pairs),
            negative_removed=sum(p['truth'] is False and p['arms'][arm]['base_alert'] and not p['arms'][arm]['changed_alert'] for p in pairs))
        if arm in ('raw', 'local'):
            changes = [p['score_delta'][arm] for p in pairs]
            query_changes = [v for p in pairs for v in p['query_score_delta'][arm]]
            values.update(mean_signed_score_change=sum(changes)/len(changes) if changes else None,
                mean_absolute_score_change=sum(abs(v) for v in changes)/len(changes) if changes else None,
                max_absolute_score_change=max(map(abs, changes)) if changes else None,
                mean_absolute_query_score_change=sum(abs(v) for v in query_changes)/len(query_changes) if query_changes else None)
        for key, value in values.items():
            same(saved[key], value, 'appearance summary '+key)
    for app in ('base', 'changed'):
        part = [r for r in rows if r['appearance'] == app]
        source_ok = source_ok and (len(part), sum(r['truth'] is True for r in part), sum(r['truth'] is False for r in part)) == (288, 128, 160)
    source_ok = bool(source_ok and not pub['issues'] and not m['appearance_pairs']['issues'])
    same(m['source_admissibility']['status'], 'PASS' if source_ok else 'NOT_EVALUABLE', 'source admission')
    geometry = m['geometry_ordering']
    for p in geometry['pairs']:
        b, o = row_by_id[p['boundary_id']], row_by_id[p['outside_id']]
        q = 1 if b['layer'] == 'BODY' else 4
        same(p['eligible'], b['truth'] is True and o['truth'] is False, 'geometry eligible positive period')
        for arm in ('raw', 'local'):
            saved = p['arms'][arm]
            bs, os = b['query_probabilities'][arm][q], o['query_probabilities'][arm][q]
            for key, value in dict(boundary_score=bs, outside_score=os, difference=bs-os,
                ordered=bs > os, tied=bs == os, reversed=bs < os,
                boundary_frame_score=b['frame_scores'][arm], outside_frame_score=o['frame_scores'][arm],
                frame_ordered=b['frame_scores'][arm] > o['frame_scores'][arm],
                query_boundary_hit=bs >= freeze['thresholds'][arm], query_outside_false=os >= freeze['thresholds'][arm],
                actual_boundary_alert=b['predictions'][arm]['alert'], actual_outside_alert=o['predictions'][arm]['alert'],
                actual_separates=b['predictions'][arm]['alert'] and not o['predictions'][arm]['alert']).items():
                same(saved[key], value, 'geometry pair '+key)
    for subset, stored in [(geometry['pairs'], geometry['summary'])] + [
            ([p for p in geometry['pairs'] if p['appearance'] == app], geometry['by_appearance'][app]) for app in ('base', 'changed')]:
        valid = [p for p in subset if p['eligible']]
        for arm in ('raw', 'local'):
            values = dict(total_pairs=len(subset), eligible_pairs=len(valid),
                mean_score_margin=sum(p['arms'][arm]['difference'] for p in valid)/len(valid) if valid else None,
                frame_ordering_fraction=sum(p['arms'][arm]['frame_ordered'] for p in valid)/len(valid) if valid else None)
            for plural, singular in [('ordered', 'ordered'), ('tied', 'tied'), ('reversed', 'reversed'), ('frame_ordered', 'frame_ordered'),
                ('query_boundary_hits', 'query_boundary_hit'), ('query_outside_false', 'query_outside_false'),
                ('actual_separates', 'actual_separates'), ('actual_outside_alerts', 'actual_outside_alert')]:
                values[plural] = sum(p['arms'][arm][singular] for p in valid)
            for key, value in values.items():
                same(stored[arm][key], value, 'geometry summary '+key)
    decision = independently_decide(rows, freeze['gate_configuration'], source_ok, pub, labels,
        m['appearance_pairs']['pairs'], geometry['pairs'])
    gates = m['gates']
    same(gates['configuration'], freeze['gate_configuration'], 'frozen gates')
    same(gates['component']['status'], decision['component'], 'component gate')
    same(gates['component']['bootstrap']['recall_delta_95'], decision['bootstrap95'], 'paired layout bootstrap')
    same(gates['appearance_stability']['status'], decision['appearance'], 'appearance gate')
    same(gates['appearance_stability']['base_rescues'], decision['base_rescues'], 'base rescues')
    same(gates['appearance_stability']['retained_rescues'], decision['retained_rescues'], 'retained rescues')
    same(gates['geometry_sensitivity']['status'], decision['geometry'], 'geometry gate')
    same(gates['strict_system']['status'], decision['strict_system'], 'strict system gate')
    same(gates['retained_transfer_component'], decision['retained'], 'transfer decision')
    for app in ('base', 'changed'):
        same(gates['component']['by_appearance'][app]['status'], decision['component_by_appearance'][app], 'appearance component')
        same(gates['geometry_sensitivity']['by_appearance'][app]['status'], decision['geometry_by_appearance'][app], 'appearance geometry')
    answer = dict(status='PASS', checks=checks, frames=576, queries=3456, decisions=decision,
        prediction_seal_sha256=digest(pred/'prediction-seal.json'), evaluation_result_sha256=digest(ev/'result.json'),
        scope='Independent saved-output arithmetic and sealed model/code/input hashes; no fitting, cutoff changes or new predictions',
        limits='Does not independently verify sklearn tree arithmetic, each RGB feature, or render geometry')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open('x', encoding='utf-8') as stream:
        json.dump(answer, stream, indent=2, allow_nan=False)
    print(json.dumps(answer), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('predictions', 'evaluated', 'frozen-run', 'observations', 'materialization', 'labels', 'protocol', 'repo', 'result'):
        parser.add_argument('--'+name, type=Path, required=True)
    audit(parser.parse_args())
