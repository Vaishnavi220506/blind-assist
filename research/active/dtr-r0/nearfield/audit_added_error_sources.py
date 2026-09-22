"""Read-only, consumed-Development attribution of saved A/B/R/U/G decisions.

No fitting, model inference, cutoff selection, policy evaluation or test access.
Existing public witnesses are cross-tabulated, never converted into alert flags.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sealed_file(folder, filename, seal_name):
    seal = read(folder / seal_name)
    hashes = seal.get('hashes', seal)
    assert sha(folder / filename) == hashes[filename], filename
    if 'protocol_sha256' in seal:
        assert sha(folder / 'protocol.json') == seal['protocol_sha256']
    return read(folder / filename)


def tally(rows, axes):
    return dict(n=len(rows), ids=[r['id'] for r in rows],
        categories=dict(Counter(axes[r['id']]['axes']['category'] for r in rows)),
        groups=dict(Counter(r['base_group_id'] for r in rows)),
        relations=dict(Counter(r['layout_relation'] for r in rows)),
        witnesses={w: sum(axes[r['id']]['public']['witnesses'][w] for r in rows)
            for w in ('possible_depth', 'contained_depth', 'compatible_contained_depth')},
        native_supported=sum(r['native_target_corridor_samples'] > 0 for r in rows))


def audit():
    source = WORK / 'ba-spatial-complement-transfer-20260921'
    spatial = WORK / 'ba-spatial-structure-20260921'
    axis = WORK / 'ba-axis-evidence-20260921'
    lateral = WORK / 'ba-lateral-pair-diagnostic-20260921'
    frames = sealed_file(spatial, 'frame-results.json', 'evaluation-seal.json')
    original = sealed_file(source, 'frame-results.json', 'evaluation-seal.json')
    axes = {r['source_id']: r for r in sealed_file(axis, 'joined-frame-results.json', 'analysis-seal.json')}
    pairs = sealed_file(lateral, 'pair-records.json', 'output-seal.json')
    assert len(frames) == len(original) == len(axes) == 1152
    assert len({r['id'] for r in frames}) == 1152
    logits = {}
    clips = defaultdict(list)
    for r, old in zip(frames, original):
        assert r['id'] == old['id']
        a = axes[r['id']]
        assert r['truth'] == old['truth'] == a['truth'] == a['axes']['truth']
        assert r['base_group_id'] == a['base_group_id']
        for mode in ('current', 'hold'):
            assert r['flags']['A_' + mode] == old['flags']['A_' + mode]
            assert r['flags']['B_' + mode] == old['flags']['C_' + mode]
        logits[r['id']] = old['logit']
        clips[r['clip_id']].append(r)
    positions = {}
    for clip in clips.values():
        positive = [r['frame_in_clip'] for r in clip if r['truth']]
        for r in clip:
            positions[r['id']] = ('all_negative_clip' if not positive else
                'positive' if r['truth'] else 'before_entry' if r['frame_in_clip'] < min(positive)
                else 'after_exit' if r['frame_in_clip'] > max(positive) else 'internal_negative')
    result = {'scope': 'CONSUMED_DEVELOPMENT_SAVED_OUTPUT_DIAGNOSTIC',
        'frames': len(frames), 'layouts': len({r['base_group_id'] for r in frames}),
        'backend': 'CPU / TASK_NOT_GPU_SUITABLE', 'fits': 0, 'inference_runs': 0,
        'cutoff_selection': False, 'new_alert_policy': False, 'arms': {}}
    for arm in ('B', 'R', 'U', 'G'):
        arm_result = {}
        for mode in ('current', 'hold'):
            added = [r for r in frames if r['flags'][arm + '_' + mode] and not r['flags']['A_' + mode]]
            assert not any(r['flags']['A_' + mode] and not r['flags'][arm + '_' + mode] for r in frames)
            table = {}
            for stratum in ('Core', 'Boundary', 'All'):
                subset = [r for r in added if stratum == 'All' or
                    (r['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
                for truth, label in ((False, 'FP'), (True, 'TP')):
                    rows = [r for r in subset if r['truth'] == truth]
                    entry = tally(rows, axes)
                    entry['timing'] = dict(Counter(positions[r['id']] for r in rows))
                    entry['trigger'] = dict(Counter('current' if r['flags'][arm + '_current'] else 'hold_only' for r in rows))
                    table[stratum + '_' + label] = entry
            arm_result[mode] = table
        result['arms'][arm] = arm_result
    positives = [r for r in frames if r['truth'] and r['flags']['B_current'] and not r['flags']['A_current']]
    negatives = [r for r in frames if not r['truth'] and r['flags']['B_current'] and not r['flags']['A_current']]
    result['B_conditioned_score_ordering'] = {}
    for category in ('ALL', 'DISTANCE_NEGATIVE', 'LATERAL_NEGATIVE'):
        ns = [r for r in negatives if category == 'ALL' or axes[r['id']]['axes']['category'] == category]
        maximum = max(ns, key=lambda r: logits[r['id']])
        wins = sum(logits[p['id']] > logits[n['id']] for p in positives for n in ns)
        ties = sum(logits[p['id']] == logits[n['id']] for p in positives for n in ns)
        result['B_conditioned_score_ordering'][category] = dict(positives=len(positives), negatives=len(ns),
            pairwise_wins=wins, ties=ties, comparisons=len(positives)*len(ns),
            auc=(wins + .5 * ties)/(len(positives)*len(ns)), max_negative_id=maximum['id'],
            max_negative_logit=logits[maximum['id']],
            positive_ids_above_max_negative=[p['id'] for p in positives if logits[p['id']] > logits[maximum['id']]])
    matched = [p for p in pairs if p['contrast'] == 'BOUNDARY_vs_OUTSIDE' and p['primary']]
    result['matched_lateral_ordering'] = dict(pairs=len(matched),
        positive_higher=sum(p['left_logit'] > p['right_logit'] for p in matched))
    result['inputs'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in
        (spatial/'frame-results.json', source/'frame-results.json', axis/'joined-frame-results.json',
         lateral/'pair-records.json', Path(__file__))}
    result['limits'] = ['Selected rescued-TP versus added-FP ranking is not whole-cohort model AUC.',
        'Witness counts are coverage diagnostics, not a tested filtered/held alert policy.',
        'Evaluator geometry is attribution only; no truth enters a predictor.',
        'No claim that all RGB plus 64-zone inputs are separable or intrinsically ambiguous.',
        'Frames cluster within 16 layouts; no independent significance or hardware claim.']
    assert result['arms']['B']['current']['All_FP']['n'] == 28
    assert result['arms']['B']['current']['All_TP']['n'] == 120
    assert result['arms']['B']['hold']['All_FP']['n'] == 39
    assert result['matched_lateral_ordering'] == dict(pairs=173, positive_higher=173)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({arm: {mode: {k: v['n'] for k, v in values.items()}
        for mode, values in data.items()} for arm, data in result['arms'].items()}, indent=2))
