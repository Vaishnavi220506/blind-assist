"""One user-requested posthoc OR of existing sealed predictions; no inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from tolerance_eval import metric, temporal, geometry, classify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT/'artifacts.local/work/corridor-public-single-20260917'
OUT = ROOT/'artifacts.local/work/corridor-astar-positive-posthoc-20260917'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    assert not OUT.exists(), 'Preserve existing analysis'
    OUT.mkdir(parents=True)
    cap = SOURCE/'source/returned-v1/capture-v1'
    conf = SOURCE/'confirmation'
    bindings = {str(p): sha(p) for p in conf.iterdir() if p.is_file()}
    for p in [SOURCE/'recipe-freeze.json', SOURCE/'bundle/config.json', cap/'receipt.json',
              cap/'raw.jsonl', cap/'evaluator.jsonl', cap/'spec.json', Path(__file__),
              HERE/'tolerance_eval.py', HERE/'ASTAR_POSITIVE_POSTHOC_PROTOCOL_20260917.md']:
        bindings[str(p)] = sha(p)
    done = read(conf/'completion.json')
    assert done['status'] == 'PASS'
    assert sha(conf/'summary.json') == done['summary_sha256']
    assert sha(conf/'prediction-seal.json') == done['prediction_seal_sha256']
    assert sha(conf/'predictions.json') == read(conf/'prediction-seal.json')['predictions_sha256']
    for p, h in read(SOURCE/'recipe-freeze.json')['bundle_files'].items():
        assert sha(p) == h
        bindings[p] = h
    receipt = read(cap/'receipt.json')
    for name, h in receipt['hashes'].items():
        assert sha(cap/name) == h
    write(OUT/'analysis-seal.json', dict(authority='POSTHOC_CONSUMED_DEVELOPMENT_EXACT_OR',
        formula='control OR positive', no_fit=True, no_inference=True, no_threshold_change=True,
        source_confirmation_preserved=True, inputs=bindings))
    pred, cases, prior = read(conf/'predictions.json'), read(conf/'cases.json'), read(conf/'summary.json')
    config = read(SOURCE/'bundle/config.json')
    assert len(pred) == len(cases) == 288
    for p, c in zip(pred, cases):
        assert p['id'] == c['id']
        assert p['A'] == (p['A_score'] >= config['A_threshold'])
        assert p['control'] == (p['control_score'] >= config['control_threshold'])
        assert p['positive'] == (p['usable_tof_returns'] > 0 and p['positive_score'] >= config['positive_threshold'])
        assert p['alert'] == (p['A'] or p['positive'])
    truth = np.array([c['truth'] for c in cases], bool)
    state = np.array([c['stratum'] for c in cases])
    clear = state != 'boundary'
    a, control, positive = [np.array([p[k] for p in pred], bool) for k in ['A', 'control', 'positive']]
    combo = control | positive
    write(OUT/'predictions.json', [dict(id=p['id'], episode_id=p['episode_id'], time_s=p['time_s'],
        A=p['A'], A_star=p['control'], positive=p['positive'], combined=bool(combo[i])) for i, p in enumerate(pred)])
    methods = {}
    for name, flags in [('A', a), ('A_plus_public', a | positive), ('A_retrained', control), ('A_star_plus_public', combo)]:
        entry = dict(clear=metric(truth[clear], flags[clear]), strict=metric(truth, flags),
            boundary=metric(truth[~clear], flags[~clear]), coverage=float(clear.mean()),
            temporal=temporal(cases, state, flags),
            strict_temporal=temporal(cases, np.where(truth, 'positive', 'negative'), flags))
        if name in prior['methods']:
            for key in ['clear', 'strict', 'boundary', 'temporal', 'strict_temporal']:
                assert entry[key] == prior['methods'][name][key], (name, key)
        methods[name] = entry
    masks = {'clear': clear, 'boundary': ~clear, 'strict': np.ones(288, bool)}
    def changes(base):
        return {tag: dict(rescued_FN=int(sum(mask & truth & ~base & combo)),
            added_FP=int(sum(mask & ~truth & ~base & combo)),
            lost_TP=int(sum(mask & truth & base & ~combo)),
            removed_FP=int(sum(mask & ~truth & base & ~combo))) for tag, mask in masks.items()}
    def onset_changes(basename, key):
        before = {e['episode']: e for e in methods[basename][key]['events']}
        return [dict(episode=e['episode'], baseline_first_s=before[e['episode']]['first_in_core_delay_s'],
            combined_first_s=e['first_in_core_delay_s']) for e in methods['A_star_plus_public'][key]['events']
            if e['first_in_core_delay_s'] != before[e['episode']]['first_in_core_delay_s']]
    added = []
    # Native labels are used only to explain the fixed composed outputs.
    sys.path.insert(0, str(HERE.parent))
    from mz171_return_labels import make_witness_labels
    raw = [json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
    native = [json.loads(s) for s in (cap/'evaluator.jsonl').read_text().splitlines()]
    for i, (p, c, r, e) in enumerate(zip(pred, cases, raw, native)):
        assert p['id'] == r['id'] == e['id']
        g = geometry(e)
        assert g['strict'] == c['truth'] and classify(g, .05) == c['stratum']
        if not control[i] and combo[i]:
            lab = make_witness_labels(r, e)
            slot = p['max_return_slot']
            added.append(dict(id=p['id'], episode=p['episode_id'], time_s=p['time_s'],
                family=c['family'], group=c['group'], truth=c['truth'], stratum=c['stratum'],
                old_A=p['A'], control_score=p['control_score'], positive_score=p['positive_score'],
                max_return_slot=slot, max_return_has_native_witness=bool(lab['known'][slot] and lab['target'][slot] > 0)))
        if p['usable_tof_returns'] == 0:
            assert combo[i] == control[i]
    groups = []
    for group in sorted({c['group'] for c in cases}):
        mask = np.array([c['group'] == group for c in cases])
        groups.append(dict(group=group, clear_frames=int(sum(mask & clear)),
            A_star=metric(truth[mask], control[mask]), combined=metric(truth[mask], combo[mask])))
    families = {}
    for family in sorted({c['family'] for c in cases}):
        f = np.array([c['family'] == family for c in cases])
        families[family] = {tag: metric(truth[f & mask], combo[f & mask]) for tag, mask in masks.items()}
    lost_old = [dict(id=p['id'], recovered=bool(combo[i]), family=cases[i]['family'])
        for i, p in enumerate(pred) if clear[i] and truth[i] and a[i] and not control[i]]
    result = dict(authority='POSTHOC_CONSUMED_DEVELOPMENT_NOT_NEW_CONFIRMATION', methods=methods,
        versus_A_star=changes(control), versus_old_A=changes(a), added_alerts=added,
        old_A_clear_TP_lost_by_A_star=lost_old, configurations=groups, combined_families=families,
        core_onsets_vs_A_star=onset_changes('A_retrained', 'temporal'),
        strict_onsets_vs_A_star=onset_changes('A_retrained', 'strict_temporal'),
        strict_onsets_vs_old_A=onset_changes('A', 'strict_temporal'),
        latency='No new inference or timing. Prior A* comparison includes old-A head overhead; no optimized combined latency claimed.')
    write(OUT/'summary.json', result)
    assert all(sha(p) == h for p, h in bindings.items())
    write(OUT/'completion.json', dict(status='PASS', inputs_unchanged=True,
        summary_sha256=sha(OUT/'summary.json'), predictions_sha256=sha(OUT/'predictions.json'),
        analysis_seal_sha256=sha(OUT/'analysis-seal.json')))
    print(json.dumps(dict(metrics={k: v['clear'] for k, v in methods.items()},
        changes=result['versus_A_star'], recovered=lost_old, added_alerts=added), indent=2))


if __name__ == '__main__':
    main()
