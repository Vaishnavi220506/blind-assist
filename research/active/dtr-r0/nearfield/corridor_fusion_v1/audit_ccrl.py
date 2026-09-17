"""Read-only replay, group and native accounting audit for the fixed CCRL run."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ccrl import CorridorResidual, log_odds
from prepare_single_a import read, write, sha
from tolerance_eval import geometry, classify
from mz171_return_labels import make_witness_labels

ROOT = HERE.parents[4]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'corridor-ccrl-20260918'
DATA = WORK/'corridor-public-single-20260917/a-control'


def main():
    torch.set_num_threads(4)
    done = read(OUT/'completion.json')
    assert done['status'] == 'PASS'
    assert sha(OUT/'summary.json') == done['summary_sha256']
    seal = read(OUT/'prediction-seal.json')
    for p, h in seal['inputs'].items():
        assert sha(p) == h, p
    assert sha(OUT/'predictions.json') == seal['predictions_sha256']
    assert sha(OUT/'report-features.npz') == seal['feature_sha256']
    config = read(OUT/'model-seal.json')
    pred, cases = read(OUT/'predictions.json'), read(OUT/'cases.json')
    report = read(OUT/'summary.json')
    x = np.load(OUT/'report-features.npz')
    meta = read(DATA/'metadata.json')
    train = np.load(DATA/'features.npz')['base'].astype(np.float32)
    oof = np.load(OUT/'oof-scores.npz')
    nonanchor = oof['indices']
    pa = read(OUT/'pair-audit.json')
    rank = pa['rank_pairs']
    assert len(rank) == 672 and pa['invariance_count'] == 0
    folds = read(OUT/'fold-receipts.json')
    covered = []
    for fold in folds:
        fi, ri = fold['residual_fit'], fold['report']
        covered += ri
        assert not set(fi) & set(ri)
        fg = {meta[i]['group'] for i in fi}
        rg = {meta[i]['group'] for i in ri}
        assert not fg & rg
        for a, b in rank:
            assert (a in fi) == (b in fi) and (a in ri) == (b in ri)
        for inner in fold['inner']:
            assert not set(inner['fit']) & (set(inner['report']) | set(ri))
            assert not {meta[i]['group'] for i in inner['fit']} & (rg | {meta[i]['group'] for i in inner['report']})
        for arm in config['thresholds']:
            ck = torch.load(OUT/f"{arm}-fold{fold['fold']}.pt", weights_only=True)
            assert len(ck['history']) == 120
            np.testing.assert_array_equal(ck['state_dict']['mean'], train[fi].mean(0))
            np.testing.assert_array_equal(ck['state_dict']['scale'], np.maximum(train[fi].std(0), .001))
    assert sorted(covered) == sorted(nonanchor.tolist())
    y_oof = np.array([meta[i]['truth'] for i in nonanchor], bool)
    clear_oof = np.array([meta[i]['stratum'] != 'boundary' for i in nonanchor])
    for arm, tau in config['thresholds'].items():
        # Independent scalar threshold reduction with the declared tie order.
        s = oof[arm]
        choices = np.r_[np.nextafter(s.max(), np.inf), np.unique(s[clear_oof])]
        tuples = []
        for t in choices:
            pairs = [(bool(yy), bool(ss >= t)) for yy, ss, cc in zip(y_oof, s, clear_oof) if cc]
            tp = sum(yy and pp for yy, pp in pairs)
            fp = sum(not yy and pp for yy, pp in pairs)
            fn = sum(yy and not pp for yy, pp in pairs)
            tuples.append((2*tp/(2*tp+fp+fn), -fp, float(t)))
        assert max(tuples)[2] == tau
        path = OUT/f'{arm}-final.pt'
        assert sha(path) == config['models'][arm]
        ck = torch.load(path, weights_only=True)
        np.testing.assert_array_equal(ck['state_dict']['mean'], train[nonanchor].mean(0))
        np.testing.assert_array_equal(ck['state_dict']['scale'], np.maximum(train[nonanchor].std(0), .001))
        model = CorridorResidual()
        model.load_state_dict(ck['state_dict'])
        with torch.inference_mode():
            score = model(torch.tensor(x['base'].astype(np.float32)),
                          torch.tensor(log_odds([p['baseline_probability'] for p in pred]))).numpy()
        np.testing.assert_array_equal(score, [p['scores'][arm] for p in pred])
        assert [bool(s >= tau) for s in score] == [p['flags'][arm] for p in pred]
    cap = WORK/'corridor-public-single-20260917/source/returned-v1/capture-v1'
    receipt = read(cap/'receipt.json')
    assert sha(cap/'evaluator.jsonl') == receipt['hashes']['evaluator.jsonl']
    native = [json.loads(s) for s in (cap/'evaluator.jsonl').read_text().splitlines()]
    raw = [json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
    for row, case, ev in zip(raw, cases, native):
        assert row['id'] == case['id'] == ev['id']
        g = geometry(ev)
        assert g['strict'] == case['truth'] and classify(g, .05) == case['stratum']
        lab = make_witness_labels(row, ev)
        assert bool(((lab['target'][:128] > 0) & lab['known'][:128]).any()) == case['sampled_witness']
    for arm, result in report['methods'].items():
        for stratum in ('clear', 'strict', 'boundary'):
            ii = [i for i, c in enumerate(cases) if stratum == 'strict' or (c['stratum'] == 'boundary') == (stratum == 'boundary')]
            counts = dict(TP=0, FP=0, FN=0, TN=0)
            for i in ii:
                truth, flag = cases[i]['truth'], pred[i]['flags'][arm]
                counts[('T' if truth == flag else 'F')+('P' if flag else 'N')] += 1
            assert all(result['metrics'][stratum][key] == value for key, value in counts.items())
        for kind in ('core', 'strict_events'):
            for event in result[kind]['events']:
                ii = [i for i, c in enumerate(cases) if c['episode_id'] == event['episode']
                      and event['start_s'] <= c['time_s'] <= event['end_last_sample_s']]
                hit = [cases[i]['time_s'] for i in ii if pred[i]['flags'][arm]]
                assert event['detected_in_core'] == bool(hit)
                assert event['first_in_core_delay_s'] == (min(hit)-event['start_s'] if hit else None)
    write(OUT/'audit.json', dict(status='PASS', frames=288, rank_pairs=672, folds=6,
        checks=['input hashes', 'pair and nested scene isolation', 'fit-only normalization',
                'last checkpoint and threshold selection', 'bitwise CPU weight replay',
                'native truth and support parity', 'independent scalar counts and event onsets'],
        limits='Native geometry helper reused; not independent geometric implementation. No retraining or fresh evidence.',
        summary_sha256=sha(OUT/'summary.json')))
    print('PASS: seals, groups, normalization, thresholds, public replay, native accounting and events')


if __name__ == '__main__':
    with threadpool_limits(4):
        main()
