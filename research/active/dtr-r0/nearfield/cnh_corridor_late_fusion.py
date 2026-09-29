"""Frozen C1: integer empirical-CDF max fusion on existing predictions only."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cnh_corridor_statistics import paired_cluster_ber

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-corridor-late-fusion-20260929'
FAMILIES = ('boundary', 'mixed_surface', 'sidewall', 'general')
ARMS = ('LOFO', 'MOTION8_PARTIAL', 'FUSION', 'EQUAL_COUNT')


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ranks(reference, scores):
    reference, scores = np.asarray(reference, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    assert np.isfinite(reference).all() and np.isfinite(scores).all()
    return np.searchsorted(np.sort(reference), scores, side='right').astype(np.int64)


def threshold(r, n):
    r = np.asarray(r, dtype=np.int64)
    assert len(r) == n and n > 0
    # Lowest integer threshold meeting the exact rational 1/10 budget.
    for k in range(n+2):
        if 10 * int((r >= k).sum()) <= n:
            return k
    raise AssertionError('unreachable')


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    fn, fp = int((y & ~p).sum()), int((~y & p).sum())
    return dict(positive=pos, negative=neg, fn=fn, fp=fp,
                ber=.5*(fn/pos+fp/neg) if pos and neg else None)


def main():
    assert (OUT/'C1_PLAN.md').exists() and not (OUT/'results.json').exists()
    manifest = {}

    def read(p):
        manifest[str(p.relative_to(ROOT))] = sha(p)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    for p in [OUT/'C1_PLAN.md', Path(__file__), Path(__file__).with_name('cnh_corridor_statistics.py')]:
        manifest[str(p.relative_to(ROOT))] = sha(p)
    ledger, calibration, comparisons, depth = [], [], [], []
    for family in FAMILIES:
        sets = {}
        for split, ids in [('calib', range(2000,2024)), ('evaluation', range(3000,3048))]:
            rows = []
            for u in ids:
                b = read(WORK/'cnh-corridor-retrain-20260929/predictions'/f'unit{u}.npz')
                l = read(WORK/'cnh-corridor-lofo-20260929'/family/'predictions'/f'unit{u}.npz')
                e = read(WORK/'cnh-corridor-equal-count-20260929'/family/'predictions'/f'unit{u}.npz')
                ix = l['config']
                assert np.array_equal(ix, e['config']) and int(l['unit']) == int(e['unit']) == u
                for key in ['labels','family','margin','group']:
                    assert np.array_equal(l[key], e[key]) and np.array_equal(l[key], b[key][ix])
                assert ((l['family'] != family).all() if split == 'calib' else (l['family'] == family).all())
                for i, config in enumerate(ix):
                    for q, group in enumerate(['HEAD','BODY']):
                        rows.append(dict(unit=u, config=int(config), group=group, family=family,
                            label=int(l['labels'][i,q]), target_group=int(l['group'][i]), margin=float(l['margin'][i]),
                            scores=dict(LOFO=float(l['lofo'][i,q]), MOTION8_PARTIAL=float(b['scores'][config,4,q]),
                                        EQUAL_COUNT=float(e['equal_count'][i,q]))))
            sets[split] = rows
        for q, group in enumerate(['HEAD','BODY']):
            cal = [r for r in sets['calib'] if r['group']==group and r['label']==0]
            ev = [r for r in sets['evaluation'] if r['group']==group]
            n = len(cal)
            cr, er = {}, {}
            for arm in ['LOFO','MOTION8_PARTIAL','EQUAL_COUNT']:
                ref = [r['scores'][arm] for r in cal]
                cr[arm] = ranks(ref, ref)
                er[arm] = ranks(ref, [r['scores'][arm] for r in ev])
            cr['FUSION'] = np.maximum(cr['LOFO'], cr['MOTION8_PARTIAL'])
            er['FUSION'] = np.maximum(er['LOFO'], er['MOTION8_PARTIAL'])
            pred = {}
            for arm in ARMS:
                k = threshold(cr[arm], n)
                fp = int((cr[arm]>=k).sum())
                assert 10*fp<=n and (k==0 or 10*int((cr[arm]>=k-1).sum())>n)
                calibration.append(dict(family=family,group=group,arm=arm,rank_threshold=k,negative=n,fp=fp,fpr=fp/n))
                pred[arm] = er[arm]>=k
            y = np.array([r['label'] for r in ev])
            units = np.array([r['unit'] for r in ev])
            assert len(np.unique(units)) == 48
            m = {a:metrics(y,pred[a]) for a in ARMS}
            boot = paired_cluster_ber(y,pred['LOFO'],pred['FUSION'],units)
            gap = m['LOFO']['ber']-m['EQUAL_COUNT']['ber']
            closure = (m['LOFO']['ber']-m['FUSION']['ber'])/gap if gap>0 else None
            gate = bool(gap>0 and closure>=.5 and boot['ci'][1]<0) if family in ('sidewall','mixed_surface') else bool(boot['point']<=.02)
            comparisons.append(dict(family=family,group=group,metrics=m,bootstrap=boot,gap=gap,gap_closure=closure,passed=gate))
            for i,r in enumerate(ev):
                r['predictions'] = {a:int(pred[a][i]) for a in ARMS}
                r['rank_numerators'] = {a:int(er[a][i]) for a in ARMS}
                r['rank_denominator'] = n
                r['role'] = 'positive_intrusion' if r['label'] else ('target_group_outside' if r['target_group']==q else 'other_height_negative')
                r['band'] = '<=5cm' if abs(r['margin'])<=.05 else ('5-15cm' if abs(r['margin'])<=.15 else '>15cm')
                ledger.append(r)
            for role in ['positive_intrusion','target_group_outside','other_height_negative']:
                for band in ['<=5cm','5-15cm','>15cm','all']:
                    rr = [r for r in ev if r['role']==role and (band=='all' or r['band']==band)]
                    for arm in ARMS:
                        depth.append(dict(family=family,group=group,role=role,band=band,arm=arm,
                                          **metrics([r['label'] for r in rr],[r['predictions'][arm] for r in rr])))
    assert len(ledger)==len({(r['unit'],r['config'],r['group']) for r in ledger})==2112
    save('input_source_hashes.json',manifest)
    save('calibration.json',calibration)
    passed = all(c['passed'] for c in comparisons)
    save('results.json',dict(c1_pass=passed,c2_authorized_by_gate=passed,comparisons=comparisons,deviations=[]))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf8')
    with (OUT/'depth_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(depth[0]));w.writeheader();w.writerows(depth)
    print(json.dumps(dict(c1_pass=passed,comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()
