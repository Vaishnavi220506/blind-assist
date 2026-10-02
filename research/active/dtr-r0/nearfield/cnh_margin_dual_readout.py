"""One fixed ECDF-max NEAR/M3 readout, calibrated on all-object clear.

Consumed Development only. Cached scores; no network inference or rendering.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_all_object_truth_audit as A
import cnh_margin_confirm_evaluate as E

OUT = A.MC.SS.WORK / 'cnh-margin-dual-readout-20261002'
ARMS = ('NEAR', 'M3', 'DUAL')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def threshold(values):
    """Exclude the whole boundary tie; guarantee <= floor(10% n) alarms."""
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) < 10 or not np.isfinite(values).all():
        raise ValueError('Insufficient or invalid calibration clear rows')
    k = int(np.floor(.10*len(values)))
    t = float(np.nextafter(values[len(values)-k-1], np.inf))
    assert int((values >= t).sum()) <= k
    return t


def values(rows, scores, arm):
    return np.asarray([scores[r['dataset']][arm][str(r['unit'])]
                       [r['config']-(22 if r['dataset']=='context' else 0), r['query']] for r in rows])


def calibrate(ledger, scores):
    rows = [r for r in ledger if r['dataset']=='source' and r['split']=='calib']
    mapped = {d:{a:{u:v.copy() for u,v in scores[d][a].items()} for a in ('NEAR','M3')}
              for d in ('source','context')}
    refs = {}
    for q in (0,1):
        keep = np.asarray([r['query']==q and r['all_category']=='clear' for r in rows])
        refs[q] = {a:np.sort(values(rows,scores,a)[keep]) for a in ('NEAR','M3')}
    for dataset in mapped:
        mapped[dataset]['DUAL'] = {}
        for u in mapped[dataset]['NEAR']:
            parts = []
            for a in ('NEAR','M3'):
                x = mapped[dataset][a][u]
                parts.append(np.column_stack([np.searchsorted(refs[q][a],x[:,q],side='right')/len(refs[q][a]) for q in (0,1)]))
            mapped[dataset]['DUAL'][u] = np.maximum(parts[0],parts[1])
    thresholds, counts = {}, {}
    for a in ARMS:
        thresholds[a], counts[a] = {}, {}
        for q in (0,1):
            keep = np.asarray([r['query']==q and r['all_category']=='clear' for r in rows])
            x = values(rows,mapped,a)[keep]
            t = threshold(x)
            thresholds[a][str(q)] = t
            counts[a][str(q)] = dict(clear_rows=len(x), alarms=int((x>=t).sum()), rate=float((x>=t).mean()))
    return mapped, thresholds, counts, {str(q):{a:v.tolist() for a,v in refs[q].items()} for q in refs}


def statistics(rows, scores, thresholds, select, units, seed):
    ui = np.asarray([{u:i for i,u in enumerate(units)}[r['unit']] for r in rows])
    boot = A.boot_weights(len(units),seed)
    masks = A.masks(rows)
    masks = {k:masks[k] for k in (*A.CONTACTS,'new_pass','new_clear')}
    alarms = {a:values(rows,scores,a)>=np.asarray([thresholds[a][str(r['query'])] for r in rows]) for a in ARMS}
    stats, brates = {}, {}
    for a in ARMS:
        stats[a], brates[a] = {}, {}
        for k,mask in masks.items():
            stats[a][k],brates[a][k] = E.summarize(mask&select,alarms[a],ui,boot)
    comparisons = {}
    for reference in ('NEAR','M3'):
        comparisons[reference] = {}
        for k,mask in masks.items():
            a,b = stats['DUAL'][k]['rate'],stats[reference][k]['rate']
            chosen = mask&select
            comparisons[reference][k] = dict(delta=None if a is None or b is None else a-b,
                ci95=E.ci95(brates['DUAL'][k]-brates[reference][k]),
                rescued=int((chosen&alarms['DUAL']&~alarms[reference]).sum()),
                lost=int((chosen&~alarms['DUAL']&alarms[reference]).sum()))
    return dict(statistics=stats, comparisons=comparisons)


def old_diagnostic(ledger, scores, thresholds):
    rows = [r for r in ledger if r['dataset']=='context' and r['family']=='low_beam']
    masks = A.masks(rows)
    alarms = {a:values(rows,scores,a)>=np.asarray([thresholds[a][str(r['query'])] for r in rows]) for a in ('NEAR','M3')}
    result = {}
    for k in (*A.CONTACTS,'new_clear','new_pass'):
        m = masks[k]
        result[k] = dict(denominator=int(m.sum()), both=int((m&alarms['NEAR']&alarms['M3']).sum()),
            neither=int((m&~alarms['NEAR']&~alarms['M3']).sum()),
            near_only=int((m&alarms['NEAR']&~alarms['M3']).sum()),
            m3_only=int((m&~alarms['NEAR']&alarms['M3']).sum()))
    raw, hashes = {}, {}
    for a in ('NEAR','M3'):
        path = A.CONTEXT/f'frame_scores_{a}.npz'
        marker = A.read(A.CONTEXT/f'inference_{a}.json')
        hashes[a] = E.sha(path)
        assert hashes[a] == marker['raw_sha256']
        with np.load(path,allow_pickle=False) as z:
            raw[a] = {k:z[k] for k in z.files}
        for u,x in raw[a].items():
            expected=(x.astype(float)*np.asarray([1,2,4,8,16])[None,:,None]).sum(1)/31
            np.testing.assert_array_equal(expected,scores['context'][a][u])
    failures=[]
    for i,r in enumerate(rows):
        if masks['contact_gt5cm'][i] and alarms['NEAR'][i]!=alarms['M3'][i]:
            entry={k:r[k] for k in ('unit','config','query','target_off','range')}
            entry['direction']='NEAR_only' if alarms['NEAR'][i] else 'M3_only'
            entry['score_excess']={a:float(values([r],scores,a)[0]-thresholds[a][str(r['query'])]) for a in ('NEAR','M3')}
            entry['last5_frame_excess']={a:(raw[a][str(r['unit'])][r['config']-22,:,r['query']]-thresholds[a][str(r['query'])]).tolist() for a in ('NEAR','M3')}
            failures.append(entry)
    return dict(family='low_beam', old_thresholds=thresholds, pairs=result, deep_disagreements=failures,
        raw_score_sha256=hashes, boundary='Threshold-excess diagnostics only; scores cannot prove absence of sensor information')


def run():
    if not (OUT/'PLAN.json').is_file(): raise RuntimeError('Plan required')
    if (OUT/'result.json').exists(): raise FileExistsError('Inspect completed result; do not rerun')
    ledger,scores,old_thresholds,_,_,provenance=A.input_data()
    mapped,thresholds,calibration,cdfs=calibrate(ledger,scores)
    result_scopes={}
    for dataset in ('source','context'):
        rows=[r for r in ledger if r['dataset']==dataset and r['split']=='evaluation']
        units=sorted({r['unit'] for r in rows})
        selections={'all':np.ones(len(rows),bool)}
        for f in sorted({r['family'] for r in rows}):
            selections['family:'+f]=np.asarray([r['family']==f for r in rows])
        if dataset=='source': selections['primary']=np.asarray([.6<=r['range']<2.1 for r in rows])
        for name,selected in selections.items():
            result_scopes[dataset+'|'+name]=statistics(rows,mapped,thresholds,selected,units,A.SEEDS[dataset])
    diff=lambda scope,cat:result_scopes[scope]['comparisons']['M3'][cat]
    checks=dict(context_deep_ci_positive=diff('context|all','contact_gt5cm')['ci95'][0]>0,
        source_deep_drop_le2pp=diff('source|primary','contact_gt5cm')['delta']>=-.02,
        source_shallow_drop_le2pp=diff('source|primary','contact_0_2cm')['delta']>=-.02,
        context_shallow_drop_le2pp=diff('context|all','contact_0_2cm')['delta']>=-.02,
        source_clear_increase_le2pp=diff('source|all','new_clear')['delta']<=.02,
        context_clear_increase_le2pp=diff('context|all','new_clear')['delta']<=.02)
    result=dict(status='COMPLETE',verdict='DUAL_SUPPORTED' if all(checks.values()) else 'NOT_SUPPORTED',
        checks=checks, thresholds=thresholds, calibration=calibration, scopes=result_scopes,
        old_low_beam_diagnostic=old_diagnostic(ledger,scores,old_thresholds),
        provenance=provenance, plan_sha256=E.sha(OUT/'PLAN.json'), evaluator_sha256=E.sha(__file__),
        role='Consumed Development, single fixed candidate; old verdicts unchanged',
        runtime='Requires both frozen five-seed ensembles; latency not benchmarked')
    assert A.input_data()[-1]==provenance
    save('calibration_cdfs.json',cdfs)
    for dataset in ('source','context'):
        np.savez_compressed(OUT/f'scores_DUAL_{dataset}.npz',**mapped[dataset]['DUAL'])
    save('result.json',result)
    print(result['verdict'],json.dumps(checks),flush=True)
    print('Original low_beam paired counts',json.dumps(result['old_low_beam_diagnostic']['pairs']),flush=True)


def check():
    for x in (np.arange(20),np.ones(20),np.r_[np.zeros(18),np.ones(2)],np.r_[np.zeros(17),np.ones(3)]):
        assert (x>=threshold(x)).sum()<=int(len(x)*.1)
    ref=np.asarray([0.,1.,1.,3.])
    np.testing.assert_array_equal(np.searchsorted(ref,[-1.,0.,1.,2.,3.,4.],side='right')/len(ref),[0,.25,.75,.75,1,1])
    print('PASS ECDF right ties/out-of-range and conservative 10% threshold')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('check','run'));args=p.parse_args()
    check() if args.stage=='check' else run()
