"""One terminal zone-local diagnosis from retained four-model scores, no fit."""
import json
from collections import defaultdict
import numpy as np
import ba_nfo_matched as m
from ba_nfo_conditional_diagnostic import curve, at_recall, ARMS

OUT = m.ROOT/'artifacts.local/work/ba-nfo-local-diagnostic-20260919'
CACHE = m.ROOT/'artifacts.local/work/ba-nfo-conditional-20260919'


def auc(c):
    # Ties advance both rates together: trapezoids give half-credit.
    return float(np.trapezoid(c['recall'], c['fpr']))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m.write(OUT/'protocol.json',dict(scope='Consumed Development diagnostic; evaluator-only per-zone thresholds',
        source_sha256=m.sha(CACHE/'subgroup-scores.npz'),code_sha256=m.sha(__file__),
        question='Is original NFO locally separable at useful recall, before any cross-zone readout?',
        selection='Original 2m finite far returns and 0<near/known<=.2; exact retained subgroup',
        target_recall=.8,local_good='AUC>=.8 and FPR at recall>=.8 <=.2',
        readout_route='More than half zones locally good AND negative-pixel-weighted local FPR<=.2',
        otherwise='One fine-scale public-ToF conditioning contrast; no Hybrid training',
        interpretation='Pragmatic route selection, not a validated statistical gate; GT-based thresholds never deploy',
        backend='numpy-cpu',placement='TASK_NOT_GPU_SUITABLE: small stored zone vectors'))
    with np.load(CACHE/'subgroup-scores.npz') as a:
        v={k:a[k] for k in a.files}
    rows=[r for r in json.loads((m.OUT/'manifest.json').read_text()) if r['split']=='test']
    records=[]
    for fi in np.unique(v['frame']):
        r=rows[int(fi)];path=m.OLD/r['prepared']
        assert m.sha(path)==r['sha256']
        with np.load(path) as a:
            known=np.isfinite(a['depth'])&(a['depth']>0)
            truth=known&(a['depth']<2)
            mask=np.zeros_like(known);zones=np.full(known.shape,-1,np.int16)
            for zi,(y0,x0,y1,x1) in enumerate(a['boxes']):
                sl=np.s_[y0:y1,x0:x1];k=known[sl];t=truth[sl]
                if np.isfinite(a['values'][zi]) and a['values'][zi]>=2 and 0<t.sum()<=.2*k.sum():
                    mask[sl]=k;zones[sl]=zi
        frame=v['frame']==fi
        np.testing.assert_array_equal(truth[mask],v['truth'][frame])
        zi=zones[mask]
        scores={a:v[a][frame] for a in ARMS}
        for z in np.unique(zi):
            dm=zi==z;y=v['truth'][frame][dm]
            record=dict(id=r['id'],scene=r['scene'],zone=int(z),positive=int(y.sum()),negative=int((~y).sum()),arms={})
            for a in ARMS:
                c=curve(scores[a][dm],y);p=at_recall(c,.8)
                record['arms'][a]=dict(auc=auc(c),at_recall80=p,
                    good=auc(c)>=.8 and p['fpr']<=.2)
            records.append(record)
    assert len(records)==449
    summary={}
    for a in ARMS:
        aa=[r['arms'][a] for r in records]
        negative=sum(r['negative'] for r in records)
        fp=sum(x['at_recall80']['fp'] for x in aa)
        tp=sum(x['at_recall80']['recall']*r['positive'] for x,r in zip(aa,records))
        summary[a]=dict(zones=len(aa),mean_auc=float(np.mean([x['auc'] for x in aa])),
            median_auc=float(np.median([x['auc'] for x in aa])),
            pair_weighted_auc=float(np.average([x['auc'] for x in aa],weights=[r['positive']*r['negative'] for r in records])),
            good_zones=sum(x['good'] for x in aa),
            median_local_fpr_at80=float(np.median([x['at_recall80']['fpr'] for x in aa])),
            local_threshold_fp=int(fp),local_threshold_tp=int(round(tp)),local_threshold_fpr=fp/negative)
    n=summary['nfo']
    route='conditional_readout' if n['good_zones']>len(records)/2 and n['local_threshold_fpr']<=.2 else 'fine_scale_tof_conditioning'
    m.write(OUT/'results.json',dict(summary=summary,selected_route=route,records=records,
        caution='Each zone independently forced to 80% recall; oracle diagnostic, not one operating threshold or achievable router'))
    print(json.dumps(dict(summary=summary,selected_route=route),indent=2))


if __name__=='__main__':main()
