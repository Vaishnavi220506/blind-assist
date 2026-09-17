"""Replay fitted weights and independently recount events without training."""
from pathlib import Path
import pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_bg_invariance import OUT, REP, DATA, ARMS, RawResidual, read, write, sha, predict_head, log_odds
from audit_representation import counts, events
from fractions import Fraction


def main():
    seal=read(OUT/'model-seal.json');summary=read(OUT/'summary.json')
    for n,h in seal['outputs'].items():assert sha(OUT/n)==h
    inputs=np.load(OUT/'training-inputs.npz');meta=read(OUT/'training-metadata.json')
    oof=np.load(OUT/'oof-scores.npz');folds=read(OUT/'folds.json')
    oldmeta=read(DATA/'metadata.json');old_indices=[i for i,m in enumerate(oldmeta) if m['cohort']!='anchor']
    newmeta=read(OUT/'new-metadata.json');new=np.load(OUT/'new-features.npz')
    ni=[i for i,m in enumerate(newmeta) if m['split']=='train']
    dev=[i for i,m in enumerate(newmeta) if m['split']=='dev']
    assert len(ni)==192 and len(dev)==96
    assert not {m['group'] for m in meta}&{newmeta[i]['group'] for i in dev}
    replay_counts=0
    for f in folds:
        fi,ri=np.array(f['fit']),np.array(f['report'])
        assert not {meta[i]['group'] for i in fi}&{meta[i]['group'] for i in ri}
        k=f['fold'];inner=np.load(OUT/f'outer{k}-inner.npz')
        oldmap={int(i):float(v) for i,v in zip(inner['indices'],inner['logits'])}
        hgb=pickle.loads((REP/f'raw-fold{k}.pkl').read_bytes())
        newbase=log_odds(hgb.predict_proba(new['raw'][ni])[:,1])
        b=np.array([inputs['base'][i] if i<1152 else newbase[i-1152] for i in ri],np.float32)
        for arm in ARMS:
            ck=torch.load(OUT/f'{arm}-fold{k}.pt',map_location='cpu',weights_only=False)
            assert len(ck['history'])==120 and ck['fit_rows']==len(fi)
            np.testing.assert_array_equal(ck['state_dict']['mean'],inputs['x'][fi].mean(0))
            np.testing.assert_array_equal(ck['state_dict']['scale'],np.maximum(inputs['x'][fi].std(0),.001))
            model=RawResidual(np.zeros(2224),np.ones(2224));model.load_state_dict(ck['state_dict']);model.eval()
            np.testing.assert_array_equal(predict_head(model,inputs['x'][ri],b),oof[arm][ri])
            replay_counts+=1
        for pairkey in ('intrusion','background'):
            local=np.array(f[pairkey],int)
            assert np.all(local>=0) and np.all(local<len(fi))
            for a,bidx in local:
                assert meta[fi[a]]['group']==meta[fi[bidx]]['group']
    y=inputs['y'].astype(bool);clear=np.array([m['stratum']!='boundary' for m in meta])
    for arm in ARMS:
        best=None
        for t in np.r_[np.nextafter(oof[arm].max(),np.inf),np.unique(oof[arm][clear])]:
            c=counts(y[clear],oof[arm][clear]>=t)
            key=(Fraction(2*c['TP'],2*c['TP']+c['FP']+c['FN']),-c['FP'],float(t))
            if best is None or key>best:best=key
        assert best[2]==seal['models'][arm]['threshold']
        ck=torch.load(OUT/seal['models'][arm]['path'],map_location='cpu',weights_only=False)
        assert sha(OUT/seal['models'][arm]['path'])==seal['models'][arm]['sha256']
        assert len(ck['history'])==120 and ck['fit_rows']==1344 and ck['background_pairs']==96
        np.testing.assert_array_equal(ck['state_dict']['mean'],inputs['x'].mean(0))
    extras={}
    for name in ('old','new'):
        cases=read(OUT/f'{name}-cases.json');pred=read(OUT/f'{name}-predictions.json')
        yy=np.array([c['truth'] for c in cases],bool);cc=np.array([c['stratum']!='boundary' for c in cases])
        native=np.array([c['sampled_witness'] for c in cases]);zero=np.array([c['usable_tof_returns']==0 for c in cases])
        masks=dict(clear=cc,strict=np.ones(len(cases),bool),boundary=~cc,native_supported=native,
            zero_return=zero,clear_native_supported=cc&native,clear_zero_return=cc&zero)
        extras[name]={}
        for arm,m in summary['methods'][name].items():
            pp=np.array([p['flags'][arm] for p in pred])
            extras[name][arm]={k:counts(yy[v],pp[v]) for k,v in masks.items()}
            for key,v in masks.items():
                if key in m['metrics']:
                    for field,num in extras[name][arm][key].items():assert m['metrics'][key][field]==num
            for key,core in [('core',True),('strict_events',False)]:
                ee=events(cases,pp,core)
                assert ee==[(e['episode'],e['start_s'],e['first_in_core_delay_s']) for e in m[key]['events']]
        x=np.load(REP/'report-features.npz')['raw'] if name=='old' else new['raw'][dev]
        b=log_odds(np.array([p['probabilities']['raw_hgb'] for p in pred]))
        for arm in ARMS:
            ck=torch.load(OUT/seal['models'][arm]['path'],map_location='cpu',weights_only=False)
            model=RawResidual(np.zeros(2224),np.ones(2224));model.load_state_dict(ck['state_dict']);model.eval()
            np.testing.assert_array_equal(predict_head(model,x,b),[p['scores'][arm] for p in pred])
    write(OUT/'audit.json',dict(status='PASS',oof_residual_replays=replay_counts,final_prediction_replays=6,
        threshold_reselections=3,independent_frames=384,fit_normalization_verified=True,group_pair_isolation=True,
        summaries=extras,summary_sha256=sha(OUT/'summary.json'),
        limitation='Uses stored native labels and torch forward; source sensor/geometry parity separately audited'))
    print('PASS:18OOF/6final replays, thresholds,384counts/events, normalization and group isolation')


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(4):main()
