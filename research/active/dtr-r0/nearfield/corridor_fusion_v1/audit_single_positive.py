"""Independent full-head/pooled-threshold and saved-confirmation accounting."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-single-20260917'
PREP=ROOT/'artifacts.local/work/corridor-public-positive-v2-20260917/preparation'
OOF=ROOT/'artifacts.local/work/corridor-public-positive-v2-20260917/bce'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def metric(y,p):return dict(TP=int(sum(y&p)),FP=int(sum(~y&p)),FN=int(sum(y&~p)),TN=int(sum(~y&~p)),frames=len(y))


def model_audit():
    torch.set_num_threads(4)
    head=HOME/'head';done=read(head/'completion.json')
    assert done['status']=='PASS'
    for n,h in done['outputs'].items():assert sha(head/n)==h,n
    freeze=read(head/'fit-freeze.json')
    for p,h in freeze['sources'].items():assert sha(p)==h,p
    d=np.load(PREP/'public-tokens.npz');m=read(PREP/'metadata.json')
    token=d['tokens'][:,:128];valid=d['valid'][:,:128]
    u=np.array([4,2,3],np.float32);lo=np.array([.2,-.3,.4],np.float32);hi=np.array([3.6,.3,2.05],np.float32)
    center=token[:,:,5:8]*u;box=token[:,:,8:14].reshape(1344,128,3,2)*u[:,None]
    x=np.concatenate([token,np.minimum(center-lo,hi-center),np.minimum(box[:,:,:,1]-lo,hi-box[:,:,:,0]),
        np.minimum(box[:,:,:,0]-lo,hi-box[:,:,:,1])],-1)
    state=torch.load(head/'model.pt',weights_only=True,map_location='cpu')['state_dict']
    np.testing.assert_array_equal(state['mean'],x[valid].mean(0))
    np.testing.assert_array_equal(state['scale'],x[valid].std(0).clip(.01))
    training=read(head/'training.json');rng=np.random.default_rng(185017)
    assert training['fit_indices']==list(range(1344)) and len(training['orders'])==120
    for order in training['orders']:assert order==rng.permutation(1344).tolist()
    pred=np.load(OOF/'predictions.npz');ii=pred['indices']
    a=np.array([m[i]['A'] for i in ii]);truth=np.array([m[i]['truth'] for i in ii]);clear=np.array([m[i]['stratum']!='boundary' for i in ii])
    v=valid[ii];score=np.where(v,pred['logits'],-30.).max(1).astype(np.float64)
    curve=[]
    for tau in np.r_[np.nextafter(score.max(),np.inf),np.unique(score[v.any(1)&clear])]:
        p=(a|(v.any(1)&(score>=tau)))[clear];y=truth[clear]
        z=metric(y,p);curve.append(dict(threshold=float(tau),TP=z['TP'],FP=z['FP'],FN=z['FN'],f1=2*z['TP']/max(1,2*z['TP']+z['FP']+z['FN'])))
    selected=max(curve,key=lambda q:(q['f1'],-q['FP'],q['threshold']))
    sel=read(head/'selection.json');assert sel['curve']==curve and sel['chosen']==selected
    config=read(HOME/'bundle/config.json');assert config['positive_threshold']==selected['threshold']
    for n,h in config['model_hashes'].items():assert sha(HOME/'bundle'/n)==h
    recipe=read(HOME/'recipe-freeze.json')
    for p,h in recipe['sources'].items():assert sha(p)==h,p
    for p,h in recipe['bundle_files'].items():assert sha(p)==h,p
    assert read(HOME/'a-control/verification.json')['status']=='PASS'
    write(HOME/'model-audit.json',dict(status='PASS',fit1344_only=True,all120orders_exact=True,
        full_fit_normalization_bitwise=True,OOF_selection_exact=True,threshold=selected['threshold'],
        one_head_no_fold_selection=True,recipe_sha256=sha(HOME/'recipe-freeze.json')))


def confirmation_audit(cap):
    out=HOME/'confirmation';done=read(out/'completion.json');assert done['status']=='PASS'
    assert sha(out/'summary.json')==done['summary_sha256']
    ps=read(out/'prediction-seal.json');assert sha(out/'predictions.json')==ps['predictions_sha256']
    cases=read(out/'cases.json');pred=read(out/'predictions.json');summary=read(out/'summary.json')
    assert len(cases)==len(pred)==288
    a=np.array([p['A'] for p in pred]);y=np.array([c['truth'] for c in cases]);clear=np.array([c['stratum']!='boundary' for c in cases])
    config=read(HOME/'bundle/config.json')
    for p in pred:
        assert p['A']==(p['A_score']>=config['A_threshold'])
        assert p['positive']==(p['usable_tof_returns']>0 and p['positive_score']>=config['positive_threshold'])
        assert p['alert']==(p['A'] or p['positive'])
        assert p['control']==(p['control_score']>=config['control_threshold'])
    results={}
    for name,key in [('A','A'),('A_plus_public','alert'),('A_retrained','control')]:
        p=np.array([r[key] for r in pred]);got=summary['methods'][name]
        results[name]={}
        for tag,mask in [('clear',clear),('strict',np.ones(288,bool)),('boundary',~clear)]:
            counts=metric(y[mask],p[mask]);results[name][tag]=counts
            assert all(got[tag][k]==v for k,v in counts.items())
        for event in got['strict_temporal']['events']:
            ii=[i for i,c in enumerate(cases) if c['episode_id']==event['episode'] and y[i]
                and event['start_s']<=c['time_s']<=event['end_last_sample_s']]
            hits=[i for i in ii if p[i]]
            delay=cases[hits[0]]['time_s']-event['start_s'] if hits else None
            assert event['detected_in_core']==bool(hits) and event['first_in_core_delay_s']==delay
    assert all(not r['A'] or r['alert'] for r in pred)
    assert all(r['alert']==r['A'] for r in pred if r['usable_tof_returns']==0)
    write(out/'independent-accounting.json',dict(status='PASS',counts=results,
        threshold_logic_exact=True,A_retained=True,unknown_fallback=True,strict_onset_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path);args=p.parse_args()
    model_audit()
    if args.capture:confirmation_audit(args.capture)
    print('PASS: single model audit'+(' and confirmation accounting' if args.capture else ''))
