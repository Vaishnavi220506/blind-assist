"""One fixed HGB structure, grouped OOF threshold, one full-data A refit."""
import pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from prepare_single_a import ROOT,WORK,OUT,read,write,sha

PARAMS=dict(max_leaf_nodes=7,max_iter=150,l2_regularization=1.,learning_rate=.05,
            min_samples_leaf=8,early_stopping=False,random_state=185017)
def metric(y,p):return dict(TP=int(sum(y&p)),FP=int(sum(~y&p)),FN=int(sum(y&~p)),TN=int(sum(~y&~p)),frames=len(y))
def main():
    assert not (OUT/'freeze.json').exists();start=time.perf_counter()
    ds=read(OUT/'data-seal.json')
    for p,h in ds['bindings'].items():assert sha(p)==h,p
    for n,h in ds['outputs'].items():assert sha(OUT/n)==h,n
    source=WORK/'corridor-depth-e1-20260917/run-v2'
    old=pickle.loads((source/'A-model.pkl').read_bytes());sel=read(source/'selection-seal.json')
    assert sha(source/'A-model.pkl')==sel['models']['A']=='d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef'
    for key,val in PARAMS.items():
        if key!='random_state':assert old.get_params()[key]==val
    assert old.get_params()['class_weight'] is None
    meta=read(OUT/'metadata.json');z=np.load(OUT/'features.npz');x=z['base'];y=np.array([m['truth'] for m in meta],bool)
    assert list(z['ids'])==[m['id'] for m in meta]
    folds=[];report_all=[i for i,m in enumerate(meta) if m['cohort']!='anchor']
    for k in range(6):
        fi=[i for i,m in enumerate(meta) if m['cohort']=='anchor' or m['scene_index']!=k]
        ri=[i for i,m in enumerate(meta) if m['cohort']!='anchor' and m['scene_index']==k]
        assert len(fi)==1152 and len(ri)==192
        assert not {meta[i]['group'] for i in fi}&{meta[i]['group'] for i in ri}
        folds.append(dict(fold=k,fit=fi,report=ri))
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('prepare_single_a.py'),source/'selection-seal.json',source/'A-model.pkl']}
    write(OUT/'freeze.json',dict(parameters=PARAMS,original_random_state=177017,random_state_change='USER_FIXED_SEED185017',
        strict_frame_labels=True,class_weight=None,sample_weight=None,folds=folds,data_seal_sha256=sha(OUT/'data-seal.json'),sources=sources,
        threshold_selection='ALL1152_GROUP_OOF_CLEAR_F1_FEWER_FP_HIGHER_THRESHOLD',backend='CPU_SKLEARN_HGB_NO_GPU_BACKEND',
        authority='SAME1344DATA_FROZEN_STRUCTURE_RETRAINED_A_CONTROL_NO_SWEEP'))
    scores=np.full(1344,np.nan);seen=np.zeros(1344,int)
    for f in folds:
        tick=time.perf_counter();m=HistGradientBoostingClassifier(**PARAMS);fi,ri=f['fit'],f['report']
        m.fit(x[fi],y[fi]);scores[ri]=m.predict_proba(x[ri])[:,1];seen[ri]+=1
        (OUT/f"fold{f['fold']}.pkl").write_bytes(pickle.dumps(m))
        write(OUT/f"fold{f['fold']}-fit.json",dict(**f,seconds=time.perf_counter()-tick,model_sha256=sha(OUT/f"fold{f['fold']}.pkl")))
        print(f"fold {f['fold']} complete",flush=True)
    assert (seen[report_all]==1).all() and np.isfinite(scores[report_all]).all()
    ii=np.array(report_all);clear=np.array([meta[i]['stratum']!='boundary' for i in ii]);ss=scores[ii];yy=y[ii]
    thresholds=np.r_[np.nextafter(float(ss.max()),np.inf),np.unique(ss[clear])];curve=[]
    for t in thresholds:
        metrics=metric(yy[clear],ss[clear]>=t);curve.append(dict(threshold=float(t),**metrics,f1=2*metrics['TP']/max(1,2*metrics['TP']+metrics['FP']+metrics['FN'])))
    chosen=max(curve,key=lambda v:(v['f1'],-v['FP'],v['threshold']))
    np.savez_compressed(OUT/'oof-scores.npz',indices=ii,ids=z['ids'][ii],scores=ss)
    write(OUT/'selection.json',dict(chosen=chosen,curve=curve,oof_sha256=sha(OUT/'oof-scores.npz'),
        warning='Pooled OOF threshold calibration is Development selection, not independent validation'))
    final=HistGradientBoostingClassifier(**PARAMS);final.fit(x,y)
    (OUT/'retrainedA.pkl').write_bytes(pickle.dumps(final))
    write(OUT/'threshold.json',dict(threshold=chosen['threshold'],comparison='score>=threshold',model_sha256=sha(OUT/'retrainedA.pkl'),
        features=2485,rows=1344,selection_sha256=sha(OUT/'selection.json')))
    report={}
    for c in sorted({m['cohort'] for m in meta if m['cohort']!='anchor'}):
        mask=np.array([meta[i]['cohort']==c for i in ii]);cc=mask&clear
        report[c]=dict(clear=metric(yy[cc],ss[cc]>=chosen['threshold']),strict=metric(yy[mask],ss[mask]>=chosen['threshold']),
            boundary=metric(yy[mask&~clear],ss[mask&~clear]>=chosen['threshold']))
    assert all(sha(p)==h for p,h in sources.items())
    write(OUT/'completion.json',dict(status='PASS',model_sha256=sha(OUT/'retrainedA.pkl'),threshold_sha256=sha(OUT/'threshold.json'),
        freeze_sha256=sha(OUT/'freeze.json'),selection_sha256=sha(OUT/'selection.json'),oof_metrics=report,
        seconds=time.perf_counter()-start,resources='No persistent process; CPU HGB',final_deliverable=['retrainedA.pkl','threshold.json']))
    print(chosen)
if __name__=='__main__':
    with threadpool_limits(4):main()
