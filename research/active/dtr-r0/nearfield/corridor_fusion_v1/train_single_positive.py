"""Fit exactly one full-data public head, select one pooled scene-OOF threshold."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from prepare_public_positive import ROOT,HERE,WORK,read,write,sha
from public_positive import PositiveHead,spatial_features,select_threshold,add_positive
from public_positive_v2 import fitting_weights
from tolerance_eval import metric

HOME=WORK/'corridor-public-single-20260917'
PREP=WORK/'corridor-public-positive-v2-20260917/preparation'
OOF=WORK/'corridor-public-positive-v2-20260917/bce'
OUT=HOME/'head'
SEED=185017


def main():
    assert not OUT.exists(),'Preserve existing fit'
    OUT.mkdir(parents=True)
    started=time.perf_counter()
    torch.set_num_threads(4)
    ds=read(PREP/'data-seal.json')
    for n,h in ds['outputs'].items():assert sha(PREP/n)==h,n
    done=read(OOF/'completion.json')
    assert done['status']=='PASS' and sha(OOF/'prediction-seal.json')==done['prediction_seal_sha256']
    ps=read(OOF/'prediction-seal.json')
    assert sha(OOF/'predictions.npz')==ps['outputs']['predictions.npz']
    assert read(OOF/'independent-audit.json')['status']=='PASS'
    data,lab=np.load(PREP/'public-tokens.npz'),np.load(PREP/'offline-targets.npz')
    meta=read(PREP/'metadata.json');pred=np.load(OOF/'predictions.npz');ci=pred['indices']
    features=spatial_features(data['tokens']);valid=data['valid'][:,:128]
    target,known=lab['target'][:,:128],lab['known'][:,:128]
    assert len(meta)==1344 and np.array_equal(ci,np.arange(192,1344))
    a=np.array([meta[i]['A'] for i in ci],bool)
    truth=np.array([meta[i]['truth'] for i in ci],bool)
    clear=np.array([meta[i]['stratum']!='boundary' for i in ci])
    chosen,curve=select_threshold(a,pred['logits'],valid[ci],truth,clear)
    write(OUT/'selection.json',dict(chosen=chosen,curve=curve,calibration_indices=ci.tolist(),
        source_oof_sha256=sha(OOF/'predictions.npz'),authority='CONSUMED_POOLED_SCENE_OOF_SELECTION_NOT_NEW_CONFIRMATION'))
    src={str(p):sha(p) for p in [Path(__file__),HERE/'PUBLIC_SINGLE_PROTOCOL_20260917.md',
        HERE/'public_positive.py',HERE/'public_positive_v2.py',HERE/'public_return_tokens.py',HERE/'public_positive_inference.py']}
    write(OUT/'fit-freeze.json',dict(sources=src,data_seal_sha256=sha(PREP/'data-seal.json'),
        selection_sha256=sha(OUT/'selection.json'),seed=SEED,epochs=120,batch=16,
        parameters=1537,device='cpu',backend_reason='CPU_FASTER_MEASURED_UNCHANGED_HEAD_LOSS',
        inherited_backend=str(OOF/'backend.json'),inherited_backend_sha256=sha(OOF/'backend.json')))
    torch.manual_seed(SEED)
    fit_values=features[valid]
    model=PositiveHead(fit_values.mean(0),fit_values.std(0).clip(.01))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    weights,_=fitting_weights(meta,list(range(len(meta))),valid,target,known)
    x,v,y,w=[torch.tensor(q) for q in [features,valid,target,weights]]
    rng=np.random.default_rng(SEED);orders=[];history=[]
    for epoch in range(120):
        model.train();order=rng.permutation(len(meta));orders.append(order.tolist());total=0.
        for start in range(0,len(meta),16):
            ii=order[start:start+16];optimizer.zero_grad(set_to_none=True)
            z=model(x[ii],v[ii])
            loss=(F.binary_cross_entropy_with_logits(z,y[ii],reduction='none')*w[ii]).sum()*len(meta)/len(ii)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite BCE')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ii)/len(meta)
        history.append(dict(epoch=epoch+1,loss=total))
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=120,seed=SEED),OUT/'model.pt')
    write(OUT/'training.json',dict(orders=orders,history=history,fit_indices=list(range(len(meta)))))
    calibrated,_,_=add_positive(a,pred['logits'],valid[ci],chosen['threshold'])
    write(OUT/'summary.json',dict(threshold=chosen['threshold'],seconds=time.perf_counter()-started,
        OOF_A_clear=metric(truth[clear],a[clear]),OOF_OR_clear=metric(truth[clear],calibrated[clear]),
        OOF_A_strict=metric(truth,a),OOF_OR_strict=metric(truth,calibrated),
        OOF_authority='DEVELOPMENT_SELECTION_FIT_METRICS_NOT_CONFIRMATION',single_model=True))
    assert all(sha(Path(p))==h for p,h in src.items())
    write(OUT/'completion.json',dict(status='PASS',model_sha256=sha(OUT/'model.pt'),
        selection_sha256=sha(OUT/'selection.json'),fit_freeze_sha256=sha(OUT/'fit-freeze.json'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},resources='Process-local optimizer/model released on exit'))
    print(read(OUT/'summary.json'))


if __name__=='__main__':main()
