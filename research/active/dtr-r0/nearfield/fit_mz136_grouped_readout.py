"""Readout-only grouped diagnostics on consumed TRAIN; fixed backbone saw all TRAIN.

Compare original L2, stronger L2, and stronger L2 plus paired midpoint penalty.
Select by out-of-fold BCE before any new dev access. This is not independent CV
of the complete network and must never be represented as such.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz136_direct_readout import DirectReadoutNet
from repair_mz136_train_fit import ROOT, load_train, describe
from run_mz136_corridor_pair import predict
from run_mz120_occupancy import batch
from run_mz107_four_sensor import truth, sha, write
from research_backend import BackendCandidate, DeviceObservation, select_backend

RECIPES = [('plain', .0001, 0.), ('ridge', .01, 0.), ('midpoint', .01, .1)]


def midpoint_loss(scores, pairs):
    return ((scores[pairs[:,0]]+scores[pairs[:,1]])*.5).square().mean()


def grouped_folds(pairs):
    # Each fold excludes one full scene per family, both intervention members.
    folds=[]
    for fold in range(4):
        train=[p for p in pairs if int(p['scene_group'].rsplit('scene',1)[1]) != fold]
        held=[p for p in pairs if int(p['scene_group'].rsplit('scene',1)[1]) == fold]
        tr=sorted(i for p in train for i in (p['a'],p['b']))
        va=sorted(i for p in held for i in (p['a'],p['b']))
        assert not set(tr)&set(va)
        assert not {p['scene_group'] for p in train}&{p['scene_group'] for p in held}
        folds.append((tr,va,train))
    assert sorted(i for _,va,_ in folds for i in va)==list(range(192))
    return folds


def fit_linear(features, base, target, train_idx, pairs, l2, midpoint, device):
    ix=torch.tensor(train_idx,device=device)
    values=features.to(device)
    mu=values[ix].mean(0);scale=values[ix].std(0,unbiased=False).clamp_min(.001)
    x=((values-mu)/scale).detach()
    b,y=base.to(device),target.to(device)
    remap={old:new for new,old in enumerate(train_idx)}
    pp=torch.tensor([[remap[p['a']],remap[p['b']]] for p in pairs],device=device)
    assert all(target[p['a']] != target[p['b']] for p in pairs)
    head=torch.nn.Linear(features.shape[1],1,device=device)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=200,tolerance_grad=1e-7,
                               tolerance_change=1e-10,line_search_fn='strong_wolfe')
    started=time.perf_counter();evaluations=0
    def closure():
        nonlocal evaluations
        assert time.perf_counter()-started < 120
        optimizer.zero_grad(set_to_none=True)
        scores=b[ix]+head(x[ix]).squeeze(-1)
        loss=F.binary_cross_entropy_with_logits(scores,y[ix])+l2*head.weight.square().sum()
        if midpoint:loss=loss+midpoint*midpoint_loss(scores,pp)
        loss.backward();evaluations+=1
        return loss
    optimizer.step(closure)
    elapsed=time.perf_counter()-started
    with torch.no_grad():scores=(b+head(x).squeeze(-1)).cpu().numpy()
    return dict(scores=scores,mean=mu,scale=scale,head=head,seconds=elapsed,evaluations=evaluations)


def run(root,out):
    root,out=root.resolve(),out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(136014)
    assert torch.cuda.is_available()
    cap=root/'source/returned-v1/capture-v1'
    checkpoint=root/'full-fits-v1/bce-model.pt'
    prior=root/'train-direct-readout-v1'
    prior_freeze=json.loads((prior/'freeze.json').read_text())
    assert sha(checkpoint)==prior_freeze['checkpoint_sha256']
    files=[Path(__file__),Path(__file__).with_name('mz136_direct_readout.py'),
           Path(__file__).with_name('mz120_occupancy.py'),Path(__file__).with_name('repair_mz136_train_fit.py')]
    (out/'source-snapshot').mkdir()
    for f in files:shutil.copyfile(f,out/'source-snapshot'/f.name)
    source={n:sha(cap/n) for n in ('raw.jsonl','evaluator.jsonl','spec.json','receipt.json')}
    assert source==prior_freeze['source']
    freeze=dict(authority='CONSUMED_TRAIN_READOUT_ONLY_GROUPED_DIAGNOSTIC_BACKBONE_SAW_ALL192',
        recipes=[dict(name=n,l2=r,midpoint=g) for n,r,g in RECIPES],folds='scene0/1/2/3 across four families;48held144fit',
        preprocessing='fold training mean/std only; std floor .001',optimizer='LBFGS200 strong_wolfe',
        selection='minimum OOF BCE; tie prefers simpler listed recipe',
        proceed='selected OOF BCE <=90% plain, accuracy and both_rate >= plain; select one final refit only',
        source=source,checkpoint_sha256=sha(checkpoint),code={f.name:sha(f) for f in files},
        dev_outcomes_used_for_fitting_or_selection=False,test_outcomes_used_for_fitting_or_selection=False,
        loader_scope='complete manifest/evaluator parsed; TRAIN records selected before truth/feature fitting',
        feature_extraction_backend_sha256=sha(prior/'backend.json'))
    write(out/'freeze.json',freeze)
    rows,es,pairs,data,original=load_train(cap)
    gt=np.array([truth(e) for e in es],bool)
    model=DirectReadoutNet().cuda().eval()
    missing,extra=model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=False)
    assert not extra and set(missing)=={'feature_mean','feature_scale','readout.weight','readout.bias'}
    bb,vv=[],[];tick=time.perf_counter()
    with torch.no_grad():
        for start in range(0,192,8):
            b,v=model.components(batch(data,np.arange(start,start+8)));bb.append(b.cpu());vv.append(v.cpu())
    base,features=torch.cat(bb),torch.cat(vv)
    extraction_seconds=time.perf_counter()-tick
    np.testing.assert_allclose(base.numpy(),np.load(root/'full-fits-v1/bce-scores.npy')[original],atol=1e-5,rtol=1e-5)
    np.savez_compressed(out/'train-features.npz',features=features.numpy(),base=base.numpy(),target=gt)
    target=torch.tensor(gt,dtype=torch.float32)
    folds=grouped_folds(pairs)
    write(out/'folds.json',[dict(train=[rows[i]['id'] for i in tr],held=[rows[i]['id'] for i in va]) for tr,va,_ in folds])
    tr,_,pp=folds[0]
    def probe(device):return fit_linear(features,base,target,tr,pp,.01,.1,device)['head'].weight
    backend=select_backend('batch-tensor',
        cpu=BackendCandidate('torch-cpu','cpu',lambda:probe('cpu'),lambda v:DeviceObservation(v.device.type,'host CPU','torch '+torch.__version__)),
        gpu=BackendCandidate('torch-cuda','cuda',lambda:probe('cuda'),lambda v:DeviceObservation(v.device.type,torch.cuda.get_device_name(),'torch '+torch.__version__),torch.cuda.synchronize),
        record_path=out/'optimizer-backend.json',warmups=0,repeats=1)
    device=backend['selected_device_type'];results={}
    for name,l2,midpoint in RECIPES:
        oof=np.full(192,np.nan);details=[]
        for fold,(tr,va,pp) in enumerate(folds):
            fitted=fit_linear(features,base,target,tr,pp,l2,midpoint,device)
            oof[va]=fitted['scores'][va]
            details.append(dict(fold=fold,seconds=fitted['seconds'],evaluations=fitted['evaluations']))
        assert np.isfinite(oof).all()
        np.save(out/(name+'-oof-scores.npy'),oof)
        report=describe(gt,oof,pairs)
        report['mean_abs_pair_midpoint']=float(np.mean([abs((oof[p['a']]+oof[p['b']])*.5) for p in pairs]))
        report['fold_runs']=details;results[name]=report
        write(out/'progress.json',results)
        print(json.dumps(dict(recipe=name,bce=report['bce'],accuracy=report['accuracy'],both=report['both_correct'],offset=report['mean_abs_pair_midpoint'])),flush=True)
    selected=min(results,key=lambda n:results[n]['bce'])
    winner,plain=results[selected],results['plain']
    passes=bool(winner['bce']<=.9*plain['bce'] and winner['accuracy']>=plain['accuracy'] and winner['both_rate']>=plain['both_rate'])
    summary=dict(oof=results,selected=selected,proceed=passes,feature_extraction_seconds=extraction_seconds,
                 optimizer_device=device,limits=freeze['authority'])
    write(out/'selection.json',summary)
    if passes:
        _,l2,midpoint=next(r for r in RECIPES if r[0]==selected)
        final=fit_linear(features,base,target,list(range(192)),pairs,l2,midpoint,device)
        model.feature_mean.copy_(final['mean']);model.feature_scale.copy_(final['scale'])
        model.readout.load_state_dict(final['head'].state_dict())
        dest=out/'selected-fit';dest.mkdir()
        torch.save(model.state_dict(),dest/'model.pt')
        replay=predict(model,data)
        np.testing.assert_allclose(replay,final['scores'],atol=2e-4,rtol=1e-4)
        np.save(dest/'after-scores.npy',replay)
        write(dest/'freeze.json',dict(ordered=False,source=source,code=freeze['code'],
            selected_recipe=selected,l2=l2,midpoint=midpoint,selection_sha256=sha(out/'selection.json'),
            parent_freeze_sha256=sha(out/'freeze.json'),dev_used=False,test_used=False))
        shutil.copyfile(prior/'backend.json',dest/'backend.json')
        write(dest/'summary.json',dict(train=describe(gt,replay,pairs),seconds=final['seconds'],
            inference_replay_agrees=True,model_sha256=sha(dest/'model.pt')))
        write(dest/'completion.json',dict(status='PASS',hashes={f.name:sha(f) for f in dest.iterdir() if f.is_file()}))
        summary['final_train']=describe(gt,replay,pairs)
    write(out/'summary.json',summary)
    write(out/'completion.json',dict(status='PASS',hashes={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
