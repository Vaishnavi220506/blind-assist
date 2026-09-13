"""Two matched fixed fits, common joint selection; transfer only after dev gain."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet
from mz122_local_fusion import LocalFusionNet,point_geometry
from mz121_joint_readout import select_joint,precision,recall
from run_mz120_occupancy import ROOT,SEED,load_source,labels,read_evaluation,split,batch,predict,evaluate
from run_mz107_four_sensor import sha,write
from research_backend import select_backend,BackendCandidate,DeviceObservation


def tensor_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def datasets(sources):
    for s in sources:
        s['inputs']['point_local']=np.stack([point_geometry(r,{k:v[i] for k,v in s['inputs'].items() if k!='rgb'})
                                            for i,r in enumerate(s['rows'])])
    return {k:torch.from_numpy(np.concatenate([s['inputs'][k] for s in sources])) for k in sources[0]['inputs']}


def curve(rows,es,y,p,indices,baseline):
    return [dict(threshold=float(t),**evaluate([rows[i] for i in indices],[es[i] for i in indices],
        y[indices],p[indices],float(t),baseline[indices])) for t in np.linspace(0,1,101)]


def envelopes(points):
    out={}
    for area in ('metrics','HEAD'):
        out[area]={}
        for target in (.8,.9,.95):
            values=[r for r in points if recall(r['spatial'][area]) is not None and
                    recall(r['spatial'][area])>=target and precision(r['spatial'][area]) is not None]
            best=max(values,key=lambda r:precision(r['spatial'][area]),default=None)
            out[area][str(target)]=None if best is None else dict(precision=precision(best['spatial'][area]),
                achieved_recall=recall(best['spatial'][area]),threshold=best['threshold'],role='CURVE_DIAGNOSTIC_NOT_SELECTED_READOUT')
    return out


def run(output):
    start=time.perf_counter();out=output.resolve();old=ROOT/'artifacts.local/work/mz120-learned-occupancy-20260913/pilot-v1'
    assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve())
    assert torch.cuda.is_available();out.mkdir(parents=True);torch.set_num_threads(4)
    frozen=json.loads((old/'freeze.json').read_text());oldsplit=json.loads((old/'split.json').read_text())
    assert sha(old/'split.json')==frozen['split_sha256']
    sources=[load_source(n,evaluation=True) for n in ('mz115','mz117')]
    for s in sources:
        assert s['hashes']==frozen['source'][s['name']]
        receipt=json.loads((s['capture']/'receipt.json').read_text())
        assert sha(s['capture']/'spec.json')==receipt['spec_sha256']
    rows=sum([s['rows'] for s in sources],[]);es=sum([s['es'] for s in sources],[])
    y=np.stack([labels(e) for e in es]);data=datasets(sources);tr,dv=split(sources)
    assert [rows[i]['id'] for i in tr]==oldsplit['train'] and [rows[i]['id'] for i in dv]==oldsplit['dev']
    assert np.array_equal(y,np.load(old/'development_predictions.npz')['labels'])
    baseline=np.concatenate([s['baseline'] for s in sources]);baseline_dev=json.loads((old/'summary.json').read_text())['development']['baseline']
    rng=np.random.default_rng(SEED);schedule=np.stack([rng.choice(tr,8,replace=True) for _ in range(800)])
    aug_rng=np.random.default_rng(SEED+122);augmentation=aug_rng.random((800,2,8,1,1,1),dtype=np.float32)
    np.savez_compressed(out/'schedule.npz',indices=schedule,augmentation=augmentation)
    torch.manual_seed(SEED);early=OccupancyNet();late=LocalFusionNet();late.load_state_dict(early.state_dict(),strict=False)
    common=tensor_digest(early.state_dict());assert common==tensor_digest({k:late.state_dict()[k] for k in early.state_dict()})
    torch.save(early.state_dict(),out/'early_initial.pt');torch.save(late.state_dict(),out/'late_initial.pt')
    names=('mz122_local_fusion.py','run_mz122_local_fusion.py','MZ122_PROTOCOL_20260913.md','mz120_occupancy.py',
           'run_mz120_occupancy.py','mz121_joint_readout.py')
    for name in names:shutil.copyfile(Path(__file__).with_name(name),out/name)
    write(out/'freeze.json',dict(seed=SEED,steps=800,batch=8,source={s['name']:s['hashes'] for s in sources},
        split_sha256=sha(old/'split.json'),schedule_sha256=sha(out/'schedule.npz'),shared_initial_digest=common,
        initial_hashes={a:sha(out/(a+'_initial.pt')) for a in ('early','late')},
        code={n:sha(out/n) for n in names},authority='MATCHED_CONSUMED_DEVELOPMENT',
        parameters={a:sum(p.numel() for p in m.parameters()) for a,m in [('early',early),('late',late)]}))
    weights=torch.tensor(float(np.clip((~y[tr]).sum()/max(1,y[tr].sum()),1,12)),device='cuda')
    target=torch.from_numpy(y.astype(np.float32)).cuda();probabilities={};results={};models={}
    for arm,model in [('early',early),('late',late)]:
        directory=out/arm;directory.mkdir();model=model.cuda();model.eval();probe=batch(data,tr[:8])
        cpu_model=copy.deepcopy(model).cpu();cpu_probe={k:v.cpu() for k,v in probe.items()}
        def gpu_run():
            with torch.no_grad():return model(probe)
        def cpu_run():
            with torch.no_grad():return cpu_model(cpu_probe)
        backend=select_backend('model-inference',cpu=BackendCandidate('torch-cpu','cpu',cpu_run,
            lambda v:DeviceObservation(v.device.type,'host CPU','torch '+torch.__version__)),
            gpu=BackendCandidate('torch-cuda','cuda',gpu_run,
                lambda v:DeviceObservation(v.device.type,torch.cuda.get_device_name(),'torch '+torch.__version__),torch.cuda.synchronize),
            record_path=directory/'backend.json')
        assert backend['selected_device_type']=='cuda';del cpu_model,cpu_probe
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        model.train();losses=[];tick=time.perf_counter()
        for step,idx in enumerate(schedule):
            assert time.perf_counter()-tick<1200,'Fixed fit exceeded 20 minute cap'
            b=batch(data,idx);contrast=torch.from_numpy(.75+.5*augmentation[step,0]).cuda()
            offset=torch.from_numpy(-.1+.2*augmentation[step,1]).cuda()
            b['rgb']=((b['rgb']-.5)*contrast+.5+offset).clamp(0,1)
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(b),target[idx],pos_weight=weights)
            loss.backward();optimizer.step()
            if (step+1)%100==0:
                torch.cuda.synchronize();losses.append(dict(step=step+1,loss=float(loss.detach()),seconds=time.perf_counter()-tick))
                write(directory/'progress.json',losses);print(json.dumps(dict(arm=arm,**losses[-1])),flush=True)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-tick
        torch.save(model.state_dict(),directory/'model.pt');tick=time.perf_counter()
        p=predict(model,data,np.arange(len(rows)));torch.cuda.synchronize();inference_seconds=time.perf_counter()-tick
        np.save(directory/'probabilities.npy',p)
        dc=curve(rows,es,y,p,dv,baseline);tc=curve(rows,es,y,p,tr,baseline)
        selected,feasible=select_joint(dc,baseline_dev)
        write(directory/'development_curve.json',dc);write(directory/'training_curve.json',tc)
        at_train=next(r for r in tc if selected and r['threshold']==selected['threshold']) if selected else None
        results[arm]=dict(selected=selected,selected_training=at_train,feasible_thresholds=[r['threshold'] for r in feasible],
            precision_envelopes=dict(train=envelopes(tc),dev=envelopes(dc)),training_seconds=training_seconds,inference_seconds=inference_seconds,
            parameters=sum(p.numel() for p in model.parameters()),final_model_sha256=sha(directory/'model.pt'))
        probabilities[arm]=p;models[arm]=model.cpu();del optimizer
    a=results['early'];b=results['late'];qualified=a['selected'] is not None and b['selected'] is not None
    checks=dict(both_jointly_feasible=qualified,
        train_iou_gain=bool(qualified and b['selected_training']['spatial']['iou']>a['selected_training']['spatial']['iou']),
        dev_iou_gain=bool(qualified and b['selected']['spatial']['iou']>a['selected']['spatial']['iou']),
        dev_alert_fp_not_increased=bool(qualified and b['selected']['metrics']['FP']<=a['selected']['metrics']['FP']))
    write(out/'development-decision.json',dict(checks=checks,thresholds={k:None if v['selected'] is None else v['selected']['threshold'] for k,v in results.items()},
        model_hashes={k:v['final_model_sha256'] for k,v in results.items()},transfer_authorized_by_fixed_protocol=all(checks.values())))
    transfer=None
    if all(checks.values()):
        s=load_source('mz119',evaluation=False);td=datasets([s]);tp={}
        for arm,model in models.items():
            tp[arm]=predict(model.cuda(),td,np.arange(len(s['rows'])));np.save(out/(arm+'_transfer.npy'),tp[arm]);model.cpu()
        write(out/'transfer-seal.json',dict(source=s['hashes'],predictions={arm:sha(out/(arm+'_transfer.npy')) for arm in tp},
            decision_sha256=sha(out/'development-decision.json'),scope='CONSUMED_TRANSFER_NO_SELECTION'))
        te=read_evaluation(s['capture']);ty=np.stack([labels(e) for e in te]);transfer={}
        for arm,p in tp.items():transfer[arm]=evaluate(s['rows'],te,ty,p,results[arm]['selected']['threshold'],s['baseline'])
    decision=('NOT_EVALUABLE_PAIRED_OPERATING_POINT' if a['selected'] is None else
              'RETAIN_DEVELOPMENT_COMPONENT_REVIEW_TRANSFER' if all(checks.values()) else 'STOP_FIXED_LOCAL_FUSION_RECIPE')
    result=dict(status='PAIRED_LOCAL_FUSION_COMPLETE',decision=decision,
        checks=checks,arms=results,transfer=transfer,frames=dict(train=len(tr),dev=len(dv)),elapsed_seconds=time.perf_counter()-start,
        limitations='Single paired seed; 1296 extra parameters; consumed related templates; no physical echo identity supervision')
    write(out/'summary.json',result);torch.cuda.empty_cache()
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),checks=checks,resources_started=['foreground Python CUDA tensors released on process exit']))
    print(json.dumps(dict(decision=result['decision'],checks=checks),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
