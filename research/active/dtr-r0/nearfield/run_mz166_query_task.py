"""One matched dense-query supervision contrast over frozen pretrained context."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from run_mz161_dense_task import ROOT,CODE,CAP,BASE,FOLDS,read,write,sha,state_hash,release
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from mz136_incumbent import public_observations
from mz161_dense_task import loss
from mz161_dense_labels import make_labels
from mz165_pretrained_task import ContextTaskNet
from mz166_query_geometry import OFFSETS,public_query,make_query_references,label_for_query
from mz166_query_task import evaluate_queries

WORK=ROOT/'artifacts.local/work/mz166-query-conditioned-context-20260916'
PARENT=ROOT/'artifacts.local/work/mz165-pretrained-task-context-20260916/run-v1'
OLD=ROOT/'artifacts.local/work/mz161-dense-task-20260916/run-v1'
ARMS=('central','queries');SEED=166016;EPOCHS=20;BATCH=4;LIMIT=1800.

def seed():
    np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def run(out):
    out=Path(out).resolve();assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    def within():
        if time.perf_counter()-start>=LIMIT:raise TimeoutError('MZ1661800s allocation exhausted')
    spec,receipt=read(CAP/'spec.json'),read(CAP/'receipt.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/n)==receipt['hashes'][n]
    frames=[f for f in spec['frames'] if f['split']=='train'];assert len(frames)==192
    metadata={f['id']:dict(partition='heldout' if f['scene_group'].endswith('_scene3') else 'fit',
        family=f['family'],scene_group=f['scene_group']) for f in frames}
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(metadata)));ids=[r['id'] for r in rows]
    oldfreeze=read(OLD/'freeze.json');inputseal=read(OLD/'input-seal.json');labelseal=read(OLD/'fit-label-seal.json')
    for p,h in oldfreeze['inputs'].items():assert sha(p)==h,('Parent cache input mismatch',p)
    oldcomplete=read(OLD/'completion.json');oldpred=read(OLD/'prediction-seal.json');oldmodel=read(OLD/'model-seal.json')
    assert oldcomplete['status']=='PASS' and sha(OLD/'prediction-seal.json')==oldcomplete['prediction_seal_sha256']
    assert sha(OLD/'model-seal.json')==oldpred['model_seal_sha256']
    assert sha(OLD/'input-seal.json')==oldpred['input_seal_sha256']
    assert sha(OLD/'fit-label-seal.json')==oldmodel['fit_label_seal_sha256']
    assert metadata==oldfreeze['metadata'] and sha(OLD/'freeze.json')==inputseal['freeze_sha256']
    assert [r['id'] for r in read(OLD/'input-audit.json')]==ids==[r['id'] for r in read(FOLDS)]
    assert sha(OLD/'public-features.npy')==inputseal['features_sha256']
    assert sha(OLD/'public-tokens.npz')==inputseal['tokens_sha256']
    assert sha(OLD/'fit-targets.npz')==labelseal['targets_sha256']
    assert sha(OLD/'fit-label-audit.json')==labelseal['audit_sha256']
    assert [r['id'] for r in read(BASE/'folds.json')[:192]]==ids
    assert sha(BASE/'oof.npz')==read(BASE/'model-seal.json')['oof_sha256']
    with np.load(BASE/'oof.npz',allow_pickle=False) as c:
        assert c['baseline'].shape==(480,);baseline=c['baseline'][:192].astype(bool)
    features=np.load(OLD/'public-features.npy',mmap_mode='r');assert features.shape==(192,22,360,640)
    with np.load(OLD/'public-tokens.npz',allow_pickle=False) as c:
        tokens,valid,yaws=c['tokens'],c['valid'],c['yaws']
    assert tokens.shape==(192,132,21) and valid.shape==(192,132) and yaws.shape==(192,)
    fitix=np.array([i for i,r in enumerate(rows) if metadata[r['id']]['partition']=='fit'])
    with np.load(OLD/'fit-targets.npz',allow_pickle=False) as c:
        assert np.array_equal(c['indices'],fitix) and len(fitix)==144
        targets,weights,frame_targets=c['target'],c['weights'],c['frame']
    assert sorted(labelseal['fit_ids'])==sorted(ids[i] for i in fitix)
    fit_position={int(i):j for j,i in enumerate(fitix)}
    parent_complete=read(PARENT/'completion.json');parent_pred=read(PARENT/'prediction-seal.json')
    parent_model=read(PARENT/'model-seal.json');parent_context=read(PARENT/'context-seal.json')
    parent_freeze=read(PARENT/'freeze.json')
    assert parent_complete['status']=='PASS'
    assert sha(PARENT/'prediction-seal.json')==parent_complete['prediction_seal_sha256']
    assert sha(PARENT/'model-seal.json')==parent_pred['model_seal_sha256']
    assert sha(PARENT/'context-seal.json')==parent_model['context_seal_sha256']
    assert sha(PARENT/'freeze.json')==parent_context['freeze_sha256']
    assert sha(PARENT/'encoder-binding.json')==parent_context['encoder_binding_sha256']
    assert sha(PARENT/'extraction-audit.json')==parent_context['extraction_audit_sha256']
    assert sha(PARENT/'pretrained-context.npy')==parent_context['outputs']['pretrained']
    assert sha(PARENT/'pretrained-predictions.json')==parent_pred['outputs']['pretrained-predictions.json']
    assert metadata==parent_freeze['metadata']
    for p,h in parent_freeze['inputs'].items():assert sha(p)==h,('Context parent input mismatch',p)
    for p,h in parent_freeze['sources'].items():assert sha(p)==h,('Context parent source mismatch',p)
    prior=read(PARENT/'pretrained-predictions.json');assert [p['id'] for p in prior]==ids
    context=np.load(PARENT/'pretrained-context.npy',mmap_mode='r')
    assert context.shape==(192,384,37,66) and context.dtype==np.float16
    sources=local_dependencies(__file__)
    for name in ('MZ166_PROTOCOL_20260916.md','test_mz166_query_geometry.py','test_mz166_query_task.py'):
        sources[str(CODE/name)]=sha(CODE/name)
    inputs={**oldfreeze['inputs'],**parent_freeze['inputs']}
    for home,names in ((OLD,('completion.json','prediction-seal.json','model-seal.json','freeze.json','input-seal.json','input-audit.json','public-features.npy','public-tokens.npz','fit-targets.npz','fit-label-seal.json','fit-label-audit.json')),
        (PARENT,('completion.json','prediction-seal.json','model-seal.json','context-seal.json','freeze.json','encoder-binding.json','extraction-audit.json','pretrained-context.npy','pretrained-predictions.json')),
        (WORK,('preflight.json','tests.log'))):
        for name in names:inputs[str(home/name)]=sha(home/name)
    for name in ('head-backend.json','head-preflight.json'):
        p=PARENT.parent/name;inputs[str(p)]=sha(p)
    for row in rows:
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        h=sha(p);assert h==receipt['hashes'][row['rgb_path']]==inputseal['rgb_sha256'][str(p)]
        inputs[str(p)]=h
    counts={};diagnostic_indices=[]
    for i in fitix:
        family=metadata[ids[i]]['family'];counts.setdefault(family,0)
        if counts[family]<6:diagnostic_indices.append(int(i));counts[family]+=1
    assert len(diagnostic_indices)==24 and set(counts.values())=={6}
    rng=np.random.default_rng(SEED);orders=[rng.permutation(fitix).tolist() for _ in range(EPOCHS)]
    schedules={a:[[3 if a=='central' else (epoch+fit_position[i])%7 for i in order]
                  for epoch,order in enumerate(orders)] for a in ARMS}
    frequency=np.bincount(np.array(schedules['queries']).ravel(),minlength=7)
    assert frequency.min()==411 and frequency.max()==412
    write(out/'training-orders.json',dict(orders=orders,queries=schedules,offsets=list(OFFSETS),frequency=frequency.tolist()))
    write(out/'freeze.json',dict(method='MZ166_DENSE_QUERY_SUPERVISION_MATCHED_PRETRAINED_CONTEXT',sources=sources,
        inputs=inputs,metadata=metadata,seed=SEED,epochs=EPOCHS,updates_per_arm=720,batch=BATCH,
        offsets=list(OFFSETS),diagnostic_indices=diagnostic_indices,diagnostic_ids=[ids[i] for i in diagnostic_indices],
        orders_sha256=sha(out/'training-orders.json'),optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        budget_seconds=LIMIT,backend='CUDA_FLOAT32_TF32_DISABLED; NUMPY_QUERY_GEOMETRY',
        backend_evidence='MZ165_IDENTICAL_HEAD_WORKLOAD_MEASURED',
        access='FIT144_NATIVE_AFTER_FREEZE; HELD48_NATIVE_AFTER_ALL_PREDICTIONS; NO_ORIGINAL_DEV_TEST_DECODE'))
    within();print('FREEZE SEALED',flush=True)
    fit_es=selected_jsonl(CAP/'evaluator.jsonl',set(ids[i] for i in fitix))
    assert [e['id'] for e in fit_es]==[ids[i] for i in fitix]
    def mapped(name,dtype,shape):
        return np.lib.format.open_memmap(out/name,mode='w+',dtype=dtype,shape=shape)
    planes=mapped('fit-query-planes.npy',np.float16,(144,7,3,360,640))
    qt=mapped('fit-query-targets.npy',bool,(144,7,360,640))
    qw=mapped('fit-query-weights.npy',np.float32,(144,7,360,640))
    qtokens=np.empty((144,7,132,21),np.float32);qframes=np.empty((144,7),np.float32)
    audits=[];query_start=time.perf_counter()
    for li,(i,e) in enumerate(zip(fitix,fit_es)):
        reference=make_query_references(rows[i],e,float(yaws[i]));base=np.asarray(features[i],np.float32)
        item=dict(id=ids[i],queries=[])
        for qi,offset in enumerate(OFFSETS):
            public=public_query(rows[i],float(yaws[i]),base,tokens[i],valid[i],offset)
            label=label_for_query(reference,offset)
            assert np.array_equal(public['valid'],valid[i]) and np.array_equal(label['known'],reference['known'])
            assert not label['weights'][~label['known']].any()
            if qi==3:
                assert np.array_equal(public['image'],base) and np.array_equal(public['tokens'],tokens[i])
                assert np.array_equal(label['target'],targets[li]) and np.array_equal(label['weights'],weights[li])
                assert label['frame']==bool(frame_targets[li])
            planes[li,qi]=public['image'][13:16];qt[li,qi]=label['target'];qw[li,qi]=label['weights']
            qtokens[li,qi]=public['tokens'];qframes[li,qi]=label['frame']
            item['queries'].append(dict(public=public['audit'],label=label['audit']))
        audits.append(item);within()
        if (li+1)%24==0:print(f'QUERY_CACHE {li+1}/144 seconds={time.perf_counter()-query_start:.2f}',flush=True)
    for array in (planes,qt,qw):array.flush()
    np.savez(out/'fit-query-tokens.npz',tokens=qtokens,frame=qframes,indices=fitix)
    write(out/'fit-query-audit.json',audits)
    query_names=('fit-query-planes.npy','fit-query-targets.npy','fit-query-weights.npy','fit-query-tokens.npz','fit-query-audit.json')
    write(out/'query-seal.json',dict(freeze_sha256=sha(out/'freeze.json'),outputs={n:sha(out/n) for n in query_names},
        zero_query_exact_matches=144,seconds=time.perf_counter()-query_start,
        frame_positive_counts=qframes.sum(axis=0).astype(int).tolist(),
        frames_with_changed_truth=int(np.any(qframes!=qframes[:,3,None],axis=1).sum())))
    del targets,weights,frame_targets,audits,fit_es,reference,label,public,base
    def batch(ii,qq=None):
        image=np.asarray(features[ii],np.float32)
        bt=tokens[ii]
        if qq is not None:
            li=[fit_position[i] for i in ii]
            image[:,13:16]=planes[li,qq];bt=qtokens[li,qq]
        return dict(image=torch.from_numpy(image).cuda(),context=torch.from_numpy(np.asarray(context[ii],np.float32)).cuda(),
            tokens=torch.from_numpy(bt).cuda(),valid=torch.from_numpy(valid[ii]).cuda())
    def record(output,row,**extra):
        f,p,t=[float(output[k][0]) for k in ('frame','local','independent')]
        assert np.isfinite([f,p,t]).all()
        return dict(id=row['id'],frame_logit=f,pixel_max=p,token_max=t,winner='pixel' if p>=t else 'token',**extra)
    predictions={};query_predictions={};initial={};final={};timings={}
    for arm in ARMS:
        seed();model=ContextTaskNet().cuda();initial[arm]=state_hash(model)
        parameters=sum(p.numel() for p in model.parameters());assert parameters==20090
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        arm_start=time.perf_counter();torch.cuda.reset_peak_memory_stats();history=[];steps=0
        for epoch,order in enumerate(orders):
            model.train();epoch_terms=[];query_order=schedules[arm][epoch]
            for j in range(0,len(order),BATCH):
                ii=order[j:j+BATCH];qq=query_order[j:j+BATCH];li=[fit_position[i] for i in ii]
                optimizer.zero_grad(set_to_none=True);output=model(batch(ii,qq))
                value,terms=loss(output,torch.from_numpy(qframes[li,qq]).cuda(),
                    torch.from_numpy(qt[li,qq].astype(np.float32)).cuda(),torch.from_numpy(qw[li,qq]).cuda(),'dense')
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                value.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();steps+=1;epoch_terms.append(terms);within()
            history.append(dict(epoch=epoch+1,steps=steps,frame_loss=float(np.mean([v['frame'] for v in epoch_terms])),
                dense_loss=float(np.mean([v['dense'] for v in epoch_terms])),seconds=time.perf_counter()-arm_start))
            write(out/(arm+'-training.json'),history)
            if (epoch+1)%5==0:
                torch.save(dict(epoch=epoch+1,steps=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f'{arm}-epoch{epoch+1:02d}.pt')
                print(f'TRAIN {arm} epoch={epoch+1} frame={history[-1]["frame_loss"]:.5f} dense={history[-1]["dense_loss"]:.5f}',flush=True)
        assert steps==720
        torch.save(model.state_dict(),out/(arm+'-model.pt'));final[arm]=sha(out/(arm+'-model.pt'))
        model.eval();maps=mapped(arm+'-pixel-logits.npy',np.float32,(192,360,640));records=[]
        diagnostic=mapped(arm+'-query-logits.npy',np.float32,(24,7,360,640));query_records=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                output=model(batch([i]));maps[i]=output['pixel'][0].cpu().numpy()
                assert np.isfinite(maps[i]).all();records.append(record(output,row));within()
            for di,i in enumerate(diagnostic_indices):
                qr=[]
                for qi,offset in enumerate(OFFSETS):
                    output=model(batch([i],[qi]));diagnostic[di,qi]=output['pixel'][0].cpu().numpy()
                    assert np.isfinite(diagnostic[di,qi]).all()
                    qr.append(record(output,rows[i],query_index=qi,offset_m=offset))
                    if qi==3:
                        assert np.array_equal(diagnostic[di,qi],maps[i])
                        assert qr[-1]['frame_logit']==records[i]['frame_logit']
                    within()
                query_records.append(qr)
        maps.flush();diagnostic.flush();write(out/(arm+'-predictions.json'),records)
        write(out/(arm+'-query-predictions.json'),query_records)
        predictions[arm]=records;query_predictions[arm]=query_records
        timings[arm]=dict(seconds=time.perf_counter()-arm_start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,optimizer,maps,diagnostic,output,value;release()
    assert initial['central']==initial['queries']
    write(out/'model-seal.json',dict(initial_states=initial,models=final,head_parameters=parameters,
        query_seal_sha256=sha(out/'query-seal.json'),orders_sha256=sha(out/'training-orders.json'),timings=timings))
    prediction_names=('pixel-logits.npy','predictions.json','query-logits.npy','query-predictions.json')
    write(out/'prediction-seal.json',dict(authority='ALL_CENTRAL_AND_DIAGNOSTIC_PREDICTIONS_BEFORE_HELD_NATIVE_PARSE',
        outputs={a+'-'+n:sha(out/(a+'-'+n)) for a in ARMS for n in prediction_names},
        model_seal_sha256=sha(out/'model-seal.json')))
    print('PREDICTIONS SEALED',flush=True);within()
    es=selected_jsonl(CAP/'evaluator.jsonl',set(ids));assert [e['id'] for e in es]==ids
    labels=[make_labels(r,e,float(y)) for r,e,y in zip(rows,es,yaws)]
    maps={a:np.load(out/(a+'-pixel-logits.npy'),mmap_mode='r') for a in ARMS}
    query_maps={a:np.load(out/(a+'-query-logits.npy'),mmap_mode='r') for a in ARMS}
    di=[fit_position[i] for i in diagnostic_indices]
    result=evaluate_queries(rows,es,labels,maps,predictions,baseline,spec,metadata,prior,
        query_maps,query_predictions,diagnostic_indices,qt[di],qframes[di].astype(bool))
    for name,key in (('cases.json','cases'),('summary.json','summary'),('query-diagnostic.json','query_diagnostic')):
        write(out/name,result[key])
    write(out/'label-audit.json',{r['id']:l['audit'] for r,l in zip(rows,labels)})
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within()
    outputs=('summary.json','cases.json','query-diagnostic.json','label-audit.json')
    write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=result['summary']['decision'],
        prediction_seal_sha256=sha(out/'prediction-seal.json'),outputs={n:sha(out/n) for n in outputs},
        original_dev_test_access=False,resources='Process-local GPU/CPU; no persistent worker'))
    print(result['summary']['decision'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1');args=parser.parse_args()
    try:run(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),resume='NO_AUTOMATIC_RESTART'))
        raise
    finally:
        if torch.cuda.is_available():release()
