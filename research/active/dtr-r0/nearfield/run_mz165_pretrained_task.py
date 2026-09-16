"""One frozen pretraining contrast using authenticated TRAIN-only cached inputs."""
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
from mz165_pretrained_task import ContextTaskNet,evaluate_context,METHOD
from mz165_dino_features import load_backbones,extract

WORK=ROOT/'artifacts.local/work/mz165-pretrained-task-context-20260916'
OLD=ROOT/'artifacts.local/work/mz161-dense-task-20260916/run-v1'
ARMS=('random','pretrained');SEED=165016;EPOCHS=20;BATCH=4;LIMIT=1800.

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
        if time.perf_counter()-start>=LIMIT:raise TimeoutError('MZ1651800s allocation exhausted')
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
    sources=local_dependencies(__file__)
    for p in (WORK/'upstream/dinov2').rglob('*.py'):sources[str(p)]=sha(p)
    for name in ('MZ165_PROTOCOL_20260916.md','test_mz165_pretrained_task.py','test_mz165_dino_features.py'):
        sources[str(CODE/name)]=sha(CODE/name)
    input_paths=[OLD/n for n in ('completion.json','prediction-seal.json','model-seal.json','freeze.json','input-seal.json','input-audit.json','public-features.npy','public-tokens.npz','fit-targets.npz','fit-label-seal.json','fit-label-audit.json')]
    input_paths += [CAP/n for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')]+[FOLDS]+[BASE/n for n in ('folds.json','oof.npz','model-seal.json')]
    input_paths += [p for p in WORK.rglob('*') if p.is_file() and (p.suffix in ('.pth','.pt') or p.parent.name=='adapter-preflight' and p.suffix in ('.json','.log'))]
    input_paths += [WORK/n for n in ('head-preflight.json','head-backend.json','head-tests.log','download-source-model-receipt.json')]
    inputs={str(p):sha(p) for p in input_paths}
    for row in rows:
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        h=sha(p);assert h==receipt['hashes'][row['rgb_path']]==inputseal['rgb_sha256'][str(p)]
        inputs[str(p)]=h
    write(out/'freeze.json',dict(method=METHOD,sources=sources,inputs=inputs,metadata=metadata,
        seed=SEED,epochs=EPOCHS,updates_per_arm=720,batch=BATCH,optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        budget_seconds=LIMIT,feature_cache_dtype='float16',head_compute_dtype='float32',
        math_flags=dict(cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_deterministic=True),
        input_authority='ADMITTED_MZ161_PUBLIC_CACHE_AND_FIT144_LABELS',
        access='Full source spec and originalTRAIN only; no original dev/test decode; no HELD fit labels'))
    models,binding=load_backbones(WORK,device='cuda');write(out/'encoder-binding.json',binding)
    model_states={a:state_hash(models[a]) for a in ARMS}
    contexts={a:np.lib.format.open_memmap(out/(a+'-context.npy'),mode='w+',dtype=np.float16,shape=(192,384,37,66)) for a in ARMS}
    extraction=[];extract_start=time.perf_counter()
    for i,row in enumerate(rows):
        image=cv2.cvtColor(cv2.imread(str(CAP/row['rgb_path'])),cv2.COLOR_BGR2RGB)
        item=dict(id=row['id'],arms={})
        for arm in ARMS:
            value,audit=extract(models[arm],image)
            assert value.shape==(384,37,66) and value.dtype==np.float32 and np.isfinite(value).all()
            contexts[arm][i]=value;item['arms'][arm]=audit
        extraction.append(item);within()
        if (i+1)%24==0:print(f'EXTRACT {i+1}/192 seconds={time.perf_counter()-extract_start:.2f}',flush=True)
    for a in ARMS:
        contexts[a].flush();assert state_hash(models[a])==model_states[a]
    del models,value;release()
    write(out/'extraction-audit.json',extraction)
    write(out/'context-seal.json',dict(outputs={a:sha(out/(a+'-context.npy')) for a in ARMS},
        encoder_binding_sha256=sha(out/'encoder-binding.json'),encoder_state_hashes=model_states,
        freeze_sha256=sha(out/'freeze.json'),seconds=time.perf_counter()-extract_start,
        extraction_audit_sha256=sha(out/'extraction-audit.json')))
    rng=np.random.default_rng(SEED);orders=[rng.permutation(fitix).tolist() for _ in range(EPOCHS)]
    write(out/'training-orders.json',orders)
    def batch(ii,arm):
        return dict(image=torch.from_numpy(np.asarray(features[ii],np.float32)).cuda(),
            context=torch.from_numpy(np.asarray(contexts[arm][ii],np.float32)).cuda(),
            tokens=torch.from_numpy(tokens[ii]).cuda(),valid=torch.from_numpy(valid[ii]).cuda())
    predictions={};initial={};final={};timings={};parameters=None
    for arm in ARMS:
        seed();model=ContextTaskNet().cuda();initial[arm]=state_hash(model)
        parameters=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        arm_start=time.perf_counter();torch.cuda.reset_peak_memory_stats();history=[];steps=0
        for epoch,order in enumerate(orders,1):
            model.train();epoch_terms=[]
            for j in range(0,len(order),BATCH):
                ii=order[j:j+BATCH];li=[fit_position[i] for i in ii]
                optimizer.zero_grad(set_to_none=True);output=model(batch(ii,arm))
                value,terms=loss(output,torch.from_numpy(frame_targets[li]).cuda(),
                    torch.from_numpy(targets[li].astype(np.float32)).cuda(),torch.from_numpy(weights[li]).cuda(),'dense')
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                value.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();steps+=1;epoch_terms.append(terms);within()
            history.append(dict(epoch=epoch,steps=steps,frame_loss=float(np.mean([v['frame'] for v in epoch_terms])),
                dense_loss=float(np.mean([v['dense'] for v in epoch_terms])),seconds=time.perf_counter()-arm_start))
            write(out/(arm+'-training.json'),history)
            if epoch%5==0:
                torch.save(dict(epoch=epoch,steps=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f'{arm}-epoch{epoch:02d}.pt')
                print(f'TRAIN {arm} epoch={epoch} frame={history[-1]["frame_loss"]:.5f} dense={history[-1]["dense_loss"]:.5f}',flush=True)
        assert steps==720
        torch.save(model.state_dict(),out/(arm+'-model.pt'));final[arm]=sha(out/(arm+'-model.pt'))
        model.eval();maps=np.lib.format.open_memmap(out/(arm+'-pixel-logits.npy'),mode='w+',dtype=np.float32,shape=(192,360,640));records=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                output=model(batch([i],arm));maps[i]=output['pixel'][0].cpu().numpy()
                f,p,t=[float(output[k][0]) for k in ('frame','local','independent')]
                assert np.isfinite([f,p,t]).all() and np.isfinite(maps[i]).all()
                records.append(dict(id=row['id'],frame_logit=f,pixel_max=p,token_max=t,winner='pixel' if p>=t else 'token'));within()
        maps.flush();write(out/(arm+'-predictions.json'),records);predictions[arm]=records
        timings[arm]=dict(seconds=time.perf_counter()-arm_start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,optimizer,maps,output,value;release()
    assert initial['random']==initial['pretrained']
    write(out/'model-seal.json',dict(initial_states=initial,models=final,head_parameters=parameters,
        context_seal_sha256=sha(out/'context-seal.json'),orders_sha256=sha(out/'training-orders.json'),timings=timings))
    write(out/'prediction-seal.json',dict(authority='ALL192_MAPS_AND_FLAGS_PER_ARM_BEFORE_THIS_NATIVE_PARSE',
        outputs={a+'-'+n:sha(out/(a+'-'+n)) for a in ARMS for n in ('pixel-logits.npy','predictions.json')},
        model_seal_sha256=sha(out/'model-seal.json')))
    print('PREDICTIONS SEALED',flush=True);within()
    es=selected_jsonl(CAP/'evaluator.jsonl',set(ids));assert [e['id'] for e in es]==ids
    labels=[make_labels(r,e,float(y)) for r,e,y in zip(rows,es,yaws)]
    maps={a:np.load(out/(a+'-pixel-logits.npy'),mmap_mode='r') for a in ARMS}
    result=evaluate_context(rows,es,labels,maps,predictions,baseline,spec,metadata)
    write(out/'cases.json',result['cases']);write(out/'summary.json',result['summary'])
    write(out/'label-audit.json',{r['id']:l['audit'] for r,l in zip(rows,labels)})
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within()
    write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=result['summary']['decision'],
        prediction_seal_sha256=sha(out/'prediction-seal.json'),outputs={n:sha(out/n) for n in ('summary.json','cases.json','label-audit.json')},
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
