"""One matched direct-return supervision contrast over authenticated public caches."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from run_mz161_dense_task import ROOT,CODE,CAP,BASE,FOLDS,read,write,sha,state_hash,release
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from mz136_incumbent import public_observations
from mz161_dense_labels import make_labels
from mz165_pretrained_task import ContextTaskNet
from mz171_return_labels import make_witness_labels
from mz171_return_supervision import loss,evaluate_witness,METHOD

WORK=ROOT/'artifacts.local/work/mz171-return-witness-20260916'
PREP=WORK/'preparation'
PARENT=ROOT/'artifacts.local/work/mz165-pretrained-task-context-20260916/run-v1'
OLD=ROOT/'artifacts.local/work/mz161-dense-task-20260916/run-v1'
ARMS=('control','witness');SEED=171016;EPOCHS=20;BATCH=4;LIMIT=1200.


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)


def run(out):
    out=Path(out).resolve();assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    def within():
        if time.perf_counter()-start>=LIMIT:raise TimeoutError('MZ1711200s allocation exhausted')
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
    assert metadata==parent_freeze['metadata']
    for p,h in parent_freeze['inputs'].items():assert sha(p)==h,('Context parent input mismatch',p)
    for p,h in parent_freeze['sources'].items():assert sha(p)==h,('Context parent source mismatch',p)
    assert [r['id'] for r in read(PARENT/'extraction-audit.json')]==ids
    context=np.load(PARENT/'pretrained-context.npy',mmap_mode='r')
    assert context.shape==(192,384,37,66) and context.dtype==np.float16
    witness_seal=read(PREP/'fit-witness-seal.json')
    assert witness_seal['fit_ids']==[ids[i] for i in fitix]
    assert witness_seal['native_records_decoded']==144 and witness_seal['held_native_records_decoded']==0
    for group in ('inputs','sources'):
        for p,h in witness_seal[group].items():assert sha(p)==h,p
    for n,h in witness_seal['outputs'].items():assert sha(PREP/n)==h,n
    with np.load(PREP/'fit-witness-targets.npz',allow_pickle=False) as c:
        assert np.array_equal(c['indices'],fitix)
        slot_target,slot_known=c['target'],c['known']
    assert slot_target.shape==slot_known.shape==(144,132)
    assert slot_target.dtype==np.float32 and slot_known.dtype==np.bool_
    assert not slot_known[:,128:].any() and not (slot_known&~valid[fitix]).any()
    sources=local_dependencies(__file__)
    for n in ('MZ171_PROTOCOL_20260916.md','test_mz171_return_supervision.py','test_mz171_return_labels.py'):
        sources[str(CODE/n)]=sha(CODE/n)
    inputs={**oldfreeze['inputs'],**parent_freeze['inputs'],**witness_seal['inputs']}
    for home,names in ((OLD,('completion.json','prediction-seal.json','model-seal.json','freeze.json','input-seal.json','input-audit.json','public-features.npy','public-tokens.npz','fit-targets.npz','fit-label-seal.json','fit-label-audit.json')),
        (PARENT,('completion.json','prediction-seal.json','model-seal.json','context-seal.json','freeze.json','encoder-binding.json','extraction-audit.json','pretrained-context.npy')),
        (PREP,('fit-witness-seal.json','fit-witness-targets.npz','fit-witness-audit.json')),
        (WORK,('supervision-tests.json','pre-run-review.json'))):
        for n in names:inputs[str(home/n)]=sha(home/n)
    for n in ('head-backend.json','head-preflight.json'):
        p=PARENT.parent/n;inputs[str(p)]=sha(p)
    backend=read(PARENT.parent/'head-backend.json');preflight=read(PARENT.parent/'head-preflight.json')
    assert backend['selected_device_type']=='cuda' and preflight['parameters']==20090
    assert preflight['torch']==torch.__version__ and preflight['cuda']==torch.cuda.get_device_name(0)
    write(out/'backend.json',dict(selection='REUSED_IDENTICAL_HEAD_MZ165_CPU_CUDA_MEASUREMENT',
        prior_sha256=sha(PARENT.parent/'head-backend.json'),actual_device=torch.cuda.get_device_name(0),
        torch=torch.__version__,precision='FLOAT32_TF32_DISABLED',encoder_inference=False,
        auxiliary_loss='SMALL_Bx132_MASKED_BCE_ON_CUDA',prior=backend))
    for row in rows:
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        h=sha(p);assert h==receipt['hashes'][row['rgb_path']]==inputseal['rgb_sha256'][str(p)]
        inputs[str(p)]=h
    rng=np.random.default_rng(SEED);orders=[rng.permutation(fitix).tolist() for _ in range(EPOCHS)]
    write(out/'training-orders.json',orders)
    write(out/'freeze.json',dict(method=METHOD,sources=sources,inputs=inputs,metadata=metadata,
        seed=SEED,epochs=EPOCHS,updates_per_arm=720,batch=BATCH,orders_sha256=sha(out/'training-orders.json'),
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),budget_seconds=LIMIT,
        backend_sha256=sha(out/'backend.json'),feature_cache_dtype='float16',head_compute_dtype='float32',
        access='FIT144_TARGETS_ONLY;ALL_NEW_PREDICTIONS_BEFORE_HELD_NATIVE;NO_ORIGINAL_DEV_TEST_DECODE',
        predictor_input_keys=['image','context','tokens','valid']))
    within();print('FREEZE SEALED',flush=True)
    def batch(ii):
        return dict(image=torch.from_numpy(np.asarray(features[ii],np.float32)).cuda(),
            context=torch.from_numpy(np.asarray(context[ii],np.float32)).cuda(),
            tokens=torch.from_numpy(tokens[ii]).cuda(),valid=torch.from_numpy(valid[ii]).cuda())
    predictions={};initial={};final={};timings={}
    for arm in ARMS:
        seed();model=ContextTaskNet().cuda();initial[arm]=state_hash(model)
        assert sum(p.numel() for p in model.parameters())==20090
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        arm_start=time.perf_counter();torch.cuda.reset_peak_memory_stats();history=[];steps=0
        for epoch,order in enumerate(orders,1):
            model.train();epoch_terms=[]
            for j in range(0,len(order),BATCH):
                ii=order[j:j+BATCH];li=[fit_position[i] for i in ii]
                optimizer.zero_grad(set_to_none=True);output=model(batch(ii))
                value,terms=loss(output,torch.from_numpy(frame_targets[li]).cuda(),
                    torch.from_numpy(targets[li].astype(np.float32)).cuda(),torch.from_numpy(weights[li]).cuda(),
                    torch.from_numpy(slot_target[li]).cuda(),torch.from_numpy(slot_known[li]).cuda(),arm)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                value.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();steps+=1;epoch_terms.append(terms);within()
            history.append(dict(epoch=epoch,steps=steps,**{k+'_loss':float(np.mean([v[k] for v in epoch_terms]))
                for k in ('frame','dense','witness')},seconds=time.perf_counter()-arm_start))
            write(out/(arm+'-training.json'),history)
            if epoch%5==0:
                torch.save(dict(epoch=epoch,steps=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f'{arm}-epoch{epoch:02d}.pt')
                print(f'TRAIN {arm} epoch={epoch} frame={history[-1]["frame_loss"]:.5f} witness={history[-1]["witness_loss"]:.5f}',flush=True)
        assert steps==720
        torch.save(model.state_dict(),out/(arm+'-model.pt'));final[arm]=sha(out/(arm+'-model.pt'))
        model.eval();maps=np.lib.format.open_memmap(out/(arm+'-pixel-logits.npy'),mode='w+',dtype=np.float32,shape=(192,360,640))
        returned=np.empty((192,132),np.float32);records=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                output=model(batch([i]));maps[i]=output['pixel'][0].cpu().numpy()
                returned[i]=output['token'][0].cpu().numpy()
                f,p,t=[float(output[k][0]) for k in ('frame','local','independent')]
                assert np.isfinite([f,p,t]).all() and np.isfinite(maps[i]).all() and np.isfinite(returned[i]).all()
                assert np.all(returned[i][~valid[i]]==-30.)
                ix=int(returned[i].argmax())
                records.append(dict(id=row['id'],frame_logit=f,pixel_max=p,token_max=t,
                    winner='pixel' if p>=t else 'token',tof_max=float(returned[i,:128].max()),
                    radar_max=float(returned[i,128:].max()),token_argmax=ix,
                    token_argmax_public_valid=bool(valid[i,ix]),token_argmax_kind='TOF' if ix<128 else 'RADAR'))
                within()
        maps.flush();np.save(out/(arm+'-token-logits.npy'),returned)
        write(out/(arm+'-predictions.json'),records);predictions[arm]=records
        timings[arm]=dict(seconds=time.perf_counter()-arm_start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,optimizer,maps,output,value,returned;release()
    assert initial['control']==initial['witness']
    write(out/'model-seal.json',dict(initial_states=initial,models=final,head_parameters=20090,
        freeze_sha256=sha(out/'freeze.json'),orders_sha256=sha(out/'training-orders.json'),timings=timings))
    write(out/'prediction-seal.json',dict(authority='ALL192_PIXEL_TOKEN_MAPS_AND_FLAGS_PER_ARM_BEFORE_HELD_NATIVE',
        outputs={a+'-'+n:sha(out/(a+'-'+n)) for a in ARMS for n in ('pixel-logits.npy','token-logits.npy','predictions.json')},
        model_seal_sha256=sha(out/'model-seal.json')))
    print('PREDICTIONS SEALED',flush=True);within()
    es=selected_jsonl(CAP/'evaluator.jsonl',set(ids));assert [e['id'] for e in es]==ids
    labels=[make_labels(r,e,float(y)) for r,e,y in zip(rows,es,yaws)]
    witnesses=[make_witness_labels(r,e) for r,e in zip(rows,es)]
    all_targets=np.stack([v['target'] for v in witnesses]);all_known=np.stack([v['known'] for v in witnesses])
    assert np.array_equal(all_targets[fitix],slot_target) and np.array_equal(all_known[fitix],slot_known)
    np.savez_compressed(out/'evaluation-witness-labels.npz',target=all_targets,known=all_known)
    write(out/'witness-label-audit.json',[dict(id=r['id'],**v['audit']) for r,v in zip(rows,witnesses)])
    maps={a:np.load(out/(a+'-pixel-logits.npy'),mmap_mode='r') for a in ARMS}
    logits={a:np.load(out/(a+'-token-logits.npy'),allow_pickle=False) for a in ARMS}
    result=evaluate_witness(rows,es,labels,maps,predictions,baseline,spec,metadata,logits,all_targets,all_known)
    write(out/'cases.json',result['cases']);write(out/'summary.json',result['summary'])
    write(out/'pixel-label-audit.json',{r['id']:l['audit'] for r,l in zip(rows,labels)})
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within()
    write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=result['summary']['decision'],
        prediction_seal_sha256=sha(out/'prediction-seal.json'),outputs={n:sha(out/n) for n in
            ('summary.json','cases.json','pixel-label-audit.json','evaluation-witness-labels.npz','witness-label-audit.json')},
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
