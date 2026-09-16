"""One matched full-resolution TRAIN-only dense-task training comparison."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[4];CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import cv2
import numpy as np
import torch
from mz161_dense_task import DenseTaskNet,encode,loss,METHOD
from mz161_dense_labels import make_labels
from mz136_incumbent import public_observations
from run_mz139_surface_fit import local_dependencies,selected_jsonl
from run_mz107_four_sensor import truth
from research_backend import BackendCandidate,DeviceObservation,select_backend

WORK=ROOT/'artifacts.local/work/mz161-dense-task-20260916'
CAP=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
BASE=ROOT/'artifacts.local/work/mz151-expanded-training-20260916/train-v1'
FOLDS=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1/train-folds.json'
SEED=161016;EPOCHS=20;BATCH=4;LIMIT=900.

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def release():gc.collect();torch.cuda.empty_cache()

def run(out):
    out=Path(out).resolve();assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True);started=time.perf_counter()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    assert torch.cuda.is_available()
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    def within():
        if time.perf_counter()-started>=LIMIT:raise TimeoutError('Fixed900s allocation consumed')
    spec,receipt=read(CAP/'spec.json'),read(CAP/'receipt.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/n)==receipt['hashes'][n]
    frames=[f for f in spec['frames'] if f['split']=='train'];assert len(frames)==192
    metadata={f['id']:dict(partition='heldout' if f['scene_group'].endswith('_scene3') else 'fit',
        family=f['family'],scene_group=f['scene_group']) for f in frames}
    ids=set(metadata);fitids={i for i,m in metadata.items() if m['partition']=='fit'}
    assert len(fitids)==144
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids));assert len(rows)==192
    assert [r['id'] for r in rows]==[r['id'] for r in read(FOLDS)]
    assert [r['id'] for r in read(BASE/'folds.json')[:192]]==[r['id'] for r in rows]
    assert sha(BASE/'oof.npz')==read(BASE/'model-seal.json')['oof_sha256']
    with np.load(BASE/'oof.npz',allow_pickle=False) as cache:
        original_baseline=cache['baseline'];assert original_baseline.shape==(480,)
        baseline=original_baseline[:192].astype(bool)
    sources=local_dependencies(__file__)
    for name in ('evaluate_mz161_dense_task.py','test_mz161_dense_task.py','test_mz161_dense_labels.py','test_evaluate_mz161_dense_task.py','MZ161_PROTOCOL_20260916.md'):
        sources.update(local_dependencies(CODE/name) if name.endswith('.py') else {str(CODE/name):sha(CODE/name)})
    inputs={str(p):sha(p) for p in [CAP/n for n in ('raw.jsonl','evaluator.jsonl','receipt.json','spec.json')]+
        [BASE/'oof.npz',BASE/'model-seal.json',BASE/'folds.json',FOLDS]}
    for p in sources:
        dest=out/'source-snapshot'/Path(p).relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(out/'freeze.json',dict(seed=SEED,epochs=EPOCHS,batch=BATCH,updates_per_arm=720,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),method=METHOD,metadata=metadata,
        sources=sources,inputs=inputs,original_dev_test_access=False,
        baseline_cache_scope='Only480TRAIN boolean member decoded; original192 selected by exact ID order',
        budget_seconds=LIMIT,feature_cache_dtype='float16',compute_dtype='float32'))
    features=np.lib.format.open_memmap(out/'public-features.npy',mode='w+',dtype=np.float16,shape=(192,22,360,640))
    tokens=np.zeros((192,132,21),np.float32);valid=np.zeros((192,132),bool);yaws=[];audit=[];rgb={}
    yaw=0.;episode=None
    for i,row in enumerate(rows):
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];yaws.append(yaw)
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        rgb[str(p)]=sha(p);assert rgb[str(p)]==receipt['hashes'][row['rgb_path']]
        image=cv2.imread(str(p));value=encode(row,image,yaw)
        features[i]=value['image'];tokens[i]=value['tokens'];valid[i]=value['valid']
        audit.append(dict(id=row['id'],**value['audit']));within()
    features.flush();np.savez_compressed(out/'public-tokens.npz',tokens=tokens,valid=valid,yaws=yaws)
    write(out/'input-audit.json',audit)
    write(out/'input-seal.json',dict(features_sha256=sha(out/'public-features.npy'),tokens_sha256=sha(out/'public-tokens.npz'),
        rgb_sha256=rgb,freeze_sha256=sha(out/'freeze.json'),inputs='PUBLIC_ONLY_BEFORE_FIT_NATIVE_LABELS'))
    fit_eval={e['id']:e for e in selected_jsonl(CAP/'evaluator.jsonl',fitids)}
    labels={};target=np.zeros((192,360,640),bool);weight=np.zeros((192,360,640),np.float32);frame_target=np.zeros(192,np.float32)
    for i,row in enumerate(rows):
        if row['id'] in fitids:
            label=make_labels(row,fit_eval[row['id']],yaws[i]);labels[i]=label
            target[i]=label['target'];weight[i]=label['weights'];frame_target[i]=truth(fit_eval[row['id']]);within()
    write(out/'fit-label-audit.json',{rows[i]['id']:l['audit'] for i,l in labels.items()})
    np.savez_compressed(out/'fit-targets.npz',indices=sorted(labels),target=target[sorted(labels)],weights=weight[sorted(labels)],frame=frame_target[sorted(labels)])
    write(out/'fit-label-seal.json',dict(audit_sha256=sha(out/'fit-label-audit.json'),targets_sha256=sha(out/'fit-targets.npz'),fit_ids=sorted(fitids),heldout_labels_parsed=False))
    fitix=np.array(sorted(labels));order_rng=np.random.default_rng(SEED)
    orders=[order_rng.permutation(fitix).tolist() for _ in range(EPOCHS)]
    write(out/'training-orders.json',orders)
    def batch(indices,device='cuda'):
        return dict(image=torch.from_numpy(np.asarray(features[indices],np.float32)).to(device),
            tokens=torch.from_numpy(tokens[indices]).to(device),valid=torch.from_numpy(valid[indices]).to(device))
    seed();probe=DenseTaskNet().eval().cuda();cpu_probe=copy.deepcopy(probe).cpu().eval()
    pg=batch([int(fitix[0])]);pc=batch([int(fitix[0])],'cpu')
    def infer(model,data):
        with torch.no_grad():return model(data)
    select_backend('model-inference',cpu=BackendCandidate('dense-task-torch-cpu','cpu',lambda:infer(cpu_probe,pc),
        lambda _:DeviceObservation('cpu','host CPU',torch.__version__)),
        gpu=BackendCandidate('dense-task-torch-cuda','cuda',lambda:infer(probe,pg),
        lambda _:DeviceObservation('cuda',torch.cuda.get_device_name(),torch.__version__),torch.cuda.synchronize),
        record_path=out/'backend.json',capabilities={'training_and_predictions':'CUDA_FLOAT32'},warmups=1,repeats=2)
    del probe,cpu_probe,pg,pc;release()
    predictions={};initial_hashes={};model_hashes={};arm_seconds={}
    for arm in ('frame','dense'):
        seed();model=DenseTaskNet().cuda();initial_hashes[arm]=state_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        training=[];step=0;arm_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for epoch,order in enumerate(orders,1):
            model.train();epoch_losses=[]
            for j in range(0,len(order),BATCH):
                ii=order[j:j+BATCH];optimizer.zero_grad(set_to_none=True)
                output=model(batch(ii))
                value,terms=loss(output,torch.from_numpy(frame_target[ii]).cuda(),
                    torch.from_numpy(target[ii].astype(np.float32)).cuda(),torch.from_numpy(weight[ii]).cuda(),arm)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                value.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();step+=1;epoch_losses.append(terms);within()
                if step%72==0:
                    progress=dict(stage='train',arm=arm,epoch=epoch,step=step,total=720,loss=float(value.detach()),seconds=time.perf_counter()-arm_start)
                    write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            training.append(dict(epoch=epoch,step=step,frame_loss=float(np.mean([v['frame'] for v in epoch_losses])),
                dense_loss=float(np.mean([v['dense'] for v in epoch_losses])),seconds=time.perf_counter()-arm_start))
            write(out/(arm+'-training.json'),training)
            if epoch%5==0:
                torch.save(dict(epoch=epoch,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()),out/f'{arm}-epoch{epoch:02d}.pt')
        assert step==720
        torch.save(model.state_dict(),out/(arm+'-model.pt'));model_hashes[arm]=sha(out/(arm+'-model.pt'))
        model.eval();maps=np.lib.format.open_memmap(out/(arm+'-pixel-logits.npy'),mode='w+',dtype=np.float32,shape=(192,360,640));records=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                output=model(batch([i]));maps[i]=output['pixel'][0].cpu().numpy()
                a,b,c=[float(output[k][0]) for k in ('frame','local','independent')]
                if not np.isfinite([a,b,c]).all() or not np.isfinite(maps[i]).all():raise FloatingPointError('Nonfinite final output')
                records.append(dict(id=row['id'],frame_logit=a,pixel_max=b,token_max=c,winner='pixel' if b>=c else 'token'))
                within()
        maps.flush();write(out/(arm+'-predictions.json'),records);predictions[arm]=records
        arm_seconds[arm]=dict(seconds=time.perf_counter()-arm_start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,optimizer,maps,output,value;release()
    assert initial_hashes['frame']==initial_hashes['dense']
    write(out/'model-seal.json',dict(initial_hashes=initial_hashes,models=model_hashes,training_orders_sha256=sha(out/'training-orders.json'),
        freeze_sha256=sha(out/'freeze.json'),fit_label_seal_sha256=sha(out/'fit-label-seal.json'),arm_seconds=arm_seconds))
    outputs=[arm+'-'+suffix for arm in ('frame','dense') for suffix in ('pixel-logits.npy','predictions.json')]
    write(out/'prediction-seal.json',dict(authority='ALL192_PUBLIC_PREDICTIONS_SEALED_BEFORE_HELDOUT_LABEL_PARSE',
        outputs={n:sha(out/n) for n in outputs},model_seal_sha256=sha(out/'model-seal.json'),input_seal_sha256=sha(out/'input-seal.json')))
    print(json.dumps(dict(stage='all_predictions_sealed',seconds=time.perf_counter()-started)),flush=True)
    # HELD references first parsed only after both final arms and maps are sealed.
    within()
    es=selected_jsonl(CAP/'evaluator.jsonl',ids);assert [e['id'] for e in es]==[r['id'] for r in rows]
    for i,row in enumerate(rows):
        if i not in labels:
            within();labels[i]=make_labels(row,es[i],yaws[i])
    from evaluate_mz161_dense_task import evaluate
    maps={arm:np.load(out/(arm+'-pixel-logits.npy'),mmap_mode='r') for arm in ('frame','dense')}
    result=evaluate(rows,es,[labels[i] for i in range(192)],maps,predictions,baseline,spec,metadata)
    write(out/'cases.json',result['cases']);write(out/'summary.json',result['summary'])
    write(out/'heldout-label-audit.json',{rows[i]['id']:labels[i]['audit'] for i in range(192) if rows[i]['id'] not in fitids})
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within()
    write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-started,
        prediction_seal_sha256=sha(out/'prediction-seal.json'),model_seal_sha256=sha(out/'model-seal.json'),
        outputs={n:sha(out/n) for n in ('summary.json','cases.json','heldout-label-audit.json')},
        original_dev_test_access=False,resources='Process-local CPU/GPU memory released on exit; durable evidence retained'))
    print(json.dumps(dict(stage='complete',seconds=time.perf_counter()-started,decision=result['summary'].get('decision'))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1');args=parser.parse_args()
    try:run(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),resume='NO_AUTOMATIC_RESTART'))
        raise
    finally:
        if torch.cuda.is_available():release()
