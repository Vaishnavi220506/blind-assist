"""Train-only fixed-recipe corridor Readout; no calibration/selection access.

NPZ per unit: z4,z1[N,8,8,16], support[N,6,8,8,16] bool,
labels[N,6], unit scalar, split='train', query_frame='travel',
scene[N], frame[N], train_mask[N] bool exactly frame>=3.
Architecture is imported unchanged from the frozen v5 source snapshot.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import time
import numpy as np

SOURCE=Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-track-a-v5-20260928/data/source'
RECIPE=dict(epochs=20,seeds=[0,1,2],optimizer='AdamW',lr=.002,weight_decay=.0001,
            scheduler='CosineAnnealingLR',effective_batch=256,micro_batch=64,
            loss='BCEWithLogitsLoss',checkpoint='final_epoch',query_frame='travel')

def frozen_module():
    sys.path.insert(0,str(SOURCE))
    module=importlib.import_module('cnh_learned_readout')
    if Path(module.__file__).resolve().parent!=SOURCE.resolve():
        raise ImportError('Readout already imported from nonfrozen code; start a fresh process')
    return module

def validate_arrays(d, require_train=True):
    required=('z4','z1','support','labels','unit','split','query_frame','scene','frame','train_mask')
    if any(k not in d for k in required):raise ValueError('missing required feature metadata')
    if require_train and str(d['split'])!='train':raise ValueError('training loader accepts train only')
    if str(d['query_frame'])!='travel':raise ValueError('labels and supports must use travel query frame')
    n=len(d['z4'])
    if d['z4'].shape!=(n,8,8,16) or d['z1'].shape!=(n,8,8,16):raise ValueError('bad feature shape')
    if d['support'].shape!=(n,6,8,8,16) or d['support'].dtype!=bool:raise ValueError('support must be unpacked bool')
    if d['labels'].shape!=(n,6) or not np.isin(d['labels'],[0,1]).all():raise ValueError('six binary travel labels required')
    if any(d[k].shape!=(n,) for k in ('scene','frame','train_mask')):raise ValueError('bad per-frame metadata')
    if d['train_mask'].dtype!=bool or not np.array_equal(d['train_mask'],d['frame']>=3):raise ValueError('train_mask must exactly select frame>=3')
    if np.asarray(d['unit']).size!=1:raise ValueError('unit must be scalar')
    if not np.isfinite(d['z4']).all() or not np.isfinite(d['z1']).all():raise ValueError('nonfinite features')
    if len(set(zip(d['scene'].tolist(),d['frame'].tolist())))!=n:raise ValueError('duplicate scene-frame sample')
    return n

def load_train(paths):
    xs=[];supports=[];ys=[];metadata=[];seen=set()
    for path in map(Path,paths):
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        n=validate_arrays(d);unit=int(np.asarray(d['unit']).item())
        if unit in seen:raise ValueError('duplicate training unit')
        seen.add(unit);m=d['train_mask'];a=d['z4'][m].astype(np.float32);b=d['z1'][m].astype(np.float32)
        xs.append(np.stack([np.sign(a)*np.log1p(abs(a)),np.sign(b)*np.log1p(abs(b))],1))
        supports.append(d['support'][m]);ys.append(d['labels'][m].astype(np.float32))
        metadata.append(dict(path=str(path.resolve()),unit=unit,frames=n,training_frames=int(m.sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not xs or not sum(len(x) for x in xs):raise ValueError('no training samples')
    return np.concatenate(xs),np.concatenate(supports),np.concatenate(ys),metadata

def training_batches(order,effective_batch=256,micro_batch=64):
    if micro_batch<1 or effective_batch<1 or micro_batch>effective_batch:raise ValueError('invalid batch sizes')
    for start in range(0,len(order),effective_batch):
        block=order[start:start+effective_batch]
        yield [(block[i:i+micro_batch],len(block[i:i+micro_batch])/len(block)) for i in range(0,len(block),micro_batch)]

def train_models(paths,out,*,device=None,epochs=20,seeds=(0,1,2),effective_batch=256,micro_batch=64):
    """Train fixed final checkpoints. Caller freezes recipe; this function reads only supplied train NPZs."""
    import torch
    torch.set_num_threads(2)
    L=frozen_module();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if epochs<1:raise ValueError('epochs must be positive')
    if len(set(seeds))!=len(seeds) or not seeds:raise ValueError('invalid seeds')
    if any((out/f'model_seed{s}.pt').exists() for s in seeds):raise FileExistsError('refusing to replace trained checkpoints')
    x,sup,y,metadata=load_train(paths)
    dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    recipe=dict(RECIPE,epochs=epochs,seeds=list(seeds),effective_batch=effective_batch,micro_batch=micro_batch)
    request=dict(recipe=recipe,inputs=metadata,samples=len(x),device=str(dev),readout_sha256=hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),calibration_access=False,evaluation_access=False)
    (out/'training_request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    histories=[];started=time.monotonic()
    for seed in seeds:
        torch.manual_seed(seed)
        if dev.type=='cuda':torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);model=L.Readout().to(dev)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        history=[]
        for epoch in range(epochs):
            model.train();total=0.;count=0;steps=0;epoch_start=time.monotonic()
            for batch in training_batches(rng.permutation(len(x)),effective_batch,micro_batch):
                optimizer.zero_grad(set_to_none=True)
                for ids,weight in batch:
                    xb=torch.as_tensor(x[ids],device=dev);sb=torch.as_tensor(sup[ids],device=dev);yb=torch.as_tensor(y[ids],device=dev)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xb,sb),yb)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    (loss*weight).backward();total+=float(loss.detach())*len(ids);count+=len(ids)
                optimizer.step();steps+=1
            scheduler.step();row=dict(epoch=epoch+1,loss=total/count,samples=count,steps=steps,elapsed_s=time.monotonic()-epoch_start)
            history.append(row);print(f'seed={seed} epoch={epoch+1}/{epochs} loss={row["loss"]:.6f} sec={row["elapsed_s"]:.1f}',flush=True)
            (out/f'history_seed{seed}.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        target=out/f'model_seed{seed}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},target)
        histories.append(dict(seed=seed,path=str(target.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),final_epoch=epochs,final_loss=history[-1]['loss']))
        del model,optimizer,scheduler
        if dev.type=='cuda':torch.cuda.empty_cache()
    result=dict(status='complete',elapsed_s=time.monotonic()-started,models=histories,samples=len(x),recipe=recipe)
    (out/'training_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def load_models(paths,device=None):
    import torch
    L=frozen_module();dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'));models=[]
    for p in paths:
        model=L.Readout().to(dev);model.load_state_dict(torch.load(p,map_location=dev,weights_only=True));models.append(model.eval())
    return models

def predict(models,z4,z1,support,batch_size=48):
    """Return ensemble mean raw logits [N,6]. No labels accepted; caller applies unchanged causal smoothing."""
    import torch
    if not models or batch_size<1:raise ValueError('models and positive batch size required')
    n=len(z4)
    if np.shape(z4)!=(n,8,8,16) or np.shape(z1)!=np.shape(z4) or np.shape(support)!=(n,6,8,8,16):raise ValueError('invalid prediction shapes')
    if np.asarray(support).dtype!=bool:raise ValueError('support must be bool')
    dev=next(models[0].parameters()).device
    if any(next(m.parameters()).device!=dev for m in models):raise ValueError('models must share device')
    output=[]
    with torch.no_grad():
        for i in range(0,n,batch_size):
            a=np.asarray(z4[i:i+batch_size],np.float32);b=np.asarray(z1[i:i+batch_size],np.float32)
            if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite prediction features')
            x=np.stack([np.sign(a)*np.log1p(abs(a)),np.sign(b)*np.log1p(abs(b))],1)
            xb=torch.as_tensor(x,device=dev);sb=torch.as_tensor(support[i:i+batch_size],device=dev)
            for m in models:m.eval()
            output.append(torch.stack([m(xb,sb) for m in models]).mean(0).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0,6),np.float32)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();train_models(sorted(a.features.glob('unit*.npz')),a.out)
