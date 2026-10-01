"""MASK-only control: raw mask, public query support and query embedding.

No CNH z4/z1 arrays are opened. Same20epoch, AdamW, cosine, batch256 recipe
as cnh_segment_learned_train; only firstConv inputchannels changes to1.
Train0..95 IDEAL masks; v2calib96..127 IDEAL macroAP selects checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
import cnh_learned_readout as L


class Readout(L.Readout):
    def __init__(self):
        super().__init__()
        self.body[0] = nn.Conv3d(1, 16, 3, padding=1)


def load_unit(feature_path, mask_path, expected_split):
    """Read labels/public support/alignment metadata, never histogram channels."""
    with np.load(feature_path, allow_pickle=False) as f:
        d = {k:f[k].copy() for k in ('labels','main','config','frame','split')}
        d['sup'] = np.unpackbits(f['sup'],axis=-1)[...,:16].astype(bool)
    if str(d['split']) != expected_split:
        raise ValueError('Feature split mismatch')
    with np.load(mask_path, allow_pickle=False) as m:
        if str(m['split']) != expected_split:
            raise ValueError('Mask split mismatch')
        for k in ('config','frame'):
            if not np.array_equal(m[k],d[k]):
                raise ValueError('Mask/metadata alignment mismatch')
        d['mask'] = m['IDEAL'].astype(np.float32)
    if d['mask'].shape != (len(d['frame']),8,8) or not np.isfinite(d['mask']).all():
        raise ValueError('Invalid maskshape or values')
    if np.any((d['mask']<0)|(d['mask']>1)):
        raise ValueError('Mask must contain coverage fractions')
    return d


def tensor_input(mask, device=None):
    m = torch.as_tensor(mask,dtype=torch.float32,device=L.DEV if device is None else device)
    if m.ndim!=3 or m.shape[1:]!=(8,8):
        raise ValueError('Expected[B,8,8] mask')
    return m[:,None,:,:,None].expand(-1,1,8,8,16)


@torch.no_grad()
def predict(net, mask, sup, batch=256):
    net.eval()
    device=next(net.parameters()).device
    return np.concatenate([net(tensor_input(mask[i:i+batch],device),
                               torch.as_tensor(sup[i:i+batch],device=device)).cpu().numpy()
                           for i in range(0,len(mask),batch)])


def load_training(features,masks):
    rows=[]
    for u in range(96):
        d=load_unit(features/f'unit{u:03d}.npz',masks/f'unit{u:03d}.npz','train')
        ix=d['main'].astype(bool)
        rows.append(dict(mask=d['mask'][ix],sup=d['sup'][ix],y=d['labels'][ix].astype(np.float32)))
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def load_calib(features,masks):
    return [load_unit(features/f'unit{u:03d}.npz',masks/f'unit{u:03d}.npz','calib') for u in range(96,128)]


def calib_ap(net,rows):
    aps={g:[] for g,_ in L.GROUPS}
    for d in rows:
        main=d['main'].astype(bool)
        pred=predict(net,d['mask'][main],d['sup'][main])
        for group,ix in L.GROUPS:
            y=d['labels'][main][:,ix].ravel()
            if 0<y.sum()<len(y):
                aps[group].append(average_precision_score(y,pred[:,ix].ravel()))
    detail={g:float(np.mean(v)) for g,v in aps.items()}
    return float(np.mean(list(detail.values()))),dict(IDEAL=detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_if_complete(out):
    models=[]
    for seed in (0,1,2):
        p=out/f'MASK_seed{seed}.json'
        if not p.exists(): return
        r=json.loads(p.read_text())
        if r['status']!='complete': return
        checkpoint=out/f'MASK_seed{seed}.pt'
        digest=sha(checkpoint)
        if digest!=r['model_sha256'] or r['code']!=sha(__file__):
            raise ValueError('Checkpoint/source changed beforefreeze')
        models.append(dict(seed=seed,path=str(checkpoint),sha256=digest,best_epoch=r['best_epoch']))
    manifest=dict(status='frozen',family='MASK',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  selection='v2calib96..127 IDEAL mainframes macroAP HEAD/BODY mean',
                  training='units0..95 IDEAL mainframes only',inputs='rawmask+querysupport+queryembedding;noCNHchannels',
                  models={r['path']:r['sha256'] for r in models},details=models,
                  source_sha256={Path(__file__).name:sha(__file__),Path(L.__file__).name:sha(L.__file__)})
    target=out/'FROZEN.json'
    if not target.exists(): target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','train-masks','calib-masks','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required;noCPUtrainingfallback')
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    stem=f'MASK_seed{a.seed}'; receipt=a.out/f'{stem}.json'
    identity=dict(mode='MASK',seed=a.seed,epochs=20,features=str(a.features),train_masks=str(a.train_masks),
                  calib_masks=str(a.calib_masks),code=sha(__file__),base_code=sha(L.__file__))
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['status']=='complete':
            if any(old.get(k)!=v for k,v in identity.items()) or old['model_sha256']!=sha(a.out/f'{stem}.pt'):
                raise ValueError('Completed checkpoint identity changed')
            freeze_if_complete(a.out); print('Alreadycomplete',stem); return
    start=time.monotonic()
    train=load_training(a.features,a.train_masks)
    calib=load_calib(a.features,a.calib_masks)
    net=Readout().to(L.DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
    history,best,best_epoch,best_state,first=[],-float('inf'),-1,None,0
    resume=a.out/f'{stem}-resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity: raise ValueError('Resumeidentitychanged')
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        history,best,best_epoch,best_state,first=[state[k] for k in ('history','best','best_epoch','best_state','next_epoch')]
        rng.bit_generator.state=state['rng']
    progress=dict(status='running',**identity,device=torch.cuda.get_device_name(),params=sum(p.numel() for p in net.parameters()),
                  train_frames=len(train['mask']),calib_units=len(calib),inputs='maskonly;publicquerysupportandembeddingunchanged',
                  batch=256,lr=.002,weight_decay=.0001,scheduler='cosine20')
    try:
        for ep in range(first,20):
            net.train(); loss_sum=0.; order=rng.permutation(len(train['mask']))
            for begin in range(0,len(order),256):
                ix=order[begin:begin+256]
                z=net(tensor_input(train['mask'][ix]),torch.as_tensor(train['sup'][ix],device=L.DEV))
                loss=nn.functional.binary_cross_entropy_with_logits(z,torch.as_tensor(train['y'][ix],device=L.DEV))
                opt.zero_grad();loss.backward();opt.step()
                loss_sum+=float(loss.detach())*len(ix)
            sched.step()
            score,ap=calib_ap(net,calib)
            if score>best:
                best,best_epoch=score,ep
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            row=dict(epoch=ep,loss_sum=loss_sum,calib=ap,score=score,elapsed_s=time.monotonic()-start)
            history.append(row);print(json.dumps(row),flush=True)
            torch.save(dict(identity=identity,net=net.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),
                            history=history,best=best,best_epoch=best_epoch,best_state=best_state,next_epoch=ep+1,
                            rng=rng.bit_generator.state),resume)
            progress.update(completed_epochs=ep+1,best_epoch=best_epoch,elapsed_s=time.monotonic()-start)
            receipt.write_text(json.dumps(progress,indent=2),encoding='utf-8')
        torch.save(best_state,a.out/f'{stem}.pt')
        progress.update(status='complete',history=history,best_calib=best,model_sha256=sha(a.out/f'{stem}.pt'),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        progress.update(status='failed',error=str(exc));raise
    finally:
        receipt.write_text(json.dumps(progress,indent=2),encoding='utf-8')
    freeze_if_complete(a.out)

if __name__=='__main__':main()
