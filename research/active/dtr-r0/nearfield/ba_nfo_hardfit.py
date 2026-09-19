"""One TRAIN-only 32-frame learning check; original NFO architecture/loss."""
import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

import ba_nfo_matched as m
from ba_nfo_zone_readout import load_base, public_boxes

BASE = m.OUT
OUT = m.ROOT/'artifacts.local/work/ba-nfo-hardfit32-20260919'
CUT = .081


def zone_info(rgb, depth, value):
    known = np.isfinite(depth) & (depth > 0)
    near = depth[known & (depth < 2)]
    far = depth[known & (depth >= 2)]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    info = dict(known=int(known.sum()), total=depth.size, near=len(near), far=len(far),
                return_m=float(value) if np.isfinite(value) else None,
                texture_std=float(gray.std()),
                laplacian_mean=float(np.abs(cv2.Laplacian(gray,cv2.CV_32F)).mean()))
    if len(near): info['q90_near'] = float(np.quantile(near,.9))
    if len(far): info['q10_far'] = float(np.quantile(far,.1))
    public_far = bool(np.isfinite(value) and value >= 2.2)
    info['positive'] = bool(public_far and known.mean() >= .9 and 4 <= len(near) <= .2*known.sum()
        and info['q90_near'] <= 1.8 and info['q10_far'] >= 2.2
        and info['q10_far']-info['q90_near'] >= .5)
    info['negative'] = bool(public_far and known.mean() >= .9 and len(near)==0
        and depth[known].min() >= 2.2 and info['texture_std'] >= 12 and info['laplacian_mean'] >= 8)
    return info


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Refuse repeat selection'
    protocol = dict(id='ba-nfo-hardfit32-20260919',phase='EXPLORE_TRAIN_ONLY_LEARNING_DIAGNOSTIC',
        question='Can retained NFO learn separated small near support and textured pure-far controls on 32 seen frames?',
        rows='16 positive and16 negative original TRAIN full frames;32 distinct scenes; original manifest order, first eligible zone; positive selection first',
        positive='known>=90%;4<=near pixels<=20% known; q90 near<=1.8m, q10 far>=2.2m, gap>=.5m; actual return>=2.2m',
        negative='known>=90%; every known depth>=2.2m; actual return>=2.2m; grayscale std>=12, mean absolute Laplacian>=8',
        selection='GT used only for disclosed TRAIN case selection/evaluation; no baseline score or fit outcome selects cases; no test/val frames',
        model='Unchanged original NFO, initialize retained trained checkpoint; full model trainable',
        inputs='Full256x192 contiguous RGB plus unchanged actual8x8 public ToF; no target-zone mask/crop/GT input',
        loss='Original full-known four-head BCE+.2 ordinal; no zone weighting, auxiliary, or target-only loss',
        training=dict(steps=512,batch=8,lr=.002,weight_decay=.0001,clip_norm=5,seed=m.SEED,
                      schedule='Constant original fit32 learning rate',checkpoint='Last step only'),
        cutoff=CUT,primary_m=2,
        gates='Target-positive recall>=.95 AND IoU>=.65; target-negative FPR<=.01; at least14/16 positive zones recall>=.90 AND IoU>=.50',
        report='Paired counts per selected zone plus full/mixed/small/all-pure-far for all4 heads; UNKNOWN excluded only from labels',
        decision='PASS supports seen-sample expressibility; FAIL only rejects this fixed full-image-loss/512-update fit; neither proves generalization or RGB ceiling',
        stop='One fit, no retuning or follow-on training, no validation/test inference, no App/default replacement',
        checkpoint_sha256=m.sha(BASE/'trained-nfo.pt'),manifest_sha256=m.sha(BASE/'manifest.json'),
        code_sha256=m.sha(__file__),backend='CUDA; reuse original same-network fit32 placement evidence',
        placement_evidence=str(BASE/'backend.json'))
    m.write(OUT/'protocol.json',protocol)
    rows=[r for r in json.loads((BASE/'manifest.json').read_text()) if r['split']=='train']
    selected=[];scenes=set()
    for kind in ['positive','negative']:
        for row in rows:
            if row['scene'] in scenes: continue
            path=m.OLD/row['prepared'];assert m.sha(path)==row['sha256']
            with np.load(path) as a:
                np.testing.assert_array_equal(a['boxes'],public_boxes())
                for zi,(y0,x0,y1,x1) in enumerate(a['boxes']):
                    info=zone_info(a['rgb'][y0:y1,x0:x1],a['depth'][y0:y1,x0:x1],a['values'][zi])
                    if info[kind]:
                        selected.append(dict(row=row,kind=kind,zone=zi,box=[int(v) for v in (y0,x0,y1,x1)],info=info))
                        scenes.add(row['scene']);break
            if sum(r['kind']==kind for r in selected)==16:break
        assert sum(r['kind']==kind for r in selected)==16, f'Insufficient {kind} cases; do not relax rules'
    assert len(scenes)==32
    m.write(OUT/'selection.json',selected)
    arrays=load_selection()
    render(selected,arrays)
    print('SELECTED',[(r['kind'],r['row']['id'],r['zone']) for r in selected],flush=True)


def load_selection():
    rows=json.loads((OUT/'selection.json').read_text());arrays=[]
    for r in rows:
        assert r['row']['split']=='train'
        path=m.OLD/r['row']['prepared'];assert m.sha(path)==r['row']['sha256']
        with np.load(path) as a:arrays.append({k:a[k].copy() for k in ['rgb','depth','boxes','values']})
    return arrays


def render(selected,arrays,before=None,after=None):
    for kind in ['positive','negative']:
        indices=[i for i,r in enumerate(selected) if r['kind']==kind]
        for page in range(2):
            panels=[]
            for i in indices[page*8:(page+1)*8]:
                row,a=selected[i],arrays[i];y0,x0,y1,x1=row['box'];sl=np.s_[y0:y1,x0:x1]
                known=np.isfinite(a['depth'])&(a['depth']>0);truth=known&(a['depth']<2)
                rgb=a['rgb'].copy();cv2.rectangle(rgb,(x0,y0),(x1-1,y1-1),(255,220,0),1)
                gt=np.zeros_like(a['rgb']);gt[truth]=[40,220,100];gt[~known]=[160,80,180]
                cols=[rgb,cv2.resize(a['rgb'][sl],(192,192),interpolation=cv2.INTER_NEAREST),
                      cv2.resize(gt[sl],(192,192),interpolation=cv2.INTER_NEAREST)]
                if before is not None:
                    for scores in [before,after]:
                        pred=scores[i,2]>=CUT;v=a['rgb'].copy()
                        for mask,color in [(pred&truth,[40,220,100]),(pred&~truth&known,[255,70,60]),(~pred&truth,[50,120,255])]:
                            v[mask]=(.25*v[mask]+.75*np.array(color)).astype(np.uint8)
                        v[~known]=[160,80,180]
                        cols.append(cv2.resize(v[sl],(192,192),interpolation=cv2.INTER_NEAREST))
                body=np.concatenate(cols,axis=1);header=np.full((30,body.shape[1],3),245,np.uint8)
                label=f'{i:02d} {row["row"]["id"]} zone {row["zone"]} | full RGB / zone RGB / near GT'
                if before is not None:label+=' / original / fit'
                cv2.putText(header,label,(4,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
                panels.extend([header,body])
            name=f'{"comparison" if before is not None else "selection"}-{kind}-{page+1}.jpg'
            cv2.imwrite(str(OUT/name),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))


@torch.inference_mode()
def scores(model,rgb,zones):
    model.eval()
    return torch.cat([m.probabilities(model(rgb[i:i+8],zones[i:i+8]),'nfo') for i in range(0,len(rgb),8)]).cpu().numpy()


def evaluate(predictions,arrays,selected):
    totals=defaultdict(lambda:np.zeros(4,np.int64));frames=[]
    for i,(a,row) in enumerate(zip(arrays,selected)):
        each={};y0,x0,y1,x1=row['box']
        target=np.zeros(a['depth'].shape,bool);target[y0:y1,x0:x1]=True
        for ti,t in enumerate(m.THRESHOLDS):
            known,truth,mixed,small=m.masks(a['depth'],a['boxes'],float(t));pure=np.zeros_like(known)
            for zi,(yy0,xx0,yy1,xx1) in enumerate(a['boxes']):
                sl=np.s_[yy0:yy1,xx0:xx1]
                if np.isfinite(a['values'][zi]) and a['values'][zi]>=t and known[sl].any() and not truth[sl].any():pure[sl]=True
            domains=dict(full=known,mixed=mixed,small=small,pure_far=pure&known)
            domains['target_'+row['kind']]=target&known
            for name,mask in domains.items():
                c=m.counts(predictions[i,ti]>=CUT,truth,mask);totals[str(float(t)),name]+=c
                if ti==2:each[name]=m.metrics(c)
        frames.append(dict(id=row['row']['id'],zone=row['zone'],kind=row['kind'],metrics=each))
    metrics={str(float(t)):{name:m.metrics(c) for (tt,name),c in totals.items() if tt==str(float(t))} for t in m.THRESHOLDS}
    return dict(metrics=metrics,frames=frames)


def fit():
    assert (OUT/'visual-check.json').exists(),'Inspect sealed selections before training'
    assert not (OUT/'fit-start.json').exists(),'One fit only'
    protocol=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    arrays=load_selection();torch.set_num_threads(4);assert torch.cuda.is_available();m.seed()
    rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
    zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
    depth=torch.from_numpy(np.stack([a['depth'] for a in arrays])).cuda()
    assert rgb.is_contiguous()
    model=load_base().cuda();before=scores(model,rgb,zones)
    base=evaluate(before,arrays,selection);m.write(OUT/'baseline.json',base)
    np.savez_compressed(OUT/'original-scores.npz',scores=before)
    m.write(OUT/'fit-start.json',dict(protocol_sha256=m.sha(OUT/'protocol.json'),selection_sha256=m.sha(OUT/'selection.json'),
        initial_checkpoint_sha256=m.sha(BASE/'trained-nfo.pt'),actual_device=torch.cuda.get_device_name()))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    rng=np.random.default_rng(m.SEED);logs=[];start=time.perf_counter();model.train()
    for step in range(512):
        ii=rng.choice(32,8,replace=False).tolist();opt.zero_grad(set_to_none=True)
        loss=m.loss_fn(model(rgb[ii],zones[ii]),depth[ii],'nfo');assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        logs.append(dict(step=step+1,loss=float(loss.detach())))
        if (step+1)%128==0:
            m.write(OUT/'progress.json',logs);print('HARD_FIT',logs[-1],flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(state_dict=model.cpu().state_dict(),protocol=protocol,selection_sha256=m.sha(OUT/'selection.json'),
                    source_checkpoint_sha256=protocol['checkpoint_sha256'],cutoff=CUT),OUT/'hardfit-nfo.pt')
    model.cuda();after=scores(model,rgb,zones);new=evaluate(after,arrays,selection)
    np.savez_compressed(OUT/'fit-scores.npz',scores=after)
    p=new['metrics']['2.0']['target_positive'];n=new['metrics']['2.0']['target_negative']
    good=sum(f['metrics']['target_positive']['recall']>=.9 and f['metrics']['target_positive']['iou']>=.5 for f in new['frames'] if f['kind']=='positive')
    gates=dict(target_recall95=p['recall']>=.95,target_iou65=p['iou']>=.65,negative_fpr1=n['false_positive_rate']<=.01,positive_zones14=good>=14)
    assert m.sha(BASE/'trained-nfo.pt')==protocol['checkpoint_sha256']
    result=dict(baseline=base,fit=new,gates=gates,pass_all=all(gates.values()),good_positive_zones=int(good),
        actual_device=torch.cuda.get_device_name(),fit_seconds=elapsed,steps=512,parameters=sum(p.numel() for p in model.parameters()),
        unknown_pixels=int((~np.isfinite(np.stack([a['depth'] for a in arrays]))).sum()),
        baseline_checkpoint_unchanged=True,scope='Seen TRAIN frames only; no validation, test, or generalization evidence')
    m.write(OUT/'results.json',result);render(selection,arrays,before,after)
    m.write(OUT/'completion.json',dict(status='COMPLETE',pass_all=result['pass_all']))
    print('HARD_FIT_RESULT',json.dumps(dict(gates=gates,before=base['metrics']['2.0'],after=new['metrics']['2.0'],good=good,seconds=elapsed)),flush=True)


@torch.inference_mode()
def infer(sample,output):
    torch.set_num_threads(4);p=torch.load(OUT/'hardfit-nfo.pt',map_location='cpu',weights_only=False)
    model=m.Net('nfo');model.load_state_dict(p['state_dict']);model.cuda().eval()
    with np.load(sample) as a:
        rgb=torch.from_numpy(a['rgb'].transpose(2,0,1).copy()[None]).cuda()
        z=torch.from_numpy(m.public_zones(a['values'])[None]).cuda()
    p=m.probabilities(model(rgb,z),'nfo')[0].cpu().numpy()
    np.savez_compressed(output,scores=p,masks=p>=CUT,cutoff=CUT)
    print('PUBLIC_INPUT_INFERENCE_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fit','infer'])
    parser.add_argument('--sample');parser.add_argument('--output');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='fit':fit()
    else:infer(args.sample,args.output)
