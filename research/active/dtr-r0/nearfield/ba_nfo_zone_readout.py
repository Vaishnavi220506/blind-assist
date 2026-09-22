"""One frozen-NFO, public-zone constant-logit readout; no backbone fit."""
import argparse
import json
import math
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ba_nfo_matched as m
from ba_nfo_conditional_diagnostic import curve, ap, at_recall

BASE = m.OUT
OUT = m.ROOT/'artifacts.local/work/ba-nfo-zone-readout-20260919'
CUT = .081


def public_boxes():
    ys=np.linspace(round(.1*m.H),round(.9*m.H),9).astype(int)
    xs=np.linspace(round(.1*m.W),round(.9*m.W),9).astype(int)
    return np.array([[ys[y],xs[x],ys[y+1],xs[x+1]] for y in range(8) for x in range(8)])


class ZoneReadout(nn.Module):
    arm='nfo'

    def __init__(self, base):
        super().__init__()
        self.base=base.eval().requires_grad_(False)
        self.readout=nn.Linear(7,1)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        zone_map=np.full((m.H,m.W),-1,np.int64)
        for zi,(y0,x0,y1,x1) in enumerate(public_boxes()):zone_map[y0:y1,x0:x1]=zi
        pixel=np.flatnonzero(zone_map.ravel()>=0)
        zone=zone_map.ravel()[pixel]
        self.register_buffer('pixels',torch.from_numpy(pixel))
        self.register_buffer('zone_ids',torch.from_numpy(zone))
        self.register_buffer('zone_map',torch.from_numpy(zone_map.clip(0)))
        self.register_buffer('inside',torch.from_numpy(zone_map>=0))
        self.register_buffer('counts',torch.from_numpy(np.bincount(zone,minlength=64)).float())

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def features(self, raw, zones):
        # Full public zone, including pixels with evaluator UNKNOWN depth.
        score=m.probabilities(raw,'nfo')[:,2].flatten(1)[:,self.pixels]
        index=self.zone_ids[None].expand(len(score),-1)
        def avg(x):
            return torch.zeros(len(score),64,device=score.device).scatter_add_(1,index,x)/self.counts
        mean=avg(score)
        std=(avg(score.square())-mean.square()).clamp_min(0).sqrt()
        maximum=torch.zeros_like(mean).scatter_reduce_(1,index,score,reduce='amax',include_self=True)
        fraction=avg((score>=CUT).float())
        z=zones.flatten(2)
        return torch.stack([z[:,0],z[:,4],z[:,5],mean,std,maximum,fraction],dim=-1)

    def adjust(self,raw,zones):
        feat=self.features(raw.detach(),zones)
        gate=(zones[:,1].flatten(1)>0)&(zones[:,0].flatten(1)>=.25)
        offset=2*self.readout(feat).squeeze(-1).tanh()*gate
        dense=offset[:,self.zone_map]*self.inside
        return raw+dense[:,None],offset

    def forward(self,rgb,zones):
        with torch.no_grad():raw=self.base(rgb,zones)
        return self.adjust(raw,zones)[0]


def load_base():
    payload=torch.load(BASE/'trained-nfo.pt',map_location='cpu',weights_only=False)
    assert payload['manifest_sha256']==m.sha(BASE/'manifest.json')
    base=m.Net('nfo');base.load_state_dict(payload['state_dict'])
    return base


def train(data, model):
    ids=[i for i,r in enumerate(data.rows) if r['split']=='train']
    original={k:v.clone() for k,v in model.base.state_dict().items()}
    m.seed();opt=torch.optim.AdamW(model.readout.parameters(),lr=.002,weight_decay=.0001)
    rng=np.random.default_rng(m.SEED);logs=[]
    start=time.perf_counter()
    for epoch in range(12):
        model.train();order=rng.permutation(ids).tolist();total=np.zeros(3)
        for group in opt.param_groups:group['lr']=.002*(.1+.9*(1+math.cos(math.pi*epoch/11))/2)
        for offset in range(0,len(order),24):
            rgb,z,d=data.batch(order[offset:offset+24],'cuda')
            opt.zero_grad(set_to_none=True);pred=model(rgb,z)
            loss=m.loss_fn(pred,d,'nfo')
            assert torch.isfinite(loss)
            ordinal=F.relu(pred.sigmoid()[:,:-1]-pred.sigmoid()[:,1:])
            valid=torch.isfinite(d)&(d>0)
            ordinal=ordinal.permute(0,2,3,1)[valid].mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.readout.parameters(),5.);opt.step()
            total += [float(loss.detach()),float((loss-.2*ordinal).detach()),float(ordinal.detach())]
        torch.cuda.synchronize()
        logs.append(dict(epoch=epoch+1,loss=total[0]/125,bce=total[1]/125,ordinal=total[2]/125,seconds=time.perf_counter()-start))
        m.write(OUT/'training-progress.json',logs)
        print('ZONE_READOUT_EPOCH',logs[-1],flush=True)
    assert all(torch.equal(original[k],v) for k,v in model.base.state_dict().items())
    torch.save(dict(state_dict=model.cpu().state_dict(),protocol=json.loads((OUT/'protocol.json').read_text()),
        base_checkpoint_sha256=m.sha(BASE/'trained-nfo.pt'),cutoff=CUT),OUT/'trained-readout.pt')
    model.cuda().eval()
    m.write(OUT/'training-receipt.json',dict(base_bit_identical=True,trained_parameters=8,
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=logs[-1]['seconds'],updates=1500,
        coefficients=model.readout.weight.detach().cpu().tolist(),bias=model.readout.bias.detach().cpu().tolist(),
        checkpoint_sha256=m.sha(OUT/'trained-readout.pt')))


@torch.inference_mode()
def evaluate(data,model):
    totals=defaultdict(lambda:np.zeros(4,np.int64));paired=defaultdict(lambda:np.zeros(4,np.int64))
    curves_data=defaultdict(list);offsets=defaultdict(list);frames=[];scene=defaultdict(lambda:np.zeros(4,np.int64))
    unknown=defaultdict(int);outside_equal=True
    for split in ['val','test']:
        ids=[i for i,r in enumerate(data.rows) if r['split']==split]
        for start in range(0,len(ids),24):
            ii=ids[start:start+24];rgb,z,d=data.batch(ii,'cuda')
            raw=model.base(rgb,z);new,off=model.adjust(raw,z)
            p={a:m.probabilities(x,'nfo').cpu().numpy() for a,x in [('nfo',raw),('readout',new)]}
            off=off.cpu().numpy()
            for j,idx in enumerate(ii):
                depth=data.depth[idx].numpy();values=data.zones[idx,0].numpy().ravel()*8
                valid=data.zones[idx,1].numpy().ravel().astype(bool)
                gate=np.zeros(depth.shape,bool)
                for zi,(y0,x0,y1,x1) in enumerate(data.boxes):
                    if valid[zi] and values[zi]>=2:gate[y0:y1,x0:x1]=True
                assert np.array_equal(p['nfo'][j,:,~gate],p['readout'][j,:,~gate])
                unknown[split]+=int((~np.isfinite(depth)).sum())
                for ti,t in enumerate(m.THRESHOLDS):
                    key=str(float(t));known,truth,mixed,small=m.masks(depth,data.boxes,float(t))
                    far=np.zeros_like(gate);pure_far=np.zeros_like(gate)
                    for zi,(y0,x0,y1,x1) in enumerate(data.boxes):
                        sl=np.s_[y0:y1,x0:x1]
                        if valid[zi] and values[zi]>=t:
                            far[sl]=True
                            if known[sl].any() and not truth[sl].any():pure_far[sl]=True
                    domains=dict(full=known,mixed=mixed,small_foreground=small,
                        public_far=far&known,pure_far=pure_far&known,far_small=far&small,
                        outside_gate=known&~gate)
                    pp={a:p[a][j,ti]>=CUT for a in p}
                    for domain,dm in domains.items():
                        for arm in p:
                            c=m.counts(pp[arm],truth,dm);totals[split,arm,key,domain]+=c
                            if split=='test' and ti==2 and domain in ['mixed','far_small','pure_far']:
                                scene[arm,domain,data.rows[idx]['scene']]+=c
                        if ti==2:
                            old,newp=pp['nfo'],pp['readout']
                            paired[split,domain]+=np.array([(dm&truth&~old&newp).sum(),(dm&truth&old&~newp).sum(),
                                (dm&~truth&~old&newp).sum(),(dm&~truth&old&~newp).sum()])
                    if split=='test' and ti==2:
                        dm=domains['far_small']
                        curves_data['truth'].append(truth[dm]);curves_data['frame'].append(np.full(int(dm.sum()),idx,np.int32))
                        for a in p:curves_data[a].append(p[a][j,ti][dm])
                        frames.append(dict(id=data.rows[idx]['id'],scene=data.rows[idx]['scene'],
                            metrics={a:{dom:m.metrics(m.counts(pp[a],truth,dd)) for dom,dd in domains.items()} for a in p}))
                        for zi,(y0,x0,y1,x1) in enumerate(data.boxes):
                            if valid[zi] and values[zi]>=2:
                                sl=np.s_[y0:y1,x0:x1];k=known[sl];n=truth[sl]
                                group='unknown' if not k.any() else ('pure_far' if not n.any() else ('small' if n.sum()<=.2*k.sum() else 'other'))
                                offsets[group].append(float(off[j,zi]))
            if start%120==0:print('ZONE_READOUT_EVAL',split,start,flush=True)
    original=json.loads((BASE/'results.json').read_text())
    for (split,arm,t,domain),c in totals.items():
        if split=='test' and arm=='nfo' and domain in ['full','mixed','small_foreground']:
            assert c.tolist()==[original['metrics'][arm][t][domain][k] for k in ['tp','fp','fn','tn']],(t,domain)
    metrics={split:{arm:{str(float(t)):{dom:m.metrics(c) for (s,a,tt,dom),c in totals.items()
        if s==split and a==arm and tt==str(float(t))} for t in m.THRESHOLDS} for arm in ['nfo','readout']} for split in ['val','test']}
    base=metrics['test']['nfo']['2.0'];new=metrics['test']['readout']['2.0']
    gate=dict(validation_recall_95=metrics['val']['readout']['2.0']['mixed']['recall']>=.95,
        far_small_recall_improves=new['far_small']['recall']>base['far_small']['recall'],
        far_small_fp_not_increased=new['far_small']['fp']<=base['far_small']['fp'],
        pure_far_fp_not_increased=new['pure_far']['fp']<=base['pure_far']['fp'],
        public_far_fp_not_increased=new['public_far']['fp']<=base['public_far']['fp'],
        mixed_iou_retained=new['mixed']['iou']>=base['mixed']['iou'],
        mixed_recall_loss_at_most_1pp=new['mixed']['recall']>=base['mixed']['recall']-.01)
    gate['pass']=all(gate.values())
    v={k:np.concatenate(x) for k,x in curves_data.items()}
    np.savez_compressed(OUT/'subgroup-scores.npz',**v)
    curves={a:curve(v[a],v['truth']) for a in ['nfo','readout']}
    expected=json.loads((m.ROOT/'artifacts.local/work/ba-nfo-conditional-20260919/results.json').read_text())
    assert all(base['far_small'][k]==expected['fixed']['nfo'][k] for k in ['tp','fp','fn','tn'])
    np.savez_compressed(OUT/'curves.npz',**{a+'_'+k:x for a,c in curves.items() for k,x in c.items()})
    result=dict(metrics=metrics,gate=gate,paired=[dict(split=s,domain=d,rescued_tp=int(c[0]),lost_tp=int(c[1]),added_fp=int(c[2]),removed_fp=int(c[3])) for (s,d),c in paired.items()],
        offset_summary={k:dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),saturated_fraction=float(np.mean(np.abs(x)>1.9))) for k,x in offsets.items()},
        ranking={a:dict(ap=ap(c),at_recall80=at_recall(c,.8),recall_at_old_fp=float(c['recall'][c['fp']<=base['far_small']['fp']].max())) for a,c in curves.items()},
        original_test_counts_exact=True,outside_gate_scores_bit_identical=True,unknown_pixels=unknown,
        scenes=[dict(arm=a,domain=d,scene=s,**m.metrics(c)) for (a,d,s),c in scene.items()])
    m.write(OUT/'results.json',result);m.write(OUT/'frames.json',frames)
    preview(data,model)
    plot_curves(curves,base,new)
    print('ZONE_READOUT_RESULT',json.dumps(dict(gate=gate,base=base,new=new,ranking=result['ranking'],offsets=result['offset_summary'])),flush=True)


def plot_curves(curves,base,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for a,label,point in [('nfo','Frozen NFO',base),('readout','Zone readout (8 parameters)',new)]:
        c=curves[a]
        for ax,metric in zip(axs,['precision','iou','fp']):
            line,=ax.plot(c['recall'][1:],c[metric][1:],label=label)
            ax.scatter(point['far_small']['recall'],point['far_small'][metric],color=line.get_color())
    for ax,name in zip(axs,['Precision','IoU','FP pixels']):
        ax.set(xlabel='Recall',ylabel=name,xlim=(0,1));ax.grid(alpha=.2)
    axs[0].set_ylim(0,.25);axs[2].set_yscale('symlog',linthresh=100);axs[2].set_ylim(bottom=0)
    axs[0].legend(fontsize=9)
    fig.suptitle('2m far-return small foreground | both original cutoff 0.081\nConsumed Development; curves diagnostic only, PR zoom 0-25%')
    fig.tight_layout();fig.savefig(OUT/'comparison-curves.png',dpi=170);plt.close(fig)


@torch.inference_mode()
def preview(data,model):
    ids=json.loads((BASE/'preview-selection.json').read_text())['small_area'];panels=[]
    for name in ids:
        idx=next(i for i,r in enumerate(data.rows) if r['id']==name)
        rgb,z,d=data.batch([idx],'cuda');raw=model.base(rgb,z);new=model.adjust(raw,z)[0]
        im=data.rgb[idx].permute(1,2,0).numpy();depth=data.depth[idx].numpy()
        known=np.isfinite(depth);truth=known&(depth<2);gt=im.copy();gt[truth]=[40,220,100]
        cols=[im.copy(),gt]
        for logits in [raw,new]:
            pred=m.probabilities(logits,'nfo')[0,2].cpu().numpy()>=CUT;vis=im.copy()
            for dm,color in [(pred&truth,[40,220,100]),(pred&~truth&known,[255,70,60]),(~pred&truth,[50,120,255])]:
                vis[dm]=(.25*vis[dm]+.75*np.array(color)).astype(np.uint8)
            vis[~known]=[160,80,180];cols.append(vis)
        row=np.concatenate(cols,1);header=np.full((30,row.shape[1],3),245,np.uint8)
        for j,label in enumerate(['RGB','GT <2m','Frozen NFO','Zone readout']):
            cv2.putText(header,label,(j*m.W+4,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        panels.extend([header,row])
    cv2.imwrite(str(OUT/'comparison.jpg'),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))


def infer(checkpoint,sample,output):
    model=ZoneReadout(m.Net('nfo'))
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['state_dict']);model.cuda().eval()
    with np.load(sample) as a:
        rgb=torch.from_numpy(a['rgb'].transpose(2,0,1).copy()[None]).cuda()
        zones=torch.from_numpy(m.public_zones(a['values'])[None]).cuda()
    with torch.inference_mode():scores=m.probabilities(model(rgb,zones),'nfo').cpu().numpy()[0]
    assert np.isfinite(scores).all() and np.all(scores[:-1]<=scores[1:])
    np.savez_compressed(output,scores=scores,masks=scores>=CUT,cutoff=np.array(CUT))
    print('PUBLIC_INFERENCE_PASS',list(scores.shape),torch.cuda.get_device_name())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Refuse repeat fit'
    torch.set_num_threads(4);assert torch.cuda.is_available()
    protocol=dict(id='ba-nfo-zone-readout-20260919',scope='One consumed synthetic Development contrast',
        hypothesis='Frozen NFO has local ranking but public zone summaries can align cross-zone score offsets',
        route_diagnostic_sha256=m.sha(m.ROOT/'artifacts.local/work/ba-nfo-local-diagnostic-20260919/results.json'),
        manifest_sha256=m.sha(BASE/'manifest.json'),baseline_sha256=m.sha(BASE/'trained-nfo.pt'),
        modification='Frozen trained NFO plus 8 trainable parameters; one bounded constant raw-logit offset per zone shared by four heads',
        features=['public range/8','public zone center x','public zone center y','original 2m score mean','std','maximum','fraction>=.081'],
        feature_scope='All pixels of public zone, no evaluator known/near/small masking; fixed natural scales, no fitted normalization',
        gate='valid public return>=2m; exact calibrated raster boxes; every eligible zone including pure far; same 2m gate for all heads',
        offset='2*tanh(linear7+bias); all8 parameters zero-initialized',
        initialization='original trained NFO frozen eval and no gradient, zero residual equals original logits',
        seed=m.SEED,train=3000,val=500,test=500,epochs=12,batch=24,updates=1500,
        optimizer='AdamW lr.002 weight_decay.0001 cosine to10% gradient clip5; last epoch only',
        loss='original all-known full-image four-head BCE + .2 ordinal; no depth auxiliary or subgroup reweighting',
        cutoff='Both .081 frozen from original NFO validation; no global recalibration; original >=95% val mixed recall required at this cutoff',
        joint_target='val mixed recall>=.95; far-small recall improves with FP<=65965; pure-far and all-public-far FP do not increase; mixed IoU retained and recall loss<=1pp',
        stop='One readout fit only; if unsuccessful do not switch to fine-scale fusion or tune offset/features/cutoff',
        backend='CUDA; reuse original encoder placement measurement',code_sha256=m.sha(__file__))
    m.write(OUT/'protocol.json',protocol)
    try:
        m.OUT=OUT;data=m.Data(json.loads((BASE/'manifest.json').read_text()))
        np.testing.assert_array_equal(data.boxes,public_boxes())
        model=ZoneReadout(load_base()).cuda()
        rgb,z,_=data.batch(list(range(2)),'cuda')
        with torch.inference_mode():assert torch.equal(model(rgb,z),model.base(rgb,z))
        train(data,model);evaluate(data,model)
        m.write(OUT/'completion.json',dict(status='COMPLETE',gate=json.loads((OUT/'results.json').read_text())['gate']))
    except BaseException as exc:
        m.write(OUT/'completion.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--sample');parser.add_argument('--output')
    args=parser.parse_args()
    if args.checkpoint:infer(args.checkpoint,args.sample,args.output)
    else:main()
