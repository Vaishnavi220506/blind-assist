"""One trained RGB-only control; identical retained NFO supervision/budget."""
import json
import math
import time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ba_nfo_matched as m

BASE=m.OUT
OUT=m.ROOT/'artifacts.local/work/ba-nfo-rgb-control-20260919'
BINS=['0-5%','5-10%','10-20%','20-50%','50-100%']

class RGBOnly(m.Net):
    def __init__(self):
        # Construct the original initialized NFO first; remove ToF and its
        # input columns without perturbing any surviving parameter values.
        super().__init__('nfo')
        del self.tof
        old=self.fuse[0]
        new=nn.Conv2d(64,64,3,padding=1)
        with torch.no_grad():
            new.weight.copy_(old.weight[:,:64]);new.bias.copy_(old.bias)
        self.fuse[0]=new

    def forward(self,rgb,zones=None):
        x0=self.rgb0(rgb.float()/255.-.5)
        x1=self.rgb1(x0);x2=self.rgb2(x1);x3=self.rgb3(x2)
        x=self.fuse(x3)
        for skip,module in [(x2,self.up2),(x1,self.up1),(x0,self.up0)]:
            x=module(torch.cat([F.interpolate(x,skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.output(x)

def train(data):
    assert not (OUT/'rgb-only.pt').exists()
    m.seed();model=RGBOnly().cuda()
    # Initialization validation against B's seed path.
    m.seed();baseline=m.Net('nfo').cuda()
    for k,v in model.state_dict().items():
        expected=baseline.state_dict()[k]
        if k=='fuse.0.weight':expected=expected[:,:64]
        assert torch.equal(v,expected),k
    del baseline
    m.seed()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    rng=np.random.default_rng(m.SEED)
    ids=[i for i,r in enumerate(data.rows) if r['split']=='train']
    logs=[];start=time.perf_counter()
    for epoch in range(12):
        model.train();order=rng.permutation(ids).tolist();total=0.
        for g in opt.param_groups:g['lr']=.002*(.1+.9*(1+math.cos(math.pi*epoch/11))/2)
        for startidx in range(0,len(order),24):
            rgb,_,d=data.batch(order[startidx:startidx+24],'cuda')
            opt.zero_grad(set_to_none=True);loss=m.loss_fn(model(rgb),d,'nfo')
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())
        logs.append(dict(epoch=epoch+1,loss=total/125,seconds=time.perf_counter()-start))
        m.write(OUT/'training-progress.json',logs);print('RGB_EPOCH',logs[-1],flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),architecture='RGBOnly',
        protocol=json.loads((OUT/'protocol.json').read_text()),logs=logs),OUT/'rgb-only.pt')
    return model.cuda().eval()

@torch.inference_mode()
def evaluate(data,model):
    val=[i for i,r in enumerate(data.rows) if r['split']=='val']
    hist=m.histogram(model,data,val,'cuda')
    cal=m.calibration(hist)
    np.save(OUT/'validation-histogram.npy',hist);m.write(OUT/'calibration.json',cal)
    ids=[i for i,r in enumerate(data.rows) if r['split']=='test']
    sums=defaultdict(lambda:np.zeros(4,np.int64));totals=defaultdict(lambda:np.zeros(4,np.int64))
    cache=OUT/'scores';cache.mkdir(exist_ok=True)
    for start in range(0,len(ids),24):
        ii=ids[start:start+24];rgb,_,_=data.batch(ii,'cuda')
        scores=m.probabilities(model(rgb),'nfo').cpu().numpy()
        for idx,p in zip(ii,scores):
            np.save(cache/f'{data.rows[idx]["id"]}.npy',p)
            d=data.depth[idx].numpy();values=data.zones[idx,0].numpy().ravel()*8
            valid=data.zones[idx,1].numpy().ravel().astype(bool)
            for ti,t in enumerate(m.THRESHOLDS):
                known,truth,mixed,thin=m.masks(d,data.boxes,float(t));pred=p[ti]>=cal['cutoff']
                for name,dom in [('full',known),('mixed',mixed),('small',thin)]:
                    totals[str(float(t)),name]+=m.counts(pred,truth,dom)
                for zi,(y0,x0,y1,x1) in enumerate(data.boxes):
                    k=known[y0:y1,x0:x1];n=truth[y0:y1,x0:x1];count=int(n.sum());den=int(k.sum())
                    if not 0<count<den:continue
                    b=BINS[int(np.searchsorted([.05,.1,.2,.5],count/den,side='left'))]
                    state='missing' if not valid[zi] else ('near' if values[zi]<t else 'far')
                    c=m.counts(pred[y0:y1,x0:x1],n,k)
                    for group in ['all',state]:sums[str(float(t)),b,group]+=c
        if start%120==0:print('EVAL',start,flush=True)
    records=[dict(threshold=t,area=b,return_state=s,**m.metrics(c)) for (t,b,s),c in sums.items()]
    small={s:m.metrics(sum((sums['2.0',b,s] for b in BINS[:3]),np.zeros(4,np.int64))) for s in ['all','near','far','missing']}
    result=dict(calibration=cal,records=records,small_2m=small,
        totals={t:{n:m.metrics(c) for (tt,n),c in totals.items() if tt==t} for t in ['1.0','1.5','2.0','3.0']},
        device=torch.cuda.get_device_name(),checkpoint_sha256=m.sha(OUT/'rgb-only.pt'))
    # Predeclared gate demands recall recovery AND meaningful discrimination;
    # otherwise broad RGB overprediction cannot justify a non-veto mechanism.
    far=small['far'];base=json.loads((BASE/'results.json').read_text())
    result['nonveto_trigger']=bool(cal['feasible'] and far['recall']>=.80 and far['iou']>=9651/(9651+65965+4213))
    m.write(OUT/'results.json',result)
    print(json.dumps(dict(small_2m=small,totals=result['totals']['2.0'],nonveto_trigger=result['nonveto_trigger']),indent=2),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    assert not (OUT/'protocol.json').exists(),'Refuse accidental repeat'
    rows=json.loads((BASE/'manifest.json').read_text())
    m.write(OUT/'protocol.json',dict(id='ba-nfo-rgb-control-20260919',phase='consumed synthetic Development',
        source_manifest_sha256=m.sha(BASE/'manifest.json'),seed=m.SEED,epochs=12,batch=24,
        optimizer='Original AdamW and cosine schedule',loss='Unchanged NFO BCE plus .2 ordinal',
        initialization='identical remaining original NFO initial parameters; remove ToF encoder and 16 fusion input columns',
        selection='last epoch only',calibration='original validation 2m mixed recall>=.95 grid; one fixed cutoff all thresholds',
        scope='one RGB-only training; no test-time modality masking substitute',
        trigger='Nonveto candidate only if RGB-only far-return <=20% recall>=80% AND IoU>=original NFO subgroup at validation-selected cutoff',
        limitations='Observational subgroup and trained ablation not decisive proof of suppression, information absence or hardware semantics',
        backend='CUDA; reuse original identical encoder-decoder workload benchmark',code_sha256=m.sha(__file__)))
    # Avoid modifying original admission/evidence files.
    m.OUT=OUT
    data=m.Data(rows)
    model=train(data);evaluate(data,model)

if __name__=='__main__':main()
