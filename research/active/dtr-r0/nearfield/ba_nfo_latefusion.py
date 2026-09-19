"""One fixed full-resolution ToF/RGB residual-branch contrast on 32 TRAIN frames."""
import copy
import json
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ba_nfo_matched as m
import ba_nfo_hardfit as h
import ba_nfo_jointfit as j
from ba_nfo_targetfit import target_masks
from ba_nfo_zone_readout import load_base

OUT = m.ROOT/'artifacts.local/work/ba-nfo-latefusion32-20260919'
AUDIT = m.ROOT/'artifacts.local/work/ba-nfo-native-support-audit-20260919'


def zone_index(height=192, width=256):
    """Exact public sensor tiling, with -1 outside coverage; no scene truth."""
    ys = np.linspace(round(.1*height), round(.9*height), 9).astype(int)
    xs = np.linspace(round(.1*width), round(.9*width), 9).astype(int)
    index = np.full((height, width), -1, np.int64)
    for y in range(8):
        for x in range(8):
            index[ys[y]:ys[y+1], xs[x]:xs[x+1]] = 8*y+x
    return torch.from_numpy(index)


class LateFusion(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.detail = m.block(44, 16)
        self.correction = nn.Conv2d(16, 4, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.register_buffer('index', zone_index())

    def broadcast(self, z):
        idx = self.index.flatten()
        projected = z.flatten(2)[:, :, idx.clamp_min(0)]
        return (projected*(idx >= 0)[None, None]).reshape(z.shape[0], 16, *self.index.shape)

    def forward(self, rgb, zones):
        assert rgb.shape[-2:] == self.index.shape
        b = self.base
        x0 = b.rgb0(rgb.float()/255.-.5)
        x1 = b.rgb1(x0); x2 = b.rgb2(x1); x3 = b.rgb3(x2)
        encoded = b.tof(zones)
        # Preserve the original coarse path, including its original projection.
        hh, ww = x3.shape[-2:]
        y0, y1, xx0, xx1 = round(.1*hh), round(.9*hh), round(.1*ww), round(.9*ww)
        z = F.interpolate(encoded, (y1-y0, xx1-xx0), mode='nearest')
        z = F.pad(z, (xx0, ww-xx1, y0, hh-y1))
        x = b.fuse(torch.cat([x3, z], 1))
        for skip, module in [(x2, b.up2), (x1, b.up1), (x0, b.up0)]:
            x = module(torch.cat([F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        residual = self.correction(self.detail(torch.cat([x, x0, self.broadcast(encoded)], 1)))
        return b.output(x)+residual


def native_errors(scores):
    totals = dict(empty_near_fp=0, original_high_coverage_fn_recovered=0, original_mixed_fn_recovered=0)
    rows = []
    for i in range(16):
        with np.load(AUDIT/f'{i:02d}-support.npz') as a:
            fraction, truth, known, previous = [a[k] for k in ['fraction','truth','known','prediction']]
        selected = json.loads((OUT/'selection.json').read_text())[i]
        y0, x0, y1, x1 = selected['box']; pred = scores[i, 2, y0:y1, x0:x1] >= h.CUT
        old_fn = ~previous & truth
        c = dict(empty_near_fp=int((pred&~truth&known&(fraction == 0)).sum()),
            original_high_coverage_fn_recovered=int((pred&old_fn&(fraction >= .5)).sum()),
            original_mixed_fn_recovered=int((pred&old_fn&(fraction < .5)).sum()))
        for k in totals: totals[k] += c[k]
        rows.append(dict(id=selected['row']['id'], **c))
    return dict(totals=totals, rows=rows)


def placement(model, rgb, zones, depth, target):
    sys.path.insert(0, str(m.ROOT/'tools'))
    from research_backend import BackendCandidate, select_backend, torch_observation
    models = {dev:copy.deepcopy(model).to(dev) for dev in ['cpu','cuda']}
    batches = {dev:[x[:8].to(dev) for x in [rgb,zones,depth,target]] for dev in models}
    def probe(dev):
        net = models[dev]; net.zero_grad(set_to_none=True)
        r,z,d,t = batches[dev]; output = net(r,z)
        j.joint_loss(output,d,t)[0].backward()
        return output.detach()
    candidates = {dev:BackendCandidate('torch-'+dev, dev, lambda d=dev:probe(d),
        lambda output:torch_observation(output=output), torch.cuda.synchronize if dev=='cuda' else lambda:None)
        for dev in models}
    receipt = select_backend('batch-tensor',cpu=candidates['cpu'],gpu=candidates['cuda'],
        record_path=OUT/'backend.json',warmups=1,repeats=2)
    assert receipt['selected_device_type']=='cuda'
    del models, batches
    torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'One fixed architecture trial only'
    torch.set_num_threads(4); assert torch.cuda.is_available(); m.seed()
    model = LateFusion(load_base())
    parent = json.loads((j.OUT/'protocol.json').read_text())
    protocol = dict(id='ba-nfo-latefusion32-20260919', scope='EXPLORE_ONE_ARCHITECTURE_TRAIN_CONTRAST',
        hypothesis='Direct late conditioning by the same regional ToF plus full-resolution RGB features can reduce spill and recover near support',
        architecture='Original NFO + block(44,16) and zero 1x1(16,4) residual logits; inputs final decoder16/RGB skip12/exactly broadcast shared encoded ToF16',
        limitations='Additional capacity and late conditioning are coupled; original network already has full-resolution RGB skips. No new sensory evidence; seen TRAIN fit only.',
        unchanged='32 cases, original base checkpoint, 256x192 RGB/public8x8ToF, exact512batchplan, 0.5full+0.5target loss, optimizer, seed, cutoff0.081',
        training=parent['training'], selection_sha256=m.sha(j.OUT/'selection.json'),
        batch_plan_sha256=m.sha(j.OUT/'batch-plan.json'), original_checkpoint_sha256=m.sha(m.OUT/'trained-nfo.pt'),
        joint_checkpoint_sha256=m.sha(j.OUT/'jointfit-nfo.pt'), joint_results_sha256=m.sha(j.OUT/'results.json'),
        audit_results_sha256=m.sha(AUDIT/'results.json'), code_sha256=m.sha(__file__),
        base_parameters=sum(p.numel() for p in model.base.parameters()), parameters=sum(p.numel() for p in model.parameters()),
        gates=parent['train_gates'], paired_guard='Also retain joint-fit target recall/IoU, full recall and pure-far FP for a positive paired result',
        diagnostics='Original empty-native-near FP125 and four high-coverage FN; descriptive only, not a new cutoff or relabeling',
        stop='One512-update last-checkpoint fit, then verification and delivery. No variant, ratio, cutoff, longer fit, val/test access or automatic successor.')
    m.write(OUT/'protocol.json',protocol)
    for name in ['selection.json','batch-plan.json']:
        shutil.copyfile(j.OUT/name,OUT/name)
    selected=json.loads((OUT/'selection.json').read_text()); batches=json.loads((OUT/'batch-plan.json').read_text())
    assert len(selected)==32 and len(batches)==512 and all(r['row']['split']=='train' for r in selected)
    assert all(r['kind']=='positive' for r in selected[:16])
    arrays=h.load_selection()
    rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays]))
    zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays]))
    depth=torch.from_numpy(np.stack([a['depth'] for a in arrays]))
    target=torch.from_numpy(target_masks(selected,arrays[0]['depth'].shape))
    for a in arrays:
        expected=np.full((192,256),-1,np.int64)
        for zi,(y0,x0,y1,x1) in enumerate(a['boxes']): expected[y0:y1,x0:x1]=zi
        np.testing.assert_array_equal(model.index.numpy(),expected)
    placement(model,rgb,zones,depth,target)
    rgb,zones,depth,target=[a.cuda() for a in [rgb,zones,depth,target]];model.cuda()
    before=h.scores(model,rgb,zones)
    with np.load(h.OUT/'original-scores.npz') as a: np.testing.assert_array_equal(before,a['scores'])
    baseline=h.evaluate(before,arrays,selected)
    assert baseline==json.loads((h.OUT/'baseline.json').read_text())
    m.write(OUT/'fit-start.json',dict(initial_scores_bit_identical=True,public_zone_mapping_exact=True,protocol_sha256=m.sha(OUT/'protocol.json')))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train(); logs=[];torch.cuda.synchronize();start=time.perf_counter()
    for step,ii in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss,full,local=j.joint_loss(model(rgb[ii],zones[ii]),depth[ii],target[ii])
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        logs.append(dict(step=step+1,loss=float(loss.detach()),full=float(full.detach()),target=float(local.detach())))
        if (step+1)%128==0:
            m.write(OUT/'progress.json',logs);print('LATE_FIT',logs[-1],flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(state_dict=model.cpu().state_dict(),protocol=protocol,cutoff=h.CUT),OUT/'latefusion-nfo.pt')
    model.cuda();after=h.scores(model,rgb,zones);fit=h.evaluate(after,arrays,selected)
    np.savez_compressed(OUT/'fit-scores.npz',scores=after)
    gates,good=j.train_gates(fit,baseline)
    previous=json.loads((j.OUT/'results.json').read_text())['joint_fit']
    old,new=previous['metrics']['2.0'],fit['metrics']['2.0']
    paired=dict(target_recall_retained=new['target_positive']['recall']>=old['target_positive']['recall'],
        target_iou_retained=new['target_positive']['iou']>=old['target_positive']['iou'],
        full_recall_retained=new['full']['recall']>=old['full']['recall'],
        pure_far_fp_retained=new['pure_far']['fp']<=old['pure_far']['fp'])
    result=dict(baseline=baseline,previous_joint=previous,late_fit=fit,gates=gates,paired_guards=paired,
        pass_all=all(gates.values()) and all(paired.values()),good_positive_zones=good,steps=512,
        seconds=elapsed,device=torch.cuda.get_device_name(),native_errors=native_errors(after),
        original_checkpoint_unchanged=m.sha(m.OUT/'trained-nfo.pt')==protocol['original_checkpoint_sha256'],
        joint_checkpoint_unchanged=m.sha(j.OUT/'jointfit-nfo.pt')==protocol['joint_checkpoint_sha256'])
    m.write(OUT/'results.json',result)
    original_out=h.OUT
    try:
        h.OUT=OUT
        h.render(selected,arrays,before,after)
    finally:h.OUT=original_out
    m.write(OUT/'preview-columns.json',dict(columns=['fullRGB','zoneRGB','nearGT','original_nfo','late_fusion']))
    m.write(OUT/'completion.json',dict(status='COMPLETE',positive_paired_result=result['pass_all'],validation_frames=0,test_frames=0))
    print('LATE_RESULT',json.dumps(dict(gates=gates,paired=paired,good=good,metrics=new,native=result['native_errors']['totals'],seconds=elapsed)),flush=True)


if __name__=='__main__':main()
