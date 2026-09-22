"""One paired hardfit contrast: change only the supervised pixel domain."""
import json
import shutil
import time

import numpy as np
import torch

import ba_nfo_matched as m
import ba_nfo_hardfit as h
from ba_nfo_zone_readout import load_base

BASE = m.OUT
REFERENCE = h.OUT
OUT = m.ROOT/'artifacts.local/work/ba-nfo-targetfit32-20260919'


def target_loss(output, depth, target):
    """Same four-head BCE and ordinal formula; exclude outside labels only."""
    assert target.shape == depth.shape and target.dtype == torch.bool
    restricted = depth.masked_fill(~target, float('nan'))
    assert (torch.isfinite(restricted) & (restricted > 0)).any()
    return m.loss_fn(output, restricted, 'nfo')


def target_masks(selection, shape):
    masks = np.zeros((len(selection), *shape), bool)
    for i, row in enumerate(selection):
        y0, x0, y1, x1 = row['box']
        masks[i, y0:y1, x0:x1] = True
    return masks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'One fit only; refuse repeated run'
    old_protocol = json.loads((REFERENCE/'protocol.json').read_text())
    assert m.sha(BASE/'trained-nfo.pt') == old_protocol['checkpoint_sha256']
    protocol = dict(id='ba-nfo-targetfit32-20260919',phase='EXPLORE_TRAIN_ONLY_PAIRED_SUPERVISION_DOMAIN',
        question='Does restricting the original loss to the same selected zones improve hard-case fitting?',
        reference=str(REFERENCE),reference_protocol_sha256=m.sha(REFERENCE/'protocol.json'),
        reference_result_sha256=m.sha(REFERENCE/'results.json'),
        selection_sha256=m.sha(REFERENCE/'selection.json'),source_checkpoint_sha256=old_protocol['checkpoint_sha256'],
        change='Only training loss support: selected complete zone known pixels instead of full-image known pixels',
        unchanged='Original32 TRAIN frames/zone IDs, full256x192RGB, actual public8x8ToF, architecture, original checkpoint initialization, four heads, BCE+.2ordinal formula, AdamW, seed, batch order,512steps, final checkpoint, cutoff.081 and diagnostic gates',
        loss='Original pixel mean across selected known pixels; both near and far labels, all16textured pure-far controls, no class weighting or learned zone weights',
        boundary='Zone mask and depth truth enter loss/evaluator only, never network forward; no validation/test read',
        training=old_protocol['training'],gates=old_protocol['gates'],cutoff=h.CUT,
        decision='PASS supports supervision-domain contribution on seen cases; FAIL ends this contrast without tuning; neither is generalization or information-ceiling proof',
        stop='One512-step fit only; no follow-on fit, loss weight, cutoff or resolution sweep',
        backend='CUDA same network fit; retain original placement evidence',placement_evidence=str(BASE/'backend.json'),
        code_sha256=m.sha(__file__))
    m.write(OUT/'protocol.json',protocol)
    shutil.copyfile(REFERENCE/'selection.json',OUT/'selection.json')
    selection=json.loads((OUT/'selection.json').read_text())
    arrays=h.load_selection()
    assert len(selection)==32 and all(r['row']['split']=='train' for r in selection)
    assert m.sha(OUT/'selection.json')==protocol['selection_sha256']
    torch.set_num_threads(4);assert torch.cuda.is_available();m.seed()
    rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
    zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
    depth=torch.from_numpy(np.stack([a['depth'] for a in arrays])).cuda()
    target=torch.from_numpy(target_masks(selection,arrays[0]['depth'].shape)).cuda()
    assert rgb.is_contiguous()
    np.savez_compressed(OUT/'loss-support.npz',target=target.cpu().numpy(),
        known_target=(target & torch.isfinite(depth) & (depth>0)).cpu().numpy())
    model=load_base().cuda()
    before=h.scores(model,rgb,zones)
    with np.load(REFERENCE/'original-scores.npz') as a:np.testing.assert_array_equal(before,a['scores'])
    baseline=h.evaluate(before,arrays,selection)
    assert baseline==json.loads((REFERENCE/'baseline.json').read_text())
    m.write(OUT/'baseline.json',baseline)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    rng=np.random.default_rng(m.SEED)
    batches=[rng.choice(32,8,replace=False).tolist() for _ in range(512)]
    m.write(OUT/'batch-plan.json',batches)
    m.write(OUT/'fit-start.json',dict(protocol_sha256=m.sha(OUT/'protocol.json'),
        selection_sha256=m.sha(OUT/'selection.json'),batch_plan_sha256=m.sha(OUT/'batch-plan.json'),
        baseline_scores_bit_identical=True,device=torch.cuda.get_device_name()))
    logs=[];start=time.perf_counter();model.train()
    for step,ii in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss=target_loss(model(rgb[ii],zones[ii]),depth[ii],target[ii]);assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        logs.append(dict(step=step+1,loss=float(loss.detach())))
        if (step+1)%128==0:
            m.write(OUT/'progress.json',logs);print('TARGET_FIT',logs[-1],flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(state_dict=model.cpu().state_dict(),protocol=protocol,cutoff=h.CUT),OUT/'targetfit-nfo.pt')
    model.cuda();after=h.scores(model,rgb,zones)
    np.savez_compressed(OUT/'fit-scores.npz',scores=after)
    new=h.evaluate(after,arrays,selection)
    reference=json.loads((REFERENCE/'results.json').read_text())['fit']
    p,n=new['metrics']['2.0']['target_positive'],new['metrics']['2.0']['target_negative']
    good=sum(f['metrics']['target_positive']['recall']>=.9 and f['metrics']['target_positive']['iou']>=.5
             for f in new['frames'] if f['kind']=='positive')
    gates=dict(target_recall95=p['recall']>=.95,target_iou65=p['iou']>=.65,
               negative_fpr1=n['false_positive_rate']<=.01,positive_zones14=good>=14)
    assert m.sha(BASE/'trained-nfo.pt')==protocol['source_checkpoint_sha256']
    result=dict(baseline=baseline,full_image_fit=reference,target_fit=new,gates=gates,
        pass_all=all(gates.values()),good_positive_zones=int(good),steps=512,fit_seconds=elapsed,
        device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
        supervised_known_pixels=int((target&torch.isfinite(depth)&(depth>0)).sum()),
        full_known_pixels=int((torch.isfinite(depth)&(depth>0)).sum()),
        unknown_pixels=int((~torch.isfinite(depth)).sum()),
        initialization_scores_exact=True,original_checkpoint_unchanged=True,
        scope='32seen TRAIN frames only; no validation/test inference or promotion')
    m.write(OUT/'results.json',result)
    try:
        h.OUT=OUT
        h.render(selection,arrays,before,after)
    finally:
        h.OUT=REFERENCE
    m.write(OUT/'completion.json',dict(status='COMPLETE',pass_all=result['pass_all']))
    print('TARGET_FIT_RESULT',json.dumps(dict(gates=gates,metrics=new['metrics']['2.0'],good=good,seconds=elapsed)),flush=True)


if __name__=='__main__':main()
