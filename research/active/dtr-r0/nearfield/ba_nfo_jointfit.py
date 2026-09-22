"""One fixed half-full, half-target supervision fit; conditional frozen validation."""
import json
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

import ba_nfo_matched as m
import ba_nfo_hardfit as h
from ba_nfo_targetfit import target_loss, target_masks, OUT as TARGET_REFERENCE
from ba_nfo_zone_readout import load_base
from ba_nfo_zcr import domains

BASE=m.OUT
REFERENCE=h.OUT
OUT=m.ROOT/'artifacts.local/work/ba-nfo-jointfit32-20260919'


def joint_loss(output,depth,target):
    full=m.loss_fn(output,depth,'nfo')
    local=target_loss(output,depth,target)
    return .5*full+.5*local,full,local


def train_gates(fit,baseline):
    d=fit['metrics']['2.0'];b=baseline['metrics']['2.0']
    good=sum(f['metrics']['target_positive']['recall']>=.9 and f['metrics']['target_positive']['iou']>=.5
             for f in fit['frames'] if f['kind']=='positive')
    return dict(target_recall95=d['target_positive']['recall']>=.95,
        target_iou65=d['target_positive']['iou']>=.65,
        negative_fpr1=d['target_negative']['false_positive_rate']<=.01,
        positive_zones14=good>=14,
        full_recall_retained=d['full']['recall']>=b['full']['recall'],
        pure_far_fp_retained=d['pure_far']['fp']<=b['pure_far']['fp']),int(good)


@torch.inference_mode()
def validate(model):
    """Only callable after the sealed training gate; no calibration anywhere."""
    training=json.loads((OUT/'results.json').read_text())
    assert training['pass_all'],'Validation blocked by predeclared train gate'
    rows=[r for r in json.loads((BASE/'manifest.json').read_text()) if r['split']=='val']
    selected=json.loads((OUT/'selection.json').read_text())
    assert len(rows)==500 and not {r['id'] for r in rows}&{r['row']['id'] for r in selected}
    assert not (OUT/'validation.json').exists(),'One validation pass only'
    baseline=load_base().cuda().eval();model.eval()
    totals=defaultdict(lambda:np.zeros(4,np.int64));frames=[];unknown=0
    for start in range(0,500,24):
        batch=rows[start:start+24];arrays=[]
        for row in batch:
            path=m.OLD/row['prepared'];assert m.sha(path)==row['sha256']
            with np.load(path) as a:arrays.append({k:a[k].copy() for k in ['rgb','depth','values','boxes']})
        rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
        z=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
        ps={name:m.probabilities(net(rgb,z),'nfo')[:,2].cpu().numpy() for name,net in [('nfo',baseline),('joint',model)]}
        for j,(row,a) in enumerate(zip(batch,arrays)):
            truth,dm=domains(a);unknown+=int((~dm['full']).sum());record={name:{} for name in ps}
            for name,p in ps.items():
                for domain,mask in dm.items():
                    c=m.counts(p[j]>=h.CUT,truth,mask);totals[name,domain]+=c;record[name][domain]=m.metrics(c)
            frames.append(dict(id=row['id'],metrics=record))
    metrics={name:{domain:m.metrics(c) for (arm,domain),c in totals.items() if arm==name} for name in ps}
    expected=json.loads((m.ROOT/'artifacts.local/work/ba-nfo-zcr-20260919/calibration.json').read_text())['baseline']
    assert metrics['nfo']['mixed']==expected
    b,n=metrics['nfo'],metrics['joint']
    flags=dict(far_small_iou_improves=n['far_small']['iou']>b['far_small']['iou'],
        far_small_recall_retained=n['far_small']['recall']>=b['far_small']['recall'],
        full_recall_retained=n['full']['recall']>=b['full']['recall'],pure_far_fp_retained=n['pure_far']['fp']<=b['pure_far']['fp'])
    m.write(OUT/'validation.json',dict(status='COMPLETE',frames=500,cutoff=h.CUT,metrics=metrics,
        diagnostic_checks=flags,joint_transfer_supported=all(flags.values()),unknown_pixels=unknown,
        scope='Previously consumed validation families; fixed model/cutoff, no fresh confirmation or tuning'))
    m.write(OUT/'validation-frames.json',frames)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'One fixed fit only'
    old=json.loads((REFERENCE/'protocol.json').read_text())
    protocol=dict(id='ba-nfo-jointfit32-20260919',scope='EXPLORE_FIXED_JOINT_LOSS_TRAIN_CHECK',
        hypothesis='Retain global supervision while emphasizing selected hard zones to improve local fit without whole-image collapse',
        formula='0.5 * original full-known four-head BCE+.2ordinal + 0.5 * identical loss on selected complete zone known pixels',
        weights=[.5,.5],unchanged='Same32 TRAIN identities, zones, full RGB/actual ToF, original checkpoint initialization, architecture,512batches/updates, optimizer and cutoff.081',
        training=old['training'],selection_sha256=m.sha(REFERENCE/'selection.json'),
        source_checkpoint_sha256=old['checkpoint_sha256'],batch_plan_sha256=m.sha(TARGET_REFERENCE/'batch-plan.json'),
        train_gates=old['gates']+'; full-image recall>=original NFO on same32; all-pure-far FP<=original NFO on same32',
        validation='Only if all6train gates pass: one frozen original500validation-frame evaluation at.081. No threshold/model selection. Compare far-small IoU/recall, full recall and pure-far FP with original NFO; all must improve/retain respectively for a positive diagnostic.',
        stop='If train gate fails, skip validation and end this supervision-allocation sequence. No other mixing ratio, extra steps, cutoff, loss or architecture trial.',
        limits='Seen32cases first; validation already consumed, not fresh confirmation; original NFO and App defaults retained',
        code_sha256=m.sha(__file__),backend='CUDA, original same-network placement evidence',placement_evidence=str(BASE/'backend.json'))
    m.write(OUT/'protocol.json',protocol)
    for filename,source in [('selection.json',REFERENCE),('batch-plan.json',TARGET_REFERENCE)]:
        shutil.copyfile(source/filename,OUT/filename)
    selection=json.loads((OUT/'selection.json').read_text());batches=json.loads((OUT/'batch-plan.json').read_text())
    arrays=h.load_selection();assert len(batches)==512 and all(r['row']['split']=='train' for r in selection)
    torch.set_num_threads(4);assert torch.cuda.is_available();m.seed()
    rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
    zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
    depth=torch.from_numpy(np.stack([a['depth'] for a in arrays])).cuda()
    target=torch.from_numpy(target_masks(selection,arrays[0]['depth'].shape)).cuda()
    model=load_base().cuda();before=h.scores(model,rgb,zones)
    with np.load(REFERENCE/'original-scores.npz') as a:np.testing.assert_array_equal(before,a['scores'])
    baseline=h.evaluate(before,arrays,selection)
    assert baseline==json.loads((REFERENCE/'baseline.json').read_text())
    assert m.sha(BASE/'trained-nfo.pt')==protocol['source_checkpoint_sha256']
    m.write(OUT/'baseline.json',baseline)
    m.write(OUT/'fit-start.json',dict(protocol_sha256=m.sha(OUT/'protocol.json'),initial_scores_exact=True,device=torch.cuda.get_device_name()))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001);logs=[];model.train();start=time.perf_counter()
    for step,ii in enumerate(batches):
        opt.zero_grad(set_to_none=True);loss,full,local=joint_loss(model(rgb[ii],zones[ii]),depth[ii],target[ii])
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        logs.append(dict(step=step+1,loss=float(loss.detach()),full_loss=float(full.detach()),target_loss=float(local.detach())))
        if (step+1)%128==0:
            m.write(OUT/'progress.json',logs);print('JOINT_FIT',logs[-1],flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(state_dict=model.cpu().state_dict(),protocol=protocol,cutoff=h.CUT),OUT/'jointfit-nfo.pt')
    model.cuda();after=h.scores(model,rgb,zones);fit=h.evaluate(after,arrays,selection)
    np.savez_compressed(OUT/'fit-scores.npz',scores=after)
    gates,good=train_gates(fit,baseline)
    result=dict(baseline=baseline,joint_fit=fit,gates=gates,pass_all=all(gates.values()),
        good_positive_zones=good,steps=512,fit_seconds=elapsed,device=torch.cuda.get_device_name(),
        initialization_scores_exact=True,original_checkpoint_unchanged=m.sha(BASE/'trained-nfo.pt')==protocol['source_checkpoint_sha256'])
    m.write(OUT/'results.json',result)
    try:
        h.OUT=OUT;h.render(selection,arrays,before,after)
    finally:h.OUT=REFERENCE
    if result['pass_all']:validate(model)
    else:m.write(OUT/'validation-status.json',dict(status='SKIPPED_TRAIN_GATE_NOT_MET',frames_read=0))
    m.write(OUT/'completion.json',dict(status='COMPLETE',train_gates_pass=result['pass_all'],
        validation='COMPLETE' if result['pass_all'] else 'SKIPPED_TRAIN_GATE_NOT_MET'))
    print('JOINT_FIT_RESULT',json.dumps(dict(gates=gates,metrics=fit['metrics']['2.0'],good=good,seconds=elapsed)),flush=True)


if __name__=='__main__':main()
