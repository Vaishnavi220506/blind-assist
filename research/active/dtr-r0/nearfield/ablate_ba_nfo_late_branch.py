"""Fixed-checkpoint branch-off ablation; no retraining, threshold or mixing sweep."""
import json
import time

import numpy as np
import torch

import ba_nfo_matched as m
import ba_nfo_hardfit as h
import ba_nfo_jointfit as j
import ba_nfo_latefusion as late
from audit_ba_nfo_spatial_tradeoff import transitions

OUT=m.ROOT/'artifacts.local/work/ba-nfo-late-branch-ablation-20260919'


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'One fixed branch-off ablation only'
    files=[late.OUT/'latefusion-nfo.pt',late.OUT/'fit-scores.npz',j.OUT/'fit-scores.npz',
           late.OUT/'selection.json',late.OUT/'results.json']
    protocol=dict(scope='EXPLORE_FIXED_CHECKPOINT_BRANCH_ABLATION_32TRAIN',
        question='Are the183 prior hits lost by final late-fusion output restored by disabling its residual branch?',
        intervention='Same final checkpoint and co-trained base weights; set only added residual logits to zero, equivalently forward through its base',
        cutoff=.081,near_threshold=2.,inputs={str(p):m.sha(p) for p in files},code_sha256=m.sha(__file__),
        limits='Test-time ablation of co-adapted components; not a counterfactual training run, causal history of drift, or a promotion result',
        decisions='If losses mostly recover: direct residual suppression warrants focus. If mostly remain: investigate jointly-trained base/branch co-adaptation before assuming output gating will fix it.',
        stop='One32-frame branch-off evaluation and exact additive-identity check; no fit, coefficient sweep, validation/test or successor',
        backend='CUDA; same architectures/input batch and GPU placement evidence as late-fusion contrast',
        placement_evidence=str(late.OUT/'backend.json'))
    m.write(OUT/'protocol.json',protocol)
    selected=json.loads((late.OUT/'selection.json').read_text())
    assert len(selected)==32 and all(r['row']['split']=='train' for r in selected)
    arrays=h.load_selection();torch.set_num_threads(4);assert torch.cuda.is_available()
    model=late.LateFusion(m.Net('nfo'))
    model.load_state_dict(torch.load(late.OUT/'latefusion-nfo.pt',map_location='cpu',weights_only=False)['state_dict'])
    model.cuda().eval()
    rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
    zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
    captured=[]
    handle=model.correction.register_forward_hook(lambda module,args,output:captured.append(output.detach()))
    full,off=[],[];max_error=0.;torch.cuda.synchronize();start=time.perf_counter()
    try:
        for i in range(0,32,8):
            logits=model(rgb[i:i+8],zones[i:i+8]);base=model.base(rgb[i:i+8],zones[i:i+8])
            residual=captured.pop()
            torch.testing.assert_close(logits,base+residual,rtol=0,atol=0)
            max_error=max(max_error,float((logits-base-residual).abs().max()))
            full.append(m.probabilities(logits,'nfo').cpu().numpy())
            off.append(m.probabilities(base,'nfo').cpu().numpy())
    finally:handle.remove()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    full,off=np.concatenate(full),np.concatenate(off)
    with np.load(late.OUT/'fit-scores.npz') as a:np.testing.assert_array_equal(full,a['scores'])
    with np.load(j.OUT/'fit-scores.npz') as a:previous=a['scores'].copy()
    assert np.isfinite(off).all() and (np.diff(off,axis=1)>=0).all()
    np.savez_compressed(OUT/'branch-off-scores.npz',scores=off)
    evaluation=h.evaluate(off,arrays,selected)
    parent=json.loads((late.OUT/'results.json').read_text())
    gates,good=j.train_gates(evaluation,parent['baseline'])
    totals={};frames=[]
    categories=dict(previous_lost_total=0,lost_restored_by_branch_off=0,lost_still_missed_branch_off=0,
        previous_rescued_total=0,rescued_require_branch=0,rescued_present_branch_off=0)
    for i,(a,row) in enumerate(zip(arrays,selected)):
        old=previous[i,2]>=.081;new=full[i,2]>=.081;base=off[i,2]>=.081
        known,truth,mixed,small=m.masks(a['depth'],a['boxes'],2.)
        y0,x0,y1,x1=row['box'];target=np.zeros_like(known);target[y0:y1,x0:x1]=True
        lost=small&truth&old&~new;rescued=small&truth&~old&new
        c=dict(previous_lost_total=int(lost.sum()),lost_restored_by_branch_off=int((lost&base).sum()),
            lost_still_missed_branch_off=int((lost&~base).sum()),previous_rescued_total=int(rescued.sum()),
            rescued_require_branch=int((rescued&~base).sum()),rescued_present_branch_off=int((rescued&base).sum()))
        for k in categories:categories[k]+=c[k]
        for name,mask in [('full',known),('small',small),('selected',target&known),('other_small',small&~target),
                          ('small_le1p8',small&truth&(a['depth']<=1.8)),('small_gt1p8',small&truth&(a['depth']>1.8))]:
            t=transitions(base,new,truth,mask)
            if name not in totals:totals[name]={k:0 for k in t}
            for k,v in t.items():totals[name][k]+=v
        frames.append(dict(id=row['row']['id'],kind=row['kind'],zone=row['zone'],**c))
    assert categories['previous_lost_total']==183 and categories['previous_rescued_total']==61
    assert categories['lost_restored_by_branch_off']+categories['lost_still_missed_branch_off']==183
    assert categories['rescued_require_branch']+categories['rescued_present_branch_off']==61
    # Zeroing only final correction in memory is a second implementation of the intervention.
    model.correction.weight.zero_();model.correction.bias.zero_()
    for i in range(0,32,8):
        p=m.probabilities(model(rgb[i:i+8],zones[i:i+8]),'nfo').cpu().numpy()
        np.testing.assert_array_equal(p,off[i:i+8])
    for path,sha in protocol['inputs'].items():assert m.sha(path)==sha
    result=dict(status='COMPLETE',branch_off=evaluation,branch_off_original_gates=gates,good_positive_zones=good,
        prior_joint=parent['previous_joint'],late_full=parent['late_fit'],previous_loss_attribution=categories,
        branch_off_to_on_transitions=totals,frames=frames,native_errors_branch_off=late.native_errors(off),
        complete_forward_matches_saved=True,zero_correction_matches_base_all32=True,additive_logits_identity_exact=True,
        subtraction_roundoff_max=max_error,device=torch.cuda.get_device_name(),paired_forward_seconds=elapsed,
        input_files_unchanged=True,validation_frames=0,test_frames=0,training_updates=0)
    m.write(OUT/'results.json',result)
    m.write(OUT/'completion.json',dict(status='COMPLETE',training_updates=0,validation_frames=0,test_frames=0))
    print(json.dumps(dict(attribution=categories,metrics=evaluation['metrics']['2.0'],good=good,
        branch_effect=totals,native=result['native_errors_branch_off']['totals'],seconds=elapsed),indent=2),flush=True)


if __name__=='__main__':main()
