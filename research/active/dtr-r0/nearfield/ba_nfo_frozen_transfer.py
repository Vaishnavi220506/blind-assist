"""One frozen three-model transfer check on the original consumed 500-val set.

This is a new descriptive Development comparison, not promotion of failed fit32
gates, fresh confirmation, checkpoint selection, training or recalibration.
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch

import ba_nfo_matched as m
import ba_nfo_jointfit as j
import ba_nfo_latefusion as late
from ba_nfo_zone_readout import load_base, public_boxes
from ba_nfo_zcr import domains
from audit_ba_nfo_spatial_tradeoff import transitions

OUT=m.ROOT/'artifacts.local/work/ba-nfo-frozen-transfer500-20260919'
ARMS=['nfo','joint','late']
CUT=.081


def load_models():
    models={'nfo':load_base(),'joint':m.Net('nfo'),'late':late.LateFusion(m.Net('nfo'))}
    models['joint'].load_state_dict(torch.load(j.OUT/'jointfit-nfo.pt',map_location='cpu',weights_only=False)['state_dict'])
    models['late'].load_state_dict(torch.load(late.OUT/'latefusion-nfo.pt',map_location='cpu',weights_only=False)['state_dict'])
    return {name:net.cuda().eval().requires_grad_(False) for name,net in models.items()}


def evaluation_domains(a):
    truth,dm=domains(a)
    separated=np.zeros_like(truth)
    for zi,(y0,x0,y1,x1) in enumerate(a['boxes']):
        sl=np.s_[y0:y1,x0:x1];mask=dm['far_small'][sl]
        if not mask.any() or not np.isfinite(a['values'][zi]) or a['values'][zi]<2.2:continue
        near=a['depth'][sl][mask&truth[sl]];far=a['depth'][sl][mask&~truth[sl]]
        if len(near) and len(far):
            q90,q10=float(np.quantile(near,.9)),float(np.quantile(far,.1))
            if q90<=1.8 and q10>=2.2 and q10-q90>=.5:separated[sl]=mask
    dm.update(far_small_separated=separated,far_small_other=dm['far_small']&~separated,
        far_small_near_le1p8=dm['far_small']&truth&(a['depth']<=1.8),
        far_small_near_gt1p8=dm['far_small']&truth&(a['depth']>1.8))
    return truth,dm


def gates(metrics,baseline):
    return dict(far_small_recall75=metrics['far_small']['recall']>=.75,
        far_small_iou_retained=metrics['far_small']['iou']>=baseline['far_small']['iou'],
        mixed_recall945=metrics['mixed']['recall']>=.945,
        pure_far_fp_retained=metrics['pure_far']['fp']<=baseline['pure_far']['fp'])


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'One sealed comparison only'
    manifest=json.loads((m.OUT/'manifest.json').read_text())
    rows=[r for r in manifest if r['split']=='val'];train=[r for r in manifest if r['split']=='train']
    fit=[r['row'] for r in json.loads((j.OUT/'selection.json').read_text())]
    assert len(rows)==500 and len(train)==3000 and len(fit)==32
    disjoint={}
    for key in ['id','scene','family']:
        disjoint[key]=dict(val_vs_fit32=len({r[key] for r in rows}&{r[key] for r in fit}),
                          val_vs_train3000=len({r[key] for r in rows}&{r[key] for r in train}))
        assert not any(disjoint[key].values())
    weights={'nfo':m.OUT/'trained-nfo.pt','joint':j.OUT/'jointfit-nfo.pt','late':late.OUT/'latefusion-nfo.pt'}
    expected_path=m.ROOT/'artifacts.local/work/ba-nfo-zone-readout-20260919/results.json'
    expected=json.loads(expected_path.read_text())['metrics']['val']['nfo']['2.0']
    protocol=dict(id='ba-nfo-frozen-transfer500-20260919',scope='EXPLORE_CONSUMED_DEVELOPMENT_FROZEN_TRANSFER',
        question='Do the selected32-frame gains transfer to the complete original task outside their training families?',
        authority='User explicitly authorized frozen-model transfer check after retrospective; independent diagnostic, not reopening previous14/16 fit gates',
        cohort='All original500val frames in original manifest order; previously consumed for calibration/evaluation, not freshblind',
        frames=500,families=len({r['family'] for r in rows}),scenes=len({r['scene'] for r in rows}),disjoint=disjoint,
        checkpoint_sha256={name:m.sha(p) for name,p in weights.items()},manifest_sha256=m.sha(m.OUT/'manifest.json'),
        expected_baseline_sha256=m.sha(expected_path),code_sha256=m.sha(__file__),arms=ARMS,cutoff=CUT,threshold_m=2.,batch_size=24,
        domains='Unchanged full/mixed/small_foreground/far_small/pure_far/public_far/outside; UNKNOWN excluded, not negative',
        primary_targets=dict(far_small_recall=.75,far_small_iou='>= same-cohort original NFO',mixed_recall=.945,pure_far_fp='<= same-cohort original NFO'),
        diagnostics='Full/all-small and six-family metrics; paired original-to-candidate and joint-to-late changes; original depth-separated and near-depth strata are descriptive only',
        decision='Report whether fixed-point gains transfer and all four user targets hold; expose all costs. No automatic model replacement even if targets pass. Failure closes these fixed candidate adaptations for this transfer role.',
        stop='One500-frame three-arm comparison; no fit, threshold search, alternate checkpoint, test access, recipe repair or automatic successor',
        verification='Exact original baseline counts; preserve all500 packed masks; independent evaluator recount and fixed first/last batch model reload',
        backend='CUDA; retained original/late architecture placement receipts; record actual model/output device; CPU metadata/count reductions',
        placement_evidence=[str(m.OUT/'backend.json'),str(late.OUT/'backend.json')])
    m.write(OUT/'protocol.json',protocol);m.write(OUT/'manifest.json',rows)
    torch.set_num_threads(4);assert torch.cuda.is_available();models=load_models()
    assert all(not p.requires_grad for net in models.values() for p in net.parameters())
    totals=defaultdict(lambda:np.zeros(4,np.int64));family=defaultdict(lambda:np.zeros(4,np.int64))
    paired=defaultdict(lambda:defaultdict(int));frames=[];packed=[];unknown=0;times={name:0. for name in ARMS}
    begin=time.perf_counter();boxes=public_boxes()
    for start in range(0,500,24):
        batch=rows[start:start+24];arrays=[]
        for row in batch:
            path=m.OLD/row['prepared'];assert m.sha(path)==row['sha256'],row['id']
            with np.load(path) as a:arrays.append({k:a[k].copy() for k in ['rgb','depth','values','boxes']})
            np.testing.assert_array_equal(arrays[-1]['boxes'],boxes)
        rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
        zones=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
        assert rgb.is_contiguous() and rgb.device.type=='cuda' and zones.device.type=='cuda'
        predictions={}
        for name,net in models.items():
            torch.cuda.synchronize();t=time.perf_counter()
            probabilities=m.probabilities(net(rgb,zones),'nfo')
            assert probabilities.device.type=='cuda' and torch.isfinite(probabilities).all()
            assert (probabilities[:,1:]>=probabilities[:,:-1]).all()
            predictions[name]=(probabilities[:,2]>=CUT).cpu().numpy()
            torch.cuda.synchronize();times[name]+=time.perf_counter()-t
        for k,(row,a) in enumerate(zip(batch,arrays)):
            truth,dm=evaluation_domains(a);unknown+=int((~dm['full']).sum())
            fm={name:{} for name in ARMS}
            for domain,mask in dm.items():
                for name in ARMS:
                    c=m.counts(predictions[name][k],truth,mask)
                    totals[name,domain]+=c;family[row['family'],name,domain]+=c;fm[name][domain]=m.metrics(c)
                for old,new in [('nfo','joint'),('nfo','late'),('joint','late')]:
                    c=transitions(predictions[old][k],predictions[new][k],truth,mask)
                    for key,v in c.items():paired[old+'->'+new,domain][key]+=v
            frames.append(dict(id=row['id'],scene=row['scene'],family=row['family'],metrics=fm))
            packed.append(np.stack([np.packbits(predictions[name][k].reshape(-1),bitorder='little') for name in ARMS]))
        if start%120==0 or start+len(batch)==500:print('FROZEN_TRANSFER',start+len(batch),'/500',flush=True)
    metrics={name:{d:m.metrics(c) for (n,d),c in totals.items() if n==name} for name in ARMS}
    for domain in ['full','mixed','small_foreground','far_small','pure_far','public_far']:
        for key in ['tp','fp','fn','tn']:assert metrics['nfo'][domain][key]==expected[domain][key],(domain,key)
    families={f:{name:{d:m.metrics(c) for (ff,n,d),c in family.items() if ff==f and n==name} for name in ARMS} for f in sorted({r['family'] for r in rows})}
    for (comparison,domain),c in paired.items():
        old,new=comparison.split('->');a,b=metrics[old][domain],metrics[new][domain]
        assert b['tp']-a['tp']==c['rescued_fn']-c['lost_tp']
        assert b['fp']-a['fp']==c['added_fp']-c['removed_fp']
    for name in ARMS:
        for key in ['tp','fp','fn','tn']:
            assert metrics[name]['far_small'][key]==metrics[name]['far_small_separated'][key]+metrics[name]['far_small_other'][key]
        assert m.sha(weights[name])==protocol['checkpoint_sha256'][name]
    result=dict(status='COMPLETE',frames=500,families=6,scenes=52,metrics=metrics,
        gates={name:gates(metrics[name],metrics['nfo']) for name in ['joint','late']},
        all_user_targets={name:all(gates(metrics[name],metrics['nfo']).values()) for name in ['joint','late']},
        paired={old+'->'+new:{domain:dict(c) for (pair,domain),c in paired.items() if pair==old+'->'+new} for old,new in [('nfo','joint'),('nfo','late'),('joint','late')]},
        family_metrics=families,unknown_pixels=unknown,baseline_counts_exact=True,all_checkpoint_hashes_unchanged=True,
        device=torch.cuda.get_device_name(),batched_model_seconds=times,total_seconds=time.perf_counter()-begin,
        training_updates=0,original_test_frames_read=0,cohort_disjoint=disjoint,
        limitation='Consumed synthetic Development at fixed operating point; no fresh confirmation, natural-distribution, phone or safety claim')
    np.savez_compressed(OUT/'predictions.npz',masks=np.stack(packed))
    m.write(OUT/'prediction-format.json',dict(shape=[500,3,192,256],arms=ARMS,packing='flatten each2D mask,row-major; np.packbits bitorder=little',cutoff=CUT))
    m.write(OUT/'frames.json',frames);m.write(OUT/'results.json',result)
    m.write(OUT/'completion.json',dict(status='COMPLETE',training_updates=0,development_frames=500,original_test_frames_read=0))
    print('TRANSFER_RESULT',json.dumps(dict(metrics=metrics,gates=result['gates'],seconds=result['total_seconds'])),flush=True)


if __name__=='__main__':main()
