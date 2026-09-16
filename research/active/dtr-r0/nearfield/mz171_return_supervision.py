"""Direct per-return supervision; predictor and public input contract unchanged."""
import numpy as np
import torch
from torch.nn import functional as F
from mz161_dense_task import loss as dense_loss

METHOD=dict(schema='MZ171_DIRECT_RETURNED_TOF_WITNESS_SUPERVISION',
    arms=['control','witness'],model='UNCHANGED_MZ165_CONTEXT_TASK_NET',
    loss='FRAME_PLUS_KNOWN_PIXEL_BCE;WITNESS_ADDS_ONE_BALANCED_KNOWN_TOF_SLOT_BCE',
    threshold_logit=0.,tof_slots=128,radar_slots=4,
    unknown='NO_SUPERVISION;NONALERT_NOT_CLEAR',
    native_authority='OFFLINE_TRAIN_TARGET_ONLY_NOT_OBSERVATION')


def slot_weights(target,known):
    if target.ndim!=2 or target.shape!=known.shape or target.shape[1]!=132:
        raise ValueError('Expected Bx132 target and known masks')
    if known.dtype!=torch.bool or not torch.isfinite(target).all():
        raise ValueError('Invalid slot targets')
    if torch.any(known[:,128:]) or torch.any((target[known]!=0)&(target[known]!=1)):
        raise ValueError('Radar labels unavailable and known labels must be binary')
    positive=known&(target>0);negative=known&~positive
    np_=positive.sum(1,keepdim=True);nn_=negative.sum(1,keepdim=True)
    both=(np_>0)&(nn_>0)
    mass=torch.where(both,.5,1.)
    return positive*mass/np_.clamp_min(1)+negative*mass/nn_.clamp_min(1)


def loss(output,frame_target,pixel_target,pixel_weight,slot_target,slot_known,arm):
    if arm not in ('control','witness'):raise ValueError('Unknown arm')
    total,terms=dense_loss(output,frame_target,pixel_target,pixel_weight,'dense')
    weights=slot_weights(slot_target,slot_known)
    witness=(F.binary_cross_entropy_with_logits(output['token'],slot_target,reduction='none')*weights).sum(1).mean()
    terms['witness']=float(witness.detach())
    return total+(witness if arm=='witness' else 0.),terms


def slot_report(logits,target,known,indices):
    ix=np.asarray(indices,int);p=np.asarray(logits)[ix]>=0
    t=np.asarray(target,bool)[ix];k=np.asarray(known,bool)[ix]
    if p.shape!=t.shape or p.shape!=k.shape or np.any(k[:,128:]):
        raise ValueError('Invalid saved slot arrays')
    c=dict(TP=int((p&t&k).sum()),FP=int((p&~t&k).sum()),
        FN=int((~p&t&k).sum()),TN=int((~p&~t&k).sum()))
    pos=c['TP']+c['FN'];neg=c['TN']+c['FP'];pred=c['TP']+c['FP']
    recall=c['TP']/pos if pos else None;specificity=c['TN']/neg if neg else None
    return dict(counts=c,known_positive=pos,known_negative=neg,
        unknown_tof_slots=int((~k[:,:128]).sum()),unknown_radar_slots=int((~k[:,128:]).sum()),
        precision=c['TP']/pred if pred else None,recall=recall,specificity=specificity,
        balanced_accuracy=(recall+specificity)/2 if pos and neg else None,
        unknown_positive_tof_slots=int((p[:,:128]&~k[:,:128]).sum()),
        unknown_positive_radar_slots=int(p[:,128:].sum()))


def evaluate_witness(rows,es,labels,maps,predictions,baseline,spec,metadata,
                     token_logits,slot_target,slot_known):
    from evaluate_mz161_dense_task import evaluate
    aliases={'frame':'control','dense':'witness'}
    value=evaluate(rows,es,labels,{k:maps[v] for k,v in aliases.items()},
        {k:predictions[v] for k,v in aliases.items()},baseline,spec,metadata)
    def rename(x):
        if isinstance(x,dict):return {aliases.get(k,k):rename(v) for k,v in x.items()}
        if isinstance(x,list):return [rename(v) for v in x]
        return x
    value=rename(value);s=value['summary'];g=s['gates'];checks=g['heldout_alert_checks']
    checks['fewer_fp_than_matched_control']=checks.pop('fewer_fp_than_frame_control')
    checks['all_control_events_without_extra_delay']=s['partitions']['heldout']['witness']['vs_control']['all_reference_events_without_extra_delay']
    g['fit_native_supported_nonalerts_zero']=not s['partitions']['fit']['witness']['native']['nonalert_with_native_corridor_contributors']
    g['heldout_alert_pass']=all(checks.values())
    g['overall_pass']=bool(g['fit_accuracy_at_least_95pct'] and g['heldout_alert_pass'] and g['fit_native_supported_nonalerts_zero'])
    g['spatial_gate_role']='DIAGNOSTIC_NOT_METRIC_SURFACE_CLAIM'
    s['slots']={part:{arm:slot_report(token_logits[arm],slot_target,slot_known,
        [i for i,r in enumerate(rows) if metadata[r['id']]['partition']==part])
        for arm in ('control','witness')} for part in ('fit','heldout')}
    c,w=s['slots']['heldout']['control'],s['slots']['heldout']['witness']
    s['branch_learning_effect']=bool(all(c[k] is not None and w[k] is not None and w[k]>c[k]
        for k in ('recall','balanced_accuracy')))
    s['legacy_evaluator_arm_aliases']=aliases;s['method']=METHOD
    s['decision']='MZ171_RETURN_WITNESS_ALERT_COMPONENT_PASS' if g['overall_pass'] else 'MZ171_RETURN_WITNESS_ALERT_GAIN_NOT_MET'
    return value
