"""Post-seal held evaluation. No model or threshold optimization here."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from query_occupancy_data import CENTRE, EDGES, read, write, sha, stage_path, new_stage_directory
from query_occupancy_learning import frame_truth, operating_point
from ba_camera_corridor_metrics import evaluate_rows

ARMS = ('A_current', 'A_hold', 'classifier', 'occupancy')


def metric_rows(identities, labels, predictions, selection):
    truth, valid = frame_truth(labels)
    rows, last = [], {}
    for k, meta in enumerate(identities):
        base = meta['baseline']
        old = last.get(meta['clip_id'], False)
        last[meta['clip_id']] = base['alert']
        flags = dict(A_current=base['alert'], A_hold=base['alert'] or old)
        for arm in ('classifier','occupancy'):
            flags[arm] = float(predictions[arm]['probability'][k, CENTRE].max()) >= float(selection[arm]['selection']['threshold'])
        rows.append(dict(id=meta['id'], clip_id=meta['clip_id'], frame_in_clip=meta['frame_in_clip'],
            time_s=meta['time_s'], base_group_id=meta['base_group_id'], type_id=meta['type_id'],
            layer=meta['layer'], layout_relation=meta['layout_relation'],
            truth=bool(truth[k]) if valid[k] else None, boundary=meta['layout_relation']=='BOUNDARY',
            predictions={a:dict(alert=bool(flag),unknown=bool(base['unknown']),ambiguous=bool(base['unknown'] and flag))
                         for a,flag in flags.items()}))
    return rows


def group_bootstrap(rows, count=1000):
    rng = np.random.default_rng(202609223)
    groups = sorted({r['base_group_id'] for r in rows})
    totals = {}
    for g in groups:
        known = [r for r in rows if r['base_group_id']==g and r['truth'] is not None]
        totals[g] = [sum(r['truth'] for r in known)]+[
            sum(r['truth'] and r['predictions'][a]['alert'] for r in known) for a in ('classifier','occupancy')]
    values = []
    for _ in range(count):
        total = np.asarray([totals[g] for g in rng.choice(groups,len(groups),replace=True)]).sum(0)
        values.append((total[2]-total[1])/max(1,total[0]))
    return dict(unit='whole base layout, with all paired clips and queries',replicates=count,
                recall_difference_percentile_95=np.quantile(values,[.025,.975]).tolist())


def evaluate(root):
    root = Path(root)
    predroot = stage_path(root,'predictions')
    seal = read(predroot/'prediction-seal.json')
    assert seal['status']=='PASS' and seal['held_labels_opened'] is False
    assert sha(stage_path(root,'fit')/'selection.json') == seal['selection_sha256']
    assert sha(stage_path(root,'fit')/'learning-source-seal.json') == seal['source_seal_sha256']
    predictions = {}
    for arm in ('classifier','occupancy'):
        assert sha(predroot/f'{arm}.npz') == seal['models'][arm]['predictions_sha256']
        predictions[arm] = dict(np.load(predroot/f'{arm}.npz',allow_pickle=False))
    # First access to held label arrays is deliberately after the above checks.
    labels = dict(np.load(stage_path(root,'prepared')/'labels/evaluation.npz',allow_pickle=False))
    indices = labels['indices']
    assert all(np.array_equal(indices,p['indices']) for p in predictions.values())
    all_ids = read(stage_path(root,'prepared')/'observations/identities.json')
    identities = [all_ids[int(i)] for i in indices]
    selection = read(stage_path(root,'fit')/'selection.json')
    rows = metric_rows(identities, labels, predictions, selection)
    metrics = evaluate_rows(rows, arms=ARMS)
    strata = {key:{v:evaluate_rows([r for r in rows if r[key]==v],arms=ARMS)
                   for v in sorted({r[key] for r in rows})}
              for key in ('type_id','layer','layout_relation')}
    curves = {}
    y,v = frame_truth(labels)
    for arm in ('classifier','occupancy'):
        _,curves[arm]=operating_point(predictions[arm]['probability'][:,CENTRE].max(1),y,v,identities)
    occ = predictions['occupancy']
    qv = labels['valid']
    pos = qv & (labels['classes']<6)
    argmax = occ['distribution'].argmax(-1)
    mid = (EDGES[:-1]+EDGES[1:])/2
    conditional = occ['distribution'][...,:6]
    mass=conditional.sum(-1,keepdims=True)
    conditional=np.divide(conditional,mass,out=np.full_like(conditional,np.nan),where=mass>0)
    mean_distance=(conditional*mid).sum(-1)
    mask = occ['mask'] >= .5
    target=labels['mask']
    cover=labels['coverage'][:,None]
    intersection=(mask*target).sum((-1,-2))
    union=(mask*cover+target-mask*target).sum((-1,-2))
    iou=intersection/np.maximum(union,1e-9)
    mask_positive=qv & (target.sum((-1,-2))>0)
    distance_evaluable=pos & np.isfinite(mean_distance)
    empty_visible_queries=qv & ~pos & (cover.sum((-1,-2))>0)
    components=dict(query_count=int(qv.sum()), query_bin_accuracy=float((argmax[qv]==labels['classes'][qv]).mean()),
        positive_query_bin_accuracy=float((argmax[pos]==labels['classes'][pos]).mean()),
        positive_query_conditional_distance_mae_m=float(np.abs(mean_distance[distance_evaluable]-labels['distances'][distance_evaluable]).mean()) if distance_evaluable.any() else None,
        positive_query_zero_predicted_occupancy_mass=int((pos & ~distance_evaluable).sum()),
        visible_positive_query_count=int(mask_positive.sum()),
        mean_visible_positive_mask_iou=float(iou[mask_positive].mean()),
        mask_iou_convention='Native occupied area, 45x80 prediction grid, invalid native pixels ignored',
        geometric_positive_without_visible_pixels=int((pos & ~mask_positive).sum()),
        empty_geometric_query_false_mask_rate=float(((mask & (cover>0)).any((-1,-2))[empty_visible_queries]).mean()) if empty_visible_queries.any() else None,
        predicted_bins_are_model_estimates_not_sensor_measurements=True)
    c,o=[metrics['arms'][a]['frames']['all_known'] for a in ('classifier','occupancy')]
    cs,os=[metrics['arms'][a]['false_alert_segment_count'] for a in ('classifier','occupancy')]
    recall_gain=o['recall']-c['recall']
    benefit=recall_gain>=.10-1e-12 and o['FP']<=c['FP']+2 and os<=cs+1
    precision_tradeoff=c['FP']>0 and o['FP']<=.7*c['FP'] and recall_gain>=-.03-1e-12
    retention={}
    for layer,m in strata['layer'].items():
        a,b=[m['arms'][a]['frames']['all_known']['recall'] for a in ('classifier','occupancy')]
        retention[layer]=b>=a-.05-1e-12
    for family,m in strata['type_id'].items():
        retention[family]=m['arms']['occupancy']['detected_events']>=m['arms']['classifier']['detected_events']-1
    keep=(benefit or precision_tradeoff) and all(retention.values())
    event_diff={}
    ce={ (e['clip_id'],e['start_frame']):e for e in metrics['arms']['classifier']['events']}
    oe={ (e['clip_id'],e['start_frame']):e for e in metrics['arms']['occupancy']['events']}
    for key in ce:
        a,b=ce[key],oe[key]
        if a['detected'] != b['detected'] or a['first_alert_time_s']!=b['first_alert_time_s']:
            event_diff['|'.join(map(str,key))]=dict(classifier=a,occupancy=b)
    exit_tails={}
    for arm in ARMS:
        tails=[]
        censored=0
        for clip in sorted({r['clip_id'] for r in rows}):
            seq=[r for r in rows if r['clip_id']==clip]
            positive=[i for i,r in enumerate(seq) if r['truth'] is True]
            if positive:
                length=0
                if max(positive)==len(seq)-1 or seq[max(positive)+1]['truth'] is None:
                    censored+=1
                for r in seq[max(positive)+1:]:
                    if r['truth'] is False and r['predictions'][arm]['alert']:
                        length+=1
                    else:
                        break
                tails.append(length*.2)
        exit_tails[arm]=dict(total_sampled_s=sum(tails),max_sampled_s=max(tails,default=0),
                            positive_clips=len(tails),censored_clips=censored,
                            convention='Consecutive known-negative alert samples after the last known positive in each clip; not an event-run denominator')
    report=dict(status='PASS', scientific_outcome='RETAIN_CHALLENGER' if keep else 'NO_JOINT_ALERT_GAIN',
        metrics=metrics,strata=strata,components=components,descriptive_curves=curves,
        selection=selection, checks=dict(recall_cost_benefit=benefit,fp_tradeoff=precision_tradeoff,retention=retention),
        layout_bootstrap=group_bootstrap(rows), event_differences=event_diff, exit_tails=exit_tails,
        held_predictions_sha256=sha(predroot/'prediction-seal.json'),
        scope='New layout same-generator controlled-object simulation Development, no natural/hardware/safety claim')
    dest=stage_path(root,'evaluated')
    new_stage_directory(dest)
    write(dest/'frame-results.json',rows)
    write(dest/'metrics.json',report)
    return dict(status='PASS', scientific_outcome=report['scientific_outcome'],
        frames=len(rows),truth_coverage=metrics['truth_coverage'], components=components,
        arms={a:dict(**m['frames']['all_known'],events=f"{m['detected_events']}/{m['event_count']}",
                    false_segments=m['false_alert_segment_count']) for a,m in metrics['arms'].items()},
        metrics_sha256=sha(dest/'metrics.json'))
