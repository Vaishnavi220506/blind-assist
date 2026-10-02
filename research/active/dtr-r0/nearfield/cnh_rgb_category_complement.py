"""Observation-only extra category layer with frozen pixel naming cost screen.

Existing named YOLO pixels are immutable. New foreground names can fill unknown
pixels, including unknown pixels owned by an unmapped detector box. This is not
an alarm, metric-depth estimator, instance matcher or a confidence-calibrated API.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import cnh_rgb_dense_semantics as D

P,S,ROOT=D.P,D.S,D.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-category-complement-20261002'
BASE=D.OUT
TRANSFER=ROOT/'artifacts.local/work/cnh-rgb-semantic-transfer-20261002'
RUN_ID='CNH_RGB_CATEGORY_COMPLEMENT_20261002'
NEW_CLASSES=tuple(sorted(set(D.MAPPING)-D.COMMON-D.CONTEXT))


def complement(yolo_names,dense_names):
    """Only two prediction arrays enter. 0 means no mapped name, not free space."""
    yolo=np.asarray(yolo_names);dense=np.asarray(dense_names)
    if yolo.shape!=dense.shape or yolo.dtype!=np.uint8 or dense.dtype!=np.uint8:
        raise ValueError('Equal-size uint8 name maps required')
    if not np.isin(yolo,[0,*D.COMMON]).all() or np.any(dense>40):
        raise ValueError('Frozen NYU name-domain contract violated')
    added=(yolo==0)&np.isin(dense,NEW_CLASSES)
    output=yolo.copy();output[added]=dense[added]
    assert np.array_equal(output[yolo>0],yolo[yolo>0])
    assert np.array_equal(output[~added],yolo[~added])
    return output,added


def prepare():
    assert not (OUT/'PLAN.json').exists()
    assert len(NEW_CLASSES)==18 and not set(NEW_CLASSES)&(D.COMMON|D.CONTEXT)
    rows=[];dependencies=[]
    for name,run in [('base343',BASE),('transfer96',TRANSFER)]:
        plan=P.read(run/'PLAN.json')
        for path,digest in plan['source_sha256'].items():assert P.sha(ROOT/path)==digest,path
        binding=P.read(run/'label-binding.json')['hashes'] if name=='transfer96' else {}
        yroot=TRANSFER/'yolo/predictions' if name=='transfer96' else S.OUT/'predictions'
        for r in plan['inputs']:
            r=deepcopy(r)
            if binding:r.update(binding[r['id']])
            r.update(cohort='transfer96' if name=='transfer96' else r['split']+'343',source_run=str(run),yolo_predictions=str(yroot))
            rows.append(r)
        dependencies += [run/'PLAN.json',run/'result.json',run/'frame-ledger.json',run/'predictions/inference-result.json',yroot/'inference-result.json']
        if binding:dependencies.append(run/'label-binding.json')
    assert len(rows)==len({r['id'] for r in rows})==439
    prereg=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; fixed439 consumed frames, cal193/eval150/transfer96 separately; preserve namedYOLO pixels, fill UNKNOWN only with18 new foreground SegFormer names; no remapping/GT candidate selection; no model inference | NOT_RUN; primary near foreground class-query/object coverage plus added-name precision on ALL NYU1..40 GT, unmapped/generic GT included, GT0 burden separate; fine1..37 sensitivity only | Keep exploratory recipe only if transfer near-query hits strictly increase and added-name precision >=80% in each of3 cohorts; otherwise STOP_FIXED_COMPLEMENT with no same-batch class/mask/threshold tuning. 80% diagnostic target, not product/safety threshold | `artifacts.local/work/cnh-rgb-category-complement-20261002/REPORT.md` |')
    body=P.RUNS.read_text(encoding='utf-8');assert f'| {RUN_ID} |' not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+prereg+'\n',encoding='utf-8')
    (OUT/'prerun-row.txt').write_text(prereg,encoding='utf-8')
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),inputs=rows,
        source_sha256={str(Path(p).relative_to(ROOT)):P.sha(p) for p in (__file__,D.__file__,S.__file__,D.A.__file__,D.V.__file__,P.__file__)},
        dependencies={str(p.relative_to(ROOT)):P.sha(p) for p in dependencies},new_classes=NEW_CLASSES,
        rule='YOLO named pixels immutable; fill only YOLO0 using exact18 new foreground classes; unmapped detector box pixels are eligible unknowns. No confidence, area, morphology, ROI/depth filtering or alternate arm.',
        cost='All annotated NYU1..40 GT rows: added correct names / all added pixels; includes not-mapped and generic38..40 truth. Wrong names here are benchmark name mismatch, not physical false alarms. GT0 not treated negative. Sensitivity1..37 does not change decision.',
        decision='Transfer foreground class-query hits strictly greater than YOLO and added-name precision on allGT1..40 >=.80 separately in cal343,eval343,transfer96 => KEEP_EXPLORATORY_COMPLEMENT, otherwise STOP_FIXED_COMPLEMENT. Undefined precision fails. No deployment claim.',
        role='All439 consumed Development, outcome-informed design but no new output peek before this fixed run; not independent transfer confirmation',preregistration=prereg))
    print('Prepared fixed439 complement plus full annotated-domain cost screen')


def validate():
    plan=P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():assert P.sha(ROOT/path)==digest,path
    for path,digest in plan['dependencies'].items():assert P.sha(ROOT/path)==digest,path
    return plan


def cost(matrix,classes):
    matrix=np.asarray(matrix,dtype=np.int64);classes=sorted(classes)
    n=int(matrix[classes,:].sum());correct=int(sum(matrix[k,k] for k in classes))
    return dict(named_pixels=n,correct_pixels=correct,incorrect_name_pixels=n-correct,
                precision=correct/n if n else None,GT_classes=classes)


def evaluate_frame(args):
    row,old,dr,yr=args
    run=Path(row['source_run']);yroot=Path(row['yolo_predictions'])
    npz=run/'predictions'/(row['id']+'.npz');receipt=npz.with_suffix('.json')
    assert P.sha(npz)==dr['npz_sha256'] and P.sha(receipt)==dr['receipt_sha256']
    assert dr['rgb_sha256']==row['rgb_sha256']==P.sha(row['rgb_path'])
    with np.load(npz,allow_pickle=False) as f:dense=D.dense_to_nyu(f['labels'])
    yp=yroot/yr['receipt'];assert P.sha(yp)==yr['receipt_sha256']
    yy=P.read(yp);assert yy['rgb_sha256']==row['rgb_sha256']
    yolo=D.yolo_to_nyu(yy['detections'])
    output,added=complement(yolo,dense)
    # All observation outputs finalized before accessing evaluator truth.
    for key in ('depth','semantic','instance'):assert P.sha(row[key+'_path'])==row[key+'_sha256']
    sem=S.hdf(row['semantic_path']).astype(np.int32);inst=S.hdf(row['instance_path']).astype(np.int32)
    radial=S.hdf(row['depth_path']);g=D.A.whole_geometry(D.V.ray_geometry(sem.shape,row['camera_matrix']))
    z=radial/g['radial_factor'];y=z*g['fy'];x=z*g['fx'];valid=(sem>=1)&(sem<=40)
    contacts=[];cq=[]
    for q in g['queries']:
        mask=g['fov_mask']&np.isfinite(z)&(z>=.6)&(z<2.1)&(abs(x)<.3)&(y>=q['y_low'])&(y<=q['y_high'])
        contacts.append(mask)
        correct=np.bincount(sem[mask&valid&(output==sem)],minlength=41)
        base_correct=np.bincount(sem[mask&valid&(yolo==sem)],minlength=41)
        for prior in [c for c in old['dense_class_queries'] if c['query']==q['name']]:
            c=deepcopy(prior);k=c['nyu40'];n=c['support_pixels']
            assert int(base_correct[k])==c['arms']['yolo_boxes']['correct_pixels']
            c['complement']=dict(correct_pixels=int(correct[k]),hit=bool(n>=16 and correct[k]>=.5*n))
            assert not c['arms']['yolo_boxes']['hit'] or c['complement']['hit']
            cq.append(c)
    union=np.logical_or.reduce(contacts);objects=[]
    for prior in old['dense_near_objects']:
        obj=deepcopy(prior);mask=(inst==obj['instance_id'])&union;k=obj['nyu40']
        assert int(mask.sum())==obj['support_pixels']
        correct=int((mask&(output==k)).sum()) if k is not None else 0
        obj['complement']=dict(correct_pixels=correct,hit=bool(k is not None and correct>=.5*obj['support_pixels']))
        assert not obj['arms']['yolo_boxes']['hit'] or obj['complement']['hit']
        objects.append(obj)
    truth=np.where(valid,sem,0)
    matrix=np.bincount((truth[added].astype(np.int64)*41+output[added]).ravel(),minlength=1681).reshape(41,41)
    return dict(id=row['id'],family=row['family'],cohort=row['cohort'],source_ledger_identity=old['id'],
        class_queries=cq,objects=objects,added_confusion=matrix,full_confusion=D.confusion(sem,output),
        named_yolo_pixels=int((yolo>0).sum()),added_pixels=int(added.sum()),unknown_GT_added_pixels=int((added&~valid).sum()),
        base_named_pixels_preserved=True,new_classes_observed=np.unique(output[added]).tolist())


def summarize(frames):
    categories={}
    for group in ('FOREGROUND','CONTEXT'):
        samples=[c for f in frames for c in f['class_queries'] if c['group']==group and c['support_pixels']>=16]
        categories[group]=dict(n=len(samples),support_pixels=sum(c['support_pixels'] for c in samples),
            arms={a:dict(hits=sum((c['complement'] if a=='complement' else c['arms'][a])['hit'] for c in samples),
                correct_pixels=sum((c['complement'] if a=='complement' else c['arms'][a])['correct_pixels'] for c in samples)) for a in ('yolo_boxes','new_seg','complement')})
    objects=[o for f in frames for o in f['objects']]
    matrix=sum((np.asarray(f['added_confusion'],np.int64) for f in frames),np.zeros((41,41),np.int64))
    per_class={}
    for k in NEW_CLASSES:
        items=[c for f in frames for c in f['class_queries'] if c['nyu40']==k and c['support_pixels']>=16]
        per_class[str(k)]=dict(n=len(items),support_pixels=sum(c['support_pixels'] for c in items),
            complement_hits=sum(c['complement']['hit'] for c in items),correct_pixels=sum(c['complement']['correct_pixels'] for c in items))
    return dict(frames=len(frames),families=len({f['family'] for f in frames}),class_queries=categories,new_class_near_support=per_class,
        objects=dict(n=len(objects),arms={a:dict(hits=sum((o['complement'] if a=='complement' else o['arms'][a])['hit'] for o in objects),
            correct_pixels=sum((o['complement'] if a=='complement' else o['arms'][a])['correct_pixels'] for o in objects)) for a in ('yolo_boxes','new_seg','complement')}),
        added_confusion=matrix,added_cost_all_known=cost(matrix,range(1,41)),added_cost_fine_GT_sensitivity=cost(matrix,range(1,38)),
        unknown_GT_added_pixels=int(matrix[0].sum()),added_pixels=int(matrix.sum()),
        added_by_class={str(k):dict(named_on_known=int(matrix[1:,k].sum()),correct=int(matrix[k,k]),
            label_mismatch=int(matrix[1:,k].sum()-matrix[k,k]),
            named_on_unknown=int(matrix[0,k]),precision=float(matrix[k,k]/matrix[1:,k].sum()) if matrix[1:,k].sum() else None) for k in NEW_CLASSES},
        full_confusion=sum((np.asarray(f['full_confusion'],np.int64) for f in frames),np.zeros((41,41),np.int64)))


def evaluate():
    plan=validate();assert not (OUT/'result.json').exists()
    jobs=[]
    for name,run in [('base343',BASE),('transfer96',TRANSFER)]:
        old={f['id']:f for f in P.read(run/'frame-ledger.json')};dense=P.read(run/'predictions/inference-result.json')['outputs']
        yroot=TRANSFER/'yolo/predictions' if name=='transfer96' else S.OUT/'predictions'
        yolo=P.read(yroot/'inference-result.json')['outputs']
        for row in plan['inputs']:
            if row['source_run']==str(run):jobs.append((row,old[row['id']],dense[row['id']],yolo[row['id']]))
    assert len(jobs)==439
    with ThreadPoolExecutor(max_workers=3) as pool:frames=list(pool.map(evaluate_frame,jobs))
    groups={cohort:summarize([f for f in frames if f['cohort']==cohort]) for cohort in ('cal343','eval343','transfer96')}
    t=groups['transfer96']['class_queries']['FOREGROUND']['arms']
    gains=t['complement']['hits']>t['yolo_boxes']['hits']
    precision={k:v['added_cost_all_known']['precision'] for k,v in groups.items()}
    decision='KEEP_EXPLORATORY_COMPLEMENT' if gains and all(v is not None and v>=.8 for v in precision.values()) else 'STOP_FIXED_COMPLEMENT'
    P.save(OUT/'frame-ledger.json',frames)
    P.save(OUT/'result.json',dict(run_id=RUN_ID,decision=decision,groups=groups,
        families={cohort:{family:summarize([f for f in frames if f['cohort']==cohort and f['family']==family])
            for family in sorted({f['family'] for f in frames if f['cohort']==cohort})} for cohort in groups},
        rule=plan['rule'],role=plan['role'],all_named_yolo_pixels_preserved=all(f['base_named_pixels_preserved'] for f in frames),
        original_named_hit_retention='Guaranteed by construction, not independent evidence of semantic superiority',
        cost_scope=plan['cost']))
    print(decision,precision,flush=True)


def selftest():
    y=np.array([[5,0,0,0,0]],np.uint8);d=np.array([[7,7,5,1,0]],np.uint8)
    pred,added=complement(y,d)
    assert pred.tolist()==[[5,7,0,0,0]] and added.tolist()==[[False,True,False,False,False]]
    m=np.zeros((41,41),np.int64);m[7,7]=8;m[40,7]=2;m[0,7]=10
    assert cost(m,range(1,41))['precision']==.8 and cost(m,range(1,38))['precision']==1
    print('PASS named-pixel retention, new-only foreground filling, unmapped GT counted, unknown GT excluded from error')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
