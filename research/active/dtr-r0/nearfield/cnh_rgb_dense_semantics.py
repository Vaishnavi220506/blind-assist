"""Fixed dense RGB category coverage; no ToF, instances predicted, or alarms.

GT instances only supply retained evaluation supports. Context surfaces are
reported separately. Precision is conditional on explicit mapped-GT domains.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json

import numpy as np

import cnh_rgb_semantic_hints as S

P,A,V,ROOT=S.P,S.A,S.V,S.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-dense-semantics-20261002'
RUN_ID='CNH_RGB_DENSE_SEMANTICS_20261002'
MAPPING={1:0,2:3,3:10,4:7,5:19,6:23,7:15,8:14,10:62,12:45,13:63,14:33,15:24,
         16:18,17:44,18:57,19:27,21:92,22:5,24:50,25:89,27:81,29:41,31:12,
         33:65,34:47,35:36,36:37,37:115}
CONTEXT={1,2,22}
ARMS=('yolo_boxes','new_seg')
COMMON=set(S.MAPPING)
SHAPE=(768,1024)


def dense_to_nyu(labels):
    labels=np.asarray(labels)
    if labels.dtype!=np.uint8 or labels.shape!=SHAPE or np.any(labels>149):
        raise ValueError('Native uint8 ADE0..149 labels required')
    lookup=np.zeros(150,np.uint8)
    assert len(set(MAPPING.values()))==len(MAPPING)
    for nyu,ade in MAPPING.items():lookup[ade]=nyu
    return lookup[labels]


def yolo_to_nyu(detections,shape=SHAPE):
    """All RGB boxes compete; unmapped COCO classes occupy pixels as unknown."""
    output=np.zeros(shape,np.uint8);occupied=np.zeros(shape,bool)
    inverse={coco:nyu for nyu,coco in S.MAPPING.items()}
    for index in sorted(range(len(detections)),key=lambda i:(-detections[i]['score'],i)):
        detection=detections[index];x0,y0,x1,y1=map(float,detection['xyxy'])
        assert np.isfinite([x0,y0,x1,y1,detection['score']]).all()
        # Integer pixel index p belongs iff x0 <= p+.5 < x1 (same for y).
        left=max(0,min(shape[1],int(np.ceil(x0-.5))));right=max(0,min(shape[1],int(np.ceil(x1-.5))))
        top=max(0,min(shape[0],int(np.ceil(y0-.5))));bottom=max(0,min(shape[0],int(np.ceil(y1-.5))))
        if right<=left or bottom<=top:continue
        area=np.s_[top:bottom,left:right];free=~occupied[area]
        output[area][free]=inverse.get(detection['class_id'],0)
        occupied[area]=True
    return output


def prepare():
    """Called by the parent only after inference source/model receipt are final."""
    assert not (OUT/'PLAN.json').exists(),'Preserve frozen plan'
    previous=P.read(S.OUT/'PLAN.json');observations=P.read(S.OUT/'observations.json')
    assert len(previous['inputs'])==len(observations)==343
    sources=[Path(__file__),Path(S.__file__),Path(A.__file__),Path(V.__file__),Path(P.__file__),
             Path(__file__).with_name('cnh_rgb_dense_semantics_infer.py')]
    dependencies={'semantic_plan':S.OUT/'PLAN.json','semantic_ledger':S.OUT/'frame-ledger.json',
                  'observations':S.OUT/'observations.json','yolo_inference':S.OUT/'predictions/inference-result.json',
                  'taxonomy':S.OUT/'nyu40-labels.csv','model_receipt':OUT/'model-receipt.json'}
    prereg=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same343 consumed Development; fixed ADE29->NYU and oldCOCO8 mapping; full RGB semantic names only; all40 class x whole HEAD/BODY contact support>=16, >=50% correct naming; retain1..15 supports, no-instance surfaces and135eval original near instances; context1/2/22 separate; full41x41 confusion and arm-mapped/common8-GT conditional precision | NOT_RUN | No ToF/alarm, pseudo-instances, remapping/tuning or new confirmation | `artifacts.local/work/cnh-rgb-dense-semantics-20261002/result.json` |')
    body=P.RUNS.read_text(encoding='utf8');assert RUN_ID not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+prereg+'\n',encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n',encoding='utf8')
    P.save(OUT/'observations.json',observations)
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),inputs=previous['inputs'],
        role='Same343 consumed Development; fixed category diagnostic, no confirmation or alarm claim',
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in sources},
        dependencies={k:dict(path=str(p),sha256=P.sha(p)) for k,p in dependencies.items()},
        mapping_nyu_to_ade=MAPPING,mapping_nyu_to_coco=S.MAPPING,common8=sorted(COMMON),context_classes=sorted(CONTEXT),
        rules=dict(primary='All40 NYU class x HEAD/BODY query contact support>=16; Z[.6,2.1), original45FOV/y bands/absX<.3; hit iff >=50% support correctly named',
            small_support='Retain every1..15-pixel class-query separately',
            foreground='NYU classes other than wall1/floor2/ceiling22; context reported separately',
            instance_views='Original visible instances and union-contact supports retained; unsupported/mixedUNKNOWN count in all-view denominator; no predicted instances or boxes from segmentation',
            yolo='All detections confidence descending, stable original index; first pixel owner wins, even unmapped class; half-open xyxy tests native pixel centres',
            burden='Full native-image41x41 NYU confusion, invalidGT->0; per-class precision only on arm-mapped-GT, or common8-GT and8predicted classes; other predictions on common8GT separately reported, never excluded from naming recall',
            exclusions='No books/book, picture/painting, window/windowpane, floormat/rug or otherprop merging',
            workers=3),prediction_contract='predictions/{id}.npz labels uint8 ADE0..149 native768x1024; inference-result outputs bind npz/rgb/receipt hashes',
            preregistration=prereg))
    print('Prepared dense semantic diagnostic and single PRE_RUN row')


def confusion(truth,predicted):
    valid=np.where((truth>=1)&(truth<=40),truth,0).astype(np.int64)
    return np.bincount((valid*41+predicted).ravel(),minlength=41*41).reshape(41,41)


def conditional_precision(matrix,classes):
    """Only stated GT rows and stated predicted classes; other names remain visible."""
    classes=sorted(classes);domain=matrix[classes,:];inside=domain[:,classes]
    per_class={str(c):dict(true_positive=int(matrix[c,c]),predicted_in_GT_domain=int(matrix[classes,c].sum()),
        precision=float(matrix[c,c]/matrix[classes,c].sum()) if matrix[classes,c].sum() else None) for c in classes}
    denominator=int(inside.sum());correct=int(sum(matrix[c,c] for c in classes))
    return dict(condition_GT_nyu40=classes,predicted_classes_described=classes,per_class=per_class,
        GT_domain_pixels=int(domain.sum()),correct_named_pixels=correct,predicted_in_scope_pixels=denominator,
        micro_precision=correct/denominator if denominator else None,
        correct_naming_pixel_recall=correct/int(domain.sum()) if domain.sum() else None,
        predicted_other_known_class_pixels=int(domain[:,1:].sum()-denominator),
        predicted_unknown_pixels=int(domain[:,0].sum()),
        limitation='Conditional on these GT classes; not full-image true precision or precision of excluded predicted classes')


def evaluate_frame(args):
    row,old,dense_record,yolo_record,plan,model_sha=args
    npz=OUT/'predictions'/(row['id']+'.npz');receipt_path=npz.with_suffix('.json')
    assert P.sha(npz)==dense_record['npz_sha256']
    assert P.sha(receipt_path)==dense_record['receipt_sha256']
    receipt=P.read(receipt_path)
    assert receipt=={k:v for k,v in dense_record.items() if k!='receipt_sha256'}
    assert receipt['id']==row['id'] and receipt['rgb_sha256']==row['rgb_sha256']==P.sha(row['rgb_path'])
    infer=Path(__file__).with_name('cnh_rgb_dense_semantics_infer.py')
    assert receipt['source_sha256']==plan['source_sha256'][str(infer.relative_to(ROOT))]
    assert receipt['model_sha256']==model_sha
    with np.load(npz,allow_pickle=False) as f:
        assert f.files==['labels'];ade=f['labels'].copy()
    assert receipt['shape']==list(SHAPE)
    assert np.bincount(ade.ravel(),minlength=150).tolist()==receipt['class_histogram']
    dense=dense_to_nyu(ade)
    ypath=S.OUT/'predictions'/yolo_record['receipt']
    assert P.sha(ypath)==yolo_record['receipt_sha256']
    yreceipt=P.read(ypath)
    assert yreceipt['id']==row['id'] and yreceipt['rgb_sha256']==row['rgb_sha256']
    predictions=dict(yolo_boxes=yolo_to_nyu(yreceipt['detections']),new_seg=dense)
    # Evaluator truth is first opened after both observation-only pixel predictions exist.
    for name in ('depth','semantic','instance'):
        assert P.sha(row[name+'_path'])==row[name+'_sha256']
    radial=S.hdf(row['depth_path']);sem=S.hdf(row['semantic_path']).astype(np.int32);inst=S.hdf(row['instance_path']).astype(np.int32)
    assert radial.shape==sem.shape==inst.shape==SHAPE
    geo=A.whole_geometry(V.ray_geometry(SHAPE,row['camera_matrix']))
    z=radial/geo['radial_factor'];y=z*geo['fy'];x=np.abs(z*geo['fx'])
    sem_valid=(sem>=1)&(sem<=40);truth_class=np.where(sem_valid,sem,0)
    contacts=[];expanded=[];class_queries=[]
    for index,q in enumerate(geo['queries']):
        near=geo['fov_mask']&np.isfinite(z)&(z>=.6)&(z<2.1)&(y>=q['y_low'])&(y<=q['y_high'])
        contact=near&(x<.3);expand=near&(x<=.4);contacts.append(contact);expanded.append(expand)
        assert int(contact.sum())==old['queries'][index]['contact_pixels']
        assert int(expand.sum())==old['queries'][index]['expanded_pixels']
        counts=np.bincount(truth_class[contact],minlength=41)
        noinst=np.bincount(truth_class[contact&(inst<0)],minlength=41)
        correct={a:np.bincount(truth_class[contact&(pred==truth_class)&sem_valid],minlength=41) for a,pred in predictions.items()}
        named={a:np.bincount(truth_class[contact&(pred>0)],minlength=41) for a,pred in predictions.items()}
        for klass in range(1,41):
            n=int(counts[klass]);original=old['queries'][index]['semantic_support'][klass-1]
            assert original['nyu40']==klass and original['contact_pixels']==n and original['contact_without_instance']==int(noinst[klass])
            class_queries.append(dict(query=q['name'],nyu40=klass,group='CONTEXT' if klass in CONTEXT else 'FOREGROUND',
                support_pixels=n,support_status='MAIN_GE16' if n>=16 else 'SMALL_1_15' if n else 'ABSENT',without_instance_pixels=int(noinst[klass]),
                arms={a:dict(correct_pixels=int(correct[a][klass]),named_pixels=int(named[a][klass]),
                    fraction=float(correct[a][klass]/n) if n else None,hit=bool(n>=16 and correct[a][klass]>=.5*n),
                    mapped=klass in (MAPPING if a=='new_seg' else S.MAPPING)) for a in ARMS}))
    contact=np.logical_or.reduce(contacts);near_objects=[]
    for obj in old['objects']+old['small_instances']:
        target=inst==obj['instance_id']
        for i,support in enumerate(obj['query_support']):
            assert int((target&contacts[i]).sum())==support['contact_pixels']
            assert int((target&expanded[i]).sum())==support['expanded_pixels']
        if not any(s['contact_pixels']>=16 for s in obj['query_support']):continue
        assert obj in old['objects']
        take=target&contact;n=int(take.sum());klass=obj['nyu40']
        near_objects.append(dict(instance_id=obj['instance_id'],nyu40=klass,support_pixels=n,
            group='CONTEXT' if klass in CONTEXT else 'UNKNOWN_CLASS' if klass is None else 'FOREGROUND',
            arms={a:dict(mapped=klass in (MAPPING if a=='new_seg' else S.MAPPING),
                correct_pixels=int((take&(pred==klass)).sum()) if klass is not None else 0,
                named_pixels=int((take&(pred>0)).sum()),
                hit=bool(klass is not None and (take&(pred==klass)).sum()>=.5*n)) for a,pred in predictions.items()}))
    frame=deepcopy(old)
    frame['dense_class_queries']=class_queries;frame['dense_near_objects']=near_objects
    frame['dense_confusion']={a:confusion(sem,pred) for a,pred in predictions.items()}
    frame['dense_unknown_GT_contact_pixels']=[int((mask&~sem_valid).sum()) for mask in contacts]
    return frame


def summarize(frames):
    samples=[c for f in frames for c in f['dense_class_queries']]
    output=dict(frames=len(frames),class_queries={},near_objects={},full_image={})
    for arm in ARMS:
        per_class={}
        for klass in range(1,41):
            allrows=[c for c in samples if c['nyu40']==klass];main=[c for c in allrows if c['support_pixels']>=16]
            n=len(main);pixels=sum(c['support_pixels'] for c in main);correct=sum(c['arms'][arm]['correct_pixels'] for c in main)
            per_class[str(klass)]=dict(n=n,hits=sum(c['arms'][arm]['hit'] for c in main),pixels=pixels,correct_pixels=correct,
                named_pixels=sum(c['arms'][arm]['named_pixels'] for c in main),without_instance_pixels=sum(c['without_instance_pixels'] for c in main),
                hit_rate=sum(c['arms'][arm]['hit'] for c in main)/n if n else None,pixel_recall=correct/pixels if pixels else None,
                small_1_15_queries=sum(0<c['support_pixels']<16 for c in allrows),
                small_1_15_pixels=sum(c['support_pixels'] for c in allrows if 0<c['support_pixels']<16),
                mapped=klass in (MAPPING if arm=='new_seg' else S.MAPPING))
        groups={}
        for group,classes in [('all40',set(range(1,41))),('FOREGROUND',set(range(1,41))-CONTEXT),('CONTEXT',CONTEXT)]:
            items=[per_class[str(k)] for k in sorted(classes)];present=[v for v in items if v['n']]
            groups[group]=dict(n=sum(v['n'] for v in items),hits=sum(v['hits'] for v in items),
                pixels=sum(v['pixels'] for v in items),correct_pixels=sum(v['correct_pixels'] for v in items),
                named_pixels=sum(v['named_pixels'] for v in items),without_instance_pixels=sum(v['without_instance_pixels'] for v in items),
                classes_with_support=len(present),macro_class_query_hit_rate=float(np.mean([v['hit_rate'] for v in present])) if present else None,
                macro_class_pixel_recall=float(np.mean([v['pixel_recall'] for v in present])) if present else None,
                mapped_class_query_ceiling=sum(v['n'] for v in items if v['mapped']),
                small_1_15_queries=sum(v['small_1_15_queries'] for v in items),small_1_15_pixels=sum(v['small_1_15_pixels'] for v in items))
        output['class_queries'][arm]=dict(per_class=per_class,groups=groups,
            macro_denominator='GT classes with >=1 supported class-query in this split; unmapped classes remain, absent classes excluded')
        objects=[o for f in frames for o in f['dense_near_objects']]
        objectgroups={}
        for group in ('all','FOREGROUND','CONTEXT','UNKNOWN_CLASS','mapped','unsupported_or_unknown'):
            items=[o for o in objects if group=='all' or o['group']==group or (group=='mapped' and o['arms'][arm]['mapped']) or
                   (group=='unsupported_or_unknown' and not o['arms'][arm]['mapped'])]
            objectgroups[group]=dict(n=len(items),hits=sum(o['arms'][arm]['hit'] for o in items),
                support_pixels=sum(o['support_pixels'] for o in items),correct_pixels=sum(o['arms'][arm]['correct_pixels'] for o in items),
                named_pixels=sum(o['arms'][arm]['named_pixels'] for o in items),mapped_ceiling_n=sum(o['arms'][arm]['mapped'] for o in items))
        output['near_objects'][arm]=objectgroups
        matrix=sum((np.asarray(f['dense_confusion'][arm],dtype=np.int64) for f in frames),np.zeros((41,41),np.int64))
        mapped=set(MAPPING if arm=='new_seg' else S.MAPPING);unmapped=sorted(set(range(1,41))-mapped)
        output['full_image'][arm]=dict(confusion41x41=matrix,axes='GT NYU0..40 rows / predicted NYU0..40 columns; 0=unknown',
            total_pixels=int(matrix.sum()),unknown_GT_pixels=int(matrix[0].sum()),predicted_named_on_unknown_GT=int(matrix[0,1:].sum()),
            known_but_unmapped_GT_pixels=int(matrix[unmapped].sum()),predicted_named_on_unmapped_GT=int(matrix[unmapped,1:].sum()),
            conditional_mapped_GT=conditional_precision(matrix,mapped),conditional_common8_GT=conditional_precision(matrix,COMMON))
    output['unknown_GT_contact_pixels']=sum(sum(f['dense_unknown_GT_contact_pixels']) for f in frames)
    return output


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan=P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():assert P.sha(ROOT/path)==digest,path
    for item in plan['dependencies'].values():assert P.sha(item['path'])==item['sha256']
    old={f['id']:f for f in P.read(S.OUT/'frame-ledger.json')}
    dense=P.read(OUT/'predictions/inference-result.json');yolo=P.read(S.OUT/'predictions/inference-result.json')
    model=P.read(OUT/'model-receipt.json');assert dense['status']=='COMPLETE' and yolo['status']=='COMPLETE'
    assert dense['model']['files']==model['files'] and dense['model']['processor']==model['processor']
    model_sha=model['files']['model.safetensors']['sha256']
    ids=[r['id'] for r in plan['inputs']]
    assert len(ids)==len(set(ids))==343 and set(ids)==set(old)==set(dense['outputs'])==set(yolo['outputs'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames=list(pool.map(evaluate_frame,[(r,old[r['id']],dense['outputs'][r['id']],yolo['outputs'][r['id']],plan,model_sha) for r in plan['inputs']]))
    for frame in frames:
        original={k:v for k,v in frame.items() if not k.startswith('dense_')}
        assert original==old[frame['id']],'Original semantic ledger changed'
    groups={split:summarize([f for f in frames if split=='all' or f['split']==split]) for split in ('cal','eval','all')}
    assert groups['eval']['near_objects']['new_seg']['all']['n']==135
    families={family:summarize([f for f in frames if f['family']==family]) for family in sorted({f['family'] for f in frames})}
    P.save(OUT/'frame-ledger.json',frames)
    P.save(OUT/'result.json',dict(run_id=RUN_ID,role=plan['role'],groups=groups,families=families,original_semantic_ledger_preserved=True,
        inference_result_sha256=P.sha(OUT/'predictions/inference-result.json'),limits=[
            'No ToF, range, alarms, pseudo-instances or new confirmation.',
            'Context wall/floor/ceiling separately reported; class macro averages include all GT-present supported classes, not only mapped ones.',
            'Pixel precision is conditional on explicitly mapped GT; unknown and unmapped GT prediction burden remains visible.',
            'YOLO rectangles and dense semantic pixels have different support shapes; this is the fixed end-to-end category coverage comparison.']))
    print(json.dumps(dict(status='COMPLETE',frames=len(frames),eval_near_objects=135)))


def selftest():
    labels=np.full(SHAPE,149,np.uint8)
    for index,(nyu,ade) in enumerate(MAPPING.items()):labels[0,index]=ade
    mapped=dense_to_nyu(labels)
    for index,nyu in enumerate(MAPPING):assert mapped[0,index]==nyu
    assert not mapped[1:].any() and len(MAPPING)==29 and len(COMMON)==8
    det=[dict(xyxy=[0,0,2,2],class_id=2,score=.9),dict(xyxy=[0,0,4,2],class_id=56,score=.8)]
    box=yolo_to_nyu(det,(2,4));assert not box[:,:2].any() and np.all(box[:,2:]==5)
    tie=yolo_to_nyu([dict(xyxy=[.5,.5,1.5,1.5],class_id=56,score=.8),dict(xyxy=[0,0,2,2],class_id=57,score=.8)],(2,2))
    assert tie[0,0]==5 and tie[0,1]==6 and tie[1,0]==6
    matrix=np.zeros((41,41),np.int64);matrix[5,5]=2;matrix[5,1]=1;matrix[0,5]=7;matrix[10,5]=9
    metrics=conditional_precision(matrix,COMMON)
    assert metrics['per_class']['5']['precision']==1 and metrics['GT_domain_pixels']==3
    assert metrics['predicted_other_known_class_pixels']==1 and metrics['correct_naming_pixel_recall']==2/3
    print('PASS strict29/8 mapping, unknown high-score box ownership, centre/score ties, conditional precision scope')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
