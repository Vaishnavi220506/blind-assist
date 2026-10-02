"""Frozen dense semantics transfer to the exact prior 96-frame Development cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import zlib
import numpy as np

import cnh_rgb_dense_semantics as D
import cnh_rgb_dense_semantics_infer as I
import cnh_rgb_semantic_detector as Y

P,S,ROOT=D.P,D.S,D.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-semantic-transfer-20261002'
OLD=ROOT/'artifacts.local/work/cnh-rgb-hypersim-transfer-20261002'
BASE=D.OUT
RUN_ID='CNH_RGB_SEMANTIC_TRANSFER_20261002'
INFER=Path(__file__).with_name('cnh_rgb_semantic_transfer_infer.py')


def prepare():
    assert not (OUT/'PLAN.json').exists()
    previous=P.read(OLD/'PLAN.json');base=P.read(BASE/'PLAN.json')
    for path,digest in base['source_sha256'].items():assert P.sha(ROOT/path)==digest,'Inherited recipe drift: '+path
    rows=[]
    for old in previous['inputs']:
        row=deepcopy(old)
        for key in ('rgb','depth'):row[key+'_path']=str(ROOT/row[key+'_path'])
        depth=Path(row['depth_path'])
        row['semantic_path']=str(depth.with_name(depth.name.replace('depth_meters','semantic')))
        row['instance_path']=str(depth.with_name(depth.name.replace('depth_meters','semantic_instance')))
        for key in ('rgb','depth','semantic','instance'):
            path=Path(row[key+'_path'])
            if key in ('semantic','instance') and not path.exists():
                row[key+'_sha256']=None
                continue
            digest=P.sha(path)
            if key+'_sha256' in row:assert row[key+'_sha256']==digest
            row[key+'_sha256']=digest
        rows.append(row)
    assert len(rows)==96 and len({r['scene'] for r in rows})==48
    assert not {r['id'] for r in rows}&{r['id'] for r in base['inputs']}
    assert not {r['scene'] for r in rows}&{r['scene'] for r in base['inputs']}
    assert all(r['original_repository_role']=='train' for r in rows)
    assert not {r['family'] for r in rows}&set(previous['protected_families'])
    sources=[Path(__file__),INFER,Path(D.__file__),Path(I.__file__),Path(Y.__file__),
             Path(S.__file__),Path(D.A.__file__),Path(D.V.__file__),Path(P.__file__)]
    dependencies=[OLD/'PLAN.json',BASE/'PLAN.json',BASE/'model-receipt.json',
                  S.OUT/'predictions/inference-result.json',S.OUT/'nyu40-labels.csv',S.MODEL,S.LABELS,OUT/'label-manifest.json']
    labels=P.read(OUT/'label-manifest.json')['files']
    expected={str((P.DATA/r['path']).resolve()) for r in labels}
    assert len(labels)==192 and expected=={str(Path(r[k+'_path']).resolve()) for r in rows for k in ('semantic','instance')}
    row=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN: exact prior nonNFO96 frames/48 scenes; frozen SegFormerB0 ADE29 and phoneYOLO8, no training/calibration; same >=16 support / >=50% naming, all40 classes, foreground/context separate; label paths/official CRC frozen, missing SHA bound separately before evaluation | NOT_RUN; foreground query hits and pooled correct-pixel recall co-primary descriptive transfer; family bootstrap1000 seed2026100223; original objects and full confusion retained | Both positive = LIMITED_TRANSFER_INCREMENT_DEV; only hits positive = MIXED_TRANSFER_DEV; hits nonpositive = CATEGORY_GAIN_NOT_REPLICATED_DEV; no alarm or fresh confirmation, no remapping | `artifacts.local/work/cnh-rgb-semantic-transfer-20261002/REPORT.md` |')
    body=P.RUNS.read_text(encoding='utf-8');assert f'| {RUN_ID} |' not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+row+'\n',encoding='utf-8')
    (OUT/'prerun-row.txt').write_text(row+'\n',encoding='utf-8')
    observations=[{k:r[k] for k in ('id','rgb_path','rgb_sha256')} for r in rows]
    P.save(OUT/'observations.json',observations)
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),inputs=rows,
        role='Exact prior nonNFO96 consumed Development; no independent confirmation',
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in sources},
        dependencies={str(p.relative_to(ROOT)):P.sha(p) for p in dependencies},
        observations_sha256=P.sha(OUT/'observations.json'),
        rules=base['rules'],mapping_nyu_to_ade=base['mapping_nyu_to_ade'],mapping_nyu_to_coco=base['mapping_nyu_to_coco'],
        comparison='Identical full RGB recipes, representations/taxonomy differ; no model fitting, query input, remapping, sample replacement or threshold search',
        label_binding='RGB inference independent of absent labels; all192 exact official member paths/CRC/sizes frozen in dependency label-manifest. Separate immutable label-binding.json required before evaluation, never change PLAN or replace inputs.',
        decision='Foreground query-hit delta>0 and pooled correct-pixel delta>0 => LIMITED_TRANSFER_INCREMENT_DEV; hit delta>0 only => MIXED_TRANSFER_DEV; otherwise CATEGORY_GAIN_NOT_REPLICATED_DEV. Descriptive, not significance or deployment.',
        bootstrap=dict(cluster='family',draws=1000,seed=2026100223,include='All39 families including no foreground support; zero-denominator draws counted/excluded'),
        overlap_with343=dict(ids=[],scenes=[],families=sorted({r['family'] for r in rows}&{r['family'] for r in base['inputs']})),
        preregistration=row))
    print('Prepared exact96 RGB/depth hashes and official label member identities; missing label SHA awaits separate binding')


def validate():
    plan=P.read(OUT/'PLAN.json')
    for k,v in plan['source_sha256'].items():assert P.sha(ROOT/k)==v,k
    for k,v in plan['dependencies'].items():assert P.sha(ROOT/k)==v,k
    assert P.sha(OUT/'observations.json')==plan['observations_sha256']
    return plan


def bind_labels():
    """Bind exact preselected official labels after authorized acquisition, no selection."""
    plan=validate();assert not (OUT/'label-binding.json').exists()
    manifest=P.read(OUT/'label-manifest.json')
    labels={str((P.DATA/r['path']).resolve()):r for r in manifest['files']}
    hashes={}
    for row in plan['inputs']:
        hashes[row['id']]={}
        for key in ('semantic','instance'):
            path=Path(row[key+'_path']);entry=labels[str(path.resolve())]
            assert entry['frame_id']==row['id']
            data=path.read_bytes()
            assert len(data)==entry['uncompressed_bytes'] and zlib.crc32(data)==entry['crc32']
            digest=P.sha(path)
            if row[key+'_sha256'] is not None:assert digest==row[key+'_sha256']
            hashes[row['id']][key+'_sha256']=digest
    P.save(OUT/'label-binding.json',dict(plan_sha256=P.sha(OUT/'PLAN.json'),
        label_manifest_sha256=P.sha(OUT/'label-manifest.json'),hashes=hashes,
        bound_at=datetime.now(timezone.utc).isoformat(),scope='All192 preselected labels, no cohort or recipe change'))
    print('Bound all192 exact official CRC/size/SHA labels')


def paired(frames):
    rows=[r for f in frames for r in f['dense_class_queries'] if r['group']=='FOREGROUND' and r['support_pixels']>=16]
    def counts(items):
        a=np.array([r['arms']['yolo_boxes']['hit'] for r in items],bool)
        b=np.array([r['arms']['new_seg']['hit'] for r in items],bool)
        return dict(n=len(items),wins=int((b&~a).sum()),losses=int((a&~b).sum()),both=int((a&b).sum()),neither=int((~a&~b).sum()))
    families=sorted({f['family'] for f in frames})
    summaries=[]
    for family in families:
        items=[r for f in frames if f['family']==family for r in f['dense_class_queries'] if r['group']=='FOREGROUND' and r['support_pixels']>=16]
        c=counts(items)
        summaries.append([c['n'],c['wins']-c['losses'],sum(r['support_pixels'] for r in items),
                          sum(r['arms']['new_seg']['correct_pixels']-r['arms']['yolo_boxes']['correct_pixels'] for r in items)])
    samples=np.random.default_rng(2026100223).multinomial(len(families),np.full(len(families),1/len(families)),size=1000)@np.array(summaries)
    bootstrap={}
    for name,den,num in [('hit_delta',0,1),('pixel_recall_delta',2,3)]:
        valid=samples[:,den]>0
        bootstrap[name]=dict(ci95=np.quantile(samples[valid,num]/samples[valid,den],[.025,.975]).tolist() if valid.any() else None,
                             valid_draws=int(valid.sum()),zero_denominator_draws=int((~valid).sum()))
    return dict(**counts(rows),bootstrap=bootstrap,family_order=families,family_sufficient_statistics=summaries)


def evaluate():
    assert not (OUT/'result.json').exists()
    plan=validate();binding=P.read(OUT/'label-binding.json')
    assert binding['plan_sha256']==P.sha(OUT/'PLAN.json')
    assert binding['label_manifest_sha256']==P.sha(OUT/'label-manifest.json')
    assert set(binding['hashes'])=={r['id'] for r in plan['inputs']}
    for row in plan['inputs']:row.update(binding['hashes'][row['id']])
    dense=P.read(OUT/'predictions/inference-result.json')
    yolo=P.read(OUT/'yolo/predictions/inference-result.json')
    assert dense['status']==yolo['status']=='COMPLETE'
    ids={r['id'] for r in plan['inputs']}
    assert len(ids)==96 and ids==set(dense['outputs'])==set(yolo['outputs'])
    assert yolo['observations_sha256']==plan['observations_sha256']
    expected=P.read(BASE/'model-receipt.json')
    assert dense['model']['files']==expected['files'] and dense['model']['processor']==expected['processor']
    prior_yolo=P.read(S.OUT/'predictions/inference-result.json')
    assert yolo['provenance']==prior_yolo['provenance'],'Freeze TFLite backend and asset too'
    # Redirect artifact paths only; shared prediction/evaluation implementations stay byte-identical.
    D.OUT=OUT;S.OUT=OUT/'yolo'
    def frame(row):
        for name in ('depth','semantic','instance'):assert P.sha(row[name+'_path'])==row[name+'_sha256']
        yr=yolo['outputs'][row['id']];path=S.OUT/'predictions'/yr['receipt']
        assert P.sha(path)==yr['receipt_sha256']
        receipt=P.read(path)
        assert receipt['provenance']==yolo['provenance'] and receipt['rgb_sha256']==row['rgb_sha256']
        detections=deepcopy(receipt['detections'])
        for d in detections:d['box']=d['xyxy']
        old=S.evaluate_frame(row,detections)
        dr=dense['outputs'][row['id']]
        assert dr['adapter_sha256']==P.sha(INFER)
        return D.evaluate_frame((row,old,dr,yr,plan,I.MODEL_SHA))
    with ThreadPoolExecutor(max_workers=3) as pool:frames=list(pool.map(frame,plan['inputs']))
    summary=D.summarize(frames);pairs=paired(frames)
    a=summary['class_queries']['yolo_boxes']['groups']['FOREGROUND']
    b=summary['class_queries']['new_seg']['groups']['FOREGROUND']
    assert a['n']==b['n'] and a['pixels']==b['pixels']
    if not a['n']:decision='NO_SUPPORTED_FOREGROUND_DEV'
    elif b['hits']<=a['hits']:decision='CATEGORY_GAIN_NOT_REPLICATED_DEV'
    elif b['correct_pixels']<=a['correct_pixels']:decision='MIXED_TRANSFER_DEV'
    else:decision='LIMITED_TRANSFER_INCREMENT_DEV'
    P.save(OUT/'frame-ledger.json',frames)
    P.save(OUT/'result.json',dict(run_id=RUN_ID,role=plan['role'],decision=decision,summary=summary,paired=pairs,
        families={family:D.summarize([f for f in frames if f['family']==family]) for family in pairs['family_order']},
        label_binding_sha256=P.sha(OUT/'label-binding.json'),
        inference_hashes=dict(dense=P.sha(OUT/'predictions/inference-result.json'),yolo=P.sha(OUT/'yolo/predictions/inference-result.json'))))
    print(decision,a,b,pairs['bootstrap'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','bind_labels','evaluate','validate'))
    globals()[parser.parse_args().action]()
