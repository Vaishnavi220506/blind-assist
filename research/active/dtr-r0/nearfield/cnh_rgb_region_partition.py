"""Evaluate fixed RGB partitions against coarse and matched spatial partitions.

No region receives an instance or range prediction from evaluator truth. Purity
is an information diagnostic, not a deployable assignment or alarm metric.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import cnh_rgb_semantic_hints as S
import cnh_rgb_region_generators as G

P, V, A, ROOT = S.P, S.V, S.A, S.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-region-partition-20261002'
RUN_ID = 'CNH_RGB_REGION_PARTITION_20261002'
ARMS = ('coarse', 'sam', 'matched_spatial')


def prepare():
    assert not (OUT/'PLAN.json').exists(), 'Preserve existing frozen plan'
    source = P.read(S.OUT/'PLAN.json')
    observations = P.read(S.OUT/'observations.json')
    pilot = sorted([r for r in source['inputs'] if r['split'] == 'cal'],
                   key=lambda r: hashlib.sha256(('rgb-region-pilot|'+r['id']).encode()).hexdigest())[:8]
    pilot_ids = {r['id'] for r in pilot}
    P.save(OUT/'observations.json', observations)
    P.save(OUT/'pilot-observations.json', [r for r in observations if r['id'] in pilot_ids])
    prereg = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; SAM2.1 small RGB-only AMG 16x16 points, batch32, no crop, predIoU>.88/stability>.95; top32 predicted-IoU, smallest-mask overlap; coarse8x8 and exact per-zone count-matched spatial KD controls; same343 consumed Development, 8 hash-cal engineering pilot cached then full343; purity>=90% and >=50% contact-pixel coverage fixed descriptive point | NOT_RUN | No GT mask selection, no distance/assignment/alarm claim; retain every object/unknown/mixed region | `artifacts.local/work/cnh-rgb-region-partition-20261002/REPORT.md` |')
    text = P.RUNS.read_text(encoding='utf8'); assert RUN_ID not in text
    P.RUNS.write_text(text.rstrip()+'\n'+prereg+'\n', encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n', encoding='utf8')
    paths = [Path(__file__), Path(G.__file__), Path(S.__file__), Path(A.__file__), Path(V.__file__),
             Path(__file__).with_name('cnh_rgb_region_infer.py')]
    P.save(OUT/'PLAN.json', dict(run_id=RUN_ID, frozen_at=datetime.now(timezone.utc).isoformat(),
        role='Consumed Development; same 343 frames as category diagnostic, no fresh confirmation',
        inputs=source['inputs'], pilot_ids=sorted(pilot_ids), observations_sha256=P.sha(OUT/'observations.json'),
        model_recovery=P.read(OUT/'model-recovery.json'),
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in paths},
        reference_ledger_sha256=P.sha(S.OUT/'frame-ledger.json'),
        rules=dict(purity=.9, coverage=.5, minimum_query_pixels=16, maximum_masks=32,
            predictor='RGB plus public camera rays only; no reference depth/instance/semantic access',
            mask_choice='predicted IoU descending, top32; pixel overlap chooses smallest area, then score descending, then stable original index',
            spatial_control='Within each public coarse zone, same number K of observed SAM labels including uncovered. Balanced geometric KD partitions all zone pixels into exactly K parts, never reads truth.',
            purity_denominator='Every pixel in region x coarse zone; unknown truth remains in denominator. Dominant instance and semantic label only score the region, never supplied to prediction.',
            primary='Per contact-supported instance-view: fraction of union contact pixels in output patches >=90% pure for that instance; report >=50% covered count/n, mapped/unsupported/unknown and paired SAM-spatial wins/losses.',
            secondary='Pass-adjacent instances; coarse patch counts, unassigned pixel cost, per-patch radial r^-2 P90-P10 spread, semantic-only wall/floor/ceiling support; clear/UNKNOWN query strata retained.',
            decision='Descriptive opportunity diagnostic, no new promotion gate; do not call RGB assignment, metric clearance or alarm gain.'),
        preregistration=prereg))
    print('Prepared fixed 343 observations and 8 cal pilot')


def weighted_spread(values):
    if len(values) == 0:
        return None
    values = np.sort(values.astype(float))
    weight = 1/(values*values); cumulative = np.cumsum(weight)/weight.sum()
    return float(values[min(len(values)-1, np.searchsorted(cumulative,.9))]-values[min(len(values)-1,np.searchsorted(cumulative,.1))])


def score_partition(labels, zone, inst, sem, radial):
    fov = zone >= 0
    covered = fov & (labels >= 0)
    combined = labels[covered].astype(np.int64)*64+zone[covered]
    keys, inverse = np.unique(combined, return_inverse=True)
    patch_map = np.full(zone.shape, -1, np.int32); patch_map[covered] = inverse
    n = len(keys); max_id = max(2,int(inst.max())+2)
    total = np.bincount(inverse,minlength=n)
    identity = np.maximum(inst[covered]+1, 0)
    table = np.bincount(inverse*max_id+identity,minlength=n*max_id).reshape(n,max_id)
    if n:
        dominant = table[:,1:].argmax(axis=1)+1
        counts = table[np.arange(n),dominant]
        purity = counts/total
    else:
        dominant = counts = purity = np.array([])
    pure_pixel = np.zeros(zone.shape,bool)
    if n:
        pure_pixel[covered] = (purity[inverse]>=.9) & (identity == dominant[inverse])
    semantic = np.where((sem[covered]>=1)&(sem[covered]<=40),sem[covered],0)
    stable = np.bincount(inverse*41+semantic,minlength=n*41).reshape(n,41)
    sem_pure = np.zeros(zone.shape,bool)
    if n:
        sd = stable[:,1:].argmax(axis=1)+1
        sem_pure[covered] = (stable[np.arange(n),sd][inverse]/total[inverse]>=.9) & (semantic==sd[inverse])
    patches = []
    # Group reference values once rather than searching the image for each patch.
    order = np.argsort(inverse,kind='stable'); flat_r = radial[covered][order]
    offsets = np.r_[0,np.cumsum(total)]
    for k in range(n):
        values = flat_r[offsets[k]:offsets[k+1]]
        good = np.isfinite(values)&(values>0)
        patches.append(dict(region=int(keys[k]//64), zone=int(keys[k]%64), pixels=int(total[k]),
            evaluator_dominant_instance=int(dominant[k]-1) if counts[k] else None, instance_purity_lower=float(purity[k]),
            unknown_instance_pixels=int(table[k,0]), radial_valid_pixels=int(good.sum()),
            radial_weighted_p90_p10_m=weighted_spread(values[good])))
    return pure_pixel, sem_pure, covered, patches


def evaluate_frame(pair):
    row, old = pair
    region_path = OUT/'predictions'/(row['id']+'.npz')
    receipt = P.read(OUT/'predictions'/(row['id']+'.json'))
    assert P.sha(region_path) == receipt['output_sha256']
    assert receipt['rgb_sha256'] == row['rgb_sha256']
    with np.load(region_path) as data:
        labels = data['labels']
    assert labels.shape == (768,1024)
    for name in ('depth','semantic','instance'):
        assert P.sha(row[name+'_path']) == row[name+'_sha256']
    radial = S.hdf(row['depth_path']); inst=S.hdf(row['instance_path']).astype(np.int32)
    sem=S.hdf(row['semantic_path']).astype(np.int32)
    geo = A.whole_geometry(V.ray_geometry(radial.shape,row['camera_matrix']))
    z=radial/geo['radial_factor']; valid=np.isfinite(z)&(z>0)
    x=np.abs(z*geo['fx']); y=z*geo['fy']; fov=geo['fov_mask']; zone=geo['zone_id']
    truth=A.visible_truth(z,geo)
    contacts=[]; expanded=[]
    for q in geo['queries']:
        near=fov&valid&(z>=.6)&(z<2.1)&(y>=q['y_low'])&(y<=q['y_high'])
        contacts.append(near&(x<.3)); expanded.append(near&(x<=.4))
    contact=np.logical_or.reduce(contacts); expansion=np.logical_or.reduce(expanded)
    spatial=G.matched_geometry_partition(labels,zone,geo['fx'],geo['fy'])
    maps=dict(coarse=np.zeros(zone.shape,np.int32),sam=labels,matched_spatial=spatial)
    selected=[]
    for o in old['objects']:
        has_contact=any(q['contact_pixels']>=16 for q in o['query_support'])
        has_expand=any(q['expanded_pixels']>=16 for q in o['query_support'])
        if not has_expand and not has_contact: continue
        target=inst==o['instance_id']
        assert all(int((target&m).sum()) == o['query_support'][i]['contact_pixels'] for i,m in enumerate(contacts))
        selected.append(dict(instance_id=o['instance_id'],nyu40=o['nyu40'],mapped=o['coco_class'] is not None,
            stratum='contact' if has_contact else 'pass_adjacent',
            support_pixels=int((target&(contact if has_contact else expansion)).sum()),arms={}))
    result=dict(id=row['id'],scene=row['scene'],family=row['family'],split=row['split'],objects=selected,
        queries=[dict(name=q['name'],category=str(truth['category'][i]),abstain=bool(truth['abstain'][i]),
            contact_pixels=int(contacts[i].sum()),expanded_pixels=int(expanded[i].sum())) for i,q in enumerate(geo['queries'])],
        raw_candidates=receipt['candidate_count'],selected_candidates=receipt['selected_count'],arms={})
    for arm,lmap in maps.items():
        pure,semantic_pure,covered,patches=score_partition(lmap,zone,inst,sem,radial)
        per_sem=[]
        for klass in (1,2,22):
            target=contact&(sem==klass)
            per_sem.append(dict(nyu40=klass,pixels=int(target.sum()),semantic_pure_pixels=int((target&semantic_pure).sum())))
        result['arms'][arm]=dict(patches=patches,patch_count=len(patches),fov_pixels=int(fov.sum()),
            budget_patches_including_uncovered=len(patches)+len(np.unique(zone[fov&~covered])),
            covered_pixels=int(covered.sum()),uncovered_pixels=int((fov&~covered).sum()),semantic_surfaces=per_sem,
            queries=[dict(covered_contact_pixels=int((m&covered).sum()),pure_contact_instance_pixels=int((m&pure).sum())) for m in contacts])
        for obj in selected:
            target=(inst==obj['instance_id'])&(contact if obj['stratum']=='contact' else expansion)
            good=int((target&pure).sum()); denominator=obj['support_pixels']
            obj['arms'][arm]=dict(pure_pixels=good,covered_pixels=int((target&covered).sum()),
                fraction=good/denominator,hit=bool(good>=.5*denominator))
    assert result['arms']['sam']['budget_patches_including_uncovered']==result['arms']['matched_spatial']['budget_patches_including_uncovered']
    return result


def summarize(frames):
    output=dict(frames=len(frames),queries=dict(Counter(q['category'] for f in frames for q in f['queries'])),arms={},paired={})
    for arm in ARMS:
        by={}
        for stratum in ('contact','pass_adjacent'):
            for group in ('all','mapped','unsupported','UNKNOWN_CLASS'):
                objects=[o for f in frames for o in f['objects'] if o['stratum']==stratum and (group=='all' or
                    (group=='mapped' and o['mapped']) or (group=='unsupported' and not o['mapped'] and o['nyu40'] is not None) or
                    (group=='UNKNOWN_CLASS' and o['nyu40'] is None))]
                by[stratum+'/'+group]=dict(n=len(objects),hits=sum(o['arms'][arm]['hit'] for o in objects),
                    pixels=sum(o['support_pixels'] for o in objects),pure_pixels=sum(o['arms'][arm]['pure_pixels'] for o in objects))
        output['arms'][arm]=dict(groups=by,patches=sum(f['arms'][arm]['patch_count'] for f in frames),
            budget_patches_including_uncovered=sum(f['arms'][arm]['budget_patches_including_uncovered'] for f in frames),
            fov_pixels=sum(f['arms'][arm]['fov_pixels'] for f in frames),uncovered_pixels=sum(f['arms'][arm]['uncovered_pixels'] for f in frames))
    for baseline in ('coarse','matched_spatial'):
        objects=[o for f in frames for o in f['objects'] if o['stratum']=='contact']
        output['paired'][baseline]=dict(n=len(objects),sam_only=sum(o['arms']['sam']['hit'] and not o['arms'][baseline]['hit'] for o in objects),
            baseline_only=sum(not o['arms']['sam']['hit'] and o['arms'][baseline]['hit'] for o in objects))
    return output


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan=P.read(OUT/'PLAN.json')
    for name,sha in plan['source_sha256'].items(): assert P.sha(ROOT/name)==sha,name
    assert P.sha(S.OUT/'frame-ledger.json')==plan['reference_ledger_sha256']
    old={f['id']:f for f in P.read(S.OUT/'frame-ledger.json')}
    frames=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for frame in pool.map(evaluate_frame,[(r,old[r['id']]) for r in plan['inputs']]):
            frames.append(frame)
            if len(frames)%25==0: print(json.dumps(dict(evaluated=len(frames))),flush=True)
    P.save(OUT/'frame-ledger.json',frames)
    result={s:summarize([f for f in frames if s=='all' or f['split']==s]) for s in ('cal','eval','all')}
    P.save(OUT/'result.json',dict(run_id=RUN_ID,role=plan['role'],groups=result))
    print(json.dumps(result))


def selftest():
    zone=np.zeros((4,4),np.int32);inst=np.zeros((4,4),np.int32);inst[:,2:]=1
    sem=np.full((4,4),5,np.int32);radial=np.ones((4,4))
    pure,_,_,_=score_partition(np.zeros_like(zone),zone,inst,sem,radial);assert not pure.any()
    labels=inst.copy();pure,_,covered,_=score_partition(labels,zone,inst,sem,radial);assert pure.all() and covered.all()
    inst[0,0]=-1; pure,_,_,patch=score_partition(labels,zone,inst,sem,radial)
    assert patch[0]['instance_purity_lower']==7/8 and not pure[:,:2].any() and pure[:,2:].all()
    labels[:,:2]=-1;pure,_,covered,_=score_partition(labels,zone,inst,sem,radial);assert covered.sum()==8
    print('PASS evaluator purity, unknown denominator and uncovered fixtures')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
