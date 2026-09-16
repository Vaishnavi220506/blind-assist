"""Frozen supplemental TRAIN-only annotation reduction; no label/gate change."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path('E:/linnan/linnan');CODE=ROOT/'research/active/dtr-r0/nearfield'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(ROOT/'tools'))
from mz162_return_labels import prepare_frame,slot_labels
from run_mz139_surface_fit import selected_jsonl
from mz136_incumbent import public_observations

PARENT=ROOT/'artifacts.local/work/mz162-return-correspondence-20260916/preflight'
OUT=PARENT/'supplement-v1'
CAP=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def zone_mask(zone,intr):
    yy,xx=np.mgrid[:intr['height'],:intr['width']]
    theta=np.degrees(np.arctan((xx-intr['cx'])/intr['fx']))
    phi=np.degrees(np.arctan((intr['cy']-yy)/intr['fy']))
    return ((theta>=zone['theta_bounds_deg'][0]) & (theta<=zone['theta_bounds_deg'][1])
        & (phi>=zone['phi_bounds_deg'][0]) & (phi<=zone['phi_bounds_deg'][1]))


def aggregate(frames,returns,pairs):
    counts=Counter()
    for f in frames:counts.update(f['counts'])
    return_counts=Counter();pair_counts=Counter()
    for r in returns:
        return_counts['usable_returns']+=1
        return_counts['complete_face_sets']+=r['complete_face_set']
        return_counts['returns_with_members']+=r['member_pixels']>0
        return_counts['returns_with_beyond_zone_members']+=r['beyond_zone_member_pixels']>0
        for name in ('member_pixels','member_columns','beyond_zone_member_pixels','beyond_zone_member_columns',
                     'risky_member_pixels','beyond_zone_risky_pixels','beyond_zone_risky_columns'):
            return_counts[name]+=r[name]
    for p in pairs:
        pair_counts['two_public_return_zones']+=1
        for name in ('both_public_usable','both_complete','both_visible','different_face_sets',
                     'different_masks','any_visible_different_masks','both_visible_different_masks'):
            pair_counts[name]+=p[name]
        pair_counts['symmetric_difference_pixels']+=p['symmetric_difference_pixels'] or 0
    return dict(frames=len(frames),frame_union_counts=dict(counts),per_return_summed_counts=dict(return_counts),
        paired_return_counts=dict(pair_counts),
        deduplication='Per-return counts repeat surfaces across returns; frame_union_counts count each image pixel/column once per frame')


def run():
    started=time.perf_counter()
    if (OUT/'scope-freeze.json').exists():raise ValueError('Supplement already frozen; no automatic rerun')
    original=read(PARENT/'scope-freeze.json');completion=read(PARENT/'completion.json')
    assert completion['status']=='PASS'
    original_receipts={str(PARENT/n):h for n,h in completion['outputs'].items()}
    original_receipts[str(PARENT/'completion.json')]=sha(PARENT/'completion.json')
    bound={**original['inputs'],**original['sources'],**original_receipts}
    for p,h in bound.items():assert sha(p)==h,p
    assert len(original['ids'])==192
    script_sha=sha(__file__)
    scope=dict(authority='SUPPLEMENTAL_CONSUMED_TRAIN_ANNOTATION_ONLY_NO_REOPENING',
        cohort='EXACT_PARENT_TRAIN192_IDS',ids=original['ids'],parent_freeze_sha256=sha(PARENT/'scope-freeze.json'),
        parent_completion_sha256=sha(PARENT/'completion.json'),script_sha256=script_sha,bound_files=bound,
        labels_unchanged=True,gate_unchanged=True,rgb_decode=False,model_or_training=False,
        original_dev_test_records=False,backend='TASK_NOT_GPU_SUITABLE',
        beyond_zone='Member pixel center angular theta/phi outside own PUBLIC zone closed bounds; no padding',
        informative_pair='Both public usable, both complete native face sets, both visible nonempty membership masks, masks differ',
        secondary_pair='Any visible differing masks also counted separately',
        freeze_before_selected_native_parse=True)
    write(OUT/'scope-freeze.json',scope)
    print(json.dumps(dict(stage='supplement_frozen',sha256=sha(OUT/'scope-freeze.json'))),flush=True)
    ids=set(original['ids']);rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids));es=selected_jsonl(CAP/'evaluator.jsonl',ids)
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    returns=[];pairs=[];frames=[];yaw=0.;episode=None
    for i,(row,e) in enumerate(zip(rows,es)):
        if episode!=row['episode_id']:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];frame=prepare_frame(row,e,yaw);ref=frame['reference'];intr=row['rgb_intrinsics']
        union=np.zeros(ref['known'].shape,bool);beyond_union=np.zeros_like(union)
        for zone in row['tof_zones']:
            zid=zone['zone_id'];footprint=zone_mask(zone,intr);zone_labels=[]
            for slot_index in range(2):
                meta=frame['slots'][2*zid+slot_index]
                if meta['public_usable']:
                    label=slot_labels(frame,meta);member=label['member'];beyond=member&~footprint
                    risky=member&ref['risky'];beyond_risky=beyond&ref['risky']
                    union|=member;beyond_union|=beyond
                    returns.append(dict(id=row['id'],family=e['family'],zone_id=zid,target_slot=slot_index,
                        public_status=meta['public_status'],annotation_status=meta['annotation_status'],
                        complete_face_set=meta['complete_face_set'],member_pixels=int(member.sum()),
                        member_columns=int(member.any(axis=0).sum()),beyond_zone_member_pixels=int(beyond.sum()),
                        beyond_zone_member_columns=int(beyond.any(axis=0).sum()),
                        risky_member_pixels=int(risky.sum()),beyond_zone_risky_pixels=int(beyond_risky.sum()),
                        beyond_zone_risky_columns=int(beyond_risky.any(axis=0).sum())))
                    zone_labels.append(member)
                else:zone_labels.append(None)
            if len(zone['targets'])==2:
                a,b=frame['slots'][2*zid:2*zid+2]
                both_usable=a['public_usable'] and b['public_usable']
                both_complete=both_usable and a['complete_face_set'] and b['complete_face_set']
                visible=[bool(v is not None and v.any()) for v in zone_labels]
                different=bool(both_complete and np.any(zone_labels[0]!=zone_labels[1]))
                pairs.append(dict(id=row['id'],family=e['family'],zone_id=zid,
                    public_statuses=[a['public_status'],b['public_status']],ranges_m=[a['range_m'],b['range_m']],
                    both_public_usable=bool(both_usable),both_complete=bool(both_complete),
                    both_visible=bool(both_complete and all(visible)),
                    different_face_sets=bool(both_complete and a['face_ids']!=b['face_ids']),different_masks=different,
                    any_visible_different_masks=bool(different and any(visible)),
                    both_visible_different_masks=bool(different and all(visible)),
                    member_pixels=[int(v.sum()) if v is not None else None for v in zone_labels],
                    symmetric_difference_pixels=int((zone_labels[0]!=zone_labels[1]).sum()) if both_complete else None))
        tr=ref['target_risky'];tc=tr.sum(axis=0);covered=(tr&beyond_union).sum(axis=0)
        frames.append(dict(id=row['id'],family=e['family'],counts=dict(
            member_pixels=int(union.sum()),member_columns=int(union.any(axis=0).sum()),
            beyond_zone_member_pixels=int(beyond_union.sum()),beyond_zone_member_columns=int(beyond_union.any(axis=0).sum()),
            visible_risky_member_pixels=int((union&ref['risky']).sum()),
            beyond_zone_visible_risky_pixels=int((beyond_union&ref['risky']).sum()),
            beyond_zone_visible_risky_columns=int((beyond_union&ref['risky']).any(axis=0).sum()),
            target_risky_columns=int((tc>0).sum()),beyond_zone_target_risky_columns_any=int(((tc>0)&(covered>0)).sum()),
            beyond_zone_target_risky_columns_half=int(((tc>0)&(2*covered>=tc)).sum()))))
        if (i+1)%48==0:print(json.dumps(dict(stage='supplement',frames=i+1,seconds=time.perf_counter()-started)),flush=True)
    summary=dict(authority=scope['authority'],scope_freeze_sha256=sha(OUT/'scope-freeze.json'),
        all=aggregate(frames,returns,pairs),families={family:aggregate(
            [v for v in frames if v['family']==family],[v for v in returns if v['family']==family],
            [v for v in pairs if v['family']==family]) for family in sorted({e['family'] for e in es})},
        limits=['Geometric label opportunity only, not learned discrimination or alert improvement.',
                'Beyond-zone correspondence extends resolved native faces, not unobserved unrelated owners.',
                'A frame union of beyond-own-zone pixels may lie inside another returned zone; per-return attribution is distinct.',
                'Native AABB/source-commanded reference and unavailable unsaved environment visibility limits persist.',
                'Parent failed gate remains unchanged; no training or new inference was run.'])
    for name,value in (('summary.json',summary),('frames.json',frames),('returns.json',returns),('pairs.json',pairs)):write(OUT/name,value)
    for p,h in bound.items():assert sha(p)==h,p
    assert sha(__file__)==script_sha
    write(OUT/'completion.json',dict(status='PASS',seconds=time.perf_counter()-started,parent_files_unchanged=True,
        outputs={n:sha(OUT/n) for n in ('scope-freeze.json','summary.json','frames.json','returns.json','pairs.json')},
        resources='CPU process exit; no models, GPU allocations or workers'))
    print(json.dumps(dict(stage='complete',seconds=time.perf_counter()-started,all=summary['all'])),flush=True)


if __name__=='__main__':run()
