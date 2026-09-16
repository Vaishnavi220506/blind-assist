"""One frozen all-TRAIN192 annotation opportunity audit; no RGB/model access."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT=Path('E:/linnan/linnan');CODE=ROOT/'research/active/dtr-r0/nearfield'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(ROOT/'tools'))
from mz162_return_labels import METHOD,prepare_frame,slot_labels
from run_mz139_surface_fit import local_dependencies,selected_jsonl
from mz136_incumbent import public_observations

OUT=ROOT/'artifacts.local/work/mz162-return-correspondence-20260916/preflight'
CAP=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def coverage(reference,support):
    tv=reference['target_visible'];tr=reference['target_risky'];rr=reference['risky']
    tc=tv.sum(axis=0);rc=tr.sum(axis=0)
    tcovered=(tv&support).sum(axis=0);rcovered=(tr&support).sum(axis=0)
    return dict(supported_known_pixels=int((reference['known']&support).sum()),
        visible_risky_supported_pixels=int((rr&support).sum()),
        target_visible_supported_pixels=int((tv&support).sum()),target_risky_supported_pixels=int((tr&support).sum()),
        target_visible_supported_columns_any=int(((tc>0)&(tcovered>0)).sum()),
        target_visible_supported_columns_half=int(((tc>0)&(2*tcovered>=tc)).sum()),
        target_risky_supported_columns_any=int(((rc>0)&(rcovered>0)).sum()),
        target_risky_supported_columns_half=int(((rc>0)&(2*rcovered>=rc)).sum()))


def aggregate(frames):
    counts=Counter();states=Counter();reasons=Counter();by_status=defaultdict(Counter)
    for f in frames:
        counts.update(f['counts']);states.update(f['annotation_states']);reasons.update(f['unknown_reasons'])
        for status,v in f['public_statuses'].items():by_status[status].update(v)
    groups={}
    for group in ('all_annotatable','pure_owner','mixed_owner'):
        c=Counter()
        for f in frames:c.update(f['coverage'][group])
        def ratio(a,b):return c[a]/counts[b] if counts[b] else None
        groups[group]=dict(counts=dict(c),visible_risky_pixel_fraction=ratio('visible_risky_supported_pixels','visible_risky_pixels'),
            target_visible_pixel_fraction=ratio('target_visible_supported_pixels','target_visible_pixels'),
            target_risky_pixel_fraction=ratio('target_risky_supported_pixels','target_risky_pixels'),
            target_visible_half_column_fraction=ratio('target_visible_supported_columns_half','target_visible_columns'),
            target_risky_half_column_fraction=ratio('target_risky_supported_columns_half','target_risky_columns'))
    return dict(frames=len(frames),counts=dict(counts),annotation_states=dict(states),unknown_reasons=dict(reasons),
        public_statuses=dict(by_status),coverage=groups,
        frames_with_visible_risky_support=sum(f['coverage']['all_annotatable']['visible_risky_supported_pixels']>0 for f in frames),
        frames_with_target_risky_support=sum(f['coverage']['all_annotatable']['target_risky_supported_pixels']>0 for f in frames),
        frames_with_visible_target_but_no_correspondence=sum(f['counts']['target_visible_pixels']>0 and f['coverage']['all_annotatable']['target_visible_supported_pixels']==0 for f in frames))


def run():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'scope-freeze.json').exists():raise ValueError('Frozen one-run preflight already exists; no automatic rerun')
    # Source metadata selection is permitted; real native records are not parsed yet.
    spec=read(CAP/'spec.json');metadata={f['id']:f for f in spec['frames'] if f['split']=='train'}
    ids=set(metadata)
    assert len(ids)==192 and len({f['scene_group'] for f in metadata.values()})==16
    assert Counter(f['family'] for f in metadata.values())==dict.fromkeys(
        ('suspended_head','substantial_body','near_rod_farwall','shallow_boundary_stress'),48)
    receipt=read(CAP/'receipt.json');assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    sources=local_dependencies(Path(__file__))
    sources[str(CODE/'test_mz162_return_labels.py')]=sha(CODE/'test_mz162_return_labels.py')
    sources[str(Path(__file__).resolve())]=sha(__file__)
    inputs={str(CAP/n):sha(CAP/n) for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')}
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/n)==receipt['hashes'][n]
    snapshots={}
    for i,(p,h) in enumerate(sources.items()):
        target=OUT/'source-snapshot'/f'{i:02d}-{Path(p).name}';target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target);assert sha(target)==h;snapshots[str(target.relative_to(OUT))]=h
    scope=dict(method=METHOD,cohort='ALL_ORIGINAL_MZ136_TRAIN192_NO_OUTCOME_SELECTION',ids=sorted(ids),
        rows=192,scene_groups=16,sources=sources,inputs=inputs,snapshots=snapshots,
        real_native_records_parsed_before_freeze=False,rgb_decoding=False,model_or_training=False,
        original_dev_test_records_decoded=False,
        authority='CONSUMED_TRAIN_LABEL_OPPORTUNITY_NOT_PREDICTIVE_VALIDATION',
        later_partition='Future FIT144/HELD48 are both consumed Development; no claim held labels were never accessed',
        backend='TASK_NOT_GPU_SUITABLE: scalar metadata and analytic NumPy label admission',
        success_claim='Only annotation availability; no learned or alert performance claim')
    write(OUT/'scope-freeze.json',scope)
    print(json.dumps(dict(stage='scope_frozen',sha256=sha(OUT/'scope-freeze.json'),frames=192)),flush=True)
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids))
    es=selected_jsonl(CAP/'evaluator.jsonl',ids)
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    frames=[];returns=[];yaw=0.;episode=None
    for i,(row,e) in enumerate(zip(rows,es)):
        if episode!=row['episode_id']:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];meta=metadata[row['id']]
        assert e['family']==meta['family']
        native={o['name']:o for o in e['native_bounds']}
        assert set(native)=={o['name'] for o in meta['objects']}
        for o in meta['objects']:
            assert np.allclose(native[o['name']]['center_m'],o['center_m'],rtol=0,atol=1e-5)
            assert np.allclose(2*np.asarray(native[o['name']]['extent_m']),o['size_m'],rtol=0,atol=1e-5)
        assert e['camera']==meta['camera'] and e['body_origin_m']==meta['body_origin_m']
        frame=prepare_frame(row,e,yaw);ref=frame['reference']
        unions={k:np.zeros(ref['known'].shape,bool) for k in ('all_annotatable','pure_owner','mixed_owner')}
        counts=Counter(all_slot_positions=128,present_returns=0,usable_returns=0,complete_face_set_returns=0,
            pure_owner_returns=0,mixed_owner_returns=0,unknown_usable_returns=0,returns_with_visible_members=0,
            native_contributors=0,native_corridor_contributors=0,native_inside_rgb_contributors=0,
            native_shape0_contributors=0,known_first_hit_pixels=int(ref['first_hit_known'].sum()),
            known_unique_face_pixels=int(ref['known'].sum()),visible_risky_pixels=int(ref['risky'].sum()),
            target_visible_pixels=int(ref['target_visible'].sum()),target_risky_pixels=int(ref['target_risky'].sum()),
            target_visible_columns=int(ref['target_visible'].any(axis=0).sum()),target_risky_columns=int(ref['target_risky'].any(axis=0).sum()))
        states=Counter();reasons=Counter();statuses=defaultdict(Counter)
        for slot in frame['slots']:
            record=dict(id=row['id'],family=e['family'],scene_group=meta['scene_group'],**slot)
            states[slot['annotation_status']]+=1;reasons.update(slot['unknown_reasons'])
            counts['present_returns']+=slot['present'];counts['usable_returns']+=slot['public_usable']
            for key in ('native_contributors','native_corridor_contributors','native_inside_rgb_contributors','native_shape0_contributors'):
                counts[key]+=slot[key]
            group='mixed_owner' if slot['complete_face_set'] and slot['owner_count']>1 else 'pure_owner' if slot['complete_face_set'] else 'unknown'
            if slot['public_usable']:
                statuses[slot['public_status']]['usable_returns']+=1
                statuses[slot['public_status']][group+'_returns']+=1
                counts['complete_face_set_returns']+=slot['complete_face_set']
                if group=='pure_owner':counts['pure_owner_returns']+=1
                elif group=='mixed_owner':counts['mixed_owner_returns']+=1
                else:counts['unknown_usable_returns']+=1
                for key in ('native_contributors','native_corridor_contributors','native_inside_rgb_contributors'):
                    counts[group+'_'+key]+=slot[key]
            if slot['complete_face_set']:
                label=slot_labels(frame,slot);member=label['member'];record.update(label['audit'])
                unions['all_annotatable']|=member;unions[group]|=member
                counts['returns_with_visible_members']+=bool(member.any())
                if member.any():
                    values=label['relative_slant_residual'][member]
                    record['relative_slant_residual']=dict(min=float(values.min()),max=float(values.max()),mean=float(values.mean()))
                else:record['relative_slant_residual']=None
            else:record.update(visible_member_pixels=0,known_pixels=0,visible_member_columns=0,relative_slant_residual=None)
            returns.append(record)
        frames.append(dict(id=row['id'],family=e['family'],scene_group=meta['scene_group'],episode=row['episode_id'],
            nominal_partition='heldout' if meta['scene_group'].endswith('_scene3') else 'fit',counts=dict(counts),
            annotation_states=dict(states),unknown_reasons=dict(reasons),public_statuses=dict(statuses),
            coverage={k:coverage(ref,u) for k,u in unions.items()},label_audit=frame['audit']))
        if (i+1)%24==0:print(json.dumps(dict(stage='annotation',frames=i+1,seconds=time.perf_counter()-start)),flush=True)
    summary=dict(authority=scope['authority'],scope_freeze_sha256=sha(OUT/'scope-freeze.json'),all=aggregate(frames),
        families={f:aggregate([v for v in frames if v['family']==f]) for f in sorted({v['family'] for v in frames})},
        consumed_partitions={p:aggregate([v for v in frames if v['nominal_partition']==p]) for p in ('fit','heldout')},
        limits=['Annotation availability is not predictive identifiability or a hardware sensor guarantee.',
                'MIXED_OWNER_SET membership is a union; pixelwise residual does not assert one owner or one slot depth.',
                'Missing owner/bound/face/lineage leaves the entire slot label unknown; raw slot remains present.',
                'Native actor AABBs plus source-commanded camera are analytic, not rendered depth/segmentation.',
                'Non-colliding texture tiles are 1 mm in front of native faces.',
                'Unlisted global environment geometry is not filled using source truth.',
                'Target shape0 is diagnostic only; every resolved named actor contributes labels.',
                'All192 TRAIN label records are now consumed, including any future nominal heldout48.'])
    write(OUT/'annotation-summary.json',summary);write(OUT/'frames.json',frames);write(OUT/'returns.json',returns)
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    products=('scope-freeze.json','annotation-summary.json','frames.json','returns.json')
    write(OUT/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,frames=192,
        outputs={p:sha(OUT/p) for p in products},sources_unchanged=True,inputs_unchanged=True,
        no_rgb_decode=True,no_models=True,no_dev_test_records=True,resources='Process exit; no workers or allocations'))
    print(json.dumps(dict(stage='complete',seconds=time.perf_counter()-start,
        counts=summary['all']['counts'],coverage=summary['all']['coverage'])),flush=True)


if __name__=='__main__':run()
