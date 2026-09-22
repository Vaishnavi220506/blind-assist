"""One immutable two-cohort source audit; no alternate return or alert arm."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from audit_calibration_footprint import read,sha
from ba_camera_corridor import sample_native
from tof_fov45_core import simulate,boxes45
from tof_corridor_calibration import score_frame,decide
from return_lineage_core import masks,audit_zone

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
CORE=WORK/'ba-core-transfer-20260920'
THIN=WORK/'ba-tof-fov45-20260920'
OLD=WORK/'ba-camera-corridor-20260919'
CAL=WORK/'ba-tof-corridor-calibration-20260920'
GAP=WORK/'ba-score-factor-collapse-20260920/results.json'
OUT=WORK/'ba-return-lineage-20260920'
DOC=Path(__file__).with_name('RETURN_LINEAGE_PROTOCOL_20260920.md')
CODE=('return_lineage_core.py','run_return_lineage_audit.py','test_return_lineage_core.py',
      'tof_fov45_core.py','ba_camera_corridor.py','tof_corridor_calibration.py')


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)


def dependencies():
    inputs=[]
    for base,names in (
        (CORE,('protocol.json','spec.json','observations.json','private-lineage.json','predictions.json','frame-results.json','observation-seal.json','prediction-seal.json','evaluation-seal.json','capture/evaluator/geometry.json')),
        (THIN,('protocol.json','observations.json','private-lineage.json','observation-seal.json','prediction-seal.json','predictions.json')),
        (OLD,('evaluator-source.json','observation-seal.json','frame-results.json')),
        (CAL,('protocol.json','frame-results.json','predictions.json','scores.json','score-seal.json','prediction-seal.json','operating-point.json'))):
        inputs.extend(base/n for n in names)
    inputs.append(GAP)
    return inputs


def check_inputs():
    for name in ('observation-seal.json','prediction-seal.json','evaluation-seal.json'):
        seal=read(CORE/name)
        assert seal['status']=='COMPLETE' and seal['frames']==432
        assert seal['protocol_sha256']==sha(CORE/'protocol.json')
        for n,h in seal['hashes'].items():assert sha(CORE/n)==h
    os=read(THIN/'observation-seal.json');ps=read(THIN/'prediction-seal.json')
    assert os['frames']==ps['frames']==96 and os['status']==ps['status']=='COMPLETE'
    assert os['protocol_sha256']==ps['protocol_sha256']==sha(THIN/'protocol.json')
    assert os['observations_sha256']==sha(THIN/'observations.json')
    assert os['private_lineage_sha256']==sha(THIN/'private-lineage.json')
    assert ps['predictions_sha256']==sha(THIN/'predictions.json')
    assert ps['observation_seal_sha256']==sha(THIN/'observation-seal.json')
    assert read(OLD/'observation-seal.json')['evaluator_source_sha256']==sha(OLD/'evaluator-source.json')
    cs=read(CAL/'prediction-seal.json')
    for key,path in [('protocol_sha256',CAL/'protocol.json'),('score_seal_sha256',CAL/'score-seal.json'),
                     ('label_sha256',OLD/'frame-results.json'),('point_sha256',CAL/'operating-point.json'),
                     ('predictions_sha256',CAL/'predictions.json')]:assert cs[key]==sha(path)
    assert read(CAL/'score-seal.json')['scores_sha256']==sha(CAL/'scores.json')
    for name in ('tof_fov45_core.py','ba_camera_corridor.py','tof_corridor_calibration.py'):
        assert sha(Path(__file__).with_name(name))==read(CORE/'protocol.json')['code_hashes'][name]


def freeze():
    assert not OUT.exists(),'One frozen audit only'
    check_inputs();OUT.mkdir(parents=True)
    (OUT/'protocol-before-run.md').write_bytes(DOC.read_bytes())
    write(OUT/'protocol.json',dict(id='ba-return-lineage-20260920',timestamp=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cohort_frames=dict(core=432,thin=96),threshold=read(CAL/'operating-point.json')['threshold'],
        code={n:sha(Path(__file__).with_name(n)) for n in CODE},
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies()},
        protocol_text_sha256=sha(OUT/'protocol-before-run.md'),
        backend='TASK_NOT_GPU_SUITABLE: CPU source replay, raster masks and small-bin aggregation',
        new_algorithms=0,new_observations=0,automatic_successor=False))
    print('FROZEN',sha(OUT/'protocol.json'))


def verify():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'protocol-before-run.md')==p['protocol_text_sha256']
    for n,h in p['code'].items():assert sha(Path(__file__).with_name(n))==h,n
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    check_inputs();return p


def cohort_records(name):
    if name=='core':
        obs=read(CORE/'observations.json');lin=read(CORE/'private-lineage.json')
        geos=read(CORE/'capture/evaluator/geometry.json');cases=read(CORE/'spec.json')['cases']
        labels=read(CORE/'frame-results.json');preds=read(CORE/'predictions.json')
        for o,l,g,c,r,p in zip(obs,lin,geos,cases,labels,preds):
            yield CORE,o,l,g,c,r,p,CORE/'capture/evaluator'/g['native_path'],l['seed'],g['native_sha256']
    else:
        obs=read(THIN/'observations.json');lin=read(THIN/'private-lineage.json')
        sources=read(OLD/'evaluator-source.json');labels=read(CAL/'frame-results.json')
        preds=read(CAL/'predictions.json');scores=read(CAL/'scores.json')
        for o,l,s,r,p,score in zip(obs,lin,sources,labels,preds,scores):
            p=dict(id=p['id'],anchors=p['anchors'],zone_scores=score['zone_scores'],predictions=dict(calibrated=p['candidate']))
            assert r['predictions']['candidate']==p['predictions']['calibrated']
            yield THIN,o,l,s['geometry'],s['case'],r,p,OLD/'capture/evaluator'/s['geometry']['native_path'],o['seed'],s['sensor_native_sha256']


def summarize(frames):
    positive=[f for f in frames if f['truth']]
    def group(fs):
        return dict(frames=len(fs),positive_frames=sum(f['truth'] for f in fs),
            TP=sum(f['truth'] and f['alert'] for f in fs),FP=sum(not f['truth'] and f['alert'] for f in fs),
            FN=sum(f['truth'] and not f['alert'] for f in fs),
            frames_with_native_target_corridor=sum(f['native_corridor_zones']>0 for f in fs),
            frames_with_sampled_target_corridor=sum(f['sampled_corridor_zones']>0 for f in fs),
            positive_corridor_zone_instances=sum(f['sampled_corridor_zones'] for f in fs if f['truth']),
            native_target_zone_instances=sum(f['native_target_zones'] for f in fs),
            sampled_target_zone_instances=sum(f['sampled_target_zones'] for f in fs),
            eligible_target_zone_instances=sum(f['eligible_target_zones'] for f in fs),
            eligible_target_ge4_zone_instances=sum(f['eligible_target_ge4_zones'] for f in fs),
            corridor_B_instances=sum(f['corridor_B'] for f in fs),corridor_B_ge4_instances=sum(f['corridor_B_ge4'] for f in fs),
            corridor_B_reported_far_gt3=sum(f['corridor_B_reported_far_gt3'] for f in fs),
            corridor_B_clear_depth_interval=sum(f['corridor_B_clear_depth_interval'] for f in fs),
            corridor_B_frames=sum(f['corridor_B']>0 for f in fs),
            corridor_B_FN_ids=[f['id'] for f in fs if f['truth'] and not f['alert'] and f['corridor_B']],
            FN_ids=[f['id'] for f in fs if f['truth'] and not f['alert']],
            low_score_frames=sum(f['score_ratio']<1 for f in fs),
            foreground_to_background_transitions=sum(f['foreground_to_background_transitions'] for f in fs),
            retained_foreground_B_transitions=sum(f['retained_foreground_B_transitions'] for f in fs),
            strict_prior_near_B_transitions=sum(f['strict_prior_near_B_transitions'] for f in fs),
            B_transition_frame_ids=[f['id'] for f in fs if f['retained_foreground_B_transitions']],
            changed_target_zone_set_frames=sum(f['target_zone_set_changed'] is True for f in fs),
            category_counts=dict(sum((Counter(f['category_counts']) for f in fs),Counter())))
    return dict(all=group(frames),positive=group(positive),
        by_stratum={key:group([f for f in positive if f['stratum']==key]) for key in sorted({f['stratum'] for f in frames})},
        grid_crossing_positive=group([f for f in positive if f['target_zone_set_changed'] is True]),
        corridor_boundary_positive=group([f for f in positive if f['boundary']]))


def run():
    p=verify();assert not (OUT/'zone-records.jsonl').exists()
    start=time.perf_counter();frames=[];gap_records=[];miss_records=[];all_summary={};previous={}
    windows=read(GAP)['windows'];gap_targets={}
    for w in windows:
        z=w['frames'][0]['dominant']['zone']
        for f in w['frames']:
            gap_targets[(f['id'],z)]=dict(role='FLANK' if f['alert'] else 'GAP',clip_id=w['clip_id'])
    assert len(gap_targets)==13
    with (OUT/'zone-records.jsonl').open('x',encoding='utf-8') as stream:
        for cohort in ('core','thin'):
            cohort_frames=[];pairs=0
            for base,obs,lin,geo,case,label,pred,path,seed,native_hash in cohort_records(cohort):
                assert obs['id']==lin['id']==label['id']==pred['id']
                assert all(obs[k]==label[k] for k in ('clip_id','frame_in_clip','time_s'))
                assert obs['clip_id']==geo['clip_id']==case['clip_id'] or (cohort=='thin' and label['source_clip_id']==geo['clip_id']==case['clip_id'])
                assert sha(path)==native_hash==geo['native_sha256']
                assert sha(base/obs['path'])==obs['sha256']
                native=np.load(path,allow_pickle=False);assert native.shape==(360,640)
                with np.load(base/obs['path'],allow_pickle=False) as data:boxes,stored=data['boxes'],data['values']
                assert np.array_equal(boxes,boxes45())
                depth=sample_native(native);values,traces=simulate(depth,seed,boxes)
                assert np.array_equal(values,stored,equal_nan=True),'Original sensor replay differs'
                serialized=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in traces]
                assert serialized==lin['traces'],'Original lineage replay differs'
                score=score_frame(boxes,stored);cal=decide(score,p['threshold'])
                assert cal==pred['predictions']['calibrated']
                assert score['anchors']==pred['anchors'] and score['zone_scores']==pred['zone_scores']
                target,bg,corridor,projected=masks(native,case,geo)
                st,sb,sc=map(sample_native,(target,bg,corridor))
                zones=[audit_zone(z,box,native,target,bg,corridor,projected,depth,st,sb,sc,serialized[z]) for z,box in enumerate(boxes)]
                clipkey=(cohort,obs['clip_id']);prev=previous.get(clipkey)
                stats=dict(foreground_to_background_transitions=0,retained_foreground_B_transitions=0,strict_prior_near_B_transitions=0)
                occupied=[z['zone'] for z in zones if z['native']['target']]
                for z in zones:
                    z.update(cohort=cohort,id=obs['id'],clip_id=obs['clip_id'],frame_in_clip=obs['frame_in_clip'],
                        time_s=obs['time_s'],truth=label['truth'],boundary=label['boundary'],calibration_alert=cal['alert'],
                        frame_score_ratio=cal['score']/p['threshold'],native_sha256=native_hash,
                        source_id=case['clip_id']+'/target',target_actor_path=next(o['actor_path'] for o in geo['objects'] if o['name']=='target'))
                    z['transition']=None
                    if prev:
                        a=prev['zones'][z['zone']];pairs+=1
                        assert a['frame_in_clip']+1==obs['frame_in_clip']
                        aw,bw=a['winner'],z['winner']
                        flip=bool(aw and aw['target_count'] and bw and not bw['target_count'] and bw['min_m']>aw['max_m'])
                        strict=bool(flip and a['reported']['distance_m']<=3)
                        is_b=bool(flip and z['category']=='B_RETURN_WINNER_FLIP_CAPABLE')
                        z['transition']=dict(previous_frame=a['id'],previous_target_pixels=a['native']['target'],
                            foreground_to_background=flip,prior_reported_near_le3=strict,
                            classification='A_GEOMETRY_TRANSPORT' if flip and z['category']=='A_ABSENT' else
                                'B_RETURN_WINNER_FLIP' if is_b else 'C_OR_OTHER' if flip else 'NO_FLIP')
                        stats['foreground_to_background_transitions']+=flip
                        stats['retained_foreground_B_transitions']+=is_b
                        stats['strict_prior_near_B_transitions']+=bool(is_b and strict)
                    stream.write(json.dumps(z,allow_nan=False)+'\n')
                    if cohort=='core' and (obs['id'],z['zone']) in gap_targets:
                        gap_records.append({**z,**gap_targets[(obs['id'],z['zone'])]})
                    if label['truth'] and not cal['alert'] and (z['native']['target'] or z['sampled']['target']):
                        miss_records.append(z)
                cz=[z for z in zones if z['sampled']['target_corridor']]
                bz=[z for z in cz if z['category']=='B_RETURN_WINNER_FLIP_CAPABLE']
                fr=dict(cohort=cohort,id=obs['id'],clip_id=obs['clip_id'],frame_in_clip=obs['frame_in_clip'],
                    truth=label['truth'],boundary=label['boundary'],alert=cal['alert'],score_ratio=cal['score']/p['threshold'],
                    stratum=case['layer'] if cohort=='core' else case['clip_id'],
                    native_corridor_zones=sum(z['native']['target_corridor']>0 for z in zones),sampled_corridor_zones=len(cz),
                    native_target_zones=sum(z['native']['target']>0 for z in zones),
                    sampled_target_zones=sum(z['sampled']['target']>0 for z in zones),
                    eligible_target_zones=sum(z['sampled']['target_eligible']>0 for z in zones),
                    eligible_target_ge4_zones=sum(z['sampled']['target_eligible']>=4 for z in zones),
                    corridor_B=len(bz),corridor_B_ge4=sum(z['sampled']['target_eligible']>=4 for z in bz),
                    corridor_B_reported_far_gt3=sum(z['reported']['distance_m']>3 for z in bz),
                    corridor_B_clear_depth_interval=sum(z['reported']['distance_m']-(.13+.06*z['reported']['distance_m'])>3 for z in bz),
                    category_counts=dict(Counter(z['category'] for z in zones)),
                    target_zone_set_changed=occupied!=prev['occupied'] if prev else None,**stats)
                frames.append(fr);cohort_frames.append(fr);previous[clipkey]=dict(zones=zones,occupied=occupied)
            assert len(cohort_frames)==p['cohort_frames'][cohort]
            assert pairs==(25344 if cohort=='core' else 5632)
            all_summary[cohort]=dict(summary=summarize(cohort_frames),zone_frames=len(cohort_frames)*64,adjacent_pairs=pairs)
            print('COHORT_COMPLETE',cohort,len(cohort_frames),flush=True)
    assert len(gap_records)==13 and sum(g['role']=='GAP' for g in gap_records)==5
    write(OUT/'frames.json',frames);write(OUT/'gap-outgoing-zones.json',gap_records);write(OUT/'fn-zone-records.json',miss_records)
    result=dict(status='COMPLETE_DIAGNOSTIC_ONLY',cohorts=all_summary,
        gap_categories=dict(Counter(r['category'] for r in gap_records if r['role']=='GAP')),
        gap_transitions=dict(Counter(r['transition']['classification'] for r in gap_records if r['role']=='GAP')),
        elapsed_s=time.perf_counter()-start,original_vectors_and_lineages_replayed=528,original_predictions_unchanged=528,
        new_algorithms=0,new_observations=0,python=sys.executable,protocol_sha256=sha(OUT/'protocol.json'))
    verify();write(OUT/'results.json',result)
    write(OUT/'result-seal.json',dict(status='COMPLETE',hashes={n:sha(OUT/n) for n in
        ('protocol.json','zone-records.jsonl','frames.json','gap-outgoing-zones.json','fn-zone-records.json','results.json')}))
    print(json.dumps(dict(status=result['status'],gap_categories=result['gap_categories'],elapsed_s=result['elapsed_s'],
        positives={n:x['summary']['positive'] for n,x in all_summary.items()})))


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=('freeze','run'));args=ap.parse_args()
    assert OUT.resolve().is_relative_to((ROOT/'artifacts.local').resolve());globals()[args.phase]()
