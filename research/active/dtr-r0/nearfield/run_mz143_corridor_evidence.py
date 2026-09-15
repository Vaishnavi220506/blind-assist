"""One TRAIN-selected corridor evidence comparison, then sealed consumed dev."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import time

import cv2
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from mz136_incumbent import public_observations
from evaluate_mz136_corridor_pair import score, retention, pair_metrics
from run_mz107_four_sensor import ROOT, sha, write, truth
from run_mz139_surface_fit import selected_jsonl, local_dependencies
from research_backend import BackendCandidate, DeviceObservation, select_backend

CODE = Path(__file__).resolve().parent
WORK = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914'
CAP = WORK/'source/returned-v1/capture-v1'
INC = WORK/'incumbent/fresh-v1'
SEED = 143016
ARMS = ('sensor_et', 'geometry_et', 'fused_et', 'fused_hgb')
PREFERENCE = ('geometry_et', 'sensor_et', 'fused_et', 'fused_hgb')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def classifier(arm):
    if arm.endswith('_et'):
        return ExtraTreesClassifier(n_estimators=384, min_samples_leaf=2,
            max_features=.5, class_weight='balanced', random_state=SEED, n_jobs=4)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
        max_leaf_nodes=7, min_samples_leaf=8, l2_regularization=1.,
        early_stopping=False, random_state=SEED)


def operating_threshold(scores, target, baseline):
    """Calibrate on OOF TRAIN values, retaining each incumbent true frame."""
    scores=np.asarray(scores,float);target=np.asarray(target,bool);baseline=np.asarray(baseline,bool)
    if scores.shape!=target.shape or target.shape!=baseline.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid matched OOF arrays')
    if not np.any(target&baseline):raise ValueError('No baseline true-positive calibration support')
    return float(scores[target&baseline].min())


def matrices(features):
    sensor=np.stack([f['sensor'] for f in features])
    geometry=np.stack([f['geometry'] for f in features])
    fused=np.concatenate([sensor,geometry],axis=1)
    assert np.isfinite(fused).all()
    return dict(sensor_et=sensor,geometry_et=geometry,fused_et=fused,fused_hgb=fused)


def extract_rows(rows, receipt, out, label):
    from mz143_corridor_features import extract
    values=[]; yaw=0.; previous=None; start=time.perf_counter(); hashes={}
    for index,row in enumerate(rows):
        if row['episode_id']!=previous: yaw=0.
        if row['imu_valid']: yaw+=row['delta_yaw']
        previous=row['episode_id']
        path=(CAP/row['rgb_path']).resolve()
        assert path.is_relative_to(CAP.resolve())
        hashes[row['rgb_path']]=sha(path)
        assert hashes[row['rgb_path']]==receipt['hashes'][row['rgb_path']]
        image=cv2.imread(str(path)); assert image is not None
        value=extract(row,image,yaw)
        assert all(np.isfinite(value[key]).all() for key in ('sensor','geometry'))
        if values:
            assert value['sensor_names']==values[0]['sensor_names']
            assert value['geometry_names']==values[0]['geometry_names']
        values.append(value)
        if (index+1)%24==0:
            print(json.dumps(dict(stage='features',split=label,frames=index+1,
                total=len(rows),seconds=time.perf_counter()-start)),flush=True)
    np.savez_compressed(out/(label+'-features.npz'),
        sensor=np.stack([v['sensor'] for v in values]),
        geometry=np.stack([v['geometry'] for v in values]))
    write(out/(label+'-feature-audit.json'),[dict(id=r['id'],**v['audit']) for r,v in zip(rows,values)])
    write(out/(label+'-feature-seal.json'),dict(
        feature_sha256=sha(out/(label+'-features.npz')),
        audit_sha256=sha(out/(label+'-feature-audit.json')),
        rgb_sha256=hashes,seconds=time.perf_counter()-start,
        authority='PUBLIC_SENSOR_AND_RGB_FEATURES_BEFORE_SELECTED_LABEL_PARSE'))
    return values,time.perf_counter()-start


def pairs_for(rows,spec,split):
    episodes={}
    for i,row in enumerate(rows):episodes.setdefault(row['episode_id'],[]).append(i)
    pairs=[]
    for pair in spec['pairs']:
        if pair['split']!=split:continue
        aa,bb=[episodes[episode] for episode in pair['episodes']]
        assert len(aa)==len(bb)
        pairs.extend(dict(a=a,b=b) for a,b in zip(aa,bb))
    return pairs


def augmented_score(rows,es,gt,values,threshold,baseline,spec,split):
    flags=values>=threshold
    report=score(rows,es,gt,flags,list(range(len(rows))),baseline)
    report['pairs']=pair_metrics(gt,values,flags,pairs_for(rows,spec,split))
    m=report['metrics']
    report['precision']=m['TP']/max(1,m['TP']+m['FP'])
    report['recall']=m['TP']/max(1,m['TP']+m['FN'])
    report['strata']={}
    for name,pressure in [('ordinary',False),('boundary_pressure',True)]:
        ix=[i for i,e in enumerate(es) if (e['family']=='shallow_boundary_stress')==pressure]
        report['strata'][name]=score(rows,es,gt,flags,ix,baseline)
    return report


def native_account(rows,es,flags,baseline):
    totals=Counter(); misses=[]; case=[]
    for row,e,p,b in zip(rows,es,flags,baseline):
        counts=Counter(); origin=np.array(e['body_origin_m'])
        zones={z['zone_id']:z for z in row['tof_zones']}
        for native in e['zonal_tof_native']:
            zone=zones[native['zone_id']]
            rays={r['subray']:r for r in native['private_rays']}
            for lineage in native['returned_lineage']:
                if lineage['target_index']>=len(zone['targets']):continue
                for index in lineage['hit_indices']:
                    ray=rays[index]
                    point=np.array(ray['hit_point_m'])-origin
                    counts['returned_contributor_samples']+=1
                    inside=(.2<=point[0]<=3.6 and -.3<=point[1]<=.3 and .4<=point[2]<=2.05)
                    counts['corridor_contributor_samples']+=int(inside)
        record=dict(id=row['id'],truth=bool(truth(e)),candidate=bool(p),baseline=bool(b),**counts)
        case.append(record);totals.update(counts)
        if not p and counts['corridor_contributor_samples']:
            misses.append(record)
    return dict(totals=dict(totals),nonalert_with_native_corridor_contributors=misses,
        cases=case,spatial_pruning='NONE: original packets and both slots remain in feature audit',
        caveat='Input retention does not certify containment or retention of an alert vote')


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);started=time.perf_counter()
    receipt=read(CAP/'receipt.json');spec=read(CAP/'spec.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for filename in ('raw.jsonl','evaluator.jsonl'):
        assert sha(CAP/filename)==receipt['hashes'][filename]
    seal=read(INC/'prediction-seal.json');done=read(INC/'completion.json')
    assert done['status']=='PASS' and sha(INC/'prediction-seal.json')==done['prediction_seal_sha256']
    assert sha(INC/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
    cache=read(INC/'nominal/predictions.json')['predictions']
    base={p['id']:bool(p['candidate']) for p in cache}
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    assert len(ids['train'])==192 and len(ids['dev'])==48
    rows={s:public_observations(selected_jsonl(CAP/'raw.jsonl',ids[s])) for s in ids}
    assert all(rows[s]==selected_jsonl(INC/'nominal/raw.jsonl',ids[s]) for s in rows)
    sources=local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'mz143_corridor_features.py'))
    for path in sources:
        source=Path(path);dest=out/'source-snapshot'/source.relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    shutil.copyfile(CODE/'MZ143_PROTOCOL_20260916.md',out/'protocol-before-fit.md')
    write(out/'freeze.json',dict(seed=SEED,arms=list(ARMS),sources=sources,
        protocol_sha256=sha(out/'protocol-before-fit.md'),
        inputs={str(p):sha(p) for p in [CAP/'raw.jsonl',CAP/'evaluator.jsonl',CAP/'spec.json',
            CAP/'receipt.json',INC/'nominal/predictions.json',INC/'nominal/raw.jsonl']},
        selected_frame_ids={s:[r['id'] for r in rows[s]] for s in rows},
        original_test_access=False,scope='CONSUMED_CONTROLLED_DEVELOPMENT'))
    from mz143_corridor_features import extract
    first=rows['train'][0];first_path=CAP/first['rgb_path']
    assert sha(first_path)==receipt['hashes'][first['rgb_path']]
    first_image=cv2.imread(str(first_path))
    select_backend('batch-tensor',cpu=BackendCandidate('sklearn-opencv-cpu','cpu',
        lambda:extract(first,first_image,first['delta_yaw'] if first['imu_valid'] else 0.),lambda _:DeviceObservation('cpu','host CPU',
            'scikit-learn '+sklearn.__version__+' OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented sklearn tree and OpenCV robust foreground operators are CPU-only'})
    train_features,train_seconds=extract_rows(rows['train'],receipt,out,'train')
    write(out/'feature-names.json',{k:train_features[0][k] for k in ('sensor_names','geometry_names')})
    x=matrices(train_features)
    # Only TRAIN labels are read here, after untrained features are sealed.
    es_train=selected_jsonl(CAP/'evaluator.jsonl',ids['train'])
    assert [e['id'] for e in es_train]==[r['id'] for r in rows['train']]
    y=np.array([truth(e) for e in es_train],bool)
    baseline=np.array([base[r['id']] for r in rows['train']],bool)
    metadata={f['id']:f for f in spec['frames'] if f['id'] in ids['train']}
    folds=np.array([int(metadata[r['id']]['scene_group'].rsplit('scene',1)[1]) for r in rows['train']])
    assert dict(Counter(folds))=={0:48,1:48,2:48,3:48}
    write(out/'train-folds.json',[dict(id=r['id'],heldout_fold=int(f)) for r,f in zip(rows['train'],folds)])
    oof={};thresholds={};models={};oof_reports={};fit_reports={};fit_times={}
    for arm in ARMS:
        tick=time.perf_counter();scores=np.empty(len(y),float)
        for fold in range(4):
            model=classifier(arm);model.fit(x[arm][folds!=fold],y[folds!=fold])
            scores[folds==fold]=model.predict_proba(x[arm][folds==fold])[:,1]
            assert time.perf_counter()-started<1200,'20 minute run budget exhausted'
        threshold=operating_threshold(scores,y,baseline)
        assert np.all(scores[y&baseline]>=threshold)
        report=augmented_score(rows['train'],es_train,y,scores,threshold,baseline,spec,'train')
        oof[arm]=scores;thresholds[arm]=threshold;oof_reports[arm]=report
        model=classifier(arm);model.fit(x[arm],y);models[arm]=model
        fitted=model.predict_proba(x[arm])[:,1]
        fit_reports[arm]=augmented_score(rows['train'],es_train,y,fitted,threshold,baseline,spec,'train')
        fit_times[arm]=time.perf_counter()-tick
        with (out/(arm+'.pkl')).open('wb') as stream:pickle.dump(model,stream)
        print(json.dumps(dict(stage='fit',arm=arm,threshold=threshold,
            oof=report['metrics'],fit=fit_reports[arm]['metrics'],seconds=fit_times[arm])),flush=True)
    def selection_key(arm):
        r=oof_reports[arm]
        return (r['metrics']['FP'],r['events'].get('false_segments',r['events'].get('false_alert_segments',0)),PREFERENCE.index(arm))
    chosen=min(ARMS,key=selection_key)
    np.savez_compressed(out/'oof-scores.npz',**oof)
    write(out/'train-summary.json',dict(oof=oof_reports,fit=fit_reports,seconds=fit_times))
    write(out/'model-seal.json',dict(selected=chosen,thresholds=thresholds,
        models={a:sha(out/(a+'.pkl')) for a in ARMS},
        oof_sha256=sha(out/'oof-scores.npz'),train_summary_sha256=sha(out/'train-summary.json'),
        selection='OOF_FP_THEN_FALSE_SEGMENTS_THEN_PREDECLARED_SIMPLICITY',
        authority='MODELS_AND_THRESHOLDS_SEALED_BEFORE_DEV_FEATURES_PREDICTIONS_AND_LABEL_PARSE'))
    print(json.dumps(dict(stage='model_sealed',selected=chosen,thresholds=thresholds)),flush=True)
    dev_features,dev_seconds=extract_rows(rows['dev'],receipt,out,'dev')
    xd=matrices(dev_features);dev_scores={};inference={}
    for arm in ARMS:
        tick=time.perf_counter();dev_scores[arm]=models[arm].predict_proba(xd[arm])[:,1]
        inference[arm]=time.perf_counter()-tick
    payload=[dict(id=row['id'],scores={a:float(dev_scores[a][i]) for a in ARMS},
        flags={a:bool(dev_scores[a][i]>=thresholds[a]) for a in ARMS}) for i,row in enumerate(rows['dev'])]
    write(out/'dev-predictions.json',payload)
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'dev-predictions.json'),
        model_seal_sha256=sha(out/'model-seal.json'),feature_seal_sha256=sha(out/'dev-feature-seal.json'),
        inference_seconds=inference,authority='DEV_PREDICTIONS_SEALED_BEFORE_DEV_TRUTH_PARSE'))
    es_dev=selected_jsonl(CAP/'evaluator.jsonl',ids['dev'])
    assert [e['id'] for e in es_dev]==[r['id'] for r in rows['dev']]
    yd=np.array([truth(e) for e in es_dev],bool)
    bd=np.array([base[r['id']] for r in rows['dev']],bool)
    br=augmented_score(rows['dev'],es_dev,yd,bd.astype(float),.5,bd,spec,'dev')
    assert br['metrics']==dict(TP=24,FP=13,FN=0,TN=11,UNKNOWN=11)
    dev_reports={}
    for arm in ARMS:
        rr=augmented_score(rows['dev'],es_dev,yd,dev_scores[arm],thresholds[arm],bd,spec,'dev')
        rr['retention']=retention(rr,br)
        rr['family_fp_noninferior']=all(rr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
        rr['joint_gain']=bool(not rr['lost_baseline_tp'] and rr['retention']['incumbent_events_and_timing_retained']
            and rr['metrics']['FP']<=10 and rr['family_fp_noninferior'])
        dev_reports[arm]=rr
    accounting=native_account(rows['dev'],es_dev,dev_scores[chosen]>=thresholds[chosen],bd)
    write(out/'native-contributors.json',accounting)
    result=dict(selected=chosen,selected_gain=dev_reports[chosen]['joint_gain'],
        baseline=br,candidates=dev_reports,thresholds=thresholds,
        feature_dimensions={a:int(xx.shape[1]) for a,xx in x.items()},
        seconds=dict(train_features=train_seconds,dev_features=dev_seconds,fit=fit_times,
            dev_inference=inference,total=time.perf_counter()-started),
        authority='CONSUMED_CONTROLLED_DEVELOPMENT_NOT_FRESH_CONFIRMATION',
        original_test_access=False,promotion='MZ129 remains baseline pending separate unchanged fresh confirmation',
        decision='TRAIN_SELECTED_DEV_GAIN' if dev_reports[chosen]['joint_gain'] else 'NO_TRAIN_SELECTED_JOINT_DEV_GAIN')
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seal_sha256=sha(out/'prediction-seal.json'),
        resource_state='CPU process exits; no persistent workers or GPU allocations'))
    print(json.dumps(dict(selected=chosen,decision=result['decision'],
        baseline=br['metrics'],candidates={a:r['metrics'] for a,r in dev_reports.items()},
        seconds=result['seconds']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.output)
