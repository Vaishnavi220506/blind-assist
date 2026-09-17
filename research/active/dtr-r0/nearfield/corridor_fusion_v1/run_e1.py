"""Run the bounded E1 paired Development experiment; existing caches authenticate.

Usage: research Python run_e1.py --work artifacts.local/work/corridor-depth-e1-20260917
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
NEAR=HERE.parent
sys.path[:0]=[str(NEAR),str(ROOT/'tools')]
import cv2
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from threadpoolctl import threadpool_limits
from depth_features import extract as depth_extract
from mz136_incumbent import public_observations
from run_mz139_surface_fit import selected_jsonl
from run_mz143_corridor_evidence import augmented_score, native_account, truth, pairs_for
from evaluate_mz136_corridor_pair import pair_metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

OLD=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
REF=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916'
CAP=REF/'source/returned-v1/capture-v1'
BASE=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
CONFIGS=[dict(max_leaf_nodes=7,max_iter=150,l2_regularization=1.),
         dict(max_leaf_nodes=7,max_iter=300,l2_regularization=1.),
         dict(max_leaf_nodes=15,max_iter=150,l2_regularization=3.)]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dataset(cap,split):
    spec=read(cap/'spec.json');receipt=read(cap/'receipt.json')
    assert receipt['status']=='PASS'
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    ids={f['id'] for f in spec['frames'] if f['split']==split}
    for name in ('raw.jsonl','evaluator.jsonl'):
        assert sha(cap/name)==receipt['hashes'][name]
    rows=public_observations(selected_jsonl(cap/'raw.jsonl',ids))
    assert len(rows)==len(ids)
    return dict(rows=rows,spec=spec,receipt=receipt,cap=cap,split=split,ids=ids)

def labels(data):
    es=selected_jsonl(data['cap']/'evaluator.jsonl',data['ids'])
    assert [e['id'] for e in es]==[r['id'] for r in data['rows']]
    return es,np.array([truth(e) for e in es],bool)

def select_threshold(y,s,high_recall=False):
    candidates=np.r_[np.nextafter(s.max(),np.inf),np.unique(s)]
    curve=[]
    for t in candidates:
        p=s>=t;tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
        curve.append(dict(threshold=float(t),TP=tp,FP=fp,FN=fn,
                          f1=2*tp/max(1,2*tp+fp+fn),recall=tp/max(1,tp+fn)))
    if high_recall:
        eligible=[r for r in curve if r['recall']>=.95]
        chosen=min(eligible,key=lambda r:(r['FP'],-r['TP'],-r['threshold']))
    else:
        chosen=max(curve,key=lambda r:(r['f1'],-r['FP'],r['threshold']))
    return chosen,curve

def quality(y,s):
    p,r,t=precision_recall_curve(y,s)
    return dict(pr_auc=float(auc(r,p)),average_precision=float(average_precision_score(y,s)),
                pr_curve=dict(precision=p.tolist(),recall=r.tolist(),thresholds=t.tolist()))

def report(data,es,y,s,flags,baseline,ranking=True):
    r=augmented_score(data['rows'],es,y,flags.astype(float),.5,baseline,data['spec'],data['split'])
    m=r['metrics'];r['f1']=2*m['TP']/max(1,2*m['TP']+m['FP']+m['FN'])
    r.update(quality(y,s) if ranking else dict(pr_auc=None,average_precision=None,
        pr_curve=None,ranking_status='NOT_EVALUABLE_BINARY_REFERENCE'))
    r['score_pairs']=pair_metrics(y,s,flags,pairs_for(data['rows'],data['spec'],data['split']))
    r['negative_time_alert_fraction']=m['FP']/max(1,int((~y).sum()))
    # Fixed 4Hz simulation bins. End-of-episode events are right censored.
    releases=[];rows=data['rows']
    for i in range(1,len(rows)):
        if rows[i]['episode_id']!=rows[i-1]['episode_id'] or not y[i-1] or y[i]:continue
        j=i
        while j<len(rows) and rows[j]['episode_id']==rows[i]['episode_id'] and not y[j] and flags[j]:j+=1
        released=j<len(rows) and rows[j]['episode_id']==rows[i]['episode_id'] and not y[j]
        releases.append(dict(episode=rows[i]['episode_id'],offset_s=rows[i]['time_s'],
            delay_s=float(rows[j]['time_s']-rows[i]['time_s']) if released else None,right_censored=not released))
    r['release_details']=releases
    for v in [r,*r['strata'].values()]:
        m=v['metrics'];v['f1']=2*m['TP']/max(1,2*m['TP']+m['FP']+m['FN'])
    return r

def main(work,run_name):
    start=time.perf_counter();out=work/run_name;out.mkdir(exist_ok=True)
    assert out.resolve().parent==work.resolve()
    assert work.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    assert not (out/'completion.json').exists(),'Completed runs are immutable; choose a new work directory'
    torch.set_num_threads(4);cv2.setNumThreads(4)
    assert torch.cuda.is_available(),'GPU required for this implementation'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(work/'upstream'))
    from depth_anything_v2.dpt import DepthAnythingV2
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(work/'depth_anything_v2_vits.pth',map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False).cuda()
    datasets={k:dataset(OLD if k!='report' else CAP, k if k!='report' else 'confirmation') for k in ('train','dev','report')}
    assert [len(datasets[k]['rows']) for k in datasets]==[192,48,288]
    groups={k:{f['scene_group'] for f in d['spec']['frames'] if f['id'] in d['ids']} for k,d in datasets.items()}
    assert not groups['train']&groups['dev'] and not groups['train']&groups['report']
    bindings={str(p):sha(p) for p in [Path(__file__),HERE/'depth_features.py',HERE/'E1_PROTOCOL_20260917.md',
            work/'depth_anything_v2_vits.pth',BASE/'train-features.npz',BASE/'dev-features.npz',REF/'learned-v1/features.npz']}
    for cap in (OLD,CAP):
        for name in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json'):bindings[str(cap/name)]=sha(cap/name)
    freeze=dict(bindings=bindings,configs=CONFIGS,seed=177017,scope='CONSUMED_CONTROLLED_DEVELOPMENT',
        ids={k:[r['id'] for r in d['rows']] for k,d in datasets.items()},groups={k:sorted(v) for k,v in groups.items()},
        original_test_access=False,upstream_commit=subprocess.check_output(['git','-C',str(work/'upstream'),'rev-parse','HEAD'],text=True).strip())
    if (out/'freeze.json').exists():assert read(out/'freeze.json')==freeze,'Resume identities changed'
    else:write(out/'freeze.json',freeze)
    x={}
    for k in ('train','dev'):
        seal=read(BASE/(k+'-feature-seal.json'));assert sha(BASE/(k+'-features.npz'))==seal['feature_sha256']
        assert freeze['ids'][k]==read(BASE/'freeze.json')['selected_frame_ids'][k]
        z=np.load(BASE/(k+'-features.npz'));x[k]=np.c_[z['sensor'],z['geometry']]
    seal=read(REF/'learned-v1/prediction-seal.json')
    assert sha(REF/'learned-v1/features.npz')==seal['context']['features_sha256']
    x['report']=np.load(REF/'learned-v1/features.npz')['views'][:,0]
    assert freeze['ids']['report']==[r['id'] for r in read(REF/'learned-v1/predictions.json')]
    first=datasets['train']['rows'][0];image=cv2.imread(str(OLD/first['rgb_path']))
    tensor,hw=model.image2tensor(image,518)
    @torch.inference_mode()
    def probe(device):
        model.to(device);return model(tensor.to(device))
    select_backend('model-inference',
        cpu=BackendCandidate('dav2-cpu','cpu',lambda:probe('cpu'),lambda v:DeviceObservation(v.device.type,'host CPU',torch.__version__)),
        gpu=BackendCandidate('dav2-cuda','cuda',lambda:probe('cuda'),lambda v:DeviceObservation(v.device.type,torch.cuda.get_device_name(0),torch.__version__),torch.cuda.synchronize),
        warmups=0,repeats=1,record_path=out/'backend.json')
    model.cuda();torch.cuda.reset_peak_memory_stats()
    write(out/'model.json',dict(parameters=sum(p.numel() for p in model.parameters()),input_shape=list(tensor.shape),
          dtype='float32',source_rgb_shape=list(image.shape),output='relative_not_metres',device=torch.cuda.get_device_name(0)))
    dep={};timings={};preview_ids=[]
    for k,data in datasets.items():
        values=[];times=[];yaw=0.;previous=None;cache=out/(k+'-cache');cache.mkdir(exist_ok=True)
        for i,row in enumerate(data['rows']):
            assert time.perf_counter()-start<3600,'3600-second E1 budget exceeded'
            if row['episode_id']!=previous:yaw=0.
            if row['imu_valid']:yaw+=row['delta_yaw']
            previous=row['episode_id'];path=data['cap']/row['rgb_path']
            assert path.resolve().is_relative_to(data['cap'].resolve())
            assert sha(path)==data['receipt']['hashes'][row['rgb_path']]
            cached=cache/(row['id']+'.npz')
            if cached.exists():
                z=np.load(cached);value=z['features'];timing=z['timing'].tolist()
            else:
                tick=time.perf_counter();im=cv2.imread(str(path));decode=time.perf_counter()-tick
                tick=time.perf_counter()
                with torch.inference_mode():depth=model.infer_image(im,518)
                torch.cuda.synchronize();inference=time.perf_counter()-tick
                tick=time.perf_counter();value,names=depth_extract(row,im,yaw,depth);stats=time.perf_counter()-tick
                timing=[decode,inference,stats]
                np.savez_compressed(cached,features=value,timing=timing)
                if k=='report':
                    # First complete episode per family: deterministic before outputs.
                    family=next(f['family'] for f in data['spec']['frames'] if f['id']==row['id'])
                    if not any(v['family']==family for v in preview_ids) or any(v['episode']==row['episode_id'] for v in preview_ids):
                        np.savez_compressed(out/(row['id']+'-preview.npz'),depth=depth)
                        preview_ids.append(dict(id=row['id'],family=family,episode=row['episode_id']))
                if not (out/'depth-feature-names.json').exists():write(out/'depth-feature-names.json',names)
            values.append(value);times.append(timing)
            if (i+1)%24==0:
                progress=dict(stage='depth_features',split=k,completed=i+1,total=len(data['rows']),seconds=time.perf_counter()-start)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        dep[k]=np.stack(values);timings[k]=np.array(times)
        np.savez_compressed(out/(k+'-features.npz'),base=x[k],depth=dep[k],ids=freeze['ids'][k],timing=timings[k])
    write(out/'feature-seal.json',{k:sha(out/(k+'-features.npz')) for k in datasets})
    es_train,ytrain=labels(datasets['train']);es_dev,ydev=labels(datasets['dev'])
    models={};selection={};scores={};fit_records=[]
    for arm in ('A','B'):
        xx={k:x[k] if arm=='A' else np.c_[x[k],dep[k]] for k in datasets}
        candidates=[]
        for idx,config in enumerate(CONFIGS):
            tick=time.perf_counter()
            fitted=HistGradientBoostingClassifier(**config,learning_rate=.05,min_samples_leaf=8,early_stopping=False,random_state=177017)
            fitted.fit(xx['train'],ytrain);ds=fitted.predict_proba(xx['dev'])[:,1]
            best,curve=select_threshold(ydev,ds);hi,_=select_threshold(ydev,ds,True)
            ap=float(average_precision_score(ydev,ds));fit=fitted.predict_proba(xx['train'])[:,1]>=best['threshold']
            rec=dict(arm=arm,config_index=idx,config=config,dev_best=best,dev_recall95=hi,dev_ap=ap,
                     train_accuracy=float((fit==ytrain).mean()),seconds=time.perf_counter()-tick)
            fit_records.append(rec);candidates.append((best['f1'],ap,-idx,fitted,ds,rec))
            print(json.dumps(rec),flush=True)
        chosen=max(candidates,key=lambda c:c[:3]);models[arm]=chosen[3];selection[arm]=chosen[-1]
        scores[arm]={'dev':chosen[4]}
        (out/(arm+'-model.pkl')).write_bytes(pickle.dumps(models[arm]))
    write(out/'development-search.json',fit_records)
    write(out/'selection-seal.json',dict(selection=selection,models={a:sha(out/(a+'-model.pkl')) for a in models},
          feature_seal_sha256=sha(out/'feature-seal.json'),report_labels_used_for_selection=False))
    for a in models:
        xx=x['report'] if a=='A' else np.c_[x['report'],dep['report']]
        scores[a]['report']=models[a].predict_proba(xx)[:,1]
    np.savez_compressed(out/'report-scores.npz',**{a:scores[a]['report'] for a in scores})
    write(out/'prediction-seal.json',dict(scores_sha256=sha(out/'report-scores.npz'),selection_sha256=sha(out/'selection-seal.json')))
    finish(work,out,start,datasets,x,dep,models,scores,model,bindings,freeze,selection)


def finish(work,out,start,datasets,x,dep,models,scores,model,bindings,freeze,selection):
    es,y=labels(datasets['report']);data=datasets['report']
    base_records=read(REF/'baseline-v1/predictions.json');old_records=read(REF/'learned-v1/predictions.json')
    for folder in ('baseline-v1','learned-v1'):
        assert sha(REF/folder/'predictions.json')==read(REF/folder/'prediction-seal.json')['predictions_sha256']
    baseline=np.array([r['candidate'] for r in base_records],bool)
    assert [r['id'] for r in base_records]==freeze['ids']['report']
    all_scores={'MZ129':baseline.astype(float),'MZ145':np.array([r['scores']['static'] for r in old_records]),
                'four_expert':np.array([r['scores']['mean'] for r in old_records])}
    flags={'MZ129':baseline,'MZ145':np.array([r['flags']['static'] for r in old_records],bool),
           'four_expert':np.array([r['flags']['mean'] for r in old_records],bool)}
    for a in models:
        for label,key in [('f1','dev_best'),('recall95','dev_recall95')]:
            name=a+'_'+label;all_scores[name]=scores[a]['report'];flags[name]=all_scores[name]>=selection[a][key]['threshold']
    reports={name:report(data,es,y,all_scores[name],p,baseline,ranking=name!='MZ129') for name,p in flags.items()}
    native={name:native_account(data['rows'],es,p,baseline) for name,p in flags.items()}
    write(out/'native.json',native)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['id','episode','time_s','family','truth']+[a+suffix for a in flags for suffix in ('_score','_alert')]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,(row,e) in enumerate(zip(data['rows'],es)):
            item=dict(id=row['id'],episode=row['episode_id'],time_s=row['time_s'],family=e['family'],truth=int(y[i]))
            for a in flags:item.update({a+'_score':float(all_scores[a][i]),a+'_alert':int(flags[a][i])})
            writer.writerow(item)
    # Real online replay timing: all stages, cache bypass, one full first episode/family.
    from mz143_corridor_features import extract as base_extract
    timing=[];selected_episodes={}
    for row,e in zip(data['rows'],es):selected_episodes.setdefault(e['family'],row['episode_id'])
    yaw=0.;previous=None;video=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(1280,480))
    assert video.isOpened()
    try:
        for i,(row,e) in enumerate(zip(data['rows'],es)):
            if row['episode_id']!=previous:yaw=0.
            if row['imu_valid']:yaw+=row['delta_yaw']
            previous=row['episode_id']
            if row['episode_id']!=selected_episodes[e['family']]:continue
            torch.cuda.synchronize();tick=time.perf_counter();im=cv2.imread(str(CAP/row['rgb_path']))
            feat=base_extract(row,im,yaw);bv=np.r_[feat['sensor'],feat['geometry']]
            assert np.array_equal(bv,x['report'][i]),'Base cache/live mismatch'
            models['A'].predict_proba(bv[None]);a_sec=time.perf_counter()-tick
            tick_depth=time.perf_counter()
            with torch.inference_mode():depth=model.infer_image(im,518)
            dv,_=depth_extract(row,im,yaw,depth)
            assert np.allclose(dv,dep['report'][i],rtol=1e-5,atol=1e-5),'Depth replay changed'
            models['B'].predict_proba(np.r_[bv,dv][None]);torch.cuda.synchronize()
            b_extra=time.perf_counter()-tick_depth
            timing.append(dict(id=row['id'],A_seconds=a_sec,B_conservative_seconds=a_sec+b_extra))
            d=((depth-depth.min())/max(float(np.ptp(depth)),1e-6)*255).astype(np.uint8)
            top=np.concatenate([im,cv2.applyColorMap(d,cv2.COLORMAP_TURBO)],axis=1)
            canvas=np.zeros((480,1280,3),np.uint8);canvas[:360]=top
            lines=[f"{e['family']}  t={row['time_s']:.2f}s  TRUTH={int(y[i])}  Relative depth (not metres)",
                   ' | '.join(f'{a}: {int(flags[a][i])}' for a in ('MZ129','MZ145','four_expert','A_f1','B_f1')),
                   'Consumed controlled Development; first episode per family, chosen without outcomes']
            for j,line in enumerate(lines):cv2.putText(canvas,line,(12,389+30*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            video.write(canvas)
            if len(timing)==1:cv2.imwrite(str(out/'preview.png'),canvas)
    finally:video.release()
    write(out/'latency-samples.json',timing)
    latency={a:{'p50_ms':float(np.quantile([t[key] for t in timing],.5)*1000),
                'p95_ms':float(np.quantile([t[key] for t in timing],.95)*1000)}
                for a,key in [('A','A_seconds'),('B','B_conservative_seconds')]}
    result=dict(reports=reports,selection=selection,latency=latency,latency_frames=len(timing),
        latency_note='Warm single-frame decode+base+head; B adds depth+stats+head and conservatively includes A head too. First episode per family. No capture/transport/event wait.',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start,
        scope='CONSUMED_CONTROLLED_DEVELOPMENT',original_test_access=False,radar_native_lineage='NOT_EVALUABLE',
        decision='REVIEW_E1_INFORMATION_INCREMENT_NO_AUTOMATIC_E2')
    write(out/'summary.json',result)
    assert all(sha(p)==h for p,h in bindings.items()),'Input/source changed during execution'
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        predictions_sha256=sha(out/'predictions.csv'),video_sha256=sha(out/'comparison.mp4'),resources='process-local; exits on return'))
    print(json.dumps(dict(result={k:{'metrics':v['metrics'],'f1':v['f1'],'pr_auc':v['pr_auc']} for k,v in reports.items()},latency=latency)),flush=True)


def recover_delivery(work,run_name):
    """Resume only post-prediction delivery; never refit or select thresholds."""
    start=time.perf_counter();out=work/run_name
    assert work.is_relative_to((ROOT/'artifacts.local').resolve()) and out.resolve().parent==work
    assert not (out/'completion.json').exists()
    freeze=read(out/'freeze.json');bindings={}
    for path,h in freeze['bindings'].items():
        p=Path(path)
        # The frozen original program remains available; only delivery code changed.
        checked=out/'source-snapshot'/p.name if p.parent==HERE else p
        assert sha(checked)==h,str(checked)
        bindings[str(checked)]=h
    selected=read(out/'selection-seal.json');seal=read(out/'prediction-seal.json')
    assert sha(out/'selection-seal.json')==seal['selection_sha256']
    assert sha(out/'report-scores.npz')==seal['scores_sha256']
    models={}
    for a,h in selected['models'].items():
        assert sha(out/(a+'-model.pkl'))==h
        models[a]=pickle.loads((out/(a+'-model.pkl')).read_bytes())
    fseal=read(out/'feature-seal.json');assert sha(out/'report-features.npz')==fseal['report']
    feature=np.load(out/'report-features.npz');data=dataset(CAP,'confirmation')
    assert list(feature['ids'])==freeze['ids']['report']==[r['id'] for r in data['rows']]
    scored=np.load(out/'report-scores.npz');scores={a:{'report':scored[a]} for a in models}
    before={str(p):sha(p) for p in [out/'selection-seal.json',out/'prediction-seal.json',out/'report-scores.npz',out/'predictions.csv']}
    write(out/'delivery-recovery.json',dict(reason='Video image concatenation axis; binary reference ranking presentation',
        no_retraining=True,no_threshold_selection=True,sealed=before,delivery_code_sha256=sha(__file__),
        model_inference='24-frame latency/replay verification only; predictions remain sealed'))
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(work/'upstream'))
    from depth_anything_v2.dpt import DepthAnythingV2
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(work/'depth_anything_v2_vits.pth',map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False).cuda();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():model.infer_image(cv2.imread(str(CAP/data['rows'][0]['rgb_path'])),518)
    finish(work,out,start,{'report':data},{'report':feature['base']},{'report':feature['depth']},
           models,scores,model,bindings,freeze,selected['selection'])
    assert all(sha(p)==h for p,h in before.items()),'Sealed predictions changed during delivery'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--run-name',default='run-v1');p.add_argument('--finish-existing',action='store_true');args=p.parse_args()
    try:
        with threadpool_limits(4):
            (recover_delivery if args.finish_existing else main)(args.work.resolve(),args.run_name)
    except Exception:
        args.work.mkdir(exist_ok=True,parents=True)
        write(args.work/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise
