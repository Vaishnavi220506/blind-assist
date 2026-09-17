"""One positioned-token E1 diagnostic, preserving the completed scalar E1."""
import argparse
import csv
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from run_e1 import (ROOT,HERE,OLD,CAP,REF,CONFIGS,read,write,sha,dataset,labels,
                    select_threshold,report,native_account,BackendCandidate,
                    DeviceObservation,select_backend)
from intermediate_features import extract,transform

PARENT=ROOT/'artifacts.local/work/corridor-depth-e1-20260917'
E1=PARENT/'run-v2'


def load_model():
    sys.path.insert(0,str(PARENT/'upstream'))
    from depth_anything_v2.dpt import DepthAnythingV2
    m=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    m.load_state_dict(torch.load(PARENT/'depth_anything_v2_vits.pth',map_location='cpu',weights_only=True),strict=True)
    return m.eval().requires_grad_(False)


def run(out):
    start=time.perf_counter();out=out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(exist_ok=True);assert not (out/'completion.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=read(E1/'completion.json')
    assert done['status']=='PASS' and sha(E1/'summary.json')==done['summary_sha256']
    assert read(E1/'delivery-audit.json')['status']=='PASS'
    ds={k:dataset(OLD if k!='report' else CAP,k if k!='report' else 'confirmation') for k in ('train','dev','report')}
    frozen=read(E1/'freeze.json');oldseal=read(E1/'feature-seal.json')
    base={};depth={};inputs={}
    for k,d in ds.items():
        p=E1/(k+'-features.npz');assert sha(p)==oldseal[k]
        z=np.load(p);assert list(z['ids'])==[r['id'] for r in d['rows']]==frozen['ids'][k]
        base[k]=z['base'];depth[k]=z['depth'];inputs[str(p)]=sha(p)
        for name in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json'):
            p=d['cap']/name;inputs[str(p)]=sha(p)
    sources=[Path(__file__),HERE/'intermediate_features.py',HERE/'depth_features.py',HERE/'run_e1.py',HERE/'INTERMEDIATE_PROTOCOL_20260917.md']
    sources.extend((PARENT/'upstream/depth_anything_v2').rglob('*.py'))
    bindings={str(p):sha(p) for p in sources}
    bindings[str(PARENT/'depth_anything_v2_vits.pth')]=sha(PARENT/'depth_anything_v2_vits.pth')
    original_weight=[h for p,h in frozen['bindings'].items()
                     if Path(p).resolve()==(PARENT/'depth_anything_v2_vits.pth').resolve()]
    assert len(original_weight)==1 and bindings[str(PARENT/'depth_anything_v2_vits.pth')]==original_weight[0]
    freeze=dict(bindings=bindings,inputs=inputs,ids=frozen['ids'],groups=frozen['groups'],
                configs=CONFIGS,seed=177017,prior_e1_summary_sha256=sha(E1/'summary.json'),
                scope='CONSUMED_CONTROLLED_DEVELOPMENT',original_test_access=False)
    if (out/'freeze.json').exists():assert read(out/'freeze.json')==freeze
    else:write(out/'freeze.json',freeze)
    snap=out/'source-snapshot';snap.mkdir(exist_ok=True)
    for p in sources[:5]:shutil.copyfile(p,snap/p.name)
    model=load_model();first=ds['train']['rows'][0];image=cv2.imread(str(OLD/first['rgb_path']))
    # Exact encoder workload on the same preprocessed tensor, not the E1 decoder benchmark.
    tensor,_=model.image2tensor(image,518)
    @torch.inference_mode()
    def probe(device):
        model.to(device)
        return model.pretrained.get_intermediate_layers(tensor.to(device),[2,11],reshape=True)[0]
    backend=select_backend('model-inference',
        cpu=BackendCandidate('dav2-token-cpu','cpu',lambda:probe('cpu'),lambda v:DeviceObservation(v.device.type,'host CPU',torch.__version__)),
        gpu=BackendCandidate('dav2-token-cuda','cuda',lambda:probe('cuda'),lambda v:DeviceObservation(v.device.type,torch.cuda.get_device_name(0),torch.__version__),torch.cuda.synchronize),
        warmups=1,repeats=1,record_path=out/'backend.json')
    assert backend['selected_device_type']=='cuda','Unexpected placement; inspect before inference'
    model.cuda();torch.cuda.reset_peak_memory_stats()
    token={};mask={}
    for k,d in ds.items():
        cache=out/(k+'-tokens');cache.mkdir(exist_ok=True)
        tt=[];mm=[];yaw=0.;episode=None
        for i,row in enumerate(d['rows']):
            assert time.perf_counter()-start<3600,'Diagnostic budget exceeded'
            if row['episode_id']!=episode:yaw=0.
            if row['imu_valid']:yaw+=row['delta_yaw']
            episode=row['episode_id'];p=d['cap']/row['rgb_path']
            digest=sha(p);assert digest==d['receipt']['hashes'][row['rgb_path']]
            cp=cache/(row['id']+'.npz')
            if cp.exists():
                z=np.load(cp);assert str(z['rgb_sha256'])==digest
                value=z['tokens'];valid=z['mask']
            else:
                im=cv2.imread(str(p));tick=time.perf_counter();value,valid,names=extract(model,im,row,yaw)
                np.savez_compressed(cp,tokens=value,mask=valid,rgb_sha256=digest,seconds=time.perf_counter()-tick)
                if not (out/'position-names.json').exists():write(out/'position-names.json',names)
            tt.append(value);mm.append(valid)
            if (i+1)%24==0:
                progress=dict(stage='positioned_tokens',split=k,completed=i+1,total=len(d['rows']),seconds=time.perf_counter()-start)
                write(out/'progress.json',progress);print(progress,flush=True)
        token[k]=np.stack(tt);mask[k]=np.stack(mm)
    # Label-free compression: all PCA fit samples come from TRAIN only.
    pcas=[]
    for layer in range(2):
        pca=PCA(n_components=8,svd_solver='full',whiten=False)
        pca.fit(token['train'][:,layer][mask['train']]);pcas.append(pca)
    (out/'pca.pkl').write_bytes(pickle.dumps(pcas))
    added={k:transform(token[k],mask[k],pcas) for k in ds}
    for k in ds:np.savez_compressed(out/(k+'-features.npz'),base=base[k],intermediate=added[k],mask=mask[k],ids=frozen['ids'][k])
    write(out/'feature-seal.json',dict(features={k:sha(out/(k+'-features.npz')) for k in ds},pca_sha256=sha(out/'pca.pkl'),
        pca_fit_ids=frozen['ids']['train'],pca_fit_token_count=int(mask['train'].sum()),
        explained_variance=[p.explained_variance_ratio_.tolist() for p in pcas],
        mask_valid_fraction={k:float(mask[k].mean()) for k in ds},feature_seconds=time.perf_counter()-start))
    _,yt=labels(ds['train']);_,yd=labels(ds['dev']);xx={k:np.c_[base[k],added[k]] for k in ds}
    candidates=[];search=[]
    for idx,config in enumerate(CONFIGS):
        tick=time.perf_counter();fit=HistGradientBoostingClassifier(**config,learning_rate=.05,min_samples_leaf=8,early_stopping=False,random_state=177017)
        fit.fit(xx['train'],yt);dev=fit.predict_proba(xx['dev'])[:,1]
        best,curve=select_threshold(yd,dev);hi,_=select_threshold(yd,dev,True)
        ap=float(average_precision_score(yd,dev));train=fit.predict_proba(xx['train'])[:,1]
        record=dict(arm='C',config_index=idx,config=config,dev_best=best,dev_recall95=hi,dev_ap=ap,
            train_accuracy=float(((train>=best['threshold'])==yt).mean()),seconds=time.perf_counter()-tick)
        search.append(record);candidates.append((best['f1'],ap,-idx,fit,record));print(record,flush=True)
    chosen=max(candidates,key=lambda c:c[:3]);head=chosen[3];selection=chosen[4]
    (out/'C-model.pkl').write_bytes(pickle.dumps(head));write(out/'development-search.json',search)
    write(out/'selection-seal.json',dict(selection=selection,head_sha256=sha(out/'C-model.pkl'),pca_sha256=sha(out/'pca.pkl'),
          feature_seal_sha256=sha(out/'feature-seal.json'),report_labels_used_for_selection=False))
    scores=head.predict_proba(xx['report'])[:,1];np.save(out/'report-scores.npy',scores)
    write(out/'prediction-seal.json',dict(scores_sha256=sha(out/'report-scores.npy'),selection_sha256=sha(out/'selection-seal.json')))
    es,y=labels(ds['report']);prior=read(E1/'summary.json');reports={k:prior['reports'][k] for k in ('MZ129','MZ145','four_expert','A_f1','B_f1')}
    baseline_records=read(REF/'baseline-v1/predictions.json')
    assert [r['id'] for r in baseline_records]==frozen['ids']['report']
    baseline=np.array([r['candidate'] for r in baseline_records],bool);flags={};native={}
    for suffix,key in [('f1','dev_best'),('recall95','dev_recall95')]:
        name='C_'+suffix;flags[name]=scores>=selection[key]['threshold']
        reports[name]=report(ds['report'],es,y,scores,flags[name],baseline)
        native[name]=native_account(ds['report']['rows'],es,flags[name],baseline)
    write(out/'reports.json',reports);write(out/'native.json',native)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['id','episode','time_s','family','truth','score','C_f1','C_recall95'])
        for i,(r,e) in enumerate(zip(ds['report']['rows'],es)):
            writer.writerow([r['id'],r['episode_id'],r['time_s'],e['family'],int(y[i]),scores[i],int(flags['C_f1'][i]),int(flags['C_recall95'][i])])
    # Narrow complete online replay, independently matching sealed feature and score vectors.
    from mz143_corridor_features import extract as base_extract
    a_model=pickle.loads((E1/'A-model.pkl').read_bytes());timing=[];first_episodes={}
    for r,e in zip(ds['report']['rows'],es):first_episodes.setdefault(e['family'],r['episode_id'])
    video=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(640,480));assert video.isOpened()
    oldpred=list(csv.DictReader((E1/'predictions.csv').open(encoding='utf-8')))
    yaw=0.;episode=None
    try:
        for i,(r,e) in enumerate(zip(ds['report']['rows'],es)):
            if r['episode_id']!=episode:yaw=0.
            if r['imu_valid']:yaw+=r['delta_yaw']
            episode=r['episode_id']
            if r['episode_id']!=first_episodes[e['family']]:continue
            torch.cuda.synchronize();tick=time.perf_counter();im=cv2.imread(str(CAP/r['rgb_path']))
            b=base_extract(r,im,yaw);b=np.r_[b['sensor'],b['geometry']];assert np.array_equal(b,base['report'][i])
            a_model.predict_proba(b[None]);a_seconds=time.perf_counter()-tick
            more=time.perf_counter();t,m,_=extract(model,im,r,yaw);v=transform(t[None],m[None],pcas)[0]
            pred=head.predict_proba(np.r_[b,v][None])[0,1];torch.cuda.synchronize();c_seconds=a_seconds+time.perf_counter()-more
            assert np.allclose(v,added['report'][i],rtol=1e-5,atol=1e-5)
            assert abs(pred-scores[i])<1e-12
            timing.append(dict(id=r['id'],A_seconds=a_seconds,C_conservative_seconds=c_seconds))
            canvas=np.zeros((480,640,3),np.uint8);canvas[:360]=im
            lines=[f"{e['family']} t={r['time_s']:.2f}s truth={int(y[i])}",
                   f"A={oldpred[i]['A_f1_alert']} B(depth)={oldpred[i]['B_f1_alert']} C(tokens)={int(flags['C_f1'][i])}",
                   'Consumed Development; first episode per family']
            for j,line in enumerate(lines):cv2.putText(canvas,line,(9,389+30*j),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            video.write(canvas)
            if len(timing)==1:cv2.imwrite(str(out/'preview.png'),canvas)
    finally:video.release()
    write(out/'latency-samples.json',timing)
    latency={a:{'p50_ms':float(np.quantile([t[key] for t in timing],.5)*1000),'p95_ms':float(np.quantile([t[key] for t in timing],.95)*1000)}
             for a,key in [('A','A_seconds'),('C','C_conservative_seconds')]}
    result=dict(reports=reports,selection=selection,latency=latency,latency_frames=len(timing),
        prior_B_latency=prior['latency']['B'],prior_B_latency_note='Prior E1 run, not simultaneous measurement',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start,
        scope='CONSUMED_CONTROLLED_DEVELOPMENT',original_test_access=False,radar_native_lineage='NOT_EVALUABLE',
        decision='ONE_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_E2',encoder_parameters=sum(p.numel() for p in model.pretrained.parameters()))
    write(out/'summary.json',result)
    assert all(sha(p)==h for p,h in {**bindings,**inputs}.items())
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seal_sha256=sha(out/'prediction-seal.json'),predictions_sha256=sha(out/'predictions.csv'),
        video_sha256=sha(out/'comparison.mp4'),seconds=time.perf_counter()-start,resources='process-local; released at exit'))
    print({k:(v['metrics'],v['f1'],v['pr_auc']) for k,v in reports.items()},flush=True)
    print(latency,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        with threadpool_limits(4):run(args.output)
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise
