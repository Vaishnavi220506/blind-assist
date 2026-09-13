"""Frozen MZ113 observable arms; labels/provenance only after prediction seal."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import mz108_competitive_association as visual
import mz113_flow_persistence as flow
from run_mz107_four_sensor import ROOT, sha, readrows, write, truth
from run_mz111_spatial_temporal import methods, score
from research_backend import BackendCandidate, DeviceObservation, select_backend


def exit_metrics(rows, labels, values):
    """Sampled false supports after true->false transitions, until next true."""
    count = 0; false_bins = 0; longest_run = 0; run = 0; in_exit = False
    episode = None; previous = False
    for row, gt, p in zip(rows, labels, values):
        if row['episode_id'] != episode:
            previous = False; in_exit = False; run = 0
        episode = row['episode_id']
        if previous and not gt:
            count += 1; in_exit = True; run = 0
        if gt:
            in_exit = False; run = 0
        if in_exit and p['candidate']:
            false_bins += 1; run += 1; longest_run = max(longest_run,run)
        else:
            run = 0
        previous = bool(gt)
    return dict(exit_transitions=count, post_exit_false_frames=false_bins,
                post_exit_false_bin_duration_s=.25*false_bins,
                longest_post_exit_false_run_s=.25*longest_run)


def evaluate(rows, nominal, images):
    values = methods(rows,nominal)
    timings = {}
    for name,use_flow in [('flow',True),('angular_unknown_velocity',False)]:
        started=time.perf_counter()
        values[name]=flow.predict(rows,nominal,lambda r:images.get(r['id']),use_flow=use_flow)
        timings[name]=(time.perf_counter()-started)/len(rows)
        combined='combined_flow' if use_flow else 'combined_angular'
        values[combined]=[dict(candidate=bool(s['candidate'] or (f['candidate'] and not n['candidate'])))
                          for s,f,n in zip(values['filtered_plane'],values[name],nominal)]
    for arm,predictions in values.items():
        assert len(predictions)==len(rows)
        assert all(not n['tof_support'] or p['candidate'] for n,p in zip(nominal,predictions)),arm
    return values,timings


def run(out, capture=None):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);work=ROOT/'artifacts.local/work'
    sources={}
    if capture:
        sources['fresh_mz113']=(capture.resolve(),None,None)
    else:
        for study in ('mz111-spatial-temporal-20260913','mz112-fresh-spatial-temporal-20260913'):
            analysis=work/study/'analysis-v1';seal=json.loads((analysis/'prediction-seal.json').read_text())
            for panel,item in seal['panels'].items():
                name='consumed_mz112' if panel=='fresh_mz112' else panel
                sources[name]=(Path(item['capture']),analysis/panel/'predictions.json',item)
    packets={};all_predictions={};seals={};costs={}
    for panel,(source,cached,item) in sources.items():
        receipt=json.loads((source/'receipt.json').read_text());assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(source/name)==digest,name
        manifest=json.loads((source/'manifest.json').read_text());assert manifest['rgb_camera_count']==1 and not manifest['depth_images_produced']
        rows=readrows(source/'raw.jsonl');assert len(rows)==receipt['frames'] and len({r['id'] for r in rows})==len(rows)
        assert all(not any(k in field for k in ('truth','actor','native','depth')) for r in rows for field in r)
        images={r['id']:cv2.imread(str(source/r['rgb_path'])) for r in rows}
        assert all(im is not None for im in images.values())
        for r in rows:assert images[r['id']].shape[:2]==(r['rgb_intrinsics']['height'],r['rgb_intrinsics']['width'])
        if cached:
            assert sha(cached)==item['predictions_sha256'] and sha(source/'raw.jsonl')==item['raw_sha256']
            original=json.loads(cached.read_text());nominal=original['nominal']
            assert all('id' not in n or n['id']==r['id'] for r,n in zip(rows,nominal))
        else:
            assert len(rows)==240
            nominal=visual.predict(rows,lambda r:images[r['id']])
        if not packets:
            select_backend('batch-tensor',cpu=BackendCandidate('opencv-lk-cpu','cpu',
                lambda:flow.predict(rows[:2],nominal[:2],lambda r:images[r['id']]),
                lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),
                cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
                capabilities={'opencv_cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
                              'cuda_sparse_lk_binding':hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create'),
                              'workload':'actual Shi-Tomasi/pyramidal LK on RGB; installed CPU OpenCV'})
        values,cost=evaluate(rows,nominal,images)
        if cached:
            assert all(values['combined'][i]['candidate']==original['combined'][i]['candidate'] for i in range(len(rows)))
        disabled=visual.predict(rows,lambda _:None)
        controls,_=evaluate(rows,disabled,{})
        assert all(all(p['candidate']==n['baseline'] for p,n in zip(preds,disabled)) for preds in controls.values())
        target=out/panel;target.mkdir();write(target/'predictions.json',values);write(target/'rgb-disabled.json',controls)
        seals[panel]=dict(capture=str(source),receipt_sha256=sha(source/'receipt.json'),raw_sha256=sha(source/'raw.jsonl'),
                          predictions_sha256=sha(target/'predictions.json'),disabled_sha256=sha(target/'rgb-disabled.json'))
        packets[panel]=rows;all_predictions[panel]=values;costs[panel]=cost
    code=(Path(__file__),Path(flow.__file__),Path(visual.__file__),Path(__file__).with_name('mz111_spatial_evidence.py'),
          Path(__file__).with_name('mz111_temporal_geometry.py'),Path(__file__).with_name('mz109_interval_extent.py'),
          Path(__file__).with_name('run_mz111_spatial_temporal.py'))
    for path in code:shutil.copyfile(path,out/path.name)
    protocol=Path(__file__).with_name('MZ113_PROTOCOL_20260913.md');shutil.copyfile(protocol,out/protocol.name)
    write(out/'prediction-seal.json',dict(status='ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE',panels=seals,
        code_sha256={p.name:sha(p) for p in out.glob('*.py')},protocol_sha256=sha(out/protocol.name)))
    summaries={};merged_rows=[];merged_es=[];merged_values={}
    for panel,rows in packets.items():
        es=readrows(sources[panel][0]/'evaluator.jsonl');assert [r['id'] for r in rows]==[e['id'] for e in es]
        values=all_predictions[panel];summary=score(rows,es,values);labels=[truth(e) for e in es]
        for arm,pred in values.items():
            summary[arm]['exit_metrics']=exit_metrics(rows,labels,pred)
            incumbent=values['combined']
            summary[arm]['vs_incumbent']=dict(gained_TP=sum(gt and p['candidate'] and not q['candidate'] for gt,p,q in zip(labels,pred,incumbent)),
                lost_TP=sum(gt and not p['candidate'] and q['candidate'] for gt,p,q in zip(labels,pred,incumbent)),
                added_FP=sum(not gt and p['candidate'] and not q['candidate'] for gt,p,q in zip(labels,pred,incumbent)),
                removed_FP=sum(not gt and not p['candidate'] and q['candidate'] for gt,p,q in zip(labels,pred,incumbent)))
        summaries[panel]=summary
        merged_rows.extend([dict(r,episode_id=panel+'/'+r['episode_id']) for r in rows]);merged_es.extend(es)
        for arm,pred in values.items():merged_values.setdefault(arm,[]).extend(pred)
    overall=score(merged_rows,merged_es,merged_values)
    for arm,pred in merged_values.items():overall[arm]['exit_metrics']=exit_metrics(merged_rows,[truth(e) for e in merged_es],pred)
    result=dict(status='FIXED_DYNAMIC_FLOW_STUDY_COMPLETE',primary='combined_flow',incumbent='combined',frames=len(merged_rows),
                scope='FRESH_DYNAMIC_CONSTRUCTED_DEVELOPMENT' if capture else 'CONSUMED_STATIC_DEVELOPMENT',
                panels=summaries,overall=overall,seconds_per_frame=costs)
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',output_hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.json'))},resources_started=[]))
    print(json.dumps({arm:dict(metrics=v['metrics'],events=v['events'],exit_metrics=v['exit_metrics'])
                      for arm,v in overall.items() if arm in ('nominal','combined','flow','combined_flow','combined_angular')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--capture',type=Path)
    args=parser.parse_args();run(args.output,args.capture)
