"""Freeze all matched-distance predictions before parsing native labels."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import mz115_spatial_allocation as method
from run_mz107_four_sensor import ROOT,sha,readrows,write,truth
from run_mz111_spatial_temporal import score
from run_mz113_dynamic_flow import exit_metrics
from research_backend import BackendCandidate,DeviceObservation,select_backend


def run(capture,out):
    capture=capture.resolve();out=out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
    for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest,name
    manifest=json.loads((capture/'manifest.json').read_text())
    assert manifest['rgb_camera_count']==1 and not manifest['depth_images_produced']
    rows=readrows(capture/'raw.jsonl');assert len(rows)==240 and len({r['id'] for r in rows})==240
    assert all('tof_zones' in r and 'tof64_range_m' not in r for r in rows)
    images={r['id']:cv2.imread(str(capture/r['rgb_path'])) for r in rows}
    assert all(im is not None for im in images.values())
    out.mkdir(parents=True)
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-lk-cpu','cpu',
        lambda:method.predict(rows[:2],lambda r:images[r['id']]),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),cpu_reason='GPU_BACKEND_UNAVAILABLE',
        record_path=out/'backend.json',capabilities={'opencv_cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
        'cuda_sparse_lk_binding':hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create'),'workload':'existing RGB LK + scalar zone geometry'})
    started=time.perf_counter();curves=method.predict(rows,lambda r:images[r['id']]);seconds=time.perf_counter()-started
    disabled=method.predict(rows,lambda _:None)
    for d,arms in disabled.items():
        assert all(a['candidate']==b['candidate']==c['candidate'] for a,b,c in zip(arms['allocation'],arms['nominal'],arms['baseline']))
    write(out/'predictions.json',curves);write(out/'rgb-disabled.json',disabled)
    names=('mz115_spatial_allocation.py','run_mz115_spatial_allocation.py','mz108_competitive_association.py',
           'mz107_rgb_association.py','mz111_spatial_evidence.py','mz111_temporal_geometry.py','mz113_flow_persistence.py',
           'mz109_interval_extent.py','run_mz111_spatial_temporal.py','run_mz107_four_sensor.py','run_mz113_dynamic_flow.py')
    for name in names:shutil.copyfile(Path(__file__).with_name(name),out/name)
    protocol=Path(__file__).with_name('MZ115_PROTOCOL_20260913.md');shutil.copyfile(protocol,out/protocol.name)
    write(out/'prediction-seal.json',dict(status='ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE',
        capture=str(capture),raw_sha256=sha(capture/'raw.jsonl'),receipt_sha256=sha(capture/'receipt.json'),
        predictions_sha256=sha(out/'predictions.json'),rgb_disabled_sha256=sha(out/'rgb-disabled.json'),
        code_sha256={p.name:sha(p) for p in out.glob('*.py')},protocol_sha256=sha(out/protocol.name)))
    es=readrows(capture/'evaluator.jsonl');assert [r['id'] for r in rows]==[e['id'] for e in es]
    labels=[truth(e) for e in es];summaries={}
    for distance,arms in curves.items():
        summary=score(rows,es,arms)
        for arm,preds in arms.items():
            summary[arm]['exit_metrics']=exit_metrics(rows,labels,preds)
            ref=arms['nominal']
            summary[arm]['vs_nominal']=dict(gained_TP=sum(g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),
                lost_TP=sum(g and not a['candidate'] and b['candidate'] for g,a,b in zip(labels,preds,ref)),
                added_FP=sum(not g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),
                removed_FP=sum(not g and not a['candidate'] and b['candidate'] for g,a,b in zip(labels,preds,ref)))
        keep=[i for i,e in enumerate(es) if e['family']!='boundary_1cm_stress']
        summary['nonstress']=score([rows[i] for i in keep],[es[i] for i in keep],
            {arm:[preds[i] for i in keep] for arm,preds in arms.items()})
        summaries[distance]=summary
    write(out/'summary.json',dict(status='FIXED_ZONAL_ALLOCATION_COMPLETE',scope='CONSTRUCTED_DEVELOPMENT_HYPOTHETICAL_SENSOR',
        frames=len(rows),primary_distance_m=3.6,curve=summaries,seconds_per_frame=seconds/len(rows),
        truth_reference='Fixed current 3.6m corridor at all operating distances; beyond-boundary alerts count as nuisance, not anticipatory benefit'))
    write(out/'completion.json',dict(status='PASS',output_hashes={p.name:sha(p) for p in out.glob('*.json')},resources_started=[]))
    print(json.dumps({k:dict(metrics=v['metrics'],events=v['events'],vs_nominal=v['vs_nominal']) for k,v in summaries['3.6'].items() if k!='nonstress'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.capture,a.output)
