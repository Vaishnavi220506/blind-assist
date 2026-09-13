"""Final-round causal parallax, serialize before native geometry evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import mz119_parallax as motion
import mz116_four_sensor as incumbent
import mz115_spatial_allocation as zonal
from run_mz107_four_sensor import ROOT,sha,readrows,write,truth
from run_mz111_spatial_temporal import score
from run_mz113_dynamic_flow import exit_metrics
from research_backend import BackendCandidate,DeviceObservation,select_backend


def readout(rows,curves,geometry):
    for d,arms in curves.items():
        arms['parallax']=[]
        for row,original,g,primary in zip(rows,arms['resolution_guard'],geometry,curves['3.6']['resolution_guard']):
            support=[];yaw=primary['integrated_yaw_deg'];dy=.5+.2*row['time_s'];pitch=row['camera_pitch_deg']
            for point in g['points']:
                u,v=point['pixel'];xyz=zonal.slant_envelope([u-.5,v-.5,u+.5,v+.5],point['range_bounds_m'],row['rgb_intrinsics'],
                    (pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
                support.append(zonal.possible(xyz,float(d)))
                if d=='3.6':point.update(localized_xyz=xyz,alert_support=support[-1])
            arms['parallax'].append(dict(candidate=bool(original['candidate'] or any(support)),
                candidate_state='ALERT' if original['candidate'] or any(support) else 'UNKNOWN',new_geometry_support=any(support),
                added=bool(not original['candidate'] and any(support))))
        assert all(not a['candidate'] or b['candidate'] for a,b in zip(arms['resolution_guard'],arms['parallax']))
    return {d:{k:arms[k] for k in ('baseline','nominal','resolution_guard','parallax')} for d,arms in curves.items()}


def run(capture,out):
    out=out.resolve();capture=capture.resolve();start=time.perf_counter()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
    for name,h in receipt['hashes'].items():assert sha(capture/name)==h
    rows=readrows(capture/'raw.jsonl');assert len(rows)==240
    images={r['id']:cv2.imread(str(capture/r['rgb_path'])) for r in rows};assert all(v is not None for v in images.values())
    out.mkdir(parents=True)
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-scipy-cpu','cpu',lambda:motion.predict(rows[:2],lambda r:images[r['id']]),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),cpu_reason='GPU_BACKEND_UNAVAILABLE',
        record_path=out/'backend.json',capabilities={'opencv_cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
        'cuda_sparse_lk_binding':hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create'),'lp_placement':'TASK_NOT_GPU_SUITABLE'})
    tick=time.perf_counter();geometry=motion.predict(rows,lambda r:images[r['id']]);motion_seconds=time.perf_counter()-tick
    for frame in geometry:
        anchor_ids={a['feature_id'] for pose in frame['poses'] for a in pose['anchor_features']}
        assert all(p['feature_id'] not in anchor_ids for p in frame['points'])
    curves=readout(rows,incumbent.predict(rows,lambda r:images[r['id']]),geometry)
    disabled_geometry=motion.predict(rows,lambda _:None);assert not any(g['poses'] or g['points'] for g in disabled_geometry)
    disabled=readout(rows,incumbent.predict(rows,lambda _:None),disabled_geometry)
    assert all(all(a['candidate']==b['candidate'] for a,b in zip(v['baseline'],v['parallax'])) for v in disabled.values())
    assert time.perf_counter()-start<1200,'Frozen replay wall budget exhausted'
    write(out/'geometry.json',geometry);write(out/'predictions.json',curves);write(out/'rgb-disabled.json',disabled)
    names=('mz119_parallax.py','run_mz119_parallax.py','mz119_parallax_audit.py','mz116_four_sensor.py','mz116_radar_resolution_guard.py',
        'mz115_spatial_allocation.py','mz108_competitive_association.py','mz107_rgb_association.py','mz111_spatial_evidence.py',
        'mz111_temporal_geometry.py','mz113_flow_persistence.py','mz109_interval_extent.py','run_mz107_four_sensor.py',
        'run_mz111_spatial_temporal.py','run_mz113_dynamic_flow.py')
    for name in names:shutil.copyfile(Path(__file__).with_name(name),out/name)
    protocol=Path(__file__).with_name('MZ119_PROTOCOL_20260913.md');shutil.copyfile(protocol,out/protocol.name)
    write(out/'prediction-seal.json',dict(status='ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE',
        capture=str(capture),raw_sha256=sha(capture/'raw.jsonl'),receipt_sha256=sha(capture/'receipt.json'),
        predictions_sha256=sha(out/'predictions.json'),geometry_sha256=sha(out/'geometry.json'),rgb_disabled_sha256=sha(out/'rgb-disabled.json'),
        code_sha256={name:sha(out/name) for name in names},protocol_sha256=sha(out/protocol.name)))
    es=readrows(capture/'evaluator.jsonl');assert [e['id'] for e in es]==[r['id'] for r in rows]
    labels=[truth(e) for e in es];summary={}
    for d,arms in curves.items():
        value=score(rows,es,arms);ref=arms['resolution_guard']
        for arm,preds in arms.items():
            value[arm]['exit_metrics']=exit_metrics(rows,labels,preds)
            value[arm]['vs_incumbent']=dict(gained_TP=sum(g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),
                lost_TP=sum(g and not a['candidate'] and b['candidate'] for g,a,b in zip(labels,preds,ref)),
                added_FP=sum(not g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),removed_FP=0)
        keep=[i for i,e in enumerate(es) if e['family']!='boundary_1cm_stress']
        value['nonstress']=score([rows[i] for i in keep],[es[i] for i in keep],{k:[v[i] for i in keep] for k,v in arms.items()})
        summary[d]=value
    write(out/'summary.json',dict(status='FINAL_PARALLAX_ROUND_COMPLETE',frames=240,curve=summary,motion_seconds=motion_seconds,
        primary_distance_m=3.6,scope='CONSTRUCTED_HYPOTHETICAL_SENSOR_NOT_HARDWARE',stop='USER_REQUESTED_STOP_AFTER_THIS_ROUND'))
    write(out/'completion.json',dict(status='PASS',output_hashes={p.name:sha(p) for p in out.glob('*.json')},resources_started=[]))
    print(json.dumps({k:v for k,v in summary['3.6'].items() if k in ('resolution_guard','parallax')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.capture,a.output)
