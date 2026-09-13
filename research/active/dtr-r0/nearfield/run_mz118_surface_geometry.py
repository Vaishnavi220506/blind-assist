"""Matched consumed finite-zone panels, sealed before native evaluator parsing."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import scipy
import mz118_surface_pipeline as method
from run_mz107_four_sensor import ROOT,sha,readrows,write,truth
from run_mz111_spatial_temporal import score
from run_mz113_dynamic_flow import exit_metrics
from research_backend import BackendCandidate,DeviceObservation,select_backend

PANELS={'mz115':'mz115-zonal-allocation-20260913','mz117':'mz117-surface-mixtures-20260913'}
CODE=('mz118_surface_pipeline.py','mz118_interval_planes.py','mz118_surface_regions.py','run_mz118_surface_geometry.py',
      'mz117_surface_intervals.py','mz117_surface_audit.py','mz116_four_sensor.py','mz116_radar_resolution_guard.py',
      'mz115_spatial_allocation.py','mz108_competitive_association.py','mz107_rgb_association.py',
      'mz111_spatial_evidence.py','mz111_temporal_geometry.py','mz113_flow_persistence.py','mz109_interval_extent.py',
      'run_mz107_four_sensor.py','run_mz111_spatial_temporal.py','run_mz113_dynamic_flow.py')


def run(out):
    started=time.perf_counter();out=out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    inputs={}
    for panel,folder in PANELS.items():
        capture=ROOT/'artifacts.local/work'/folder/'capture-v1'
        receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest
        rows=readrows(capture/'raw.jsonl');assert len(rows)==240
        inputs[panel]=(capture,rows)
    out.mkdir(parents=True)
    for name in CODE:shutil.copyfile(Path(__file__).with_name(name),out/name)
    protocol=Path(__file__).with_name('MZ118_PROTOCOL_20260913.md');shutil.copyfile(protocol,out/protocol.name)
    write(out/'freeze.json',dict(status='CONSUMED_SOURCE_CODE_FROZEN_BEFORE_NEW_READOUT',
        code_sha256={name:sha(out/name) for name in CODE},protocol_sha256=sha(out/protocol.name),
        panels={panel:dict(capture=str(c),raw_sha256=sha(c/'raw.jsonl'),receipt_sha256=sha(c/'receipt.json')) for panel,(c,_) in inputs.items()}))
    select_backend('scalar-scoring',cpu=BackendCandidate('scipy-highs-cpu','cpu',
        lambda:method.interval.linprog([1.,0.,0.],bounds=[(1,None),(0,0),(0,0)],method='highs'),
        lambda _:DeviceObservation('cpu','host CPU','SciPy '+scipy.__version__)),cpu_reason='TASK_NOT_GPU_SUITABLE',
        record_path=out/'backend.json',capabilities={'opencv_cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
        'cuda_sparse_lk_binding':hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create'),
        'rgb_backend':'Existing CPU OpenCV; GPU_BACKEND_UNAVAILABLE; unchanged frontend workload',
        'workload':'small three-variable LP octants, CPU solver; complete comparator frontends included'})
    seals={};timings={}
    for panel,(capture,rows) in inputs.items():
        target=out/panel;target.mkdir();images={r['id']:cv2.imread(str(capture/r['rgb_path'])) for r in rows}
        assert all(im is not None for im in images.values())
        tick=time.perf_counter();curves=method.predict(rows,lambda r:images[r['id']]);timings[panel]=time.perf_counter()-tick
        disabled=method.predict(rows,lambda _:None)
        for arms in disabled.values():assert all(a['candidate']==b['candidate']==c['candidate'] for a,b,c in zip(arms['baseline'],arms['expanded_fit'],arms['interval_lp']))
        if panel=='mz117':
            old_seal=json.loads((capture.parent/'analysis-v1/prediction-seal.json').read_text())
            assert sha(capture.parent/'analysis-v1/predictions.json')==old_seal['predictions_sha256']
            old=json.loads((capture.parent/'analysis-v1/predictions.json').read_text())
            for d,arms in curves.items():assert [p['candidate'] for p in arms['resolution_guard']]==[p['candidate'] for p in old[d]['resolution_guard']]
        write(target/'predictions.json',curves);write(target/'rgb-disabled.json',disabled)
        seals[panel]=dict(capture=str(capture),raw_sha256=sha(capture/'raw.jsonl'),receipt_sha256=sha(capture/'receipt.json'),
            predictions_sha256=sha(target/'predictions.json'),rgb_disabled_sha256=sha(target/'rgb-disabled.json'))
        del curves,disabled,images
        assert time.perf_counter()-started<1200,'Frozen replay wall budget exhausted'
    write(out/'prediction-seal.json',dict(status='ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE',panels=seals,
        code_sha256={name:sha(out/name) for name in CODE},protocol_sha256=sha(out/protocol.name)))
    summaries={}
    for panel,(capture,rows) in inputs.items():
        es=readrows(capture/'evaluator.jsonl');assert [e['id'] for e in es]==[r['id'] for r in rows]
        labels=[truth(e) for e in es];curves=json.loads((out/panel/'predictions.json').read_text());sums={}
        for d,arms in curves.items():
            summary=score(rows,es,arms);ref=arms['resolution_guard']
            for arm,preds in arms.items():
                summary[arm]['exit_metrics']=exit_metrics(rows,labels,preds)
                summary[arm]['vs_incumbent']=dict(gained_TP=sum(g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),
                    lost_TP=sum(g and not a['candidate'] and b['candidate'] for g,a,b in zip(labels,preds,ref)),
                    added_FP=sum(not g and a['candidate'] and not b['candidate'] for g,a,b in zip(labels,preds,ref)),
                    removed_FP=sum(not g and not a['candidate'] and b['candidate'] for g,a,b in zip(labels,preds,ref)))
            keep=[i for i,e in enumerate(es) if e['family']!='boundary_1cm_stress']
            summary['nonstress']=score([rows[i] for i in keep],[es[i] for i in keep],{k:[v[i] for i in keep] for k,v in arms.items()})
            sums[d]=summary
        summaries[panel]=dict(frames=len(rows),curve=sums,seconds_per_frame=timings[panel]/len(rows))
    write(out/'summary.json',dict(status='CONSUMED_AUXILIARY_SURFACE_STUDY_COMPLETE',panels=summaries,
        scope='CONSTRUCTED_DEVELOPMENT_UNCHANGED_HYPOTHETICAL_SENSOR',primary_distance_m=3.6))
    write(out/'completion.json',dict(status='PASS',output_hashes={p.name:sha(p) for p in out.glob('*.json')},resources_started=[]))
    print(json.dumps({panel:{k:dict(metrics=v['metrics'],vs_incumbent=v['vs_incumbent'],events=v['events']) for k,v in summary['curve']['3.6'].items() if k in ('resolution_guard','expanded_fit','interval_lp')} for panel,summary in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
