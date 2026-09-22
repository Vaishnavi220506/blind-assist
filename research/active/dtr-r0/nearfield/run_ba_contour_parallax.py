"""Single CPU-only bounded prediction pass; evaluator data are not loaded."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import psutil

import ba_contour_parallax as m

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/ba-contour-parallax-20260919'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    def scalar(item):
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(f'Unsupported result type: {type(item).__name__}')
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,default=scalar)


def run():
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    process=psutil.Process();previous_affinity=process.cpu_affinity();process.cpu_affinity([0])
    try:
        protocol=read(OUT/'protocol.json');observations=read(OUT/'observations.json')
        assert protocol['configuration']==m.CONFIG
        assert sha(OUT/'observations.json')==protocol['observations_sha256']
        for name,digest in protocol['code_hashes'].items():
            current=sha(Path(__file__).with_name(name))
            repairs=read(OUT/'mechanical-repairs.json')['repairs'] if (OUT/'mechanical-repairs.json').exists() else []
            assert current==digest or any(name==Path(__file__).name and
                r['file']==name and r['original_sha256']==digest and r['repaired_sha256']==current and
                r['kind']=='RESULT_SERIALIZATION' and r['scientific_settings_unchanged'] and
                r['completed_predictions_before_repair']==0 for r in repairs),name
        assert len(observations)==12 and [r['id'] for r in observations]==[f's{i:02d}' for i in range(12)]
        assert not (OUT/'predictions.json').exists() and not (OUT/'prediction-seal.json').exists()
        write(OUT/'launch.json',dict(protocol_sha256=sha(OUT/'protocol.json'),observations_sha256=sha(OUT/'observations.json'),
            runner_sha256=sha(__file__),pid=process.pid,affinity=process.cpu_affinity(),opencv_threads=cv2.getNumThreads(),
            opencv_opencl=cv2.ocl.useOpenCL(),numpy=np.__version__,opencv=cv2.__version__,python=platform.python_version(),
            cpu=platform.processor(),cpu_reason='FROZEN_PROTOCOL_CPU_ONLY',model_calls=0,training_updates=0,
            target_hardware_verified=False,scope='One desktop logical core development proxy; excludes pose estimation and acquisition'))
        sys.path.insert(0,str(ROOT/'tools'))
        from research_backend import BackendCandidate,DeviceObservation,select_backend
        candidate=BackendCandidate('bounded-opencv-numpy-cpu','cpu',
            lambda:m.project([[10.,10.]],[.5],observations[0]['frames'][0]['pose'],observations[0]['frames'][2]['pose'],observations[0]['calibration']),
            lambda _:DeviceObservation('cpu',platform.processor(),'OpenCV/NumPy',('CPU',)))
        select_backend('point-cloud-matching',cpu=candidate,cpu_reason='FROZEN_PROTOCOL_CPU_ONLY',
            record_path=OUT/'backend.json',capabilities={'cpu_only_by_frozen_protocol':True,'affinity':[0],
            'probe_scope':'Projection plumbing only; actual workload timed once in the sealed twelve-window pass'})
        rows=[];started=time.perf_counter();peak_rss=process.memory_info().rss
        for row in observations:
            tick=time.perf_counter();images=[]
            for frame in row['frames']:
                path=Path(frame['path']);assert path.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
                assert sha(path)==frame['sha256'];images.append(cv2.imread(str(path),cv2.IMREAD_COLOR))
            decoded=time.perf_counter();cpu_start=time.process_time()
            result=m.match(images,[f['pose'] for f in row['frames']],row['calibration'])
            cpu_seconds=time.process_time()-cpu_start;finished=time.perf_counter()
            memory=process.memory_info();peak_rss=max(peak_rss,memory.rss,getattr(memory,'peak_wset',0))
            item=dict(id=row['id'],endpoint_index=row['endpoint_index'],matcher=result,
                timing=dict(decode_ms=(decoded-tick)*1000,algorithm_ms=(finished-decoded)*1000,
                            total_ms=(finished-tick)*1000,cpu_algorithm_ms=cpu_seconds*1000),rss_bytes=memory.rss)
            write(OUT/(row['id']+'.json'),item);rows.append(item)
            print('CONTOUR_PREDICTED',row['id'],'lines',result['candidate_count'],'ms',round(item['timing']['algorithm_ms'],2),flush=True)
        summary=dict(status='COMPLETE',rows=rows,protocol_sha256=sha(OUT/'protocol.json'),
            observations_sha256=sha(OUT/'observations.json'),launch_sha256=sha(OUT/'launch.json'),
            peak_rss_bytes=peak_rss,elapsed_s=time.perf_counter()-started,model_calls=0,training_updates=0,
            matcher_calls=len(rows),cpu=dict(affinity=process.cpu_affinity(),opencv_threads=cv2.getNumThreads(),
            reason='FROZEN_PROTOCOL_CPU_ONLY'),uses_native_depth=False,uses_ideal_metric_poses=True)
        write(OUT/'predictions.json',summary)
        write(OUT/'prediction-seal.json',dict(status='COMPLETE',clips=12,frames=36,
            predictions_sha256=sha(OUT/'predictions.json'),protocol_sha256=sha(OUT/'protocol.json'),
            observations_sha256=sha(OUT/'observations.json'),
            outputs={r['id']+'.json':sha(OUT/(r['id']+'.json')) for r in rows},runner_sha256=sha(__file__)))
        print('CONTOUR_PREDICTIONS_SEALED',flush=True)
    finally:
        process.cpu_affinity(previous_affinity)
        write(OUT/'inference-release.json',dict(pid=process.pid,affinity_restored=process.cpu_affinity()==previous_affinity,
            background_workers_started=0,model_allocations=0))


if __name__=='__main__':
    run()
