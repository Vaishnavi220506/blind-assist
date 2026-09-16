"""Public-only dense parallax and frozen incumbent; compose before evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[4];CODE=Path(__file__).resolve().parent
WORK=ROOT/'artifacts.local/work/mz160-dense-parallax-20260916'
CAP=ROOT/'artifacts.local/work/mz119-tof-scaled-parallax-20260913/capture-v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def freeze():
    from run_mz139_surface_fit import local_dependencies
    path=WORK/'recipe-freeze.json';assert not path.exists()
    sources={}
    for name in ('run_mz160_dense_parallax.py','mz160_dense_parallax.py','evaluate_mz160_dense_parallax.py','mz136_incumbent.py'):
        sources.update(local_dependencies(CODE/name))
    sources[str(CODE/'MZ160_PROTOCOL_20260916.md')]=sha(CODE/'MZ160_PROTOCOL_20260916.md')
    for name,h in sources.items():
        dest=WORK/'source-snapshot'/Path(name).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest);assert sha(dest)==h
    receipt=read(CAP/'receipt.json');assert receipt['status']=='PASS'
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name,h in receipt['hashes'].items():assert sha(CAP/name)==h
    inputs={str(CAP/n):sha(CAP/n) for n in ('raw.jsonl','evaluator.jsonl','receipt.json','spec.json')}
    write(path,dict(sources=sources,inputs=inputs,frames=240,
        scope='CONSUMED_MZ119_ALL_FRAMES_NO_NEW_CAPTURE',inference='PUBLIC_RGB_TOF_IMU_PLUS_UNCHANGED_MZ129',
        forbidden='NATIVE_OR_COMMANDED_TRANSLATION_OR_OBJECT_IDENTITY_IN_PREDICTOR',
        parameters='MZ160_PROTOCOL_AND_SOURCE_BEFORE_ANY_REAL_MATCHING',public_inference_budget_seconds=1200))
    print(json.dumps(dict(stage='frozen',source_files=len(sources),sha256=sha(path))),flush=True)

def admit():
    from mz136_incumbent import public_observations
    sealed=read(WORK/'recipe-freeze.json')
    for p,h in sealed['sources'].items():assert sha(p)==h,p
    for p,h in sealed['inputs'].items():assert sha(p)==h,p
    rows=public_observations([json.loads(s) for s in (CAP/'raw.jsonl').read_text().splitlines()])
    assert len(rows)==240 and len({r['id'] for r in rows})==240
    return rows

def seal(out,outputs,seconds,context):
    inputs={str(CAP/n):sha(CAP/n) for n in ('raw.jsonl','receipt.json')}
    write(out/'prediction-seal.json',dict(status='SEALED_PUBLIC_BEFORE_EVALUATOR_PARSE',
        capture=str(CAP),raw_sha256=sha(CAP/'raw.jsonl'),receipt_sha256=sha(CAP/'receipt.json'),
        recipe_freeze_sha256=sha(WORK/'recipe-freeze.json'),outputs={n:sha(out/n) for n in outputs},
        inputs=inputs,seconds=seconds,context=context))

def baseline():
    import cv2
    from mz136_incumbent import predict_incumbent
    from research_backend import BackendCandidate,DeviceObservation,select_backend
    assert cv2.__version__=='5.0.0'
    out=WORK/'baseline-v1';assert not out.exists();rows=admit();out.mkdir();start=time.perf_counter()
    previous=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/incumbent-v1/prediction-seal.json'
    bound=read(previous)['source_hashes']
    for p,h in bound.items():assert sha(p)==h,p
    receipt=read(CAP/'receipt.json')
    def loader(row):
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        assert sha(p)==receipt['hashes'][row['rgb_path']]
        return cv2.imread(str(p))
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-incumbent-cpu','cpu',
        lambda:loader(rows[0]),lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Frozen MZ129 CPU implementation'})
    result=predict_incumbent(rows,loader)
    preds=[dict(id=r['id'],candidate=bool(p['candidate'])) for r,p in zip(rows,result['predictions'])]
    assert len(preds)==240;write(out/'predictions.json',preds);write(out/'incumbent-details.json',result)
    admit();seconds=time.perf_counter()-start;assert seconds<1200
    seal(out,['predictions.json','incumbent-details.json'],seconds,dict(mode='UNCHANGED_MZ129',prior_source_seal_sha256=sha(previous)))
    write(out/'completion.json',dict(status='PASS',seconds=seconds,prediction_seal_sha256=sha(out/'prediction-seal.json'),resources='Process exits'))
    print(json.dumps(dict(stage='baseline_sealed',seconds=seconds)),flush=True)

def geometry():
    import cv2
    import numpy as np
    from mz160_dense_parallax import predict,dense_pair,METHOD
    from research_backend import BackendCandidate,DeviceObservation,select_backend
    assert cv2.__version__=='4.10.0';cv2.setNumThreads(4)
    out=WORK/'inference-v1';assert not out.exists();rows=admit();out.mkdir();start=time.perf_counter()
    receipt=read(CAP/'receipt.json');images={};rgb={}
    for row in rows:
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        rgb[str(p)]=sha(p);assert rgb[str(p)]==receipt['hashes'][row['rgb_path']]
        image=cv2.imread(str(p));assert image.shape==(360,640,3);images[row['id']]=image
    first=cv2.cvtColor(images[rows[0]['id']],cv2.COLOR_BGR2GRAY)
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-sgbm-cpu','cpu',
        lambda:dense_pair(rows[0],rows[0],first,first,0.,0.),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented StereoSGBM and robust regional calibration are CPU-only'})
    (out/'disparities').mkdir()
    def sink(i,lag,d,valid):
        np.savez_compressed(out/'disparities'/f'{i:03d}-lag{lag}.npz',disparity=d,valid=valid)
        assert time.perf_counter()-start<1200
        if lag==3 and (i+1)%12==0:print(json.dumps(dict(stage='dense_episode',frames=i+1)),flush=True)
    values=predict(rows,lambda r:images[r['id']],sink)
    write(out/'geometry.json',values)
    maps={str(p.relative_to(out)):sha(p) for p in sorted((out/'disparities').glob('*.npz'))}
    write(out/'geometry-seal.json',dict(geometry_sha256=sha(out/'geometry.json'),disparities=maps,rgb_sha256=rgb,
        recipe_freeze_sha256=sha(WORK/'recipe-freeze.json'),method=METHOD,
        authority='PUBLIC_GEOMETRY_SEALED_BEFORE_EVALUATOR_PARSE',seconds=time.perf_counter()-start))
    admit();assert time.perf_counter()-start<1200
    print(json.dumps(dict(stage='geometry_sealed',point_frames=sum(bool(x['points']) for x in values),
        points=sum(len(x['points']) for x in values),seconds=time.perf_counter()-start)),flush=True)

def compose():
    rows=admit();out=WORK/'inference-v1';baseline=WORK/'baseline-v1'
    assert not (out/'prediction-seal.json').exists()
    gs=read(out/'geometry-seal.json');bs=read(baseline/'prediction-seal.json')
    assert sha(out/'geometry.json')==gs['geometry_sha256']
    assert sha(baseline/'predictions.json')==bs['outputs']['predictions.json']
    assert sha(baseline/'prediction-seal.json')==read(baseline/'completion.json')['prediction_seal_sha256']
    for p,h in gs['disparities'].items():assert sha(out/p)==h
    for p,h in gs['rgb_sha256'].items():assert sha(p)==h
    geo=read(out/'geometry.json');bb=read(baseline/'predictions.json')
    assert [r['id'] for r in rows]==[g['id'] for g in geo]==[b['id'] for b in bb]
    pp=[dict(id=r['id'],baseline=b['candidate'],candidate=bool(b['candidate'] or any(p['alert_support'] for p in g['points'])),
        new_geometry_support=any(p['alert_support'] for p in g['points'])) for r,b,g in zip(rows,bb,geo)]
    write(out/'predictions.json',pp)
    seal(out,['geometry.json','geometry-seal.json','predictions.json'],gs['seconds'],
        dict(mode='UNCHANGED_MZ129_OR_CONDITIONAL_DENSE_SUPPORT',baseline_prediction_seal_sha256=sha(baseline/'prediction-seal.json')))
    write(out/'completion.json',dict(status='PASS',prediction_seal_sha256=sha(out/'prediction-seal.json'),
        seconds=gs['seconds'],resources='Process-local maps/models released; no workers'))
    print(json.dumps(dict(stage='composed',frames=len(pp),extra_alerts=sum(p['candidate'] and not p['baseline'] for p in pp))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('freeze','baseline','geometry','compose'),required=True)
    globals()[parser.parse_args().mode]()
