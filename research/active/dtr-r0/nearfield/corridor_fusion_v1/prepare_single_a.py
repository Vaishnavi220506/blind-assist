"""Authenticated 2485-column cache for same-data frozen-structure A control."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];WORK=ROOT/'artifacts.local/work'
PREP=WORK/'corridor-public-positive-v2-20260917/preparation'
OUT=WORK/'corridor-public-single-20260917/a-control'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'data-seal.json').exists()
    bindings={}
    def bind(p,h=None):
        actual=sha(p);assert h is None or actual==h,str(p);bindings[str(p)]=actual
    seal=read(PREP/'data-seal.json');bind(PREP/'data-seal.json')
    for p,h in seal['bindings'].items():bind(p,h)
    for n,h in seal['outputs'].items():bind(PREP/n,h)
    meta=read(PREP/'metadata.json');E1=WORK/'corridor-depth-e1-20260917/run-v2'
    bind(E1/'feature-seal.json');fs=read(E1/'feature-seal.json')
    values={}
    for cohort,split in [('anchor','train'),('old','report')]:
        p=E1/(split+'-features.npz');bind(p,fs[split]);z=np.load(p)
        values.update(zip(z['ids'].tolist(),z['base']))
    p=WORK/'corridor-intrusion-20260917/diagnosis/new-base-features.npz'
    bind(p.with_name('new-base-feature-seal.json'));bind(p,read(p.with_name('new-base-feature-seal.json'))['feature_sha256'])
    z=np.load(p);values.update(zip(z['ids'].tolist(),z['base']))
    p=WORK/'corridor-depth-specialist-20260917/meta-inputs.npz'
    bind(p,read(p.with_name('meta-feature-seal.json'))['sha256'])
    ids=[m['id'] for m in meta if m['cohort']=='mz146'];z=np.load(p);assert len(ids)==len(z['base'])
    for i,identifier in enumerate(ids):
        np.testing.assert_array_equal(z['base'][i],np.load(p.parent/'meta-features'/(identifier+'.npz'))['base'])
    values.update(zip(ids,z['base']))
    home=WORK/'mz158-crossview-agreement-20260916/learned-v1';ps=read(home/'prediction-seal.json')
    bind(home/'prediction-seal.json');bind(home/'features.npz',ps['context']['features_sha256']);bind(home/'predictions.json',ps['predictions_sha256'])
    ids=[r['id'] for r in read(home/'predictions.json')];values.update(zip(ids,np.load(home/'features.npz')['static']))
    x=np.stack([values[m['id']] for m in meta]);assert x.shape==(1344,2485) and np.isfinite(x).all()
    names=WORK/'mz143-corridor-evidence-20260916/run-v1/feature-names.json';bind(names)
    bind(Path(__file__));np.savez_compressed(OUT/'features.npz',base=x,ids=[m['id'] for m in meta])
    write(OUT/'metadata.json',meta)
    assert all(sha(p)==h for p,h in bindings.items())
    write(OUT/'data-seal.json',dict(bindings=bindings,outputs={n:sha(OUT/n) for n in ['features.npz','metadata.json']},
        shape=list(x.shape),no_rgb_recomputation=True,no_depth=True,no_original_test=True))
    print(json.dumps(dict(status='PASS',shape=list(x.shape))))
if __name__=='__main__':main()
