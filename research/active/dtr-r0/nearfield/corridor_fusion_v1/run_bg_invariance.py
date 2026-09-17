"""Bounded background-invariance pilot with separate crossfit/admit/train/report."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parent), str(ROOT/'tools')]
from prepare_single_a import read, write, sha
from prepare_positive_v2 import CAPS
from train_single_a import PARAMS
from ccrl import pairing, local_pairs, log_odds
from bg_invariance import RawResidual, loss, pair_report
from bg_invariance_source import pair_indices, check_source
from mz143_corridor_features import extract
from representation_features import raw_indices
from run_representation import summarize, select, lines, bind, verify
from tolerance_eval import geometry, classify
from mz171_return_labels import make_witness_labels
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK = ROOT/'artifacts.local/work'
OUT = WORK/'corridor-bg-invariance-20260918'
REP = WORK/'corridor-representation-20260918'
DATA = WORK/'corridor-public-single-20260917/a-control'
CAP = OUT/'source/returned-v1/capture-v1'
ARMS = ('B0', 'B1', 'B2')
SEED = 189018


def old_data():
    fs = read(REP/'feature-seal.json')
    verify(fs['inputs'])
    assert sha(REP/'train-features.npz') == fs['outputs']['train-features.npz']
    z = np.load(REP/'train-features.npz')
    meta = read(DATA/'metadata.json')
    assert list(z['ids']) == [m['id'] for m in meta]
    return z['raw'], meta


def crossfit():
    assert not (OUT/'crossfit-seal.json').exists()
    x, meta = old_data()
    y = np.array([m['truth'] for m in meta], bool)
    anchor = np.array([m['cohort'] == 'anchor' for m in meta])
    scene = np.array([m['scene_index'] for m in meta])
    bindings = {str(p): sha(p) for p in (Path(__file__), HERE/'bg_invariance.py', HERE/'BG_INVARIANCE_PROTOCOL_20260918.md',
        HERE/'bg_invariance_source.py', REP/'train-features.npz', DATA/'metadata.json', REP/'model-seal.json')}
    write(OUT/'training-code-freeze.json', dict(inputs=bindings, params=PARAMS, seed=SEED,
        steps=120, max_inner_hgb=30, max_residual=21, backend_hgb='CPU_GPU_BACKEND_UNAVAILABLE'))
    started = time.perf_counter()
    for k in range(6):
        inner = np.full(len(meta), np.nan, np.float32)
        records = []
        for j in range(6):
            if j == k:
                continue
            fi = np.flatnonzero(anchor | ((scene != k) & (scene != j)))
            ri = np.flatnonzero(~anchor & (scene == j))
            assert not {meta[i]['group'] for i in fi} & {meta[i]['group'] for i in ri}
            assert not set(ri) & set(np.flatnonzero(~anchor & (scene == k)))
            path = OUT/f'outer{k}-inner{j}.pkl'
            rp = path.with_suffix('.json')
            if rp.exists():
                assert sha(path) == read(rp)['sha256']
                model = pickle.loads(path.read_bytes())
            else:
                assert not path.exists()
                model = HistGradientBoostingClassifier(**PARAMS).fit(x[fi], y[fi])
                path.write_bytes(pickle.dumps(model))
                write(rp, dict(sha256=sha(path), fit=fi.tolist(), report=ri.tolist()))
            inner[ri] = log_odds(model.predict_proba(x[ri])[:, 1])
            records.append(dict(path=path.name, sha256=sha(path), fit=fi.tolist(), report=ri.tolist()))
        use = np.flatnonzero(~anchor & (scene != k))
        assert len(use) == 960 and np.isfinite(inner[use]).all()
        np.savez_compressed(OUT/f'outer{k}-inner.npz', indices=use, logits=inner[use])
        write(OUT/f'outer{k}-inner.json', dict(models=records, scores_sha256=sha(OUT/f'outer{k}-inner.npz')))
        print(f'crossfit outer{k} complete', flush=True)
    verify(bindings)
    write(OUT/'crossfit-seal.json', dict(status='PASS', fits=30, seconds=time.perf_counter()-started,
        code_freeze_sha256=sha(OUT/'training-code-freeze.json'),
        outputs={f'outer{k}-inner.npz': sha(OUT/f'outer{k}-inner.npz') for k in range(6)}))


def admit():
    assert not (OUT/'new-data-seal.json').exists()
    receipt = read(CAP/'receipt.json')
    assert receipt['status'] == 'PASS' and receipt['frames'] == 288
    spec = read(CAP/'spec.json')
    assert sha(CAP/'spec.json') == receipt['spec_sha256'] == read(OUT/'source/capture-bundle-v1/freeze.json')['files']['spec.json']
    check_source(spec)
    bindings = {}
    bind(bindings, CAP/'spec.json', receipt['spec_sha256'])
    for n, h in receipt['hashes'].items():
        bind(bindings, CAP/n, h)
    rows, native = lines(CAP/'raw.jsonl'), lines(CAP/'evaluator.jsonl')
    assert [r['id'] for r in rows] == [f['id'] for f in spec['frames']] == [e['id'] for e in native]
    values, meta, rgbs, yaw, episode = [], [], [], 0., None
    for row, e, f in zip(rows, native, spec['frames']):
        if row['episode_id'] != episode:
            yaw = 0.
        if row['imu_valid']:
            yaw += row['delta_yaw']
        episode = row['episode_id']
        image = cv2.imread(str(CAP/row['rgb_path']))
        public = extract(row, image, yaw)
        vector = np.r_[public['sensor'], public['geometry']]
        idx = raw_indices(public['sensor_names']+public['geometry_names'])
        values.append(vector[idx]); rgbs.append(image)
        g = geometry(e)
        assert len(e['native_bounds']) == len(f['objects'])
        for obj, bounds in zip(f['objects'], e['native_bounds']):
            np.testing.assert_allclose(obj['center_m'], bounds['center_m'], rtol=0, atol=1e-5)
            np.testing.assert_allclose(obj['size_m'], 2*np.array(bounds['extent_m']), rtol=0, atol=1e-5)
        labels = make_witness_labels(row, e)
        meta.append(dict(id=row['id'], episode_id=row['episode_id'], time_s=row['time_s'],
            group=f['physical_group'], scene_index=f['scene_index'], split=f['split'],
            family=f['family'], truth=g['strict'], stratum=classify(g, .05),
            background_variant=f['background_variant'], sampled_witness=bool(((labels['target'][:128]>0)&labels['known'][:128]).any()),
            usable_tof_returns=int(public['geometry'][5]+sum(s['status']=='SIM_MERGED' and s['usable'] for s in public['audit']['slots']))))
    x = np.stack(values)
    rank, inv = pair_indices(spec)
    rank, inv = np.array(rank, int), np.array(inv, int)
    diagnostics = []
    for a, b in inv:
        assert meta[a]['truth'] == meta[b]['truth'] and meta[a]['stratum'] == meta[b]['stratum']
        np.testing.assert_array_equal(x[a, :2217], x[b, :2217])
        pa, pb = [dict(r) for r in (rows[a], rows[b])]
        for rr in (pa, pb):
            for key in ('id', 'episode_id', 'rgb_path', 'rgb_sha256'):
                rr.pop(key, None)
        assert pa == pb, 'Background intervention changed public non-RGB observation'
        assert meta[a]['sampled_witness'] == meta[b]['sampled_witness']
        delta = float(np.mean(np.abs(rgbs[a].astype(float)-rgbs[b].astype(float))))
        assert delta > 0, 'No observable RGB intervention'
        diagnostics.append(dict(ids=[meta[a]['id'], meta[b]['id']], rgb_mean_absolute_change=delta))
    np.savez_compressed(OUT/'new-features.npz', raw=x, ids=[m['id'] for m in meta], intrusion=rank, background=inv)
    write(OUT/'new-metadata.json', meta)
    write(OUT/'new-pair-audit.json', dict(status='PASS', background_pairs=144, intrusion_pairs=144,
        identical_public_sensors=True, identical_native_truth_and_witness=True, rgb_changes=diagnostics,
        target_observability='Identical target+camera; all changed surfaces behind target; no new occluders'))
    write(OUT/'new-data-seal.json', dict(inputs=bindings, outputs={n: sha(OUT/n)
        for n in ('new-features.npz', 'new-metadata.json', 'new-pair-audit.json')}))
    print('PASS new source: all144 texture pairs preserve sensors, truth and witness; RGB changes', flush=True)


def placement(x, y, base, rank, inv):
    objects = {}
    for dev in ('cpu', 'cuda'):
        if dev == 'cuda' and not torch.cuda.is_available():
            continue
        torch.manual_seed(SEED)
        model = RawResidual(x.mean(0), np.maximum(x.std(0), .001)).to(dev)
        objects[dev] = (model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01),
            *[torch.as_tensor(v, device=dev) for v in (x, y, base, rank, inv)])
    def step(dev):
        m, opt, xx, yy, bb, rr, ii = objects[dev]
        opt.zero_grad(set_to_none=True)
        value = loss(m(xx, bb), yy, rr, ii, 'B2'); value.backward(); opt.step()
        return value
    def candidate(dev):
        return BackendCandidate('bg-residual-'+dev, dev, lambda: step(dev),
            lambda value: DeviceObservation(value.device.type, torch.cuda.get_device_name(0) if dev=='cuda' else 'host CPU', str(torch.__version__)),
            torch.cuda.synchronize if dev=='cuda' else lambda: None)
    result = select_backend('batch-tensor', cpu=candidate('cpu'), gpu=candidate('cuda') if 'cuda' in objects else None,
        cpu_reason=None if 'cuda' in objects else 'ACCELERATOR_UNAVAILABLE', warmups=2, repeats=5, record_path=OUT/'backend.json')
    return result['selected_device_type']


def fit_head(x, y, base, rank, inv, arm, dev, path):
    assert not path.exists()
    torch.manual_seed(SEED)
    model = RawResidual(x.mean(0), np.maximum(x.std(0), .001)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    xx, yy, bb, rr, ii = [torch.as_tensor(v, device=dev) for v in (x, y, base, rank, inv)]
    history = []
    start = time.perf_counter()
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        value = loss(model(xx, bb), yy, rr, ii, arm)
        assert torch.isfinite(value)
        value.backward(); opt.step()
        history.append(float(value.detach().cpu()))
    model.cpu().eval()
    torch.save(dict(state_dict=model.state_dict(), arm=arm, history=history, seed=SEED,
        fit_rows=len(x), intrusion_pairs=len(rank), background_pairs=len(inv), seconds=time.perf_counter()-start), path)
    return model


def predict_head(model, x, base):
    with torch.inference_mode():
        return model(torch.as_tensor(x), torch.as_tensor(base)).numpy().astype(float)


def train():
    assert not (OUT/'model-seal.json').exists()
    verify(read(OUT/'training-code-freeze.json')['inputs'])
    verify(read(OUT/'new-data-seal.json')['inputs'])
    for n, h in read(OUT/'new-data-seal.json')['outputs'].items():
        assert sha(OUT/n) == h
    oldx, oldmeta = old_data()
    newz = np.load(OUT/'new-features.npz'); newmeta = read(OUT/'new-metadata.json')
    new_train = np.array([i for i, m in enumerate(newmeta) if m['split']=='train'])
    old_nonanchor = np.array([i for i,m in enumerate(oldmeta) if m['cohort']!='anchor'])
    meta = [oldmeta[i] for i in old_nonanchor]+[newmeta[i] for i in new_train]
    x = np.r_[oldx[old_nonanchor], newz['raw'][new_train]].astype(np.float32)
    y = np.array([m['truth'] for m in meta], np.float32)
    groups = np.array([m['scene_index'] for m in meta])
    clear = np.array([m['stratum']!='boundary' for m in meta])
    oldframes = {}
    for cohort, stem in CAPS.items():
        path=WORK/stem/'source/returned-v1/capture-v1/spec.json'
        assert sha(path)==read(path.with_name('receipt.json'))['spec_sha256']
        oldframes.update({f['id']:f for f in read(path)['frames'] if cohort!='anchor' or f['split']=='train'})
    rp, _, rejected = pairing(oldmeta, oldframes)
    assert len(rp)==672 and not rejected
    rank = np.r_[local_pairs(rp,old_nonanchor), local_pairs(newz['intrusion'],new_train)+1152]
    inv = local_pairs(newz['background'],new_train)+1152
    assert len(rank)==672 and len(inv)==96 and len(meta)==1344
    raw_model=pickle.loads((REP/'raw-final.pkl').read_bytes())
    saved=np.load(REP/'oof-scores.npz')
    base=np.r_[log_odds(saved['raw']),log_odds(raw_model.predict_proba(newz['raw'][new_train])[:,1])]
    dev=placement(x,y,base,rank,inv)
    oof={a:np.full(len(meta),np.nan) for a in ARMS}; receipts=[]
    for k in range(6):
        fi=np.flatnonzero(groups!=k); ri=np.flatnonzero(groups==k)
        assert not {meta[i]['group'] for i in fi}&{meta[i]['group'] for i in ri}
        zz=np.load(OUT/f'outer{k}-inner.npz')
        assert sha(OUT/f'outer{k}-inner.npz')==read(OUT/'crossfit-seal.json')['outputs'][f'outer{k}-inner.npz']
        old_logits={int(i):float(v) for i,v in zip(zz['indices'],zz['logits'])}
        outer=pickle.loads((REP/f'raw-fold{k}.pkl').read_bytes())
        newbase=log_odds(outer.predict_proba(newz['raw'][new_train])[:,1])
        fit_base=np.array([old_logits[int(old_nonanchor[i])] if i<1152 else newbase[i-1152] for i in fi],np.float32)
        report_base=np.array([base[i] if i<1152 else newbase[i-1152] for i in ri],np.float32)
        rr,ii=local_pairs(rank,fi),local_pairs(inv,fi)
        for arm in ARMS:
            path=OUT/f'{arm}-fold{k}.pt'
            model=fit_head(x[fi],y[fi],fit_base,rr,ii,arm,dev,path)
            oof[arm][ri]=predict_head(model,x[ri],report_base)
        receipts.append(dict(fold=k,fit=fi.tolist(),report=ri.tolist(),intrusion=rr.tolist(),background=ii.tolist()))
        print(f'residual fold{k} all3 complete',flush=True)
    selections={}; configs={}
    for arm in ARMS:
        assert np.isfinite(oof[arm]).all()
        chosen,curve=select(oof[arm],y.astype(bool),clear)
        selections[arm]=dict(chosen=chosen,curve=curve)
        path=OUT/f'{arm}-final.pt'
        fit_head(x,y,base,rank,inv,arm,dev,path)
        configs[arm]=dict(path=path.name,sha256=sha(path),threshold=chosen['threshold'])
    write(OUT/'selection.json',selections)
    write(OUT/'training-metadata.json',meta)
    write(OUT/'folds.json',receipts)
    np.savez_compressed(OUT/'training-inputs.npz',x=x,y=y,base=base,intrusion=rank,background=inv)
    np.savez_compressed(OUT/'oof-scores.npz',**oof)
    write(OUT/'model-seal.json',dict(models=configs,backend=dev,old_rows=1152,new_rows=192,
        outputs={n:sha(OUT/n) for n in ('selection.json','training-inputs.npz','training-metadata.json','folds.json','oof-scores.npz')},
        baseline_models={str(p):sha(p) for p in (REP/'raw-final.pkl',REP/'multi-final.pkl')},
        source_freeze_sha256=sha(OUT/'recipe-freeze.json'),code_freeze_sha256=sha(OUT/'training-code-freeze.json')))


def prediction():
    assert not (OUT/'prediction-seal.json').exists()
    configs=read(OUT/'model-seal.json')
    verify(configs['baseline_models'])
    sets={'old':np.load(REP/'report-features.npz')}
    new=np.load(OUT/'new-features.npz'); meta=read(OUT/'new-metadata.json')
    ii=np.array([i for i,m in enumerate(meta) if m['split']=='dev'])
    # Full multi vector is recomputed only from public new report rows for A*.
    rawrows=lines(CAP/'raw.jsonl'); lookup={r['id']:r for r in rawrows}
    full=[]; yaw=0.;episode=None
    for i in ii:
        row=lookup[meta[i]['id']]
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id']
        f=extract(row,cv2.imread(str(CAP/row['rgb_path'])),yaw)
        v=np.r_[f['sensor'],f['geometry']];full.append(v)
        np.testing.assert_array_equal(v[raw_indices(f['sensor_names']+f['geometry_names'])],new['raw'][i])
    sets['new']=dict(ids=new['ids'][ii],raw=new['raw'][ii],multi=np.stack(full))
    rawmodel=pickle.loads((REP/'raw-final.pkl').read_bytes());astar=pickle.loads((REP/'multi-final.pkl').read_bytes())
    saved=read(REP/'model-seal.json')['models']
    outputs={}
    for name,z in sets.items():
        prob={'raw_hgb':rawmodel.predict_proba(z['raw'])[:,1], 'A_star':astar.predict_proba(z['multi'])[:,1]}
        logits={a:log_odds(v).astype(float) for a,v in prob.items()}
        flags={'raw_hgb':prob['raw_hgb']>=saved['raw']['threshold'],'A_star':prob['A_star']>=saved['multi']['threshold']}
        for arm,cfg in configs['models'].items():
            assert sha(OUT/cfg['path'])==cfg['sha256']
            ck=torch.load(OUT/cfg['path'],map_location='cpu',weights_only=False)
            model=RawResidual(np.zeros(2224),np.ones(2224));model.load_state_dict(ck['state_dict']);model.eval()
            logits[arm]=predict_head(model,z['raw'].astype(np.float32),log_odds(prob['raw_hgb']))
            prob[arm]=1/(1+np.exp(-logits[arm]));flags[arm]=logits[arm]>=cfg['threshold']
        predictions=[dict(id=str(identifier),scores={a:float(v[i]) for a,v in logits.items()},
            probabilities={a:float(v[i]) for a,v in prob.items()},flags={a:bool(v[i]) for a,v in flags.items()}) for i,identifier in enumerate(z['ids'])]
        write(OUT/f'{name}-predictions.json',predictions)
        outputs[f'{name}-predictions.json']=sha(OUT/f'{name}-predictions.json')
    write(OUT/'prediction-seal.json',dict(outputs=outputs,model_seal_sha256=sha(OUT/'model-seal.json'),
        report_labels_joined=False,public_feature_inputs={str(p):sha(p) for p in (REP/'report-features.npz',OUT/'new-features.npz')}))


def evaluate():
    assert not (OUT/'completion.json').exists()
    ps=read(OUT/'prediction-seal.json')
    for n,h in ps['outputs'].items():assert sha(OUT/n)==h
    oldcases=read(REP/'cases.json');newmeta=read(OUT/'new-metadata.json');newz=np.load(OUT/'new-features.npz')
    ni=np.array([i for i,m in enumerate(newmeta) if m['split']=='dev'])
    oldspec=read(WORK/'corridor-public-single-20260917/source/returned-v1/capture-v1/spec.json')
    oldpairs,_,_=pairing(oldcases,{f['id']:f for f in oldspec['frames']})
    datasets=dict(old=(oldcases,oldpairs,np.empty((0,2),int)),new=([newmeta[i] for i in ni],local_pairs(newz['intrusion'],ni),local_pairs(newz['background'],ni)))
    result={}
    for name,(cases,pairs,background) in datasets.items():
        pp=read(OUT/f'{name}-predictions.json')
        assert [p['id'] for p in pp]==[c['id'] for c in cases]
        scores={a:np.array([p['scores'][a] for p in pp]) for a in pp[0]['scores']}
        flags={a:np.array([p['flags'][a] for p in pp]) for a in scores}
        if name=='old':
            np.testing.assert_array_equal([p['probabilities']['raw_hgb'] for p in pp],[c['ablation']['scores']['raw'] for c in cases])
            np.testing.assert_array_equal([p['probabilities']['A_star'] for p in pp],[c['ablation']['scores']['multi'] for c in cases])
        # Reuse accounting with multi key standing for raw-HGB reference only.
        ss={'multi':scores['raw_hgb'],**{a:v for a,v in scores.items() if a!='raw_hgb'}}
        ff={'multi':flags['raw_hgb'],**{a:v for a,v in flags.items() if a!='raw_hgb'}}
        methods=summarize(cases,ss,ff,pairs);methods['raw_hgb']=methods.pop('multi')
        for arm in scores:
            methods[arm]['interventions']=pair_report(scores[arm],np.array([p['probabilities'][arm] for p in pp]),flags[arm],pairs,background)
        result[name]=methods
        write(OUT/f'{name}-cases.json',[dict(**c,bg=p) for c,p in zip(cases,pp)])
    decision={}
    for arm in ARMS:
        m=result['old'][arm];baseline=result['old']['raw_hgb']
        no_delay=all(c['baseline'] is None or(c['current'] is not None and c['current']<=c['baseline'])
            for kind in ('core_changes','strict_events_changes') for c in m[kind])
        clear=m['metrics']['clear']
        decision[arm]=dict(old_target_met=clear['TP']>=99 and clear['FP']<=3 and m['changes']['clear']['lost_TP']==0 and no_delay,
            no_lost_or_delayed_raw_event=no_delay)
    write(OUT/'summary.json',dict(authority='CONSUMED288_PLUS_HELD96_TEXTURE_FACTORIAL_DEVELOPMENT',methods=result,
        decisions=decision,models=read(OUT/'model-seal.json')['models'],
        warning='No causal factor disentanglement or natural-scene claim; same generator and four held physical groups'))
    write(OUT/'completion.json',dict(status='PASS',summary_sha256=sha(OUT/'summary.json'),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'),residual_fits=21,inner_hgb_fits=30,
        resources='Local Python process exits; worker capture release separately verified'))
    print(json.dumps({name:{a:m['metrics']['clear'] for a,m in methods.items()} for name,methods in result.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['crossfit','admit','train','prediction','evaluate']);a=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(4):globals()[a.stage]()
