"""Finite forward response-shape diagnostic; no training. Frozen source only."""
import argparse, hashlib, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data'
OUT=ROOT/'artifacts.local/work/cnh-attribution-diagnostic-20260929/response'
CALIB=list(range(4)); AUDIT=list(range(32,40))
CONDITIONS=[('nominal',{}),('pulse3',dict(pulse_sigma_bins=3.)),('pulse4',dict(pulse_sigma_bins=4.)),('zero2',dict(range_zero_m=.0750696)),('zero4',dict(range_zero_m=.1501392)),('xtalk15',dict(crosstalk_range_m=.15)),('xtalk30',dict(crosstalk_range_m=.30))]
FEATURE_ARMS=('NONE','SIMPLE')

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def setup():
    sys.path.insert(0,str(DATA/'source'))
    import torch,cnh_track_a_scale_evaluate as se,cnh_track_a_gpu_readout as g,cnh_learned_readout as L,cnh_track_a_readout as R
    from cnh_learned_memory_fusion import causal_ewma
    torch.set_num_threads(2)
    se.sensor_module.FAMILY=json.loads((DATA/'request.json').read_text())['family']
    return torch,se,g,L,R,causal_ewma

def render_unit(u,pilot=False):
    import cnh_track_a_v13_sensor as S
    from cnh_route_sensor import angular_rays,synthesize_response
    from cnh_track_a_geometry import raycast
    S.FAMILY=json.loads((DATA/'request.json').read_text())['family']
    params=S.reference_parameters()[0][6]; directions,weights=angular_rays(16)
    geom=json.loads((DATA/f'geometry/unit{u:02d}/unit{u:02d}.json').read_text())
    with np.load(DATA/f'sensor/unit{u:02d}-mount-10-observations.npz') as z:
        step=2 if int(z['rate'])==10 else 1
        poses=z['world_from_tof'][::step]; configs=z['config'][::step]
    oracle_path=DATA/f'sensor/unit{u:02d}-mount-10-oracle.npz'
    oracle=None
    if oracle_path.exists():
        with np.load(oracle_path) as z: oracle={k:z[k][::step] for k in ('raydistance','raycos','object_id')}
    rows=[]; start=time.monotonic()
    for c in geom['configs'][:1 if pilot else 32]:
        ids=np.flatnonzero(configs==c['config']); triangles=np.concatenate([np.asarray(o['triangles_world']) for o in c['objects']])
        objids=np.concatenate([np.full(len(o['triangles_world']),o['id']) for o in c['objects']]); rhos=np.concatenate([np.full(len(o['triangles_world']),o['rho']) for o in c['objects']])
        rho_map={o['id']:o['rho'] for o in c['objects']}; hist=[]
        for t,i in enumerate(ids):
            if oracle is None:
                hit=raycast(poses[i,:3,3],directions@poses[i,:3,:3].T,triangles,objids,rhos)
                distance=hit['distance']; rho=hit['rho']; cos=hit['cos']
            else:
                oid=oracle['object_id'][i]; distance=np.where(oid>=0,oracle['raydistance'][i],np.inf)
                rho=np.zeros_like(distance)
                for oi,rr in rho_map.items(): rho[oid==oi]=rr
                cos=oracle['raycos'][i]
            seed=S.seed_for(u,c['config'],'noise',t10=2*t,mount=-10,snr=6)
            one=[]
            for name,change in CONDITIONS:
                response=synthesize_response(distance,rho,np.clip(cos,0,1),weights,params=replace(params,**change),seed=seed)
                one.append(S._h3(response['histogram']).astype(np.float32))
            hist.append(one)
        rows.append(np.asarray(hist).transpose(1,0,2,3,4))
    values=np.concatenate(rows,axis=1)
    elapsed=time.monotonic()-start
    if not pilot: np.savez_compressed(OUT/f'render{u:03d}.npz',hist=values)
    return dict(unit=u,configs=len(rows),elapsed_s=elapsed,oracle_reused=oracle is not None,shape=list(values.shape))

def fit_bias(se):
    labels=[]; hist=[]
    for u in CALIB:
        _,records,_=se.unit_records(DATA/'geometry',DATA/'sensor',u,-10,1)
        labels.append(np.concatenate([r['labels'] for r in records]));
        with np.load(OUT/f'render{u:03d}.npz') as z: hist.append(z['hist'])
    labels=np.concatenate(labels); h=np.concatenate(hist,axis=1)
    safe=np.flatnonzero((labels==0).all(1)&(np.tile(np.arange(12),len(labels)//12)>=3))
    if len(safe)<128: raise ValueError('safe calibration budget unavailable')
    idx=safe[np.linspace(0,len(safe)-1,128).astype(int)]
    med=np.median(h[:,idx],axis=1); delta=med-med[0]
    np.savez_compressed(OUT/'bias_fit.npz',delta=delta,indices=idx)
    write(OUT/'calibration.json',dict(safe_available=len(safe),used=128,units=CALIB,threshold_sequences=128,selection='known all-six-query-empty; not unlabeled'))
    return {c[0]:delta[i] for i,c in enumerate(CONDITIONS)}


def infer_unit(u, fit, modules, models, bias):
    torch, se, g, L, R, smooth = modules
    split, records, _ = se.unit_records(DATA/'geometry', DATA/'sensor', u, -10, 1)
    keys = [(c[0], arm) for c in CONDITIONS for arm in FEATURE_ARMS]
    rows = {k: [] for k in ('logits','spread','y','main','w','config','frame')}
    with np.load(OUT/f'render{u:03d}.npz') as zz: rendered=zz['hist']
    for ri,rec in enumerate(records):
        noisy = R.noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        # One common geometry transport for all response/calibration arms.
        pos = torch.as_tensor(np.asarray(noisy), dtype=g.D64, device=g.DEV)
        pairs = [(i,j) for i in range(1,12) for j in range(max(0,i-3),i)]
        rel = torch.stack([torch.linalg.inv(pos[i]) @ pos[j] for i,j in pairs])
        mat = g.transport(rel,1).to(g.DT)
        residual, variance = [], []
        for ci,c in enumerate(CONDITIONS):
            h, a = rendered[ci,ri*12:(ri+1)*12], rec['ambient']
            for arm in FEATURE_ARMS:
                if arm == 'NONE':
                    hh, aa, bb = h, a, bias
                else:
                    hh,aa,bb=h,a,bias+fit[c[0]]
                residual.append(hh-bb)
                variance.append(16*aa[...,None]+np.maximum(bb,0))
        rf = torch.as_tensor(np.asarray(residual), dtype=g.DT, device=g.DEV).reshape(len(keys),12,1024)
        vf = torch.as_tensor(np.asarray(variance), dtype=g.DT, device=g.DEV).reshape(len(keys),12,1024)
        total, var = rf.clone(), vf.clone()
        for k,(i,j) in enumerate(pairs):
            total[:,i] += rf[:,j] @ mat[k].T
            var[:,i] += vf[:,j] @ (mat[k]*mat[k]).T
        z4 = (total/var.clamp_min(1e-9).sqrt()).reshape(len(keys),12,8,8,16)
        z1 = (rf/vf.clamp_min(1e-9).sqrt()).reshape(len(keys),12,8,8,16)
        # Match frozen float16 feature storage round-trip.
        z = torch.stack([z4,z1],dim=2).half().float()
        x = (z.sign()*z.abs().log1p()).reshape(-1,2,8,8,16)
        sup = g.query_weights(torch.as_tensor(np.asarray(rec['tq']), dtype=g.D64, device=g.DEV)) >= .75
        sup = sup.reshape(12,6,8,8,16).repeat(len(keys),1,1,1,1)
        predictions = []
        with torch.no_grad():
            for net in models:
                # Modest batch avoids [batch,query,channel,grid] pooling memory blowup.
                predictions.append(torch.cat([net(x[i:i+48],sup[i:i+48]) for i in range(0,len(x),48)]).cpu().numpy())
        predictions = np.asarray(predictions).reshape(3,len(keys),12,6)
        rows['logits'].append(predictions.mean(axis=0))
        rows['spread'].append(predictions.std(axis=0))
        for k,v in dict(y=rec['labels'],main=rec['main'],w=rec['witness'],
                        config=np.full(12,rec['config']),frame=np.arange(12)).items():
            rows[k].append(v)
    d = {k:np.concatenate(v,axis=1 if k in ('logits','spread') else 0) for k,v in rows.items()}
    d['strata'] = se.strata_of(records)
    d['split'] = np.array(split)
    d['keys'] = np.array([f'{c}|{a}' for c,a in keys])
    d['scores'] = np.stack([smooth(x,d['config'],d['frame'],alpha=.5,window=5) for x in d['logits']])
    np.savez_compressed(OUT/f'unit{u:03d}.npz',**d)
    with np.load(DATA/'predictions'/f'unit{u:03d}.npz') as old:
        diff = float(np.max(np.abs(d['logits'][0]-old['NN'])))
    return diff


def generate():
    start=time.monotonic(); modules=setup(); torch,se,g,L,R,smooth=modules
    receipt=[]
    for u in CALIB+AUDIT:
        if not (OUT/f'render{u:03d}.npz').exists(): receipt.append(render_unit(u))
        print('rendered',u,flush=True)
        if time.monotonic()-start>1200: raise TimeoutError('20min budget')
    write(OUT/'render_receipt.json',receipt)
    fit=fit_bias(se); bias=np.load(DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    expected=['b7ffb180ddde81a6c28e1e7c3922d9653e948a9023ba88fea9d1b8d7a9ee0209','7bafc3201a395f60790dbdfc95b834e5cea43580acd15a61611c972656e33c8d','9e8296de5a0af23ecb752299c4ea1f9f729237b560c3549c6bd472be34e13083']
    models=[]
    for i,h in enumerate(expected):
        path=ROOT/f'artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed{i}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h
        net=L.Readout().to(L.DEV); net.load_state_dict(torch.load(path,map_location=L.DEV,weights_only=True));models.append(net.eval())
    differences={}
    for u in CALIB+AUDIT:
        differences[u]=infer_unit(u,fit,modules,models,bias); print('inferred',u,differences[u],flush=True)
        if time.monotonic()-start>1200: raise TimeoutError('20min budget')
    write(OUT/'generation_receipt.json',dict(elapsed_s=time.monotonic()-start,device=str(L.DEV),nominal_original_logit_max_differences=differences,source='v5 frozen source snapshot',model_hashes=expected))

def evaluate():
    from sklearn.metrics import average_precision_score
    import importlib.util
    spec=importlib.util.spec_from_file_location('eval',Path(__file__).with_name('cnh_v5_evaluate.py')); E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    units={}
    for u in CALIB+AUDIT:
        with np.load(OUT/f'unit{u:03d}.npz') as z: units[u]={k:z[k] for k in z.files}
    old=json.loads((DATA/'analysis/demo_thresholds.json').read_text())['A2_thresholds'];rows=[]
    for j,key in enumerate(units[0]['keys']):
        condition,arm=str(key).split('|');ds={u:dict(d,A0=d['scores'][j],A1=d['scores'][j],A2=d['scores'][j]) for u,d in units.items()}
        for group,ix in E.GROUPS:
            cal=E.pack(ds,CALIB,ix);aud=E.pack(ds,AUDIT,ix)
            threshold=old[group] if arm=='NONE' else E.threshold(E.empty_peaks(cal,'A2'),.1)
            valid=np.isfinite(aud['A2']);stats=E.alert_stats(aud,(aud['A2']>=threshold)&valid)
            aps=[float(average_precision_score(ds[u]['y'][ds[u]['main']][:,ix].ravel(),ds[u]['A2'][ds[u]['main']][:,ix].ravel())) for u in AUDIT]
            rows.append(dict(condition=condition,arm=arm,group=group,threshold=threshold,macro_ap=float(np.mean(aps)),per_unit_ap=aps,valid=int(valid.sum()),query_frames=int(valid.size),main_query_frames=sum(int(ds[u]['main'].sum())*3 for u in AUDIT),stats=stats))
    lookup={(r['condition'],r['arm'],r['group']):r for r in rows}
    for r in rows:
        ref=lookup['nominal',r['arm'],r['group']];r['delta_ap']=r['macro_ap']-ref['macro_ap']
        r['restored']=r['delta_ap']>=-.02 and r['stats']['all']['timely']>=ref['stats']['all']['timely']-.05 and r['stats']['false_pair_rate']<=ref['stats']['false_pair_rate']+.03
    result=dict(scope='finite forward response-shape Development; no training',calib=CALIB,audit=AUDIT,conditions=CONDITIONS,rows=rows)
    write(OUT/'results.json',result)
    for r in rows:
        st=r['stats']; n=st['all'];print(r['condition'],r['arm'],r['group'],round(r['macro_ap'],4),n['timely_count'],n['near'],n['never_count'],st['false_pairs'],st['empty_pairs'],r['restored'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['generate','evaluate','all'],default='all');a=p.parse_args()
    if a.stage in ('generate','all'): generate()
    if a.stage in ('evaluate','all'): evaluate()
