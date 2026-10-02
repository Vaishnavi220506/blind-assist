"""Development-only local distance compatibility, not surface identity or alarms.

Fixed public angular links receive coarse simulated returns and RGB features.
Native depth is confined to the sensor simulator and supervised/evaluation truth.
No coarse return is declared to belong to an endpoint; this screen does not
propagate a range or fit a connected region to a plane.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import cnh_rgb_visible_query as V
import cnh_rgb_dense_semantics_infer as I
import cnh_rgb_link_features as F

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-rgb-link-compatibility-20261002'
INVENTORY = ROOT/'artifacts.local/work/cnh-rgb-affinity-inventory-20261002'
DATA = ROOT/'artifacts.local/datasets/hypersim-ba-nfo'
SOURCE = ROOT/'artifacts.local/work/ba-nfo-20260919'
RUNS = ROOT/'research/active/dtr-r0/RUNS.md'
RUN_ID = 'CNH_RGB_LINK_COMPATIBILITY_20261002'
SEED = 2026100225
ARMS = ('q10_difference', 'median_difference', 'tof_learned', 'tof_color', 'tof_color_deep')
TOF_ARMS = ARMS[:3]
RGB_ARMS = ARMS[3:]


def sha(path): return I.sha(path)
def read(path): return I.read(path)
def save(path, obj): return I.save(path, obj)


def public_links(shape, matrix):
    """16x16 angular cell centres, horizontal/vertical half/full coarse-cell links."""
    h,w = shape
    angles = np.deg2rad(-22.5+(np.arange(16)+.5)*45/16)
    ax,ay = np.meshgrid(angles, angles)
    rays = np.stack((np.tan(ax),-np.tan(ay),-np.ones_like(ax)),axis=-1).reshape(-1,3)
    homogeneous = rays@np.linalg.inv(np.asarray(matrix)).T
    uv = homogeneous[:,:2]/homogeneous[:,2,None]
    xy = np.stack(((uv[:,0]+1)*w/2-.5, (1-uv[:,1])*h/2-.5),axis=1)
    inside = (homogeneous[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
    nodes = np.rint(xy[:,::-1]).astype(int)
    nodes[:,0] = np.clip(nodes[:,0],0,h-1); nodes[:,1] = np.clip(nodes[:,1],0,w-1)
    pairs=[];steps=[]
    for step in (1,2):
        for y in range(16):
            for x in range(16):
                for dy,dx in ((0,step),(step,0)):
                    if y+dy<16 and x+dx<16:
                        pairs.append((16*y+x,16*(y+dy)+x+dx));steps.append(step)
    pairs=np.asarray(pairs,np.int16)
    angular=np.stack((ax.ravel(),ay.ravel()),axis=1)
    features=np.column_stack((angular[pairs[:,0]],angular[pairs[:,1]],steps))
    return dict(nodes=nodes,pairs=pairs,steps=np.array(steps,np.uint8),public_features=features,
                observable=inside[pairs].all(1),angular=angular)


def prepare():
    assert not (OUT/'PLAN.json').exists()
    rows=read(INVENTORY/'pilot-proposal.json')
    cameras={r['scene_name']:r for r in csv.DictReader((SOURCE/'metadata_camera_parameters.csv').open())}
    roles=read(SOURCE/'hypersim-selection.json')['families']
    seen={}
    for r in rows:
        assert roles[r['family']]=='train' and r['split']=='train'
        assert seen.setdefault(r['family'],r['proposed_role'])==r['proposed_role']
        r['camera_matrix']=[[float(cameras[r['scene']][f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)]
        for kind in ('rgb','depth'): assert sha(r[kind+'_path'])==r[kind+'_sha256']
    assert len(rows)==154 and [sum(r['proposed_role']==s for r in rows) for s in ('train','cal','eval')]==[90,32,32]
    observations=[{k:r[k] for k in ('id','rgb_path','rgb_sha256','camera_matrix')} for r in rows]
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'observations.json',observations)
    sources=[Path(__file__),Path(F.__file__),Path(I.__file__),Path(V.__file__)]
    row=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; repository-train consumed Development154 frames, family-disjoint train90/cal32/eval32 within this run; exclude old NFO500/RGB96/pair scenes; fixed16x16 public angular nodes,928 links/frame at half/full coarse-cell spacing; same HGB80x15 leaves, coarseq10/median neighborhoods vs +RGB color vs +frozen SegFormer4stage cosine; no alarm propagation/download | NOT_RUN; cal each arm >=80% near-compatible retention, select lowest incompatible acceptance separately ToF/RGB; eval <=5cm retained and >=10cm misaccepted with 5–10cm/UNKNOWN and all928 denominators; 1000 family bootstrap | Continue local association only if near incompatible acceptance drops>=3pp and compatible retention loss<=2pp, cal/eval good+bad each>=50; otherwise stop this fixed recipe, no eval retuning; coarse simulated sensor not hardware/M3; no collision gain claim | `artifacts.local/work/cnh-rgb-link-compatibility-20261002/REPORT.md` |')
    body=RUNS.read_text(encoding='utf-8'); assert f'| {RUN_ID} |' not in body
    RUNS.write_text(body.rstrip()+'\n'+row+'\n',encoding='utf-8')
    (OUT/'prerun-row.txt').write_text(row+'\n',encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,at=datetime.now(timezone.utc).isoformat(),inputs=rows,
        observations_sha256=sha(OUT/'observations.json'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        inventory_sha256=sha(INVENTORY/'pilot-proposal.json'),model_sha256=I.MODEL_SHA,
        rules=dict(links='928 fixed links before depth; unavailable native FOV endpoints abstain and remain in all-link denominator',
            geometry='nearest native pixel centre to fixed public ray; native radial reference converted to optical Z',
            sensor='all valid native pixels per8x8 angular zone; r^-2 empirical q10 and median; no object/query truth; endpoint does NOT own its zone return',
            truth='compatible abs(deltaZ)<=.05m; incompatible>=.10m; intermediate strictly(.05,.10), invalid/nonpositive/outside native FOV UNKNOWN',
            near='evaluator-only at least one valid endpoint Z in[.6,2.1); target corridor stricter stratum separately',
            training='train only all observable valid good/bad links, intermediate excluded; class-balanced sample weights; no near/GT sampling; no frame/scene IDs in features',
            learner=dict(max_iter=80,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=SEED),
            calibration='cal only, for each arm largest threshold retaining >=80% near compatible links; ties accepted; minimum incompatible acceptance chooses ToF/RGB arms separately in declared order',
            gate='cal and eval near good and bad each>=50; selectedRGB minus selectedToF incompatible acceptance<=-3pp AND compatible retention>=-2pp; no CI threshold; no next solver when fail',
            bootstrap='1000 resamples of all8 eval families including zero support, paired conditional fixed model/threshold; zero denominator samples counted',
            limitation='proxy link compatibility only, not same surface, return ownership, depth reconstruction, M3, three-level collision performance or independent confirmation'),
        role='Prior BA-NFO train pool; all39 families previously used by RGB96; scene-separated, consumed Development',preregistration=row))
    print('PREPARED154/928 links per frame',flush=True)


def validate():
    plan=read(OUT/'PLAN.json')
    for p,h in plan['source_sha256'].items(): assert sha(ROOT/p)==h,p
    assert sha(OUT/'observations.json')==plan['observations_sha256']
    return plan


def sensor_features(geometry, returns, links):
    """Only 2x64 sensor values, public native rays and endpoints enter this function."""
    nodes,pairs=links['nodes'],links['pairs']
    node_zone=geometry['zone_id'][nodes[:,0],nodes[:,1]]
    factors=geometry['radial_factor'][nodes[:,0],nodes[:,1]]
    zones=np.maximum(node_zone,0)
    parts=[links['public_features']]
    difference=[]
    for values in returns:
        node_z=values[zones]/factors
        node_z[node_zone<0]=np.nan
        diff=np.abs(node_z[pairs[:,0]]-node_z[pairs[:,1]])
        difference.append(diff)
        parts.append(np.column_stack((node_z[pairs[:,0]],node_z[pairs[:,1]],diff)))
        neighborhoods=[]
        for delta_y in (-1,0,1):
            for delta_x in (-1,0,1):
                y,x=zones//8+delta_y,zones%8+delta_x
                valid=(y>=0)&(y<8)&(x>=0)&(x<8)&(node_zone>=0)
                a=values[np.clip(8*y+x,0,63)].copy();a[~valid]=np.nan
                neighborhoods.extend((a[pairs[:,0]],a[pairs[:,1]]))
        parts.append(np.column_stack(neighborhoods))
    return np.column_stack(parts).astype(np.float32), np.stack(difference,axis=1)


def simulate_and_truth(row):
    """Depth access is confined here; raw endpoints never enter sensor_features."""
    import h5py
    assert sha(row['depth_path'])==row['depth_sha256']
    with h5py.File(row['depth_path'],'r') as f: radial=f['dataset'][:].astype(np.float64)
    assert radial.shape==(768,1024)
    geometry=V.ray_geometry(radial.shape,row['camera_matrix'])
    links=public_links(radial.shape,row['camera_matrix'])
    returns=np.stack([V.coarse_returns(radial,geometry,q)['zone_return_radial'].reshape(64) for q in (.1,.5)])
    features,difference=sensor_features(geometry,returns,links)
    yy,xx=links['nodes'].T;pairs=links['pairs']
    z=radial[yy,xx]/geometry['radial_factor'][yy,xx]
    endpoint_z=z[pairs];valid=links['observable']&np.isfinite(endpoint_z).all(1)&(endpoint_z>0).all(1)
    error=np.abs(endpoint_z[:,0]-endpoint_z[:,1])
    label=np.full(len(pairs),-1,np.int8);label[valid]=1
    label[valid&(error<=.05)]=0;label[valid&(error>=.10)]=2
    near=((endpoint_z>=.6)&(endpoint_z<2.1)).any(1)&valid
    x=z*geometry['fx'][yy,xx];y=z*geometry['fy'][yy,xx]
    corridor=((z>=.6)&(z<2.1)&(np.abs(x)<.3)&(y>=-.2)&(y<.9))[pairs].any(1)&valid
    return dict(tof=features,difference=difference,returns=returns,label=label,near=near,corridor=corridor,
        error=error,endpoint_z=endpoint_z,observable=links['observable'],steps=links['steps'])


def extract():
    from PIL import Image
    plan=validate();assert not (OUT/'features-result.json').exists()
    dest=OUT/'features';dest.mkdir(exist_ok=True)
    observations=read(OUT/'observations.json')
    loaded=I.load(); records={};started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={r['id']:pool.submit(simulate_and_truth,r) for r in plan['inputs']}
        for index,row in enumerate(observations):
            path=dest/(row['id']+'.npz');assert not path.exists()
            assert set(row)=={'id','rgb_path','rgb_sha256','camera_matrix'}
            assert sha(row['rgb_path'])==row['rgb_sha256']
            with Image.open(row['rgb_path']) as im: rgb=np.array(im.convert('RGB'))
            assert rgb.shape==(768,1024,3)
            links=public_links(rgb.shape[:2],row['camera_matrix'])
            color=F.rgb_features(rgb,links['nodes'],links['pairs'])
            deep=F.deep_features(Image.fromarray(rgb),links['nodes'],links['pairs'],loaded)
            packet=futures[row['id']].result()
            assert color.shape[0]==deep.shape[0]==928 and np.isfinite(color).all() and np.isfinite(deep).all()
            with path.open('xb') as f:np.savez_compressed(f,**packet,color=color,deep=deep)
            records[row['id']]=dict(sha256=sha(path),links=928,color_features=color.shape[1],deep_features=deep.shape[1])
            if (index+1)%10==0:print(f'features {index+1}/154',flush=True)
    save(OUT/'features-result.json',dict(status='COMPLETE',records=records,seconds=time.perf_counter()-started,
        model_loader_metadata=loaded[3],operation='four encoder hidden stages; endpoint cosine; no logits interpolation or argmax',
        color_names=F.RGB_COLUMNS,deep_names=F.DEEP_COLUMNS))
    print('FEATURES COMPLETE154',flush=True)


def matrix(packet,arm):
    values=[packet['tof']]
    if arm in RGB_ARMS:values.append(packet['color'])
    if arm=='tof_color_deep':values.append(packet['deep'])
    return np.column_stack(values)


def fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import sklearn
    import pickle
    plan=validate();assert not (OUT/'models-result.json').exists()
    receipts=read(OUT/'features-result.json');assert receipts['status']=='COMPLETE'
    packets={}
    for row in plan['inputs']:
        if row['proposed_role']!='train':continue
        path=OUT/'features'/(row['id']+'.npz');assert sha(path)==receipts['records'][row['id']]['sha256']
        with np.load(path) as f:packets[row['id']]={k:f[k] for k in f.files}
    y=np.concatenate([p['label'] for p in packets.values()]);take=(y==0)|(y==2)
    truth=(y[take]==0).astype(np.uint8);counts=np.bincount(truth,minlength=2)
    assert np.all(counts>0)
    weights=len(truth)/(2*counts[truth]);records={};started=time.perf_counter()
    with threadpool_limits(limits=4):
        for arm in ARMS[2:]:
            X=np.concatenate([matrix(p,arm) for p in packets.values()])[take]
            model=HistGradientBoostingClassifier(**plan['rules']['learner']).fit(X,truth,sample_weight=weights)
            path=OUT/(arm+'.pkl')
            with path.open('xb') as f:pickle.dump(model,f)
            records[arm]=dict(sha256=sha(path),features=X.shape[1],iterations=model.n_iter_)
            print('FIT',arm,X.shape,flush=True)
    save(OUT/'models-result.json',dict(status='COMPLETE',sklearn=sklearn.__version__,records=records,
        training_links=len(y),used_links=int(take.sum()),compatible=int(counts[1]),incompatible=int(counts[0]),
        intermediate=int((y==1).sum()),unknown=int((y<0).sum()),seconds=time.perf_counter()-started))


def threshold(scores,labels,near):
    good=scores[near&(labels==0)]
    if not len(good):return None
    return float(np.sort(good)[len(good)-math.ceil(.8*len(good))])


def tally(packet,accepted,subset):
    label=packet['label'];take=subset
    out=dict(all_links=int(take.sum()),accepted=int((take&accepted).sum()))
    for key,value in (('compatible',0),('intermediate',1),('incompatible',2),('unknown',-1)):
        mask=take&(label==value);n=int(mask.sum());yes=int((mask&accepted).sum())
        out[key]=dict(n=n,accepted=yes,rate=yes/n if n else None)
    return out


def evaluate():
    import pickle
    from threadpoolctl import threadpool_limits
    plan=validate();assert not (OUT/'result.json').exists()
    receipts=read(OUT/'features-result.json');models_receipt=read(OUT/'models-result.json')
    models={}
    for arm in ARMS[2:]:
        path=OUT/(arm+'.pkl');assert sha(path)==models_receipt['records'][arm]['sha256']
        with path.open('rb') as f:models[arm]=pickle.load(f)
    frames=[]
    with threadpool_limits(limits=4):
        for row in plan['inputs']:
            if row['proposed_role']=='train':continue
            path=OUT/'features'/(row['id']+'.npz');assert sha(path)==receipts['records'][row['id']]['sha256']
            with np.load(path) as f:packet={k:f[k] for k in f.files}
            scores={ARMS[i]:-np.nan_to_num(packet['difference'][:,i],nan=1e6) for i in range(2)}
            for arm,model in models.items(): scores[arm]=model.predict_proba(matrix(packet,arm))[:,1]
            for value in scores.values():value[~packet['observable']]=-1e9
            frames.append(dict(row=row,packet=packet,scores=scores))
    cal=[f for f in frames if f['row']['proposed_role']=='cal']
    labels=np.concatenate([f['packet']['label'] for f in cal]);near=np.concatenate([f['packet']['near'] for f in cal])
    thresholds={a:threshold(np.concatenate([f['scores'][a] for f in cal]),labels,near) for a in ARMS}
    save(OUT/'thresholds.json',dict(thresholds=thresholds,source='32cal frames only;80%compatible retention with tie acceptance'))
    ledger=[];score_path=OUT/'scores.npz';score_arrays={}
    for f in frames:
        p=f['packet'];strata=dict(all=np.ones(928,bool),near=p['near'],corridor=p['corridor'],
            near_half=p['near']&(p['steps']==1),near_full=p['near']&(p['steps']==2))
        stats={}
        for arm in ARMS:
            score_arrays[f['row']['id']+'__'+arm]=f['scores'][arm]
            accepted=f['scores'][arm]>=thresholds[arm] if thresholds[arm] is not None else np.zeros(928,bool)
            stats[arm]={s:tally(p,accepted,m) for s,m in strata.items()}
        ledger.append(dict(id=f['row']['id'],family=f['row']['family'],role=f['row']['proposed_role'],stats=stats))
    with score_path.open('xb') as f:np.savez_compressed(f,**score_arrays)
    save(OUT/'frame-ledger.json',ledger)
    def aggregate(items):
        result={}
        for arm in ARMS:
            result[arm]={}
            for stratum in ('all','near','corridor','near_half','near_full'):
                groups=[r['stats'][arm][stratum] for r in items]
                result[arm][stratum]={k:sum(g[k] for g in groups) for k in ('all_links','accepted')}
                for category in ('compatible','intermediate','incompatible','unknown'):
                    n=sum(g[category]['n'] for g in groups);a=sum(g[category]['accepted'] for g in groups)
                    result[arm][stratum][category]=dict(n=n,accepted=a,rate=a/n if n else None)
        return result
    summaries={role:aggregate([r for r in ledger if r['role']==role]) for role in ('cal','eval')}
    def choose(arms):
        return min(arms,key=lambda a:(summaries['cal'][a]['near']['incompatible']['rate'] if summaries['cal'][a]['near']['incompatible']['rate'] is not None else 2,ARMS.index(a)))
    best_tof,best_rgb=choose(TOF_ARMS),choose(RGB_ARMS)
    differences={}
    for stratum in ('near','corridor','near_half','near_full'):
        differences[stratum]={}
        for category in ('compatible','incompatible'):
            a=summaries['eval'][best_rgb][stratum][category]['rate'];b=summaries['eval'][best_tof][stratum][category]['rate']
            differences[stratum][category+'_pp']=100*(a-b) if a is not None and b is not None else None
    families=sorted({r['family'] for r in ledger if r['role']=='eval'})
    family={f:aggregate([r for r in ledger if r['role']=='eval' and r['family']==f]) for f in families}
    draws=np.random.default_rng(SEED).multinomial(len(families),np.ones(len(families))/len(families),size=1000)
    for category in ('compatible','incompatible'):
        den=draws@np.array([family[f][best_tof]['near'][category]['n'] for f in families])
        net=draws@np.array([family[f][best_rgb]['near'][category]['accepted']-family[f][best_tof]['near'][category]['accepted'] for f in families])
        valid=den>0;delta=100*net[valid]/den[valid]
        differences['near'][category+'_ci95_pp']=np.percentile(delta,[2.5,97.5]).tolist() if len(delta) else None
        differences['near'][category+'_bootstrap_valid']=int(valid.sum())
    supported=all(summaries[role][best_tof]['near'][category]['n']>=50 for role in ('cal','eval') for category in ('compatible','incompatible'))
    diff=differences['near']
    passed=supported and diff['incompatible_pp']<=-3 and diff['compatible_pp']>=-2
    verdict='LOCAL_LINK_OPPORTUNITY_DEV' if passed else ('STOP_FIXED_LINK_RECIPE' if supported else 'NOT_EVALUABLE_LOW_SUPPORT')
    paired={}
    for name,value in (('compatible',0),('intermediate',1),('incompatible',2)):
        counts=dict(both=0,rgb_only=0,tof_only=0,neither=0)
        for f in frames:
            if f['row']['proposed_role']!='eval':continue
            mask=f['packet']['near']&(f['packet']['label']==value)
            a=f['scores'][best_tof]>=thresholds[best_tof] if thresholds[best_tof] is not None else np.zeros(928,bool)
            b=f['scores'][best_rgb]>=thresholds[best_rgb] if thresholds[best_rgb] is not None else np.zeros(928,bool)
            for k,m in dict(both=a&b,rgb_only=b&~a,tof_only=a&~b,neither=~a&~b).items():counts[k]+=int((mask&m).sum())
        paired[name]=counts
    save(OUT/'result.json',dict(verdict=verdict,role=plan['role'],frames=dict(train=90,cal=32,eval=32),
        links_per_frame=928,thresholds=thresholds,selected_tof=best_tof,selected_rgb=best_rgb,
        summaries=summaries,differences=differences,family_summaries=family,support_gate=supported,
        scores_sha256=sha(score_path),paired_near=paired,limit=plan['rules']['limitation']))
    print(json.dumps(dict(verdict=verdict,tof=best_tof,rgb=best_rgb,near=differences['near'])),flush=True)


def selftest():
    matrix=np.diag([.6,.6,-1.]);links=public_links((768,1024),matrix)
    assert links['observable'].all() and links['pairs'].shape==(928,2)
    assert np.bincount(links['steps']).tolist()==[0,480,448]
    g=V.ray_geometry((768,1024),matrix)
    y,x=links['nodes'].T
    np.testing.assert_allclose(np.arctan(g['fx'][y,x]),links['angular'][:,0],atol=.002)
    np.testing.assert_allclose(np.arctan(g['fy'][y,x]),links['angular'][:,1],atol=.002)
    t=threshold(np.array([.9,.8,.7,.6,.5]),np.zeros(5),np.ones(5,bool));assert t==.6
    p=dict(label=np.array([0,1,2,-1]));q=tally(p,np.ones(4,bool),np.ones(4,bool))
    assert sum(q[k]['n'] for k in ('compatible','intermediate','incompatible','unknown'))==4
    print('PASS_PUBLIC_LINKS_THRESHOLD_DENOMINATORS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selftest','prepare','extract','fit','evaluate'))
    globals()[parser.parse_args().action]()
