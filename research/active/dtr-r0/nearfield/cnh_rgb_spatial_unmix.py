"""Two-spectrum spatial unmixing of object-blind coarse range distributions.

EXPLORE only: known target cells and frozen clean edges, 52 repeated records.
The public fit accepts RGB support counts and aggregate sensor histograms only.
It neither accepts nor reconstructs native pixelwise sensor/target matches.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_geometry import camera_geometry
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_spatial_association import select_spatial
from cnh_rgb_zone_range import summarize_distribution

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
PARENT = ROOT/'artifacts.local/work/cnh-rgb-association-expanded-20261002'
OUT = ROOT/'artifacts.local/work/cnh-rgb-spatial-unmix-20261002'
WIDTH=.05
BINS=256
MIN_SINGULAR_RATIO=.05
SEEDS=(2026100247,2026100248,2026100249)
CONDITIONS=('clean',)+tuple(f'slab_{s}' for s in SEEDS)+tuple(f'zone_{s}' for s in SEEDS)
ARMS=('unmix_mode','unmix_median','support_shuffle_mode','ordinal','binned_q10')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def support_counts(predicted_z,zones,foreground_prior):
    """Two columns: fixed [0.9*prior,1.1*prior] optical band vs complement.

    Counts use prediction validity alone. They do not share a GT validity mask.
    """
    predicted_z=np.asarray(predicted_z)
    zones=np.asarray(zones)
    if predicted_z.shape!=zones.shape or not np.isfinite(foreground_prior) or foreground_prior<=0:
        raise ValueError('Invalid predicted depth/zone/prior')
    valid=np.isfinite(predicted_z)&(predicted_z>0)&(zones>=0)&(zones<64)
    band=valid&(predicted_z>=.9*foreground_prior)&(predicted_z<=1.1*foreground_prior)
    fg=np.bincount(zones[band],minlength=64)
    total=np.bincount(zones[valid],minlength=64)
    return np.column_stack((fg,total-fg)).astype(float)


def nnls_two_columns(A,B):
    """Exact 2-variable nonnegative least squares, independently per bin.

    Normalize column scale, then enumerate the full feasible stationary point,
    the two one-dimensional boundary optima and the origin. No regularizer,
    learned parameter, target-range prior or ground-truth criterion is used.
    """
    A=np.asarray(A,dtype=float); B=np.asarray(B,dtype=float)
    if A.ndim!=2 or A.shape[1]!=2 or B.ndim!=2 or len(A)!=len(B):
        raise ValueError('Expected Nx2 and NxBins')
    if not np.isfinite(A).all() or not np.isfinite(B).all() or (A<0).any() or (B<0).any():
        raise ValueError('Finite nonnegative observations required')
    norms=np.linalg.norm(A,axis=0)
    diagnostic=dict(rows=len(A),column_norms=norms.tolist(),rank=0,singular_ratio=0.,
                    relative_residual=None,relative_per_zone_residual=None,
                    status='UNIDENTIFIABLE_SUPPORT')
    if len(A)<3 or np.any(norms<=0):
        return None,diagnostic
    U=A/norms
    singular=np.linalg.svd(U,compute_uv=False)
    ratio=float(singular[-1]/singular[0])
    diagnostic.update(rank=int(np.linalg.matrix_rank(U)),singular_values=singular.tolist(),singular_ratio=ratio,
                      normalized_column_cosine=float(U[:,0]@U[:,1]))
    if ratio<MIN_SINGULAR_RATIO:
        return None,diagnostic
    gram=U.T@U
    rhs=U.T@B
    unconstrained=np.linalg.solve(gram,rhs)
    hypotheses=np.zeros((4,2,B.shape[1]))
    hypotheses[0]=unconstrained
    hypotheses[1,0]=np.maximum(rhs[0]/gram[0,0],0)
    hypotheses[2,1]=np.maximum(rhs[1]/gram[1,1],0)
    errors=((U[None,:,:]@hypotheses-B[None,:,:])**2).sum(axis=1)
    errors[0,np.any(unconstrained<0,axis=0)]=np.inf
    choice=errors.argmin(axis=0)
    coefficients=hypotheses[choice,:,np.arange(B.shape[1])].T/norms[:,None]
    fitted=A@coefficients
    residual=fitted-B
    denominator=float(np.linalg.norm(B))
    per_zone_norm=np.linalg.norm(B,axis=1)
    per_zone=np.divide(np.linalg.norm(residual,axis=1),per_zone_norm,
                       out=np.zeros(len(B)),where=per_zone_norm>0)
    diagnostic.update(status='OK',relative_residual=float(np.linalg.norm(residual)/denominator) if denominator else 0.,
                      relative_per_zone_residual=per_zone.tolist(),
                      active_counts={str(i):int((choice==i).sum()) for i in range(4)},
                      foreground_spectral_mass=float(coefficients[0].sum()),
                      background_spectral_mass=float(coefficients[1].sum()))
    return coefficients,diagnostic


def select_unmix(rgb_counts,sensor_histograms,zone_id,shuffle=False):
    """Observation-only API: two 64-row count arrays and a public target cell."""
    A=np.asarray(rgb_counts,float); B=np.asarray(sensor_histograms,float)
    if A.shape!=(64,2) or B.shape!=(64,BINS) or not 0<=zone_id<64:
        raise ValueError('Invalid aggregate observation shapes/zone')
    rr,cc=np.indices((8,8))
    local=((abs(rr-zone_id//8)<=1)&(abs(cc-zone_id%8)<=1)).ravel()
    # Available sensor count is an observed aggregate; its per-pixel mask never
    # gates RGB support. No truth-defined target/foreground enters the fit.
    available=local&(A.sum(1)>0)&(B.sum(1)>0)
    indices=np.flatnonzero(available)
    local_A=A[indices].copy()
    if shuffle:
        local_A=local_A[np.random.default_rng(2026100246).permutation(len(local_A))]
    coefficients,diag=nnls_two_columns(local_A,B[indices])
    diag.update(neighborhood=indices.tolist(),rgb_counts=local_A.tolist(),
                sensor_zone_mass=B[indices].sum(1).tolist(),spatially_shuffled=bool(shuffle))
    result=dict(mode_radial_m=None,median_radial_m=None,diagnostic=diag,foreground_spectrum=None)
    if coefficients is None:
        return result
    foreground=coefficients[0]
    result['foreground_spectrum']=foreground.tolist()
    if not foreground.sum()>0:
        diag['status']='NO_FOREGROUND_SPECTRUM'
        return result
    result['mode_radial_m']=float((np.argmax(foreground)+.5)*WIDTH)
    median_bin=int(np.searchsorted(np.cumsum(foreground),foreground.sum()/2,side='left'))
    result['median_radial_m']=float((median_bin+.5)*WIDTH)
    return result


def histogram_quantile(hist,q):
    hist=np.asarray(hist,float)
    positive=np.flatnonzero(hist>0)
    if not len(positive): return None
    cumulative=np.cumsum(hist)
    target=float(np.clip(q,0,1))*cumulative[-1]
    if q>=1: return float((positive[-1]+1)*WIDTH)
    i=int(np.searchsorted(cumulative,target,side='right'))
    prior=cumulative[i-1] if i else 0.
    return float((i+(target-prior)/hist[i])*WIDTH)


def make_sensor(radial,zones):
    values=np.asarray(radial,dtype=float)
    valid=np.isfinite(values)&(values>0)&(values<WIDTH*BINS)&(zones>=0)
    indexes=zones[valid]*BINS+np.floor(values[valid]/WIDTH).astype(int)
    return np.bincount(indexes,minlength=64*BINS).reshape(64,BINS).astype(float)


def weighted_sensor(B,condition):
    """Fixed simulator stress; selector sees aggregates, not hidden weights.

    slab: range-only .5m slabs share rho across cells. zone: a whole cell has
    its own gain, violating a common cross-zone mixture spectrum assumption.
    Discrete gains .25,1,4 are uncalibrated stress factors, not NIR materials.
    """
    if condition=='clean': return B.copy(),dict(kind='clean')
    kind,seed=condition.split('_'); rng=np.random.default_rng(int(seed))
    if kind=='slab':
        weights=rng.choice([.25,1.,4.],size=(BINS+9)//10).repeat(10)[:BINS]
        return B*weights[None,:],dict(kind=kind,seed=int(seed),bin_weights=weights.tolist())
    weights=rng.choice([.25,1.,4.],size=64)
    return B*weights[:,None],dict(kind=kind,seed=int(seed),zone_weights=weights.tolist())


def self_check():
    from scipy.optimize import nnls
    A=np.column_stack(([10,20,40,70,90],[90,80,60,30,10])).astype(float)
    X=np.zeros((2,BINS)); X[0,40]=.25; X[1,80]=4.
    B=A@X
    recovered,diag=nnls_two_columns(A,B)
    np.testing.assert_allclose(recovered,X,atol=1e-12,rtol=0)
    assert diag['relative_residual']<1e-12
    assert np.argmax(recovered[0])==40
    assert nnls_two_columns(np.tile([20.,80.],(5,1)),B)[0] is None
    # Different component amplitudes are free parameters, unlike pixel-CDF
    # transport; a 64x foreground brightness change preserves its range peak.
    brighter=X.copy();brighter[0]*=64
    np.testing.assert_allclose(nnls_two_columns(A,A@brighter)[0],brighter,atol=1e-11,rtol=0)
    rng=np.random.default_rng(2026100245)
    random_A=rng.uniform(.1,1.,(9,2)); random_B=rng.uniform(0,2.,(9,17))
    ours,diag=nnls_two_columns(random_A,random_B)
    for k in range(17):
        expected,_=nnls(random_A,random_B[:,k])
        np.testing.assert_allclose(ours[:,k],expected,atol=1e-11,rtol=0)
    assert nnls_two_columns(A,np.zeros_like(B))[0].sum()==0
    z=np.repeat(np.arange(64),10).reshape(64,10)
    dp=np.linspace(1,4,640).reshape(64,10)
    np.testing.assert_array_equal(support_counts(dp,z,2),support_counts(dp*3,z,6))
    assert np.isclose(histogram_quantile(np.eye(1,BINS,40)[0],1),2.05)
    return dict(status='PASS',checks=['exact two-spectrum mixing with 16x brightness contrast',
        'shared foreground brightness invariance','collinear support abstention',
        'independent scipy NNLS parity 17 random bins','zero spectrum','global RGB scale support invariance',
        'quantile endpoint last occupied bin'])


def prepare(out):
    if (out/'PLAN.json').exists(): raise FileExistsError(out/'PLAN.json')
    save(out/'PLAN.json',dict(question='Can two-spectrum spatial unmixing retain local range under unknown shared brightness?',
        scope='52 original eligible repeated records, 19 frames,8 scenes; original15+additional37, consumed Development',
        primary='unmix_mode: maximum foreground nonnegative coefficient, nearest-bin tie, 5cm bin center',
        secondary='unmix_median: lower crossing weighted median bin center; reported alongside, never best-of-two',
        design='For each frozen RGB edge foreground optical prior, each zone counts predicted depth within [0.9prior,1.1prior] and complementary valid predictions',
        fit='fixed3x3 neighborhood; exact two-variable NNLS per radial bin using raw counts; normalize design columns only for stability',
        identification='>=3 observed nonempty zones, nonzero columns, normalized singular-value ratio>=0.05; otherwise missing. No residual threshold.',
        controls='fixed neighborhood RGB-support row permutation; original spatial-cosine shape clean baseline; ordinal and binnedq10 on identical stressed counts',
        stresses=dict(conditions=CONDITIONS,seeds=SEEDS,gains=[.25,1.,4.],
                      slab='shared per-.5m range-slab gain across all zones',zone='independent gain per zone'),
        aggregation='Report clean and each fixed stress seed separately. Never select best seed or pool seeds as independent data.',
        metrics='all-case<=2cm with missing failures; valid-only error quantiles; rank, normalized singular ratio, residual, spectral mass; scene-cluster paired CI',
        observation_limit='ideal uniform-pixel first-visible range histograms and arbitrary stress weights, not measured photon data',
        model_limit='foreground and background each share a radial spectrum over3x3; slant/radial variation/multiple surfaces may violate this',
        evidence_limit='GT-derived target cell/floor/clean-edge selection retained; no independent detection, target-ID proof or safety claim',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PARENT/'ordinal-ledger.json',PARENT/'selection.json',
            HERE/'cnh_rgb_clearance_edge.py',HERE/'cnh_rgb_clearance_geometry.py',HERE/'cnh_rgb_spatial_association.py']}))


def paired(rows,arm,baseline):
    scenes=sorted({r['scene'] for r in rows})
    counts=np.array([sum(r['scene']==s for r in rows) for s in scenes])
    def hit(r,a):return r['errors_m'][a] is not None and abs(r['errors_m'][a])<=.02
    diff=np.array([sum(int(hit(r,arm))-int(hit(r,baseline)) for r in rows if r['scene']==s) for s in scenes])
    samples=np.random.default_rng(2026100201).integers(len(scenes),size=(2000,len(scenes)))
    estimates=diff[samples].sum(1)/counts[samples].sum(1)*100
    return dict(delta_pp=float(diff.sum()/counts.sum()*100),ci95_pp=np.percentile(estimates,[2.5,97.5]).tolist())


def run(out):
    plan=read(out/'PLAN.json')
    assert all(sha(ROOT/p)==s for p,s in plan['source_sha256'].items())
    if (out/'result.json').exists(): raise FileExistsError(out/'result.json')
    save(out/'self-check.json',self_check())
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend,Workload,BackendCandidate,DeviceObservation
    select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    cases=read(PARENT/'ordinal-ledger.json')
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    grouped={}
    for row in cases:grouped.setdefault(row['frame_id'],[]).append(row)
    ledger=[];input_hashes={};start=time.perf_counter()
    for frame_id,frame_cases in grouped.items():
        camera=cameras[frame_id];source=manifest[frame_id]
        geom=camera_geometry(ROOT,source,camera);zones=zone_map(camera)
        depth_path=ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/source['depth']
        prediction_path=CACHE/'predictions/native'/f'{frame_id}.npz'
        for path in (depth_path,prediction_path):input_hashes[str(path.relative_to(ROOT))]=sha(path)
        with h5py.File(depth_path,'r') as handle:radial=handle['dataset'][:]
        with np.load(prediction_path,allow_pickle=False) as handle:predicted=handle['native_depth']
        sensor=make_sensor(radial,zones)
        distributions=[summarize_distribution(np.asarray(radial[zones==z],dtype=float)) for z in range(64)]
        coarse=dict(counts=sensor.tolist(),distributions=distributions)
        observations=[]
        for original in frame_cases:
            prediction=original['prediction'];zone=original['zone_id']
            design=support_counts(predicted,zones,prediction['foreground_depth_m'])
            y,xhalf=prediction['edge_pixel'];x=int(np.floor(xhalf))
            optics=float(geom['optical_z_per_radial'][y,x:x+2].mean())
            factor=prediction['side']*prediction['lateral_factor']*optics
            assert abs(factor-original['details']['radial_factor'])<1e-12
            prior=prediction['foreground_depth_m']/optics
            pred_cell=(predicted/geom['optical_z_per_radial'])[zones==zone]
            pred_cell=pred_cell[np.isfinite(pred_cell)&(pred_cell>0)]
            rank=float((np.sum(pred_cell<prior)+.5*np.sum(pred_cell==prior))/len(pred_cell))
            assert rank==original['details']['predicted_rank']
            spatial=select_spatial(predicted,zones,geom['optical_z_per_radial'],coarse,zone,prediction)
            conditions={}
            for condition in CONDITIONS:
                B,weights=weighted_sensor(sensor,condition)
                selected=select_unmix(design,B,zone)
                shuffled=select_unmix(design,B,zone,shuffle=True)
                radii=dict(unmix_mode=selected['mode_radial_m'],unmix_median=selected['median_radial_m'],
                    support_shuffle_mode=shuffled['mode_radial_m'],ordinal=histogram_quantile(B[zone],rank),
                    binned_q10=histogram_quantile(B[zone],.1))
                estimates={a:factor*r-.3 if r is not None else None for a,r in radii.items()}
                if condition=='clean':
                    estimates.update(original_ordinal=original['estimates']['ordinal_transport'],original_q10=original['estimates']['zone_q10'],
                        spatial_shape=spatial['ranges']['spatial']*factor-.3 if spatial['ranges']['spatial'] is not None else None)
                conditions[condition]=dict(estimates=estimates,selection=selected,shuffled_selection=shuffled,weights=weights)
            observations.append(dict(id=original['id'],zone_id=zone,rank=rank,radial_factor=factor,conditions=conditions,
                predicted_zone_counts=design.tolist(),original_spatial_shape=spatial))
        # Persist observation-only predictions before attaching evaluator truth.
        save(out/'observations'/f'{frame_id}.json',dict(frame_id=frame_id,clean_sensor_histograms=sensor.tolist(),cases=observations))
        by_id={r['id']:r for r in frame_cases}
        for observation in observations:
            original=by_id[observation['id']]
            for condition,record in observation['conditions'].items():
                ledger.append(dict(id=original['id'],frame_id=frame_id,scene=original['scene'],
                    original_record=original['original_record'],condition=condition,gt_clearance_m=original['gt_clearance_m'],
                    estimates=record['estimates'],errors_m={a:v-original['gt_clearance_m'] if v is not None else None for a,v in record['estimates'].items()},
                    diagnostic=record['selection']['diagnostic'],shuffled_diagnostic=record['shuffled_selection']['diagnostic']))
        print('unmix',frame_id,len(ledger),'/',52*len(CONDITIONS),flush=True)
    save(out/'case-ledger.json',ledger)
    summaries={};comparisons={};diagnostics={}
    for condition in CONDITIONS:
        rows=[r for r in ledger if r['condition']==condition]
        summaries[condition]={}
        for subset,rs in [('all52',rows),('original15',[r for r in rows if r['original_record']]),
                          ('additional37',[r for r in rows if not r['original_record']])]:
            summaries[condition][subset]={a:summarize(rs,a) for a in rs[0]['estimates']}
        comparisons[condition]={a:paired(rows,'unmix_mode',a) for a in ('ordinal','binned_q10','support_shuffle_mode')}
        diagnostics[condition]=dict(statuses=dict(Counter(r['diagnostic']['status'] for r in rows)),
            rank_counts=dict(Counter(str(r['diagnostic']['rank']) for r in rows)),
            singular_ratio_quantiles=np.percentile([r['diagnostic']['singular_ratio'] for r in rows],[0,50,100]).tolist(),
            residual_quantiles=np.percentile([r['diagnostic']['relative_residual'] for r in rows if r['diagnostic']['relative_residual'] is not None],[0,50,95,100]).tolist())
    result=dict(status='COMPLETE',unique_records=52,frames=len(grouped),scenes=len({r['scene'] for r in ledger}),
        primary='unmix_mode',summaries=summaries,paired=comparisons,diagnostics=diagnostics,seconds=time.perf_counter()-start,
        limits=[plan['observation_limit'],plan['model_limit'],plan['evidence_limit']],source_sha256=input_hashes,
        plan_sha256=sha(out/'PLAN.json'),code_sha256=sha(Path(__file__)))
    save(out/'result.json',result)
    lines=['# RGB空间双分量分解：固定52条相关边缘诊断','',
           '主方法固定为前景谱mode；加权中位数是预先声明的并列对照，不从两者择优。每个压力seed单列，缺失计精度失败。',
           '均匀像素几何直方图与未标定增益假设；不是实测ToF。52条边/19帧/8场景，保留GT已知目标格与干净边缘条件。','',
           '|条件|范围|方法|≤2cm/全部|有效|P50/P95 cm|','|---|---|---|---|---|---|']
    for condition,subsets in summaries.items():
        for subset,stats in subsets.items():
            for arm,s in stats.items():
                q=s['absolute_error_cm_quantiles']
                lines.append(f"|{condition}|{subset}|{arm}|{round(s['within_cm_all']['2']*s['n'])}/{s['n']}|{s['valid']}|{q['p50']:.2f}/{q['p95']:.2f}|")
    lines+=['','列归一化奇异值比小于0.05、少于3格或零列均不可辨识；不因残差大小事后删除样本。',
        '共有前景/背景径向谱是关键模型假设；同物体表面变化、跨格径向变化、不同背景均会破坏它。空间系数可辨识不等于目标物理身份正确。',
        '共同距离slab增益对各bin共享；独立格增益破坏共享谱假设。没有噪声、脉冲、光学配准误差或硬件标定。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({c:{a:round(s['n']*s['within_cm_all']['2']) for a,s in v['all52'].items()} for c,v in summaries.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','check'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    if args.action=='prepare':prepare(args.out)
    elif args.action=='run':run(args.out)
    else:print(json.dumps(self_check()))
