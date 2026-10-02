"""Fixed-mode raw spatial support and unknown-amplitude sensitivity assay.

Changing normalization is a new mechanism, not a retune of the frozen spatial
assay. Uniform-pixel geometry counts and all weighted variants are synthetic.
"""
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from cnh_rgb_spatial_association import ROOT,CACHE,read,save,sha,coarse_observations
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_geometry import camera_geometry,hdf
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_association_expanded import DEFAULT_OUT,paired_difference

OUT=ROOT/'artifacts.local/work/cnh-rgb-spatial-gain-20261002'
STRATEGIES=('normalized','raw','raw_r2comp')


def prepare_public(dp,zones,optics,zone_id,prediction,modes):
    """No truth arrays, target labels or target distances accepted."""
    prior=prediction['foreground_depth_m']
    rgb_counts=np.zeros(64);pixels=np.zeros(64);factors=np.zeros(64)
    for z in range(64):
        cell=zones==z;valid=cell&np.isfinite(dp)&(dp>0)
        pixels[z]=valid.sum()
        rgb_counts[z]=np.count_nonzero(valid&(dp>=prior/1.1)&(dp<prior*1.1))
        if cell.any(): factors[z]=np.median(optics[cell])
    y,xhalf=prediction['edge_pixel'];x=int(np.floor(xhalf))
    edge_optics=float(np.mean(optics[y,x:x+2]))
    rr,cc=np.indices((8,8))
    local=((abs(rr-zone_id//8)<=1)&(abs(cc-zone_id%8)<=1)).ravel()
    centres=(np.arange(256)+.5)*.05
    masks=[]
    for mode in modes:
        expected=mode['radial_m']*edge_optics
        bins=factors[:,None]*centres[None,:]
        masks.append((bins>=expected/1.1)&(bins<expected*1.1))
    return dict(rgb_counts=rgb_counts,pixels=pixels,local=local,factors=factors,masks=np.asarray(masks),
                ranges=np.asarray([m['radial_m'] for m in modes]))


def select_gain(public,counts,strategy):
    counts=np.asarray(counts,float)
    if strategy not in STRATEGIES: raise ValueError(strategy)
    if counts.shape!=(64,256) or np.any(counts<0) or not np.isfinite(counts).all():
        raise ValueError('nonfinite or invalid coarse counts')
    if not len(public['ranges']): return dict(radial=None,index=None,scores=[],evidence='NO_MODE')
    totals=counts.sum(axis=1)
    local=public['local']&(totals>0)&(public['factors']>0)
    rgb=public['rgb_counts'].copy()
    if strategy=='raw_r2comp':
        counts=counts*((np.arange(256)+.5)*.05)[None,:]**2
    mass=(public['masks']*counts[None,:,:]).sum(axis=2)
    if strategy=='normalized':
        rgb=np.divide(rgb,public['pixels'],out=np.zeros(64),where=public['pixels']>0)
        mass=np.divide(mass,totals[None,:],out=np.zeros_like(mass),where=totals[None,:]>0)
    a=rgb[local];b=mass[:,local]
    den=np.linalg.norm(a)*np.linalg.norm(b,axis=1)
    scores=np.divide(b@a,den,out=np.full(len(b),-1.),where=den>0)
    index=int(np.argmax(scores));ordered=np.sort(scores)
    evidence=('SINGLE_MODE' if len(scores)==1 else 'NO_SUPPORT' if scores[index]<0 else
              'TIE' if ordered[-1]-ordered[-2]<=1e-12 else 'UNIQUE')
    return dict(radial=float(public['ranges'][index]),index=index,scores=scores.tolist(),evidence=evidence)


def fixture():
    # A common unknown gain per isolated surface does not alter raw cosine;
    # normalizing by total cell brightness mixes foreground/background gains.
    rgb=np.array([.1,.2,.4,.6,.8,.9])
    fg=rgb*100;bg=(1-rgb)*100
    cosine=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    assert abs(cosine(fg,fg*7)-1)<1e-12
    assert cosine(rgb,(fg*7)/(fg*7+bg))<.99
    # An unknown independent gain in each zone can map either positive shape
    # to any other: no amplitude-only observation identifies the surface.
    gains=bg/fg
    np.testing.assert_allclose(fg*gains,bg)
    return dict(status='PASS',checks=['raw common surface gain invariance',
        'cell-total normalization confounds different surface gains',
        'cell-specific gains can make competing shapes indistinguishable'])


def main():
    assert not (OUT/'result.json').exists(),'Preserve completed run'
    previous=read(DEFAULT_OUT/'ordinal-ledger.json')
    previous.sort(key=lambda r:r['id'])
    files=[Path(__file__),Path(__file__).with_name('cnh_rgb_spatial_association.py'),
           DEFAULT_OUT/'ordinal-ledger.json',CACHE/'manifest.json',CACHE/'observations.json']
    save(OUT/'PLAN.json',dict(
        purpose='New cross-cell raw-count cosine mechanism; fixed52 records/19frames/8scenes, original15 and additional37',
        mechanism='RGB relative-depth band raw pixel-count vector vs each fixed mode range-band raw sensor-count vector over3x3; same1.10band and original modes; tie strongest-order',
        strategies=STRATEGIES,
        controls='Original per-cell total-normalized spatial cosine; raw support cosine; raw support with bin-centre r^2 compensation',
        scenarios='clean uniform pixel counts; inverse-r2/inverse-r4;10 shared seeds common1m radial slab gain {.25,1,4} across all64cells plus inverse-r2;10 same seeds independent cellx1m slab gains plus inverse-r2',
        key_limit='Fixed ORIGINAL geometric candidate modes retained under all pressure scenarios; isolates association only, unrealistically assumes range proposals survive photometric weighting',
        identifiability='Common gain per isolated range/surface cancels raw cosine; mixed range bands and multiple materials need not share gain. Arbitrary cell-specific surface gains make shape intrinsically ambiguous.',
        inference='Selector sees public RGB support, ray factors, coarsecounts, modes only. No targetmask/GT clear/range/identity accepted.',
        metrics='all52 incl missing within2cm,p50,p95;fixed original15/additional37; scene bootstrap2000 seed2026100201; all10 seeds retained',
        scope='Consumed synthetic Development, not calibrated reflectance/photon response, no hardware/collision/generalization claim',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    save(OUT/'fixture.json',fixture())
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend,Workload,BackendCandidate,DeviceObservation
    backend=select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=OUT/'backend.json')
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    groups={}
    for event in previous: groups.setdefault(event['frame_id'],[]).append(event)
    ledger=[];centres=(np.arange(256)+.5)*.05;started=time.perf_counter()
    for frame_id,events in groups.items():
        row=manifest[frame_id];camera=cameras[frame_id]
        geom=camera_geometry(ROOT,row,camera);zones=zone_map(camera)
        radial=hdf(ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth'])
        coarse=coarse_observations(radial,zones)
        del radial
        hist=np.asarray(coarse['counts'],float)
        scenarios=dict(clean=hist,inv_r2=hist/centres[None,:]**2,inv_r4=hist/centres[None,:]**4)
        frame_seed=int(hashlib.sha256(frame_id.encode()).hexdigest()[:8],16)
        for seed in range(10):
            common=np.random.default_rng(frame_seed+seed).choice([.25,1.,4.],size=13)
            cell=np.random.default_rng(frame_seed+seed).choice([.25,1.,4.],size=(64,13))
            slabs=np.minimum(centres.astype(int),12)
            scenarios[f'common_{seed}']=hist*common[slabs][None,:]/centres[None,:]**2
            scenarios[f'cell_{seed}']=hist*cell[:,slabs]/centres[None,:]**2
        with np.load(CACHE/'predictions/native'/f'{frame_id}.npz',allow_pickle=False) as handle: dp=handle['native_depth']
        for old in events:
            # Select using explicit public fields. old contains truth used only
            # after outputs below are complete; never pass event to selector.
            modes=coarse['distributions'][old['zone_id']]['modes']
            public=prepare_public(dp,zones,geom['optical_z_per_radial'],old['zone_id'],old['prediction'],modes)
            outputs={f'{name}__{strategy}':select_gain(public,counts,strategy)
                     for name,counts in scenarios.items() for strategy in STRATEGIES}
            estimates={k:old['details']['radial_factor']*v['radial']-.30 if v['radial'] is not None else None for k,v in outputs.items()}
            estimates.update({a:old['estimates'][a] for a in ('zone_q10','rgb_guided_mode','ordinal_transport')})
            r={k:old[k] for k in ('id','frame_id','scene','zone_id','gt_clearance_m','original_record')}
            r.update(estimates=estimates,errors_m={a:v-old['gt_clearance_m'] if v is not None else None for a,v in estimates.items()},
                     outputs=outputs,modes=modes)
            ledger.append(r)
        print('spatial gain',len(ledger),'/52',flush=True)
    save(OUT/'case-ledger.json',ledger)
    subsets=dict(all=ledger,original15=[r for r in ledger if r['original_record']],additional37=[r for r in ledger if not r['original_record']])
    summaries={s:{a:summarize(rows,a) for a in rows[0]['estimates']} for s,rows in subsets.items()}
    result=dict(status='COMPLETE',n=len(ledger),frames=len(groups),scenes=len({r['scene'] for r in ledger}),
        summaries=summaries,paired={a:paired_difference(ledger,'clean__raw',a) for a in ('clean__normalized','zone_q10','rgb_guided_mode','ordinal_transport')},
        seconds=time.perf_counter()-started,backend=backend,
        sensitivity={s:{kind:{strategy:[round(summaries[s][f'{kind}_{seed}__{strategy}']['within_cm_all']['2']*len(rows)) for seed in range(10)]
                        for strategy in STRATEGIES} for kind in ('common','cell')} for s,rows in subsets.items()})
    save(OUT/'result.json',result)
    old15={r['id']:r for r in read(ROOT/'artifacts.local/work/cnh-rgb-spatial-association-20261002/run-v2/case-ledger.json')}
    for r in subsets['original15']:
        assert abs(r['estimates']['clean__normalized']-old15[r['id']]['estimates']['spatial'])<1e-12
    # Pure r^-2 with exact declared compensation must restore every raw score.
    for r in ledger:
        np.testing.assert_allclose(r['outputs']['inv_r2__raw_r2comp']['scores'],r['outputs']['clean__raw']['scores'],atol=1e-12)
    assert (len(ledger),len(groups),len({r['scene'] for r in ledger}))==(52,19,8)
    save(OUT/'verification.json',dict(status='PASS',checks=['original15 normalizedspatial reproduced exactly',
         '52records19frames8scenes retained','inverse-r2 compensation restores raw scores all52',
         'common-gain invariance and unknown cell gain ambiguity fixture'],fixture=fixture()))
    lines=['# 空间支持的幅度不变性小试','',
           '固定52条边缘、19帧、8场景。空间模式距离完全固定，所有压力只改变聚合计数的幅度；不验证模式在压力下仍可检测。','',
           '|观测/方法|全部≤2cm|原15例|新增37例|全部P50/P95 cm|','|---|---|---|---|---|']
    for scenario in ('clean','inv_r2','inv_r4'):
        for strategy in STRATEGIES:
            arm=f'{scenario}__{strategy}';s=summaries['all'][arm];q=s['absolute_error_cm_quantiles']
            hits=[round(summaries[sub][arm]['within_cm_all']['2']*len(rows)) for sub,rows in subsets.items()]
            lines.append(f'|{arm}|{hits[0]}/52|{hits[1]}/15|{hits[2]}/37|{q["p50"]:.2f}/{q["p95"]:.2f}|')
    lines+=['','十seed幅度压力（每个seed固定；不按结果选seed）：']
    for subset,v in result['sensitivity'].items():
        for kind,strategies in v.items():
            for strategy,hits in strategies.items(): lines.append(f'- {subset}/{kind}/{strategy}: {hits}')
    lines+=['','理论可辨识性：同一孤立表面跨格共同未知幅度会被raw cosine归一化消掉；原每格除总能量会掺入背景幅度。若幅度逐格独立，任意正形状都可通过逐格增益变成另一形状，无法仅从幅度恢复对象。1m距离slab是无身份粗代理，不等于同材质表面，因此common压力不保证严格增益不变；band可能同时跨slab/跨材质。','',
        '原模式先验来自像素等权参考几何，并在压力下保留原距离、数量和顺序；这里有意只问关联选择，不代表弱回波可检测或峰位稳定。每个格像素数不是实测角响应/光子数；r^-2、r^-4、{.25,1,4}只是未标定敏感性。保留全部旧失败与新压力失败，不据此训练/部署或做安全否决。','',
        '验证：原15例normalized完全复现10/15；纯r^-2乘r²逐例恢复raw得分；共同幅度不变性与任意逐格幅度不可辨识夹具通过。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'source').mkdir(exist_ok=True)
    for p in files[:2]:(OUT/'source'/p.name).write_bytes(p.read_bytes())
    save(OUT/'terminal.json',dict(status='complete'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(OUT/'terminal.json',dict(status='failed',error=repr(error)));raise
