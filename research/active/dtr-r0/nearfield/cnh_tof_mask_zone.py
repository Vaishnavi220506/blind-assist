"""Select a neighboring H3 zone by predicted-mask overlap, then read its range.

No new sensor synthesis: the prior 64-zone clean/stress observations are replayed
unchanged. Selection accepts only public calibration and a cached RGB-depth mask.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from cnh_tof_dither_pilot import prediction_mask
from cnh_rgb_zone_sensor import quadrature
from cnh_tof_subbin_probe import dictionary, fit_peak, self_check as fit_check
from cnh_route_sensor import SensorParameters
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_multizone_calibration import read, save, sha

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT/'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
EDGE = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT = ROOT/'artifacts.local/work/cnh-tof-mask-zone-20261002'
SOURCES = [Path(__file__), *[HERE/name for name in ('cnh_tof_dither_pilot.py',
    'cnh_rgb_zone_sensor.py','cnh_tof_subbin_probe.py','cnh_route_sensor.py',
    'cnh_rgb_clearance_probe.py','cnh_rgb_clearance_geometry.py','cnh_rgb_clearance_edge.py',
    'cnh_rgb_multizone_calibration.py','cnh_rgb_zone_range.py')]]


def mask_overlap(mask, camera):
    """Solid-angle overlap using the exact original H3 16x16 sampling pattern.

    Dummy positive radial values only expose the public ray projection. No GT
    depth, pixel validity from GT, or target identity is used in selection.
    Outside-image samples remain in each full-zone denominator.
    """
    mask=np.asarray(mask)
    if mask.shape != (768,1024) or mask.dtype != np.bool_:
        raise ValueError('Need native Boolean predicted mask')
    samples=quadrature(np.ones(mask.shape),camera)
    yy,xx=samples['sampled_yx'][...,0],samples['sampled_yx'][...,1]
    inside=samples['in_image']; hit=np.zeros(inside.shape,bool)
    hit[inside]=mask[yy[inside],xx[inside]]
    weight=samples['solid_angle_weights']
    weight=weight/weight.sum(-1,keepdims=True)
    return dict(mask_fraction=(weight*hit).sum(-1).reshape(64).tolist(),
                image_fraction=(weight*inside).sum(-1).reshape(64).tolist(),
                projection='unchanged public full-matrix, same-zero-yaw 16x16 H3 quadrature')


def select_zone(coverage, query_zone):
    """Max mask fraction in fixed 3x3; ties nearest query then lower zone ID."""
    if not isinstance(query_zone,int) or not 0<=query_zone<64:
        raise ValueError('Need valid query zone')
    fraction=np.asarray(coverage,float)
    if fraction.shape!=(64,) or not np.isfinite(fraction).all() or np.any((fraction<0)|(fraction>1+1e-12)):
        return dict(status='INVALID_MASK_COVERAGE',zone=None)
    row,col=divmod(query_zone,8)
    candidates=[r*8+c for r in range(max(0,row-1),min(8,row+2)) for c in range(max(0,col-1),min(8,col+2))]
    if max(fraction[candidates])<=0:
        return dict(status='NO_MASK_COVERAGE',zone=None,candidates=candidates)
    distance=lambda z:(z//8-row)**2+(z%8-col)**2
    selected=min(candidates,key=lambda z:(-fraction[z],distance(z),z))
    return dict(status='OK',zone=selected,candidates=candidates,
                selected_fraction=float(fraction[selected]),query_fraction=float(fraction[query_zone]),
                grid_distance_squared=distance(selected))


def read_zone(source, zone):
    """Only the original readout validity, histogram and declared proxy model."""
    if zone is None:
        return dict(status='NO_SELECTED_ZONE',bin_range_m=None,fit_range_m=None,fit=None)
    if not source['valid'][zone]:
        return dict(status='PRIOR_SNR_INVALID',bin_range_m=None,fit_range_m=None,fit=None)
    hist=np.asarray(source['histogram'][zone],float)
    distance=source['distance_m'][zone]
    if hist.shape!=(16,) or not np.isfinite(hist).all() or distance is None or not np.isfinite(distance) or distance<=0:
        return dict(status='INVALID_OBSERVATION',bin_range_m=None,fit_range_m=None,fit=None)
    fitted=fit_peak(hist,dictionary(SensorParameters(**source['params'])),
                    source['ambient'][zone],source['params']['noise_scale'])
    return dict(status='OK',bin_range_m=distance,fit_range_m=fitted['range_m'],fit=fitted)


def estimate(observation):
    """Pure replay boundary: no GT, instance, semantic or clearance labels."""
    chosen=select_zone(observation['mask_fraction'],observation['query_zone'])
    ranges={};decoders={}
    for condition in ('clean','stress'):
        source=observation['electronics'][condition]
        for choice,zone in (('query',observation['query_zone']),('maskbest',chosen['zone'])):
            key=f'{condition}_{choice}'
            decoded=read_zone(source,zone);decoders[key]=decoded
            ranges[key+'_bin']=decoded['bin_range_m'];ranges[key+'_fit']=decoded['fit_range_m']
    factor=observation['radial_edge_factor']
    return dict(selection=chosen,readouts=decoders,radial_ranges_m=ranges,
                estimates={a:factor*r-.30 if r is not None else None for a,r in ranges.items()})


def self_check():
    f=np.zeros(64);f[27]=.5;f[26]=.8;f[18]=.8
    assert select_zone(f,27)['zone']==26
    f[27]=.8;assert select_zone(f,27)['zone']==27
    f=np.zeros(64);f[63]=1.;assert select_zone(f,27)['zone'] is None
    assert select_zone(np.zeros(64),0)['candidates']==[0,1,8,9]
    assert select_zone(np.full(64,np.nan),27)['status']=='INVALID_MASK_COVERAGE'
    matrix=np.diag([.6,.45,-1.])
    full=mask_overlap(np.ones((768,1024),bool),matrix)
    np.testing.assert_allclose(full['mask_fraction'],1.,atol=1e-12)
    blank=mask_overlap(np.zeros((768,1024),bool),matrix)
    assert blank['mask_fraction']==[0.]*64
    half=np.zeros((768,1024),bool);half[:,512:]=True
    overlap=np.asarray(mask_overlap(half,matrix)['mask_fraction']).reshape(8,8)
    np.testing.assert_allclose(overlap[:,:4],0.);np.testing.assert_allclose(overlap[:,4:],1.)
    depth=np.full((20,40),4.);depth[:,20:]=2.
    mask,meta=prediction_mask(depth,dict(edge_pixel=[10,19.5],side=1,foreground_depth_m=2.))
    np.testing.assert_array_equal(mask,depth==2.);assert meta['status']=='OK'
    missing=read_zone(dict(valid=[False]*64),26)
    assert missing['bin_range_m'] is None and missing['fit_range_m'] is None
    return dict(status='PASS',fit=fit_check(),checks=['3x3_membership','coverage_priority','distance_tie',
        'no_GT_fallback_empty','invalid_coverage','full_and_blank_projection','left_right_projection',
        'RGB_connected_mask','original_SNR_invalid_stays_missing'])


def hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in SOURCES+[PRIOR/'case-ledger.json',
            EDGE/'evaluation-input-seal.json',CACHE/'observations.json']}


def main(out):
    if out.exists():raise FileExistsError('Preserve outputs; choose fresh directory')
    out.mkdir(parents=True);started=time.perf_counter()
    plan=dict(scope='EXPLORE same consumed15cases/8scenes, original predicted edges and query cells',
        question='Does choosing an RGB-mask-covered neighboring zone improve existing H3 readouts?',
        mask='unchanged cached DepthPro 25percent relative band, 4-connected, original edge foreground seed',
        projection='existing H3 public full-matrix16x16 quadrature at zero yaw; solid-angle mask/full-zone ratio',
        selector='fixed clipped3x3 query neighborhood, max mask coverage; exact ties squared grid distance then ID; no GT fallback',
        observation='unaltered saved 64zone clean/stress H3 histograms/status/ranges, no new simulation or noise draw',
        readout='original strongest-bin center or imported fit_peak using matching declared proxy pulse; SNR invalid remains missing',
        transfer='unchanged original edge radial factor applied to selected-zone radial readout; common surface range remains an assumption',
        limits=['known query cell and selected clean visible edges; only2/15 near body line; no contact recall claim',
                'DepthPro connected region can merge objects and surfaces; no mask GT used for selection',
                'neighboring same-object surface can have different radial range or slope',
                'perfect camera/sensor projection and existing uncalibrated H3 electronic proxy',
                'known pulse shape and bin zero gifted to fit; not ST scalar firmware or hardware precision',
                'stress uses one retained draw per frame; scene CI only8clusters, not independent confirmation'],
        input_sha256=hashes())
    save(out/'PLAN.json',plan)
    (out/'source').mkdir()
    for p in SOURCES:(out/'source'/p.name).write_bytes(p.read_bytes())
    save(out/'self-check.json',self_check())
    seal=read(EDGE/'evaluation-input-seal.json')['input_sha256']
    assert all(sha(ROOT/p)==h for p,h in seal.items())
    save(out/'input-seal.json',dict(parent_input_sha256=seal,plan_sha256=sha(out/'PLAN.json')))
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-scipy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy-scipy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    old=read(PRIOR/'case-ledger.json');cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    ledger=[]; observations=[]
    for event in old:
        with np.load(CACHE/'predictions/native'/f"{event['frame_id']}.npz",allow_pickle=False) as blob:
            mask,meta=prediction_mask(blob['native_depth'],event['prediction'])
        coverage=mask_overlap(mask,cameras[event['frame_id']])
        observation=dict(id=event['id'],query_zone=event['zone_id'],radial_edge_factor=event['radial_edge_factor'],
                         electronics=event['electronics'],**coverage)
        decoded=estimate(observation)
        estimates={a:event['estimates'][a] for a in ('depthpro','zone_q10')};estimates.update(decoded['estimates'])
        for condition in ('clean','stress'):
            assert estimates[f'{condition}_query_bin']==event['estimates'][f'h3_{condition}']
        row=dict(id=event['id'],scene=event['scene'],frame_id=event['frame_id'],query_zone=event['zone_id'],
                 gt_clearance_m=event['gt_clearance_m'],mask=meta,decoded=decoded,
                 estimates=estimates,errors_m={a:v-event['gt_clearance_m'] if v is not None else None for a,v in estimates.items()})
        ledger.append(row);observations.append(observation)
        print(f'evaluated {len(ledger)}/15; query={event["zone_id"]}, selected={decoded["selection"]["zone"]}',flush=True)
    save(out/'observations.json',observations);save(out/'case-ledger.json',ledger)
    pairs={}
    for condition in ('clean','stress'):
        for kind in ('bin','fit'):
            a=f'{condition}_maskbest_{kind}';b=f'{condition}_query_{kind}'
            success=lambda r,k:r['errors_m'][k] is not None and abs(r['errors_m'][k])<=.02
            pairs[a]=dict(rescued=[r['id'] for r in ledger if success(r,a) and not success(r,b)],
                          lost=[r['id'] for r in ledger if not success(r,a) and success(r,b)])
    result=dict(status='COMPLETE',n=len(ledger),scenes=len({r['scene'] for r in ledger}),
                changed_zone=sum(r['decoded']['selection']['zone']!=r['query_zone'] for r in ledger),
                arms={a:summarize(ledger,a) for a in ledger[0]['estimates']},paired_vs_query=pairs,
                seconds=time.perf_counter()-started,limits=plan['limits'])
    save(out/'result.json',result);verify(out)
    lines=['# RGB mask选择邻格：旧H3观测重放','',
        '固定原15例/8场景/预测边缘。仅用既有DepthPro连通mask覆盖率选择查询格3×3邻域中的测距格；没有新增采集、模型推理或电子观测。选择不接收目标身份或真实净距。','',
        '|条件|≤2cm /15（95%场景区间）|缺失|误差中位/P95 cm|接触误判为畅通/7|擦身误判为接触/8|',
        '|---|---|---|---|---|---|']
    for a,s in result['arms'].items():
        q=s['absolute_error_cm_quantiles'];ci=s['within_cm_all_ci95']['2'];sign=s['sign_counts']
        qs=f"{q['p50']:.2f}/{q['p95']:.2f}" if q else 'UNKNOWN'
        lines.append(f"|{a}|{round(s['within_cm_all']['2']*15)}/15 [{ci[0]:.1%},{ci[1]:.1%}]|{s['missing']}|{qs}|{sign['contact_wrongly_clear']}|{sign['pass_by_wrongly_contact']}|")
    lines+=['',f"选择改变{result['changed_zone']}/15例的格；并列覆盖优先距查询格近，再按zone ID。空覆盖和原SNR无效保留缺失，不用真值回退。",'',
        '相对同readout查询格的成败配对：']
    for a,p in pairs.items():lines.append(f"- {a}：补回{len(p['rescued'])}例、丢失{len(p['lost'])}例；完整ID在result.json。")
    lines+=['','bin是旧H3最强聚合箱中心；fit复用已写单脉冲+非负平坦背景拟合，赠予该代理的pulse shape与bin zero。邻格读数仍投到原RGB边缘射线，同一物体也可能有不同表面/倾斜，不能据此保证真实距离归属。','',
        '此前11/15来自另一种未卷积逆平方几何谱，不能替代这里的电子观测，也不是硬件效果。本轮clean/stress均为未标定模拟，一次保留压力抽样不能估计真实稳健性。','',
        '限制：']+['- '+x for x in plan['limits']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(out/'terminal.json',dict(status='COMPLETE',seconds=time.perf_counter()-started))


def verify(out):
    plan=read(out/'PLAN.json');assert hashes()==plan['input_sha256']
    seal=read(out/'input-seal.json');assert sha(out/'PLAN.json')==seal['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in seal['parent_input_sha256'].items())
    for p in SOURCES:assert sha(out/'source'/p.name)==sha(p)
    observations=read(out/'observations.json');ledger=read(out/'case-ledger.json');result=read(out/'result.json')
    assert len(ledger)==len(observations)==15 and len({r['scene'] for r in ledger})==8
    assert len({r['id'] for r in ledger})==15
    prior={r['id']:r for r in read(PRIOR/'case-ledger.json')}
    for o,r in zip(observations,ledger):
        assert o['id']==r['id'] and o['electronics']==prior[r['id']]['electronics']
        assert estimate(o)==r['decoded']
        for a,v in r['estimates'].items():
            if v is not None:assert abs(v-r['gt_clearance_m']-r['errors_m'][a])<1e-12
    for a in result['arms']:assert summarize(ledger,a)==result['arms'][a]
    save(out/'verification.json',dict(status='PASS',checks=['input_and_snapshot_hashes','same15cases8scenes',
        'unchanged_saved64zone_electronics','original_query_bin_exactly_reproduced',
        'pure_observation_replay_without_truth','summary_and_scene_CI_recomputed','error_arithmetic']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check()))
    elif args.verify:verify(args.out)
    else:
        try:main(args.out)
        except BaseException as error:
            if args.out.exists() and not (args.out/'terminal.json').exists():
                save(args.out/'terminal.json',dict(status='FAILED',error=repr(error)))
            raise
