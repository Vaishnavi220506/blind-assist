"""Public-ray optical-Z resampling before fixed two-component spatial NNLS.

Consumes only the earlier frozen RGB support counts and object-blind radial
histograms. No native GT depth, label, per-pixel reference mapping or new RGB
candidate enters the selector. Geometry is an approximate per-cell median.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_spatial_unmix import read,save,sha,select_unmix,paired,WIDTH,BINS

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'artifacts.local/work/cnh-rgb-spatial-unmix-20261002'
EXPANDED=ROOT/'artifacts.local/work/cnh-rgb-association-expanded-20261002'
CACHE=ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT=ROOT/'artifacts.local/work/cnh-rgb-optical-unmix-20261002'


def public_cell_factors(camera_matrix,shape=(768,1024)):
    """Only calibrated rays; median and within-cell angular spread, no depth."""
    height,width=shape
    yy,xx=np.indices(shape,dtype=float)
    uv=np.stack(((xx+.5)*2/width-1,1-(yy+.5)*2/height,np.ones(shape)),axis=-1)
    rays=uv@np.asarray(camera_matrix,dtype=float).T
    factors=-rays[...,2]/np.linalg.norm(rays,axis=-1)
    zones=zone_map(camera_matrix,shape)
    median=np.zeros(64);diagnostics=[]
    for zone in range(64):
        values=factors[zones==zone]
        if not values.size:
            diagnostics.append(dict(zone=zone,n_pixels=0,median=None,max_absolute_deviation=None))
            continue
        median[zone]=float(np.median(values))
        diagnostics.append(dict(zone=zone,n_pixels=len(values),median=median[zone],
            minimum=float(values.min()),maximum=float(values.max()),
            q05=float(np.percentile(values,5)),q95=float(np.percentile(values,95)),
            max_absolute_deviation=float(abs(values-median[zone]).max()),
            angle_only_bound_at_2m_cm=float(2*abs(values-median[zone]).max()*100)))
    return median,diagnostics


def rebin_optical(histograms,cell_factors):
    """Deposit each transformed radial-bin center onto adjacent Z centers.

    This linear barycentric transport conserves nonnegative mass per cell and
    preserves the first moment except explicit endpoint clamping. It assumes
    each radial bin's mass sits at its center, and uses one median public ray
    factor per cell. It cannot recover the missing within-cell correspondence.
    """
    radial=np.asarray(histograms,dtype=float)
    factors=np.asarray(cell_factors,dtype=float)
    if radial.shape!=(64,BINS) or factors.shape!=(64,):raise ValueError('Invalid aggregate geometry shapes')
    if not np.isfinite(radial).all() or (radial<0).any():raise ValueError('Invalid histogram')
    active=radial.sum(1)>0
    if not np.isfinite(factors).all() or np.any(factors[active]<=0) or np.any(factors[active]>1):
        raise ValueError('Need positive optical/radial factor<=1 for observed cells')
    centers=(np.arange(BINS)+.5)*WIDTH
    transformed=factors[:,None]*centers[None,:]
    index=transformed/WIDTH-.5
    clipped=np.clip(index,0,BINS-1)
    lower=np.floor(clipped).astype(int)
    upper=np.minimum(lower+1,BINS-1)
    fraction=clipped-lower
    optical=np.zeros_like(radial)
    for zone in range(64):
        np.add.at(optical[zone],lower[zone],radial[zone]*(1-fraction[zone]))
        np.add.at(optical[zone],upper[zone],radial[zone]*fraction[zone])
    input_mass=radial.sum(1);output_mass=optical.sum(1)
    clamped=(index<0)|(index>BINS-1)
    moment_error=(optical*centers).sum(1)-(radial*transformed).sum(1)
    expected_clip_bias=(radial*(np.clip(transformed,centers[0],centers[-1])-transformed)).sum(1)
    np.testing.assert_allclose(output_mass,input_mass,atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(moment_error,expected_clip_bias,atol=1e-8,rtol=1e-10)
    return optical,dict(mass_error_max=float(abs(output_mass-input_mass).max()),
        clamped_mass_per_zone=(radial*clamped).sum(1).tolist(),
        first_moment_error_per_zone=moment_error.tolist(),
        expected_clamp_moment_bias_per_zone=expected_clip_bias.tolist())


def select_optical(rgb_counts,radial_histograms,cell_factors,zone_id,shuffle=False):
    """Independent observation interface with only public aggregate geometry."""
    optical,transport=rebin_optical(radial_histograms,cell_factors)
    # The inherited NNLS uses only bin counts, so after rebinning its numerical
    # range outputs are optical-Z. Rename them before exposing the result.
    selection=select_unmix(rgb_counts,optical,zone_id,shuffle=shuffle)
    return dict(mode_optical_z_m=selection.pop('mode_radial_m'),
        median_optical_z_m=selection.pop('median_radial_m'),
        spectral_axis='optical_z_m',resampling=transport,**selection)


def self_check():
    counts=np.zeros((64,BINS));counts[0,40]=17;counts[1,0]=3;counts[2,200]=29
    factors=np.ones(64);factors[:3]=[.91,.94,.85]
    transformed,diag=rebin_optical(counts,factors)
    np.testing.assert_allclose(transformed.sum(1),counts.sum(1),atol=1e-12,rtol=0)
    assert np.count_nonzero(transformed[0])==2
    assert diag['clamped_mass_per_zone'][1]==3
    centres=(np.arange(BINS)+.5)*WIDTH
    np.testing.assert_allclose((transformed[0]*centres).sum()/17,centres[40]*.91,atol=1e-12,rtol=0)
    identity,_=rebin_optical(counts,np.ones(64))
    np.testing.assert_allclose(identity,counts,atol=1e-11,rtol=0)
    rng=np.random.default_rng(2026100254)
    random=rng.uniform(0,1,(64,BINS));factors=rng.uniform(.8,1,64)
    output,_=rebin_optical(random,factors)
    assert np.all(output>=0)
    # Independent overlap-free scalar deposition verifies the vector transport.
    expected=np.zeros_like(output)
    for zone in range(64):
        for k,mass in enumerate(random[zone]):
            coordinate=np.clip((k+.5)*factors[zone]-.5,0,BINS-1)
            low=int(coordinate);fraction=coordinate-low
            expected[zone,low]+=mass*(1-fraction)
            expected[zone,min(low+1,BINS-1)]+=mass*fraction
    np.testing.assert_allclose(output,expected,atol=1e-12,rtol=0)
    med,stats=public_cell_factors(np.diag([.6,.45,-1.]),(48,64))
    assert np.all((med>0)&(med<=1))
    assert all(s['minimum']<=s['median']<=s['maximum'] for s in stats)
    return dict(status='PASS',checks=['mass conservation','linear barycentric first moment',
        'explicit endpoint clamp bias','identity geometry','nonnegative transport',
        'independent scalar deposition parity','public ray factor range'])


def prepare(out):
    if (out/'PLAN.json').exists():raise FileExistsError(out/'PLAN.json')
    inputs=[Path(__file__),HERE/'cnh_rgb_spatial_unmix.py',HERE/'cnh_rgb_clearance_edge.py',
        BASE/'case-ledger.json',BASE/'PLAN.json',EXPANDED/'ordinal-ledger.json',CACHE/'observations.json']
    inputs+=sorted((BASE/'observations').glob('*.json'))
    save(out/'PLAN.json',dict(question='Does a public-ray common optical-Z coordinate improve the fixed two-spectrum fit?',
        scope='same52 edges/19frames/8scenes, frozen RGB support counts and target cells; consumed Development',
        inheritance='Same[.9prior,1.1prior] band, fixed3x3, two-variable NNLS, normalized singular ratio>=.05, original predicted edge',
        change='Each cell radial5cm-bin center multiplied by median public optical/radial ray factor; mass split linearly between adjacent optical5cm-bin centers',
        primary='optical_mode foreground peak bin center projected with original edge lateral-per-opticalZ factor',
        secondary='optical_median lower median bin center; always reported, no best-arm selection',
        controls='identical RGB support shuffle; frozen radial mode/median/shuffle; frozen ordinal/q10/shape',
        diagnostics='full-spectrum relative residual before/after; public factor spread; per-zone angle-only bound at2m and maximum observed radial bin',
        limits=['Median cell ray is an approximate geometry map, not native GT correspondence',
            'Ray-spread bounds describe geometry approximation only, not real sensor distance precision or final estimator errors',
            'Original5cm-bin mass-center assumption and new-grid interpolation remain; no measured subbin timing',
            'Same first-visible uniform-pixel geometry proxy; no noise/reflectance/hardware claim',
            'Shared optical-Z spectra still fail on slanted and multiple surfaces; knownGTcell/cleanedges retained'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs}))


def run(out):
    plan=read(out/'PLAN.json')
    assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    save(out/'self-check.json',self_check())
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend,Workload,BackendCandidate,DeviceObservation
    select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    original={r['id']:r for r in read(EXPANDED/'ordinal-ledger.json')}
    radial_baseline={r['id']:r for r in read(BASE/'case-ledger.json') if r['condition']=='clean'}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    ledger=[];start=time.perf_counter()
    for path in sorted((BASE/'observations').glob('*.json')):
        frame=read(path);frame_id=frame['frame_id']
        median,spread=public_cell_factors(cameras[frame_id])
        radial=np.asarray(frame['clean_sensor_histograms'],float)
        optical,transport=rebin_optical(radial,median)
        for zone,diag in enumerate(spread):
            occupied=np.flatnonzero(radial[zone]>0)
            max_range=float((occupied[-1]+1)*WIDTH) if len(occupied) else 0.
            diag['max_observed_bin_upper_radial_m']=max_range
            diag['angle_only_bound_at_observed_window_cm']=float(max_range*diag['max_absolute_deviation']*100) if diag['max_absolute_deviation'] is not None else None
        observations=[]
        for case in frame['cases']:
            A=np.asarray(case['predicted_zone_counts'],float);zone=case['zone_id']
            selected=select_optical(A,radial,median,zone)
            shuffled=select_optical(A,radial,median,zone,shuffle=True)
            frozen=original[case['id']]['prediction']
            edge_factor=float(frozen['side']*frozen['lateral_factor'])
            zvalues=dict(optical_mode=selected['mode_optical_z_m'],optical_median=selected['median_optical_z_m'],
                optical_shuffle_mode=shuffled['mode_optical_z_m'])
            estimates={a:edge_factor*z-.3 if z is not None else None for a,z in zvalues.items()}
            old=radial_baseline[case['id']]
            estimates.update(radial_mode=old['estimates']['unmix_mode'],radial_median=old['estimates']['unmix_median'],
                radial_shuffle_mode=old['estimates']['support_shuffle_mode'],ordinal=old['estimates']['original_ordinal'],
                zone_q10=old['estimates']['original_q10'],spatial_shape=old['estimates']['spatial_shape'])
            observations.append(dict(id=case['id'],zone_id=zone,estimates=estimates,predicted_zone_counts=A.tolist(),
                edge_lateral_per_optical_z=edge_factor,selection=selected,shuffled_selection=shuffled,
                original_radial_diagnostic=old['diagnostic']))
        save(out/'observations'/f'{frame_id}.json',dict(frame_id=frame_id,public_cell_factor_median=median.tolist(),
            public_factor_diagnostics=spread,radial_histograms=radial.tolist(),optical_histograms=optical.tolist(),
            resampling=transport,cases=observations))
        for obs in observations:
            truth=original[obs['id']]
            diag=obs['selection']['diagnostic'];before=obs['original_radial_diagnostic']
            ledger.append(dict(id=obs['id'],frame_id=frame_id,scene=truth['scene'],original_record=truth['original_record'],
                gt_clearance_m=truth['gt_clearance_m'],estimates=obs['estimates'],
                errors_m={a:v-truth['gt_clearance_m'] if v is not None else None for a,v in obs['estimates'].items()},
                optical_diagnostic=diag,radial_diagnostic=before,
                target_cell_factor=spread[obs['zone_id']],
                residual_difference=diag['relative_residual']-before['relative_residual'] if diag['relative_residual'] is not None and before['relative_residual'] is not None else None))
        print('optical',frame_id,len(ledger),'/52',flush=True)
    save(out/'case-ledger.json',ledger)
    summary={name:{a:summarize(rows,a) for a in ledger[0]['estimates']} for name,rows in [
        ('all52',ledger),('original15',[r for r in ledger if r['original_record']]),('additional37',[r for r in ledger if not r['original_record']])]}
    differences=np.array([r['residual_difference'] for r in ledger if r['residual_difference'] is not None])
    result=dict(status='COMPLETE',n=len(ledger),frames=len({r['frame_id'] for r in ledger}),scenes=len({r['scene'] for r in ledger}),
        primary='optical_mode',summaries=summary,
        paired_mode={a:paired(ledger,'optical_mode',a) for a in ('radial_mode','ordinal','zone_q10','optical_shuffle_mode')},
        paired_median_vs_radial_median=paired(ledger,'optical_median','radial_median'),
        fit=dict(statuses=dict(Counter(r['optical_diagnostic']['status'] for r in ledger)),
            residual_decreased=int((differences<0).sum()),paired_cases=len(differences),
            residual_difference_quantiles=np.percentile(differences,[0,50,95,100]).tolist(),
            optical_residual_quantiles=np.percentile([r['optical_diagnostic']['relative_residual'] for r in ledger],[0,50,95,100]).tolist(),
            radial_residual_quantiles=np.percentile([r['radial_diagnostic']['relative_residual'] for r in ledger],[0,50,95,100]).tolist()),
        public_geometry=dict(target_cell_angle_only_bound_at2m_cm_quantiles=np.percentile([r['target_cell_factor']['angle_only_bound_at_2m_cm'] for r in ledger],[0,50,95,100]).tolist(),
            target_cell_factor_range=[min(r['target_cell_factor']['median'] for r in ledger),max(r['target_cell_factor']['median'] for r in ledger)]),
        plan_sha256=sha(out/'PLAN.json'),code_sha256=sha(Path(__file__)),limits=plan['limits'],seconds=time.perf_counter()-start)
    save(out/'result.json',result)
    lines=['# 公共射线坐标下的光学Z双谱分解','',
        '固定原52条边、RGB band、3×3邻格和NNLS；仅将每格径向箱按公共射线因子中位数线性保质量映射到共同5cm光学Z网格。主mode与median并列报告。',
        '', '|范围|方法|≤2cm/全部|P50/P95 cm|','|---|---|---|---|']
    for subset,stats in summary.items():
        for arm,s in stats.items():
            q=s['absolute_error_cm_quantiles'];lines.append(f"|{subset}|{arm}|{round(s['n']*s['within_cm_all']['2'])}/{s['n']}|{q['p50']:.2f}/{q['p95']:.2f}|")
    lines+=['',f"完整谱相对残差降低{result['fit']['residual_decreased']}/52；中位数{result['fit']['radial_residual_quantiles'][1]:.4f}→{result['fit']['optical_residual_quantiles'][1]:.4f}。",'',
        '公共射线spread只给每格近似几何映射的诊断界，不能称测距或最终估计精度。线性重分箱保存计数质量和未clamp部分一阶矩，不恢复逐像素传感器对应。',
        '5cm原箱质量被假设在箱中心；分配到目标网格的相邻中心是数值插值，不是观测到的亚箱测距。光学坐标下的斜面/不同表面仍可能不共享谱。',
        '52条相关边/19帧/8场景，仍由GT指定目标格与干净边缘；合成Development，不是硬件/安全或独立泛化。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:{a:round(s['n']*s['within_cm_all']['2']) for a,s in v.items()} for k,v in summary.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','check']);parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.out)
    elif args.action=='run':run(args.out)
    else:print(json.dumps(self_check()))
