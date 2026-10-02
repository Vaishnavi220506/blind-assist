"""Falsification controls for local RGB rank transport on consumed geometry.

Range weighting and coarse bins are explicit uncalibrated sensitivity models.
They cannot establish a hardware operating point or a physical ceiling.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from cnh_rgb_association_expanded import (
    ROOT, CACHE, DEFAULT_OUT, coarse_histograms, paired_difference, cpu_backend,
)
from cnh_rgb_clearance_probe import context, read, save, sha, summarize
from cnh_rgb_clearance_geometry import camera_geometry, hdf
from cnh_rgb_clearance_edge import zone_map

OUT = ROOT/'artifacts.local/work/cnh-rgb-ordinal-stress-20261002'


def histogram_quantile(counts, rank, width):
    counts = np.asarray(counts, float)
    if not np.isfinite(rank) or not 0 <= rank <= 1 or not counts.sum():
        return None
    cumulative = np.cumsum(counts)
    mass = rank*cumulative[-1]
    index = min(int(np.searchsorted(cumulative, mass, side='right')), int(np.flatnonzero(counts)[-1]))
    before = cumulative[index-1] if index else 0.
    fraction = (mass-before)/counts[index] if counts[index] else .5
    return float(max(.5*width if index == 0 and fraction <= 0 else 0.,
                     (index+np.clip(fraction,0,1))*width))


def rank_from_true_local_range(counts, radius, width):
    """Evaluator-only diagnostic, never a selector input."""
    if not counts.sum(): return None
    loc = radius/width
    i = int(np.clip(np.floor(loc), 0, len(counts)-1))
    return float((counts[:i].sum()+counts[i]*np.clip(loc-i,0,1))/counts.sum())


def run(out):
    if (out/'result.json').exists():
        raise FileExistsError('Retain old run')
    previous = read(DEFAULT_OUT/'ordinal-ledger.json')
    cpu_backend(out)
    previous.sort(key=lambda r:r['id'])
    quantiles = [.01,.05,.10,.25,.50,.75,.90,.99]
    save(out/'PLAN.json',dict(
        lane='EXPLORE successor stress after observed ordinal gain; not fresh confirmation',
        source_ledger_sha256=sha(DEFAULT_OUT/'ordinal-ledger.json'),
        code_sha256=sha(Path(__file__)),
        fixed_quantile_controls=quantiles,
        shuffled_rank='52 original ranks permuted with seed2026100204, same set and all52 denominator',
        coarse_bins_cm=[5,10,30],
        weighting='Raw proxy photon mass ~ uniform-pixel area/r^2 or /r^4 using bin centres; '
                  'plus10 seeds of independent1m radial slabs reflectance {.25,1,4} with r^-2; '
                  'range compensation multiplies by r^2 but cannot remove unknown reflectance',
        rationale='Separate RGB ordinal contribution from a fixed near quantile and expose area-to-photon mismatch',
        restrictions='No outcome-dependent thresholds, no hardware claims, all controls retained',
        matched_controls='Each bin/weight/material condition also evaluates fixed q10 on the identical observation',
    ))
    assert abs(histogram_quantile(np.array([0,1,1,0]), .5, .05)-.1) < 1e-12
    assert abs(histogram_quantile(np.array([0,1,1,0]), 1., .05)-.15) < 1e-12
    source=out/'source'; source.mkdir(exist_ok=True)
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest,cameras,_=context(); manifest={r['id']:r for r in manifest}
    ranks=np.array([r['details'].get('predicted_rank',np.nan) for r in previous])
    shuffled=ranks[np.random.default_rng(2026100204).permutation(len(ranks))]
    shuffled=dict(zip([r['id'] for r in previous],shuffled))
    grouped={}
    for r in previous: grouped.setdefault(r['frame_id'],[]).append(r)
    ledger=[]
    for frame_id, events in grouped.items():
        row=manifest[frame_id]; camera=cameras[frame_id]
        zones=zone_map(camera)
        radial=hdf(ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth'])
        hist=coarse_histograms(radial,zones)
        # Alternative widths share exactly the same radial admission (0,12.8).
        valid=np.isfinite(radial)&(radial>0)&(radial<12.8)&(zones>=0)
        hists={.05:hist}
        for width in (.1,.3):
            n=int(np.ceil(12.8/width))
            codes=zones[valid]*n+np.floor(radial[valid]/width).astype(int)
            hists[width]=np.bincount(codes,minlength=64*n).reshape(64,n)
        for old in events:
            zone=old['zone_id']; rank=old['details'].get('predicted_rank')
            factor=old['details'].get('radial_factor')
            estimates=dict(old['estimates'])
            estimates['ordinal_rank_shuffle']=None
            estimates.update({f'fixed_q{round(q*100):02}':None for q in quantiles})
            estimates.update({a:None for a in ('ordinal_bin10cm','ordinal_bin30cm','ordinal_inv_r2','ordinal_inv_r4')})
            estimates.update({a:None for a in ('q10_bin10cm','q10_bin30cm','q10_inv_r2','q10_inv_r4')})
            estimates.update({f'ordinal_material_{comp}_{seed}':None for comp in ('raw','r2comp') for seed in range(10)})
            estimates.update({f'q10_material_{comp}_{seed}':None for comp in ('raw','r2comp') for seed in range(10)})
            if rank is not None and factor is not None:
                radii={f'fixed_q{round(q*100):02}':histogram_quantile(hist[zone],q,.05) for q in quantiles}
                radii['ordinal_rank_shuffle']=histogram_quantile(hist[zone],float(shuffled[old['id']]),.05)
                for width in (.1,.3):
                    radii[f'ordinal_bin{round(width*100)}cm']=histogram_quantile(hists[width][zone],rank,width)
                    radii[f'q10_bin{round(width*100)}cm']=histogram_quantile(hists[width][zone],.10,width)
                centres=(np.arange(256)+.5)*.05
                for power in (2,4):
                    radii[f'ordinal_inv_r{power}']=histogram_quantile(hist[zone]/centres**power,rank,.05)
                    radii[f'q10_inv_r{power}']=histogram_quantile(hist[zone]/centres**power,.10,.05)
                frame_seed=int(hashlib.sha256(frame_id.encode()).hexdigest()[:8],16)
                for seed in range(10):
                    material=np.random.default_rng(frame_seed+seed).choice([.25,1.,4.],size=(64,13))
                    weights=material[zone,np.minimum(centres.astype(int),12)]
                    for comp in ('raw','r2comp'):
                        counts=hist[zone]*weights/(centres**2 if comp=='raw' else 1.)
                        radii[f'ordinal_material_{comp}_{seed}']=histogram_quantile(counts,rank,.05)
                        radii[f'q10_material_{comp}_{seed}']=histogram_quantile(counts,.10,.05)
                estimates.update({a:float(factor*v-.30) if v is not None else None for a,v in radii.items()})
            r={k:v for k,v in old.items() if k not in ('errors_m','estimates')}
            r.update(estimates=estimates,errors_m={a:v-old['gt_clearance_m'] if v is not None else None for a,v in estimates.items()})
            ledger.append(r)
        print('stress',frame_id,len(ledger),'/',len(previous),flush=True)
        del radial,zones,hist,hists
    save(out/'case-ledger.json',ledger)
    arms=list(ledger[0]['estimates'])
    summaries={subset:{a:summarize(rows,a) for a in arms} for subset,rows in (
        ('all',ledger),('original15',[r for r in ledger if r['original_record']]),
        ('additional37',[r for r in ledger if not r['original_record']]))}
    comparisons=['zone_q10','rgb_guided_mode','ordinal_rank_shuffle']+[f'fixed_q{round(q*100):02}' for q in quantiles]
    result=dict(status='COMPLETE',n=len(ledger),scenes=len({r['scene'] for r in ledger}),
        summaries=summaries,paired={a:paired_difference(ledger,'ordinal_transport',a) for a in comparisons},
        material_sensitivity={subset:{comp:{
            'hits_each_seed':[round(summaries[subset][f'ordinal_material_{comp}_{seed}']['within_cm_all']['2']*len(rows)) for seed in range(10)],
            'q10_hits_each_seed':[round(summaries[subset][f'q10_material_{comp}_{seed}']['within_cm_all']['2']*len(rows)) for seed in range(10)],
            'n':len(rows)} for comp in ('raw','r2comp')}
            for subset,rows in [('all',ledger),('original15',[r for r in ledger if r['original_record']])]},
        interpretation='Uncalibrated mass/range-bin sensitivity only; no physical feasibility assertion')
    result['paired_matched_pressure']={a:paired_difference(ledger,a,a.replace('ordinal_','q10_',1))
        for a in arms if a.startswith(('ordinal_bin','ordinal_inv_r','ordinal_material_'))}
    save(out/'result.json',result)
    rows=['# RGB排序传递的证伪对照与观测压力','',
          '同一52条已消费Development边缘；原理想像素等权直方图并非传感器回波。'
          '固定分位与排序置乱检验RGB排序贡献，距离权重/反射率/粗箱检验观测依赖。','',
          '|方法|全部≤2cm|原15例≤2cm|全部P50/P95 cm|','|---|---|---|---|']
    for a in arms:
        if a.startswith(('ordinal_material_','q10_material_')): continue
        s=summaries['all'][a]; s15=summaries['original15'][a]; q=s['absolute_error_cm_quantiles']
        rows.append(f"|{a}|{round(s['within_cm_all']['2']*s['n'])}/{s['n']}|{round(s15['within_cm_all']['2']*s15['n'])}/{s15['n']}|{q['p50']:.2f}/{q['p95']:.2f}|")
    rows+=['','十组分段反射权重压力（非物理标定）：']
    for subset,v in result['material_sensitivity'].items():
        for comp,x in v.items():
            rows.append(f"- {subset}/{comp}: ordinal {x['hits_each_seed']} / {x['n']}; matched q10 {x['q10_hits_each_seed']}")
    (out/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
