"""Consumed-Development spatial support assay; no measured ToF or training.

The public selector accepts only Depth Pro optical depth, public ray factors,
coarse zone IDs and object-blind, uniform-pixel range histogram summaries.
Labels and pixelwise reference ranges are confined to simulation/evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_geometry import reference_frame
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_zone_range import summarize_distribution

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
BASE = ROOT/'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT = ROOT/'artifacts.local/work/cnh-rgb-spatial-association-20261002'
WIDTH = .05
BAND = 1.10
ARMS = ('spatial', 'spatial_shift', 'spatial_shuffle', 'rgb_guided_mode', 'zone_q10', 'strongest_mode')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coarse_observations(radial, zones):
    """Simulator boundary: returns 64 histograms/modes, never pixel correspondence."""
    counts, distributions = [], []
    for z in range(64):
        vals = np.asarray(radial, dtype=np.float64)[zones == z]
        eligible = vals[np.isfinite(vals) & (vals > 0) & (vals < 12.8)]
        counts.append(np.bincount(np.floor(eligible/WIDTH).astype(int), minlength=256).tolist())
        distributions.append(summarize_distribution(vals.astype(np.float64)))
    return dict(counts=counts, distributions=distributions)


def _cosine(a, b):
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a, b)/denom) if denom > 0 else -1.0


def select_spatial(depthpro_optical_z, zones, optical_z_per_radial, coarse, zone_id, prediction):
    """Truth-blind one-case API. Returns radial ranges and all mode scores.

    Input coarse = {counts: 64x256, distributions: 64 old range summaries}.
    Depth Pro and ray factors are pixel arrays. No evaluator event is accepted.
    Candidate identity/range is fixed by the original target-cell mode detector.
    The RGB prior sets only a RELATIVE depth band, never distance proximity.
    """
    dp = np.asarray(depthpro_optical_z, dtype=np.float64)
    zones, optics = np.asarray(zones), np.asarray(optical_z_per_radial)
    if dp.shape != zones.shape or dp.shape != optics.shape:
        raise ValueError('prediction, zone, calibration shapes differ')
    if set(coarse) != {'counts', 'distributions'}:
        raise ValueError('coarse boundary accepts only aggregated counts/distributions')
    counts = np.asarray(coarse['counts'], dtype=np.float64)
    if counts.shape != (64, 256) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError('invalid aggregate counts')
    modes = coarse['distributions'][zone_id]['modes']
    if prediction['status'] != 'OK' or not modes:
        return dict(status='MISSING', ranges={a:None for a in ARMS[:3]}, selected={}, scores=[])
    y, xhalf = prediction['edge_pixel']
    x = int(np.floor(xhalf))
    edge_optics = float(np.mean(optics[y, x:x+2]))
    prior = prediction['foreground_depth_m']
    rgb_support = np.zeros(64)
    cell_optics = np.zeros(64)
    for z in range(64):
        cell = zones == z
        valid = cell & np.isfinite(dp) & (dp > 0)
        if valid.any():
            rgb_support[z] = np.mean((dp[valid] >= prior/BAND) & (dp[valid] < prior*BAND))
        if cell.any():
            cell_optics[z] = float(np.median(optics[cell]))
    rows, cols = np.indices((8, 8))
    neighborhood = ((abs(rows-zone_id//8) <= 1) & (abs(cols-zone_id%8) <= 1)).ravel()
    valid_cells = neighborhood & (counts.sum(axis=1) > 0) & (cell_optics > 0)
    centres = (np.arange(256)+.5)*WIDTH
    sensor_support = []
    for mode in modes:
        # Approximate per-cell optical-Z using its median ray factor. The full
        # reference frame cannot enter here; no subcell pixel matching exists.
        z_candidate = mode['radial_m']*edge_optics
        optical_bins = cell_optics[:, None]*centres[None, :]
        inside = (optical_bins >= z_candidate/BAND) & (optical_bins < z_candidate*BAND)
        support = np.divide((counts*inside).sum(axis=1), counts.sum(axis=1),
                            out=np.zeros(64), where=counts.sum(axis=1) > 0)
        sensor_support.append(support)
    # These perturb RGB support LOCATION only; candidate ranges/counts unchanged.
    shifted = np.roll(np.roll(rgb_support.reshape(8, 8), 1, axis=0), 1, axis=1).ravel()
    shuffled = rgb_support[np.random.default_rng(2026100203).permutation(64)]
    maps = dict(spatial=rgb_support, spatial_shift=shifted, spatial_shuffle=shuffled)
    all_scores = {arm:[_cosine(vec[valid_cells], s[valid_cells]) for s in sensor_support]
                  for arm, vec in maps.items()}
    selected = {arm:int(np.argmax(scores)) for arm, scores in all_scores.items()}
    evidence={a:('SINGLE_MODE_NO_ASSOCIATION_CHOICE' if len(s)==1 else
                'NO_SPATIAL_SUPPORT' if max(s)<0 else
                'INDISTINGUISHABLE_MODES' if sorted(s,reverse=True)[0]-sorted(s,reverse=True)[1] <= 1e-12 else
                'UNIQUE_SPATIAL_WINNER') for a,s in all_scores.items()}
    return dict(status='OK', ranges={a:float(modes[i]['radial_m']) for a,i in selected.items()},
                selected=selected, scores=all_scores, neighborhood=np.flatnonzero(valid_cells).tolist(),
                evidence=evidence,
                rgb_support=rgb_support.tolist(), mode_support=[s.tolist() for s in sensor_support],
                edge_optical_factor=edge_optics,
                score_margin={a:float(sorted(s, reverse=True)[0]-sorted(s, reverse=True)[1]) if len(s)>1 else None
                              for a,s in all_scores.items()})


def self_check():
    # Exact support shapes identify the near surface despite 2x metric RGB bias.
    zones = np.repeat(np.arange(64), 100).reshape(80,80)
    sensor = np.full(zones.shape, 4.025)
    predicted = np.full(zones.shape, 8.05)
    # Distinct footprint, including mixed source zone, over a 3x3 neighborhood.
    for zone, fraction in ((18, .2), (19, .4), (20, .7), (26, .8), (27, .6), (28, .3), (34,.1), (35,.5), (36,.9)):
        indexes=np.flatnonzero(zones.ravel()==zone)[:int(fraction*100)]
        sensor.ravel()[indexes]=2.025
        predicted.ravel()[indexes]=4.05
    pred=dict(status='OK',edge_pixel=[33, 60.5],foreground_depth_m=4.05)
    coarse=coarse_observations(sensor,zones)
    out=select_spatial(predicted,zones,np.ones_like(sensor),coarse,27,pred)
    assert abs(out['ranges']['spatial']-2.025)<1e-10
    changed=select_spatial(predicted*3,zones,np.ones_like(sensor),coarse,27,{**pred,'foreground_depth_m':12.15})
    assert changed == out, 'RGB global scale must not affect relative support selection'
    assert out['scores']['spatial'][out['selected']['spatial']] > .999999
    assert _cosine(np.zeros(2),np.ones(2)) == -1.
    try:
        select_spatial(predicted,zones,np.ones_like(sensor),{**coarse,'instance':sensor},27,pred)
    except ValueError:
        pass
    else:
        raise AssertionError('truth extra key accepted')
    return dict(status='PASS',checks=['2x biased RGB exact spatial footprint recovery',
                'global RGB scale invariance','zero-support deterministic fallback',
                'coarse extra-field isolation guard','JSON finite outputs'])


def paired(ledger, arm, baseline):
    scenes=sorted({r['scene'] for r in ledger})
    den=np.array([sum(r['scene']==s for r in ledger) for s in scenes])
    def success(r, a):
        return r['errors_m'][a] is not None and abs(r['errors_m'][a]) <= .02
    diff=np.array([sum(int(success(r,arm))-int(success(r,baseline)) for r in ledger if r['scene']==s) for s in scenes])
    rng=np.random.default_rng(2026100121)
    w=np.array([np.bincount(rng.integers(len(scenes),size=len(scenes)),minlength=len(scenes)) for _ in range(1000)])
    rates=w@diff/(w@den)
    return dict(difference_pp=float(diff.sum()/den.sum()*100),
                ci95_pp=(np.percentile(rates,[2.5,97.5])*100).tolist(),
                scene_success_differences=dict(zip(scenes,diff.tolist())))


def source_seal():
    paths=[Path(__file__),HERE/'cnh_rgb_clearance_edge.py',HERE/'cnh_rgb_clearance_geometry.py',
           HERE/'cnh_rgb_clearance_probe.py',HERE/'cnh_rgb_zone_range.py',
           PRIOR/'selection.json',PRIOR/'case-ledger.json',PRIOR/'evaluation-input-seal.json',
           BASE/'case-ledger.json',BASE/'result.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def prepare(out):
    assert not (out/'PLAN.json').exists()
    save(out/'PLAN.json',dict(
        scope='Consumed synthetic Hypersim Development; same 15 cases / 8 scenes and original predicted edges',
        mechanism='3x3 spatial support cosine: relative RGB foreground optical-Z band [prior/1.10,prior*1.10); each original retained mode optical-Z hypothesis produces same relative band per coarse histogram; strongest order tie',
        observations='Full native cached DepthPro, public camera ray factors, original coarse cell, object-blind 64x256 uniform-pixel 5cm radial geometry counts and old <=4 modes per cell',
        prohibited_selector_inputs='pixelwise reference range, target masks, semantic/instance IDs, truth edges/clearance, GT target distance',
        baseline='immutable v2 q10 8/15, RGB guided7/15, strongest6/15; fixed existing candidate modes',
        controls='RGB support maps shifted +1 row,+1column toroidally; deterministic 64-cell permutation seed2026100203. Same sensor candidates and same local cells. Exact ties (margin<=1e-12), no support and single mode are separately reported; all-case main keeps declared strongest-order fallback, not claimed as successful spatial identification.',
        metrics='all15 within2cm incl missing; median/P95/bias/sign counts; paired scene bootstrap1000 seed2026100121; no fitted threshold, no outcome-driven rerun',
        limits=['Known GT-derived target cell and clean edge selection inherited; not independent detection',
                'Uniform-pixel first-visible geometric distributions are not photon return histograms or real ToF',
                '5cm bins and +/-10% band are diagnostic assumptions; per-cell median optical factor approximation',
                'Relative RGB depth band may merge different surfaces; no semantic target input',
                'Only15 cases/8 scenes consumed Development, only2 near2cm body-line; no hardware or safety claim'],
        code_and_input_sha256=source_seal()))


def run(out):
    started=time.perf_counter()
    plan=read(out/'PLAN.json')
    assert plan['code_and_input_sha256']==source_seal()
    assert not (out/'result.json').exists()
    checks=self_check()
    parent=read(PRIOR/'evaluation-input-seal.json')['input_sha256']
    assert all(sha(ROOT/p)==h for p,h in parent.items())
    save(out/'self-check.json',checks)
    save(out/'input-seal.json',dict(plan_sha256=sha(out/'PLAN.json'),parent_sha256=parent))
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    backend=select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    old={r['id']:r for r in read(BASE/'case-ledger.json')}
    selection=read(PRIOR/'selection.json')
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    ledger=[]
    for event in selection:
        baseline=old[event['id']]
        ref=reference_frame(ROOT,manifest[event['frame_id']],cameras[event['frame_id']])
        zones=zone_map(cameras[event['frame_id']])
        coarse=coarse_observations(ref['radial'],zones)
        assert coarse['distributions'][event['zone_id']]['modes']==baseline['distribution']['modes']
        with np.load(CACHE/'predictions/native'/f'{event["frame_id"]}.npz',allow_pickle=False) as handle:
            dp=handle['native_depth']
        selected=select_spatial(dp,zones,ref['optical_z_per_radial'],coarse,event['zone_id'],baseline['prediction'])
        # Serialize predictions BEFORE any evaluator target/clearance join.
        observation_name=hashlib.sha256(event['id'].encode()).hexdigest()[:16]+'.json'
        save(out/'observations'/observation_name,dict(case_id=event['id'],zone_id=event['zone_id'],selection=selected,
             counts=coarse['counts'],modes=coarse['distributions'][event['zone_id']]['modes']))
        estimates={a:baseline['estimates'][a] for a in ARMS[3:]}
        estimates.update({a:(r*baseline['radial_edge_factor']-.30) if r is not None else None
                          for a,r in selected['ranges'].items()})
        result={k:v for k,v in event.items() if k!='support_yx'}
        result.update(estimates=estimates,errors_m={a:v-event['gt_clearance_m'] if v is not None else None for a,v in estimates.items()},
                      selection=selected,prediction=baseline['prediction'],radial_edge_factor=baseline['radial_edge_factor'],
                      attribution={a:baseline['mode_attribution'][i] for a,i in selected['selected'].items()})
        ledger.append(result)
        print('spatial',len(ledger),'/',len(selection),flush=True)
    save(out/'case-ledger.json',ledger)
    result=dict(status='COMPLETE',n=len(ledger),scenes=len({r['scene'] for r in ledger}),
                arms={a:summarize(ledger,a) for a in ARMS},
                paired={b:paired(ledger,'spatial',b) for b in ARMS[1:]},
                identifiability={a:{s:sum(r['selection'].get('evidence',{}).get(a)==s for r in ledger)
                    for s in ('SINGLE_MODE_NO_ASSOCIATION_CHOICE','NO_SPATIAL_SUPPORT','INDISTINGUISHABLE_MODES','UNIQUE_SPATIAL_WINNER')}
                    for a in ARMS[:3]},
                elapsed_seconds=time.perf_counter()-started,backend=backend,
                limits=plan['limits'])
    save(out/'result.json',result)
    verify(out)
    report(out,result,ledger)
    for p in (Path(__file__),HERE/'cnh_rgb_zone_range.py',HERE/'cnh_rgb_clearance_edge.py',HERE/'cnh_rgb_clearance_geometry.py',HERE/'cnh_rgb_clearance_probe.py'):
        (out/'source').mkdir(exist_ok=True)
        (out/'source'/p.name).write_bytes(p.read_bytes())


def verify(out):
    assert read(out/'PLAN.json')['code_and_input_sha256']==source_seal()
    ledger,result=read(out/'case-ledger.json'),read(out/'result.json')
    selection=read(PRIOR/'selection.json')
    assert [r['id'] for r in ledger]==[r['id'] for r in selection] and len(ledger)==15
    assert len({r['scene'] for r in ledger})==8
    for a in ARMS:
        assert summarize(ledger,a)==result['arms'][a]
    for b in ARMS[1:]:
        assert paired(ledger,'spatial',b)==result['paired'][b]
    for r in ledger:
        for a in ARMS[:3]:
            if r['selection']['ranges'][a] is not None:
                assert abs(r['selection']['ranges'][a]*r['radial_edge_factor']-.30-r['estimates'][a])<1e-12
    assert result['arms']['zone_q10']['within_cm_all']['2']==8/15
    assert result['arms']['rgb_guided_mode']['within_cm_all']['2']==7/15
    save(out/'verification.json',dict(status='PASS',checks=['same15 ordered identities,8scenes',
         'parent labels/calibration/cache hashes','same original retained candidate modes',
         'baseline q10 8/15 RGB7/15 reproduced','radial-clearance arithmetic',
         'all-case summaries and paired bootstrap recalculated','pure selector fixtures']))


def report(out,result,ledger):
    lines=['# 跨格空间支持归属小试','',
        '固定原15例、8场景、原预测边缘与45°等角8×8格；没有训练/下载/新增模型推理。用RGB相对前景深度带在邻近3×3格的占比形状，选择原保留几何模式；不使用Depth Pro绝对距离选最近模式。','',
        '|方法|≤2cm（全部15例）|绝对误差中位/P95（cm）|', '|---|---|---|']
    for a in ARMS:
        s=result['arms'][a];q=s['absolute_error_cm_quantiles']
        lines.append(f"|{a}|{round(s['within_cm_all']['2']*15)}/15|{q['p50']:.2f}/{q['p95']:.2f}|")
    lines+=['','场景配对差（空间法减对照；百分点，95%场景bootstrap区间）：']
    for a,p in result['paired'].items():
        lines.append(f"- {a}：{p['difference_pp']:+.2f}，区间[{p['ci95_pp'][0]:+.2f},{p['ci95_pp'][1]:+.2f}]。")
    lines+=['','可辨识性诊断（单模式无需关联；同分或无支持使用原强度顺序，但不能声称空间识别成功）：',json.dumps(result['identifiability'],ensure_ascii=False),'',
            '每例输出完整支持图、各候选得分与消融选择，见observations和case-ledger.json。相对深度带固定[foreground/1.10,foreground×1.10)，粗格直方图5cm箱中心，按公开射线中位因子作光学Z转换；最大cosine选模式，同分保持原强度顺序。空间扰动只移动RGB占比位置，不改变传感器输入/模式/目标格。','',
            '数据边界与局限：']+['- '+s for s in result['limits']]
    lines+=['','验证：原输入哈希、原15例和模式复现；RGB全局尺度不变性与偏尺度形状夹具通过；all-case误差/场景配对回算通过。观察与选择函数不接收目标mask、实例、GT距离或GT净距。模拟器用无对象筛选的整格可见深度生成计数，仍不是实测ToF。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','run','verify','check'),required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if args.stage=='check':
        print(json.dumps(self_check()));return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='verify':verify(args.output)
    else:
        try:
            run(args.output);save(args.output/'terminal.json',dict(status='complete'))
        except BaseException as error:
            save(args.output/'terminal.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
