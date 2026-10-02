"""Fixed 52-record comparison: frozen distribution and spatial public selectors.

Additional37 records share scenes/frames with original15, not fresh confirmation.
Truth joins only after new public numerical predictions have been persisted.
"""
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np

from cnh_rgb_distribution_alignment import cell_quantiles, predict_ranges, read, save, sha, ROOT, HERE, CACHE, self_check as distribution_check
from cnh_rgb_spatial_association import coarse_observations, select_spatial, paired, self_check as spatial_check
from cnh_rgb_clearance_geometry import camera_geometry, hdf
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_probe import summarize

OUT=ROOT/'artifacts.local/work/cnh-rgb-association-compare-20261002/run-v2'
INPUT=ROOT/'artifacts.local/work/cnh-rgb-association-expanded-20261002/ordinal-ledger.json'
DIST=ROOT/'artifacts.local/work/cnh-rgb-distribution-alignment-20261002'
SPACE=ROOT/'artifacts.local/work/cnh-rgb-spatial-association-20261002/run-v2'
ARMS=('depthpro','zone_q10','strongest_mode','rgb_guided_mode','ordinal_transport',
      'affine','scale_only','affine_leave_cell_out','inverse_affine','affine_leave_3x3_out',
      'spatial','spatial_shift','spatial_shuffle')


def main():
    started=time.perf_counter()
    assert not (OUT/'PLAN.json').exists(),'Preserve existing output'
    paths=[Path(__file__),HERE/'cnh_rgb_distribution_alignment.py',HERE/'cnh_rgb_spatial_association.py',
           HERE/'cnh_rgb_clearance_geometry.py',HERE/'cnh_rgb_clearance_edge.py',HERE/'cnh_rgb_zone_range.py',
           HERE/'cnh_rgb_clearance_probe.py',INPUT,DIST/'case-ledger.json',SPACE/'case-ledger.json']
    seal={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(OUT/'PLAN.json',dict(lane='EXPLORE',input_sha256=seal,
        scope='all52 original eligible records/19frames/8scenes including original15+additional37; shared scene/frame conditional scope',
        selection='fixed preexisting52-record inventory and all previous13 arms; no outcome-driven removal or threshold change',
        public='per-frame object-blind64-cell geometry distributions, cached DepthPro and camera geometry; fixed old edges',
        evaluator='original truth fields stripped before processing; numerical outputs persisted before evaluation join',
        checks='original15 estimates agree with both frozen previous runs; fixed main affine retained; all52/15/37/scene paired summaries',
        claims='consumed synthetic Development, ideal geometry proxy, no new inference/training/hardware'))
    save(OUT/'self-check.json',dict(distribution=distribution_check(),spatial=spatial_check()))
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend,Workload,BackendCandidate,DeviceObservation
    backend=select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=OUT/'backend.json')
    grouped=defaultdict(list)
    for r in read(INPUT):
        grouped[r['frame_id']].append(dict(id=r['id'],frame_id=r['frame_id'],zone_id=r['zone_id'],prediction=r['prediction']))
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    public=[]; frame_inputs={}
    for frame,cases in sorted(grouped.items()):
        row=manifest[frame];camera=cameras[frame]
        geometry=camera_geometry(ROOT,row,camera);optics=geometry['optical_z_per_radial'];zones=zone_map(camera)
        depth_path=ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth']
        prediction_path=CACHE/'predictions/native'/f'{frame}.npz'
        frame_inputs.update({str(p.relative_to(ROOT)):sha(p) for p in [depth_path,prediction_path]})
        radial=hdf(depth_path)
        # Preserve each frozen predecessor's adapter precision exactly: the
        # distribution run summarized native HDF dtype; spatial used float32.
        coarse=coarse_observations(radial.astype(np.float32),zones);sensor=cell_quantiles(radial,zones)
        del radial
        with np.load(prediction_path,allow_pickle=False) as data: depth=data['native_depth']
        predicted=cell_quantiles(depth/optics,zones)
        save(OUT/'observations'/f'{frame}.json',dict(coarse=coarse,
            sensor_quantiles=[[float(v) if np.isfinite(v) else None for v in q] for q in sensor],
            predicted_quantiles=[[float(v) if np.isfinite(v) else None for v in q] for q in predicted]))
        for case in cases:
            p=case['prediction']; y,xhalf=p['edge_pixel'];x=int(np.floor(xhalf))
            edge_optic=float(np.mean(optics[y,x:x+2]));factor=p['side']*p['lateral_factor']*edge_optic
            dist=predict_ranges(predicted,sensor,p['foreground_depth_m']/edge_optic,case['zone_id'])
            spatial=select_spatial(depth,zones,optics,coarse,case['zone_id'],p)
            radii={a:v['radial_m'] for a,v in dist.items()};radii.update(spatial['ranges'])
            public.append(dict(**case,radial_edge_factor=factor,ranges=radii,
                estimates={a:(r*factor-.30 if r is not None else None) for a,r in radii.items()},distribution=dist,spatial=spatial))
        print('frame',len({r['frame_id'] for r in public}),'/',len(grouped),'cases',len(public),flush=True)
    save(OUT/'public-predictions.json',public)
    save(OUT/'frame-input-seal.json',frame_inputs)
    # Evaluation-only join. Previous baseline/ordinal outputs copied unchanged.
    truth={r['id']:r for r in read(INPUT)}
    ledger=[]
    for prediction in public:
        old=truth[prediction['id']]
        estimates={**old['estimates'],**prediction['estimates']}
        ledger.append(dict(id=old['id'],frame_id=old['frame_id'],scene=old['scene'],original_record=old['original_record'],
            gt_clearance_m=old['gt_clearance_m'],range_bin=old['range_bin'],estimates=estimates,
            errors_m={a:(v-old['gt_clearance_m'] if v is not None else None) for a,v in estimates.items()}))
    original=[r for r in ledger if r['original_record']]
    assert len(ledger)==52 and len(original)==15 and len(grouped)==19 and len({r['scene'] for r in ledger})==8
    comparisons=0
    for folder,arms in ((DIST,ARMS[5:10]),(SPACE,ARMS[10:])):
        baseline={r['id']:r for r in read(folder/'case-ledger.json')}
        for r in original:
            for arm in arms:
                np.testing.assert_allclose(r['estimates'][arm],baseline[r['id']]['estimates'][arm],rtol=0,atol=1e-12)
                comparisons+=1
    subsets={'all52':ledger,'original15':original,'additional37':[r for r in ledger if not r['original_record']]}
    result=dict(status='COMPLETE',n=52,frames=19,scenes=8,backend=backend,seconds=time.perf_counter()-started,
        subset={name:dict(n=len(rows),arms={a:summarize(rows,a) for a in ARMS},
            paired={a:{b:paired(rows,a,b) for b in ('depthpro','zone_q10')} for a in ARMS[4:]}) for name,rows in subsets.items()},
        scene={s:{a:summarize([r for r in ledger if r['scene']==s],a) for a in ARMS} for s in sorted({r['scene'] for r in ledger})})
    save(OUT/'case-ledger.json',ledger);save(OUT/'result.json',result)
    for p in paths:
        assert sha(p)==seal[str(p.relative_to(ROOT))]
    for key,rows in subsets.items():
        for a in ARMS: assert summarize(rows,a)==result['subset'][key]['arms'][a]
    save(OUT/'verification.json',dict(status='PASS',original15_arm_comparisons=comparisons,absolute_tolerance=1e-12,
        checks=['original15 frozen selector result agreement','all52=15+37 identity and denominator','all aggregate metrics recomputed','fixed source/input identity']))
    (OUT/'source').mkdir(exist_ok=True)
    for p in paths:
        if p.suffix=='.py':(OUT/'source'/p.name).write_bytes(p.read_bytes())
    report(result)
    save(OUT/'terminal.json',dict(status='complete'))


def report(result):
    lines=['# RGB固定机制扩展复核：52记录/19帧/8场景','',
        '这是原候选库存的完整52条条件记录（原15+同域新增37），不构成新场景验证。原目标格/边缘条件、所有方法与阈值固定；主仿射方法不因其他臂较好而改名为成功。新数值预测保存后才做GT评价。','',
        '|方法|全部52 ≤2cm; P50/P95 cm|原15 ≤2cm; P50/P95 cm|新增37 ≤2cm; P50/P95 cm|','|---|---|---|---|']
    for a in ARMS:
        cells=[]
        for key in ('all52','original15','additional37'):
            s=result['subset'][key]['arms'][a];q=s['absolute_error_cm_quantiles']
            cells.append(f'{round(s["within_cm_all"]["2"]*s["n"])}/{s["n"]}; {q["p50"]:.2f}/{q["p95"]:.2f}')
        lines.append('|'+a+'|'+'|'.join(cells)+'|')
    lines+=['','固定臂相对q10的≤2cm配对百分点差（1000次场景簇重采样95%区间；8场景相关证据）：','',
        '|方法|all52|original15|additional37|','|---|---|---|---|']
    for a in ARMS[4:]:
        cells=[]
        for key in ('all52','original15','additional37'):
            p=result['subset'][key]['paired'][a]['zone_q10'];ci=p['ci95_pp']
            cells.append(f'{p["difference_pp"]:+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]')
        lines.append('|'+a+'|'+'|'.join(cells)+'|')
    lines+=['','逐场景成功例数/分母：','',
        '|场景|DepthPro|q10|ordinal|affine|inverse affine|spatial|shift|shuffle|','|---|---|---|---|---|---|---|---|---|']
    for scene,arms in result['scene'].items():
        entries=[]
        for a in ('depthpro','zone_q10','ordinal_transport','affine','inverse_affine','spatial','spatial_shift','spatial_shuffle'):
            s=arms[a];entries.append(f'{round(s["within_cm_all"]["2"]*s["n"])}/{s["n"]}')
        lines.append('|'+scene+'|'+'|'.join(entries)+'|')
    lines+=['','原15的分布5臂和空间3臂共120项估计与原运行逐项一致（绝对容差1e-12m）。全部13臂保留，未删失败例或改阈值。',
        '', '边界：完美首可见几何分布按像素等权聚合，不是实测ToF直方图；同一帧/场景存在多条相关记录；粗格由原GT条件选择；只评价清晰边缘净距，不能推断全图候选能力、真实安全性、硬件收益或解除报警。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'terminal.json',dict(status='failed',error=repr(e)));raise
