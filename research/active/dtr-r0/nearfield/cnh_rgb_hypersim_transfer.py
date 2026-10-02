"""Frozen RGB/coarse-query transfer to 48 outside-NFO500 Development scenes.

New native RGB inference, no fitting, threshold changes, target gifts or fusion.
"""
import cnh_rgb_visible_query_gate as P
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
import cnh_rgb_visible_query as V
import cnh_rgb_range_anchor as A
import cnh_rgb_range_anchor_gate as R

ROOT=P.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-hypersim-transfer-20261002'
RUN_ID='CNH_RGB_HYPERSIM_TRANSFER_20261002'
SOURCE=ROOT/'artifacts.local/work/ba-nfo-20260919'
OLD=ROOT/'artifacts.local/work/cnh-rgb-range-anchor-20261002'
INFER=Path(__file__).with_name('cnh_rgb_hypersim_transfer_infer.py')
DEPTHPRO=Path(__file__).with_name('ba_nfo_depthpro.py')
BOOT_SEED=2026100214


def order(value):
    return hashlib.sha256(('rgb-hypersim-transfer-20261002|'+value).encode()).hexdigest()


def prepare(out):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Preserve plan')
    prereg=[s for s in P.RUNS.read_text(encoding='utf8').splitlines() if RUN_ID in s]
    assert len(prereg)==1
    manifest=SOURCE/'prepared-manifest.json'
    selection=SOURCE/'hypersim-selection.json'
    nfo=P.SOURCE/'manifest.json'
    pair_run=ROOT/'artifacts.local/work/cnh-rgb-nearfield-pair-supply-20261002/result.json'
    pair_start=ROOT/'artifacts.local/work/cnh-rgb-sequence-preflight-20261002/result.json'
    roles=P.read(selection)['families']
    excluded={r['scene'] for r in P.read(nfo)}
    excluded.update(r['metadata']['scene'] for r in P.read(pair_run)['pairs'])
    excluded.update(r['scene'] for r in P.read(pair_start)['candidates'])
    groups=defaultdict(list)
    for item in P.read(manifest):
        if item['source']!='hypersim' or roles.get(item['scene'][:6])!='train' or item['scene'] in excluded:
            continue
        groups[item['scene']].append(item)
    groups={s:sorted(rows,key=lambda r:order(r['id'])) for s,rows in groups.items() if len(rows)>=2}
    by_family=defaultdict(list)
    for scene in groups:
        by_family[scene[:6]].append(scene)
    families=sorted(by_family,key=order)
    for f in families:
        by_family[f].sort(key=order)
    selected=[]; level=0
    while len(selected)<48:
        before=len(selected)
        for f in families:
            if len(by_family[f])>level:
                selected.append(by_family[f][level])
                if len(selected)==48:
                    break
        assert len(selected)>before,'Insufficient scenes; no outcome-dependent fallback'
        level+=1
    cameras={r['scene_name']:r for r in csv.DictReader((SOURCE/'metadata_camera_parameters.csv').open())}
    inputs=[];observations=[]
    for scene in selected:
        matrix=[[float(cameras[scene][f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)]
        for item in groups[scene][:2]:
            assert item['split']=='train'
            rgb,depth=P.DATA/item['rgb'],P.DATA/item['depth']
            assert P.sha(rgb)==item['rgb_sha256'] and P.sha(depth)==item['depth_sha256']
            rgbrow=dict(id=item['id'],rgb_path=str(rgb.relative_to(ROOT)),rgb_sha256=item['rgb_sha256'],camera_matrix=matrix)
            observations.append(rgbrow)
            inputs.append(dict(**rgbrow,scene=scene,family=scene[:6],split='eval',original_repository_role='train',
                depth_path=str(depth.relative_to(ROOT)),depth_sha256=item['depth_sha256']))
    assert len(inputs)==len({r['id'] for r in inputs})==96
    P.save(out/'observations.json',observations)
    dependencies=[Path(__file__),INFER,DEPTHPRO,Path(P.__file__),Path(V.__file__),Path(A.__file__),Path(R.__file__),
        manifest,selection,nfo,pair_run,pair_start,SOURCE/'metadata_camera_parameters.csv',OLD/'result.json']
    dependencies+=sorted((P.SOURCE/'upstream/src').rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):P.sha(p) for p in dependencies}
    backend=P.SOURCE/'backend.json'
    weight=P.SOURCE/'depth_pro.pt'
    weight_sha=P.sha(weight)
    assert weight_sha=='3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
    inference=dict(observations_sha256=P.sha(out/'observations.json'),weight_sha256=weight_sha,
        source_sha256=hashes,backend_reference=dict(path=str(backend.relative_to(ROOT)),sha256=P.sha(backend)))
    P.save(out/'inference-plan.json',inference)
    policies={b:dict(policies=d['policies'],selected_tof=d['selected_tof']) for b,d in P.read(OLD/'result.json')['budgets'].items()}
    plan=dict(run_id=RUN_ID,frozen_at_utc=datetime.now(timezone.utc).isoformat(),preregistration_row=prereg[0],
        role='Consumed Hypersim Development repository-train inputs, evaluated at old frozen policies; outsideNFO500 is not independent confirmation',
        selection='Repository train only; exact NFO500 and prior pair-screen scenes excluded. Family sha256 round-robin to48scenes, first2 sha256 frameIDs per scene. No depth/image outcomes inspected for choice; no replacements.',
        selected_scenes=selected,excluded_scenes=sorted(excluded),families=sorted({r['family'] for r in inputs}),
        protected_families=sorted(f for f,s in roles.items() if s=='test'),inputs=inputs,
        observations_sha256=inference['observations_sha256'],inference_plan_sha256=P.sha(out/'inference-plan.json'),
        source_sha256=hashes,old_policies=policies,
        arms=list(R.ARMS),anchor_role='Retained negative control only; unchanged global-scale method, no repair or new mechanism',
        queries='HEAD/BODY visible camera-relative0.6–2.1m, native16pixel D16/95%coverage, same frozen readout',
        comparisons='RawDepthPro and anchored controls vs old-cal-selectedq10; median separately. Paired rescue/loss and extra clear alarms; no OR/fusion deployment or inference of matched eval false-positive burden.',
        missing='No sample replacement or dropping. Missing inference counted prediction abstention on same truth denominator; UNKNOWN retained separately. Engine failures pause evaluation pending integrity check, not outcome filtering.',
        bootstrap=dict(cluster='family',draws=1000,seed=BOOT_SEED,selection='all fixed; paired family weights; zero denominators excluded and counted'),
        interpretation='Descriptive frozen transfer. No new pass threshold; shallow support/actual clear cost must accompany recall. No new calibration, training, fusion, download, pose/floor/object identity.',
        limits=['Coarseq10/median are reference-depth/r^-2 geometry proxies, not M3 or hardware',
            'Visible camera-relative query is not wearer or hidden complete collision truth',
            'Selected repository-train data and model pretraining overlap prevent fresh confirmation'])
    P.save(out/'PLAN.json',plan)
    print('PREPARED',len(inputs),'frames',len(selected),'scenes',len(plan['families']),'families',flush=True)


def summarize(rows,policies,best,families):
    draws=np.random.default_rng(BOOT_SEED).multinomial(len(families),np.full(len(families),1/len(families)),size=1000)
    result={}
    for stratum in P.STRATA:
        group=P.subset(rows,'whole',stratum)
        stats={a:P.tally(group,a,policies) for a in R.ARMS}
        rates={}
        for arm in R.ARMS:
            cells=[P.tally([r for r in group if r['family']==f],arm,policies) for f in families]
            den=draws@np.array([c['n'] for c in cells]);num=draws@np.array([c['alarms'] for c in cells])
            valid=(den>0)&((draws@np.array([c['uncalibrated_query_rows'] for c in cells]))==0)
            rates[arm]=np.divide(num,den,out=np.full(1000,np.nan),where=valid)
            stats[arm].update(ci95=np.percentile(rates[arm][valid],[2.5,97.5]).tolist() if valid.any() else None,
                bootstrap_valid_draws=int(valid.sum()),bootstrap_excluded_draws=int((~valid).sum()))
        differences={}
        for arm in ('depthpro','anchored_depthpro'):
            delta=(rates[arm]-rates[best])*100;valid=np.isfinite(delta)
            a,b=stats[arm]['rate'],stats[best]['rate']
            gained=[r for r in group if P.alarm(r,arm,policies) is True and P.alarm(r,best,policies) is False]
            lost=[r for r in group if P.alarm(r,arm,policies) is False and P.alarm(r,best,policies) is True]
            differences[arm]=dict(pp=(a-b)*100 if a is not None and b is not None else None,
                ci95_pp=np.percentile(delta[valid],[2.5,97.5]).tolist() if valid.any() else None,
                paired_valid_draws=int(valid.sum()),paired_excluded_draws=int((~valid).sum()),
                gains=len(gained),losses=len(lost),gained=[dict(id=r['id'],query=r['query_name']) for r in gained],
                lost=[dict(id=r['id'],query=r['query_name']) for r in lost])
        result[stratum]=dict(arms=stats,minus_frozen_q10=differences)
    return result


def evaluate(out):
    if (out/'result.json').exists():
        raise FileExistsError('Preserve evaluation')
    plan=P.read(out/'PLAN.json');start=time.perf_counter()
    for path,expected in plan['source_sha256'].items():
        assert P.sha(ROOT/path)==expected,path
    assert P.sha(out/'inference-plan.json')==plan['inference_plan_sha256']
    inference=P.read(out/'inference-result.json')
    assert inference['status']=='COMPLETE'
    inputs=[]
    for item in plan['inputs']:
        path=out/'predictions'/(item['id']+'.npz')
        receipt=P.read(path.with_suffix('.json'))
        assert receipt['output_sha256']==P.sha(path) and receipt['rgb_sha256']==item['rgb_sha256']
        assert receipt['plan_sha256']==plan['inference_plan_sha256'] and receipt['camera_matrix']==item['camera_matrix']
        inputs.append(dict(item,prediction_path=str(path.relative_to(ROOT)),prediction_sha256=receipt['output_sha256']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames=list(pool.map(R.process_frame,inputs))
    P.save(out/'frame-ledger.json',frames)
    rows=[r for f in frames for r in f['rows']]
    assert len(rows)==192
    result=dict(role=plan['role'],plan_sha256=P.sha(out/'PLAN.json'),frames=96,scenes=48,families=len(plan['families']),queries=192,
        truth_counts=dict(Counter(r['category'] for r in rows)),native_partial_frames=sum(not f['geometry']['native_covers_nominal_fov'] for f in frames),
        budgets={},limits=plan['limits'],verdict='FROZEN_TRANSFER_CHECK_COMPLETE')
    for b,old in plan['old_policies'].items():
        assert old['selected_tof']=='coarse_q10'
        result['budgets'][b]=dict(selected_tof=old['selected_tof'],policies=old['policies'],evaluation=summarize(rows,old['policies'],old['selected_tof'],plan['families']))
    result['evaluation_seconds']=time.perf_counter()-start
    P.save(out/'result.json',result)
    lines=['# 非NFO500场景：固定RGB查询工作点迁移','',result['verdict'],'',
        f"96帧/48场景/{len(plan['families'])}family，192查询；全部来自已消费仓库train角色，非新确认。按标识hash选取，无真值挑样。",
        '原native DepthPro预处理及阈值保持，q10继承原cal选择。10/20%为旧校准预算，不保证本批清晰误报相等。',
        '全局尺度锚仅保留原失败配方作对照，无修补；无新模型、训练、融合、下载或目标格赠送。','',
        '|旧cal预算|类别|RGB报警/n|q10报警/n|median报警/n|尺度RGB报警/n|RGB补回/丢失对q10|',
        '|---|---|---|---|---|---|---|']
    for b,d in result['budgets'].items():
        for name,s in d['evaluation'].items():
            counts=[f"{s['arms'][a]['alarms']}/{s['arms'][a]['n']}" for a in ('depthpro','coarse_q10','coarse_median','anchored_depthpro')]
            diff=s['minus_frozen_q10']['depthpro']
            lines.append('|'+ '|'.join([b,name,*counts,f"{diff['gains']}/{diff['losses']}"] )+'|')
    lines+=['','所有family配对区间、全部补回/丢失行、UNKNOWN/缺测/视域和尺度保留JSON。',
        '粗测距是几何代理，非M3或实测ToF。可见相机query不是隐藏物体或穿戴碰撞真值。',
        '这是一组固定方法的迁移诊断；补回不自动成为融合收益，旧失败结果保持。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(result['verdict'],result['truth_counts'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','evaluate'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    (prepare if args.stage=='prepare' else evaluate)(args.out)
