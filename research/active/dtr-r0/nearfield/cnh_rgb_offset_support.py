"""Fixed installation-offset support audit, not a new RGB comparison.

Every old eval frame receives every public offset. Old ToF maps, thresholds and
cal-selected arms are replayed unchanged; no RGB recipe is reopened or retuned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import cnh_rgb_plane_alarm as Q
import cnh_rgb_offset_query as O

ROOT=Q.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-offset-support-20261002'
RUN_ID='CNH_RGB_OFFSET_SUPPORT_20261002'
OFFSETS=(-.2,-.1,0.,.1,.2)
ARMS=Q.TOF_ARMS
read,save,sha=Q.P.read,Q.P.save,Q.P.sha


def prepare():
    assert not (OUT/'PLAN.json').exists()
    original=read(Q.OUT/'PLAN.json');baseline=read(Q.OUT/'result.json')
    rows=[r for r in original['inputs'] if r['split']=='eval']
    assert len(rows)==150 and len({r['id'] for r in rows})==150
    paths=[Path(__file__),Path(O.__file__),Path(Q.__file__),Path(Q.T.__file__),Path(Q.A.__file__),Path(Q.V.__file__)]
    artifacts=[Q.OUT/'PLAN.json',Q.OUT/'result.json',Q.OUT/'frame-ledger.json',Q.T.OUT/'frame-ledger.json']
    OUT.mkdir(parents=True,exist_ok=True)
    row=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same150 oldeval frames, each offset[-.2,-.1,0,.1,.2]m x HEAD/BODY=1500whole0.6–2.1 visible queries; x_body=x_camera-offset; recompute public slab coverage/entryZ/clipping. Replay original3ToF maps and old193cal10/20%policies+selected arm; no fitting/RGB scoring | NOT_RUN; all offsets, shallow0–2/2–5 support, distinct frame/scene/family, misses and actual clear/pass/UNKNOWN; offset0 all300 rows reproduce original | Support/headroom audit only; n<20 or<3families in either shallow bin => LOW_SHALLOW_SUPPORT; otherwise zero misses in both => NO_SHALLOW_HEADROOM; else CONDITIONAL_HEADROOM. No RGB efficacy, no new view/independent units/true full-volume clearance; retain cropped/occluded full-volume UNKNOWN separately | `artifacts.local/work/cnh-rgb-offset-support-20261002/REPORT.md` |')
    text=Q.P.RUNS.read_text(encoding='utf-8');assert RUN_ID not in text
    Q.P.RUNS.write_text(text.rstrip()+'\n'+row+'\n',encoding='utf-8')
    (OUT/'prerun-row.txt').write_text(row+'\n',encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,at=datetime.now(timezone.utc).isoformat(),inputs=rows,offsets_m=OFFSETS,
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
        frozen_baselines={k:dict(policies={a:v['policies'][a] for a in ARMS},selected=v['cal_selected_tof']['selected_arm']) for k,v in baseline['budgets'].items()},
        role='Same150 consumed Development eval frames; five known query-body installation offsets per image, not new images or views',
        truth='Visible points only, unchanged D16 and body0.3m plus acceptable0.1m margin; any cropped/occluded full physical volume cannot be claimed clear',
        scope='Whole0.6–2.1 support only, not the far1.2–2.1 clearance main gate. No resampling of offsets, RGB method, new calibration, rendering, model inference, downloads or training.',
        decision='Both bins >=20 queries and>=3families, then at least one frozen baseline miss for potential headroom; descriptive proxy, no independent evidence growth from repeated offsets.',preregistration=row))
    print('PREPARED150x5x2',flush=True)


def scalar(value):return float(value) if np.isfinite(value) else None


def one(args):
    item,cached,legacy,policies=args
    base=Q.V.ray_geometry((768,1024),item['camera_matrix'])
    values=np.array([np.nan if v is None else v for v in cached['range_prediction']['sensor_q10_radial']],float)
    q10,constant=Q.coarse_maps(base,values)
    block,_=Q.replay_plane(Q.T.block_labels(base),base,q10,cached['tof_block_diagnostics'])
    maps=dict(coarse_q10=q10,coarse_constant_z=constant,tof_block_plane=block)
    assert sha(item['depth_path'])==item['depth_sha256']
    z=Q.S.hdf(item['depth_path'])/base['radial_factor']
    out=[];checks=0
    for offset in OFFSETS:
        geometry=O.offset_geometry(base,offset)
        scores={a:O.query_scores(d,geometry) for a,d in maps.items()}
        truth=O.visible_truth(z,geometry)
        for i,q in enumerate(geometry['queries']):
            row=dict(id=item['id'],scene=item['scene'],family=item['family'],split='eval',query=q['name'],offset_m=offset,
                category=str(truth['category'][i]),contact_bin=str(truth['contact_bin'][i]),truth_abstain=bool(truth['abstain'][i]),
                truth_contact_pixels=int(truth['contact_count'][i]),truth_expanded_pixels=int(truth['expanded_count'][i]),
                truth_score=scalar(truth['score'][i]),truth_coverage=float(truth['coverage'][i]),
                native_clipped=bool(geometry['query_native_clipped'][i]),nominal_clipped=bool(geometry['query_nominal_fov_clipped'][i]),
                full_volume_category=str(truth['full_volume_category'][i]),
                full_volume_hidden_tail_ray_count=int(truth['full_volume_hidden_tail_ray_count'][i]),
                truth_missing_possible_count=int(truth['missing_possible_count'][i]),
                foreground_occluded_possible_count=int(truth['occluded_possible_count'][i]),scores={})
            for arm,s in scores.items():
                row['scores'][arm]=dict(score=scalar(s['score'][i]),abstain=bool(s['abstain'][i]),coverage=float(s['coverage'][i]),
                    return_count=int(s['return_count'][i]),contact_count=int(s['contact_count'][i]),expanded_count=int(s['expanded_count'][i]),
                    missing_possible_count=int(s['missing_possible_count'][i]))
            row['alarms']={k:{a:Q.fired(row,a,p['policies']) for a in ARMS} for k,p in policies.items()}
            if offset==0:
                old=next(r for r in legacy['rows'] if r['query']==row['query'])
                for key in ('category','contact_bin','truth_abstain','truth_contact_pixels','truth_expanded_pixels','truth_score','truth_coverage','native_clipped'):
                    assert row[key]==old[key],(item['id'],key,row[key],old[key]);checks+=1
                for a in ARMS:
                    assert row['scores'][a]==old['scores'][a],(item['id'],a,row['scores'][a],old['scores'][a]);checks+=len(row['scores'][a])
                    for b in policies:assert row['alarms'][b][a]==old['alarms'][b][a];checks+=1
            out.append(row)
    return dict(id=item['id'],rows=out,zero_offset_scalar_checks=checks)


def summarize(rows,policies):
    support={s:dict(queries=len(Q.subset(rows,s)),frames=len({r['id'] for r in Q.subset(rows,s)}),
        scenes=len({r['scene'] for r in Q.subset(rows,s)}),families=sorted({r['family'] for r in Q.subset(rows,s)})) for s in Q.STRATA}
    budgets={}
    for key,p in policies.items():
        arms={a:{s:Q.tally(Q.subset(rows,s),a,p['policies']) for s in Q.STRATA} for a in ARMS}
        best=arms[p['selected']]
        for s in ('contact0-2','contact2-5','contact'):
            c=best[s];c['misses']=c['n']-c['alarms'];c['headroom_pp']=100*c['misses']/c['n'] if c['n'] else None
        low=any(support[s]['queries']<20 or len(support[s]['families'])<3 for s in ('contact0-2','contact2-5'))
        verdict='LOW_SHALLOW_SUPPORT' if low else ('NO_SHALLOW_HEADROOM' if sum(best[s]['misses'] for s in ('contact0-2','contact2-5'))==0 else 'CONDITIONAL_HEADROOM')
        budgets[key]=dict(selected_tof=p['selected'],arms=arms,status=verdict)
    return dict(queries=len(rows),support=support,budgets=budgets,
        native_clipped=sum(r['native_clipped'] for r in rows),nominal_clipped=sum(r['nominal_clipped'] for r in rows),
        occluded=sum(r['foreground_occluded_possible_count']>0 for r in rows),
        full_volume_categories={s:sum(r['full_volume_category']==s for r in rows) for s in ('contact','pass','clear','UNKNOWN')})


def run():
    plan=read(OUT/'PLAN.json');assert not (OUT/'result.json').exists()
    for p,h in {**plan['sources'],**plan['artifacts']}.items():assert sha(ROOT/p)==h,p
    cached={r['id']:dict(range_prediction=r['range_prediction'],tof_block_diagnostics=r['tof_block_diagnostics']) for r in read(Q.T.OUT/'frame-ledger.json')}
    legacy={r['id']:r for r in read(Q.OUT/'frame-ledger.json')}
    start=time.perf_counter();frames=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        args=[(r,cached[r['id']],legacy[r['id']],plan['frozen_baselines']) for r in plan['inputs']]
        for i,frame in enumerate(pool.map(one,args)):
            frames.append(frame)
            if (i+1)%10==0:print(f'audited {i+1}/150',flush=True)
    rows=[r for f in frames for r in f['rows']];assert len(rows)==1500
    save(OUT/'frame-ledger.json',frames)
    result=dict(run_id=RUN_ID,role=plan['role'],all_offsets=summarize(rows,plan['frozen_baselines']),
        by_offset={str(c):summarize([r for r in rows if r['offset_m']==c],plan['frozen_baselines']) for c in OFFSETS},
        by_family={f:summarize([r for r in rows if r['family']==f],plan['frozen_baselines']) for f in sorted({r['family'] for r in rows})},
        zero_offset_scalar_checks=sum(f['zero_offset_scalar_checks'] for f in frames),seconds=time.perf_counter()-start,
        paired_units='All5offsets share frame; no independent sample or fresh confirmation claim',
        limits=plan['scope'])
    save(OUT/'result.json',result)
    print('COMPLETE1500',result['all_offsets']['support'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'))
    globals()[parser.parse_args().action]()
