"""Three-level whole-range query alarms from frozen reconstructed plane outputs.

No fitting: accepted parameters are replayed on original public region rays.
Whole [.6,2.1) visible-point truth is not a far-range metric-clearance gate.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import cnh_rgb_tof_plane_control as T

H,M,C,P,R,S,A,V,G,ROOT = T.H,T.M,T.C,T.P,T.R,T.S,T.A,T.V,T.G,T.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-plane-alarm-20261002'
RUN_ID = 'CNH_RGB_PLANE_ALARM_20261002'
ARMS = ('coarse_q10','coarse_constant_z','rgb_plane','tof_block_plane')
TOF_ARMS = ('coarse_q10','coarse_constant_z','tof_block_plane')
STRATA = ('contact0-2','contact2-5','contact>5','contact','clear','pass','UNKNOWN')
BUDGETS = (.10,.20)


def coarse_maps(geometry, returns):
    values = np.asarray(returns,float).reshape(64)
    values = np.where(np.isfinite(values)&(values>0),values,np.nan)
    factor = geometry['radial_factor']; zone = geometry['zone_id']; fov = geometry['fov_mask']
    coefficient = V.coarse_returns(factor,geometry,.1)['zone_return_radial'].reshape(64)
    radial = np.full(zone.shape,np.nan); constant = radial.copy()
    radial[fov] = values[zone[fov]]/factor[fov]
    constant[fov] = (values/coefficient)[zone[fov]]
    return radial,constant


def replay_plane(labels, geometry, fallback, diagnostics):
    """Cached parameters only. Neither a solver nor reference data enters here."""
    output = fallback.copy(); accepted = np.zeros(labels.shape,bool); seen = set()
    for item in diagnostics:
        region = item['region']; assert region not in seen; seen.add(region)
        if item['status'] != 'accepted':
            continue
        params = np.asarray(item['parameters'],float)
        assert params.shape == (3,) and np.isfinite(params).all()
        mask = geometry['fov_mask'] & (labels == region); assert mask.any()
        inverse = params[0]*geometry['fx'][mask]+params[1]*geometry['fy'][mask]+params[2]
        assert np.isfinite(inverse).all() and np.all(inverse > 0)
        output[mask] = 1./inverse; accepted[mask] = True
    return output,accepted


def reconstruct(geometry, returns, sam_labels, rgb_diagnostics, block_diagnostics):
    q10,constant = coarse_maps(geometry,returns)
    rgb,ra = replay_plane(sam_labels,geometry,q10,rgb_diagnostics)
    block,ba = replay_plane(T.block_labels(geometry),geometry,q10,block_diagnostics)
    return dict(coarse_q10=q10,coarse_constant_z=constant,rgb_plane=rgb,tof_block_plane=block),dict(rgb_plane=ra,tof_block_plane=ba)


def artifacts():
    return dict(control_plan=T.OUT/'PLAN.json',control_result=T.OUT/'result.json',
                control_ledger=T.OUT/'frame-ledger.json',inference_audit=R.OUT/'inference-audit.json')


def prepare():
    assert not (OUT/'PLAN.json').exists()
    previous = P.read(T.OUT/'PLAN.json')
    for path,digest in previous['source_sha256'].items():
        assert P.sha(ROOT/path) == digest,path
    for item in previous['artifacts'].values():
        assert P.sha(item['path']) == item['sha256']
    modules = [T,H,M,C,R,S,A,V,G,P]
    paths = [Path(__file__)]+[Path(mod.__file__) for mod in modules]
    prereg = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; replay cached accepted RGB/block planes only, no fitting; same343/686 whole0.6-2.1m queries; q10/constantZ/RGBplane/ToFblock; exact P.calibrate perquery lowest feasible cal breakpoint at10/20%clear; pass zero cost/UNKNOWN retained; one global ToF candidate perbudget selected only on all cal by shallow0-2,2-5,contact,lowclear,fixed order; no bootstrap; eval actual clear and all bins retained | NOT_RUN | DESCRIPTIVE if shallow n<20 or <3families; cal budget is not eval matched cost; not far clearance gate or confirmation | `artifacts.local/work/cnh-rgb-plane-alarm-20261002/result.json` |')
    body = P.RUNS.read_text(encoding='utf8'); assert RUN_ID not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+prereg+'\n',encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n',encoding='utf8')
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),inputs=previous['inputs'],
        role='Same consumed Development, descriptive cached-output diagnostic; no confirmation',
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in paths},
        artifacts={k:dict(path=str(p),sha256=P.sha(p)) for k,p in artifacts().items()},preregistration=prereg,
        rules=dict(arms=list(ARMS),tof_fixed_order=list(TOF_ARMS),budgets=list(BUDGETS),workers=3,queries_per_frame=2,
            truth='Original A.visible_truth whole0.6-2.1; pass costs0; UNKNOWN retained, excluded from clear budget/contact recall',
            threshold='P.calibrate unchanged: ascending finite scores from all cal rows, then all-off; lowest clear-budget-feasible threshold perquery; no cal clear NOT_EVALUABLE',
            breakpoints='All calibration scores supply candidate breakpoints; contact outcomes do not optimize thresholds',
            tof_selection='One global arm perbudget, all cal whole queries; priority recalls0-2,2-5,all-contact,lower clear rate, fixed arm order',
            reconstruction='Only cached accepted inverseZ plane parameters on original region masks; all failed regions keep q10; no fitting',
            guards='Any shallow bin n<20 or<3families implies DESCRIPTIVE_LOW_SHALLOW_SUPPORT; eval clear over budget cannot be called same-cost gain',
            bootstrap=False),limits=['Whole0.6-2.1 only; not far1.2-2.1 or metric-clearance main gate.',
                'A calibration-selected ToF candidate is not a theoretical optimum.',
                'Calibration budget does not imply equal or in-budget evaluation clear alarms.',
                'Visible truth excludes hidden surfaces and outside-FOV space; abstained positive predictions remain misses.',
                'Reference data is read only after reconstruction, for original truth and old object-metric parity checks.']))
    print('Prepared frozen cached plane alarm readout')


def validate_reconstruction(old, maps, accepted, reference_z, instance, contacts, expanded):
    contact = np.logical_or.reduce(contacts); expansion = np.logical_or.reduce(expanded)
    for obj in old['objects']:
        target = (instance == obj['instance_id']) & (contact if obj['stratum']=='contact' else expansion)
        assert int(target.sum()) == obj['support_pixels']
        expected = dict(coarse_q10=obj['range_arms']['coarse_q10'],coarse_constant_z=obj['range_arms']['coarse_constant_z'],
                        rgb_plane=obj['plane_range'],tof_block_plane=obj['tof_block_range'])
        for arm in ARMS:
            current = C.error_stats(maps[arm],reference_z,target)
            assert all(current[key] == expected[arm][key] for key in current), (old['id'],obj['instance_id'],arm)
            if arm in accepted:
                assert int((target & accepted[arm]).sum()) == expected[arm]['accepted_pixels']


def evaluate_frame(args):
    row,old,audit = args
    path = R.OUT/'predictions'/(row['id']+'.npz')
    assert P.sha(path) == audit['output_sha256'] and P.sha(path.with_suffix('.json')) == audit['receipt_sha256']
    with np.load(path,allow_pickle=False) as f:
        labels = f['labels'].copy()
    assert labels.shape == (768,1024)
    geometry = A.whole_geometry(V.ray_geometry((768,1024),row['camera_matrix']))
    returns = np.asarray([np.nan if v is None else v for v in old['range_prediction']['sensor_q10_radial']],float)
    maps,accepted = reconstruct(geometry,returns,labels,old['plane_diagnostics'],old['tof_block_diagnostics'])
    scores = {arm:A.query_scores(maps[arm],geometry) for arm in ARMS}
    # Evaluator starts here; no reference is passed to reconstruction or scoring.
    for name in ('depth','instance'):
        assert P.sha(row[name+'_path']) == row[name+'_sha256']
    radial = S.hdf(row['depth_path']); instance = S.hdf(row['instance_path'])
    z = radial/geometry['radial_factor']; truth = A.visible_truth(z,geometry)
    contacts,expanded = [],[]
    for q in geometry['queries']:
        take = geometry['fov_mask'] & np.isfinite(z) & (z >= .6) & (z < 2.1) & (z*geometry['fy'] >= q['y_low']) & (z*geometry['fy'] <= q['y_high'])
        contacts.append(take & (np.abs(z*geometry['fx']) < .3)); expanded.append(take & (np.abs(z*geometry['fx']) <= .4))
    assert old['queries'] == [dict(name=q['name'],category=str(truth['category'][i]),abstain=bool(truth['abstain'][i]),
        contact_pixels=int(contacts[i].sum()),expanded_pixels=int(expanded[i].sum())) for i,q in enumerate(geometry['queries'])]
    validate_reconstruction(old,maps,accepted,z,instance,contacts,expanded)
    result = []
    for i,q in enumerate(geometry['queries']):
        entry = dict(id=row['id'],scene=row['scene'],family=row['family'],split=row['split'],query=q['name'],
            distance='whole0.6-2.1',category=str(truth['category'][i]),contact_bin=str(truth['contact_bin'][i]),
            truth_abstain=bool(truth['abstain'][i]),truth_contact_pixels=int(truth['contact_count'][i]),
            truth_expanded_pixels=int(truth['expanded_count'][i]),truth_score=float(truth['score'][i]),
            truth_coverage=float(truth['coverage'][i]),native_clipped=bool(geometry['query_native_clipped'][i]),scores={})
        for arm in ARMS:
            item = scores[arm]
            entry['scores'][arm] = dict(score=float(item['score'][i]) if np.isfinite(item['score'][i]) else None,
                abstain=bool(item['abstain'][i]),coverage=float(item['coverage'][i]),
                return_count=int(item['return_count'][i]),contact_count=int(item['contact_count'][i]),
                expanded_count=int(item['expanded_count'][i]),missing_possible_count=int(item['missing_possible_count'][i]))
        result.append(entry)
    return dict(id=row['id'],rows=result,reconstruction_object_metrics_verified=True)


def subset(rows,stratum):
    return [r for r in rows if r['contact_bin']==stratum] if stratum.startswith('contact') and stratum!='contact' else [r for r in rows if r['category']==stratum]


def fired(row,arm,policies):
    policy = policies[arm][row['query']]
    if policy['status'] != 'READY':
        return None
    value = row['scores'][arm]['score']
    return bool(value is not None and not row['scores'][arm]['abstain'] and policy['threshold'] is not None and value >= policy['threshold'])


def tally(rows,arm,policies):
    flags = [fired(r,arm,policies) for r in rows]; n=len(rows); missing=sum(v is None for v in flags)
    alarms=sum(v is True for v in flags)
    return dict(n=n,alarms=alarms,rate=alarms/n if n and not missing else None,uncalibrated_rows=missing,
                prediction_abstain=sum(r['scores'][arm]['abstain'] for r in rows),families=len({r['family'] for r in rows}))


def select_tof(cal,policies):
    candidates=[]
    for index,arm in enumerate(TOF_ARMS):
        ready=all(p['status']=='READY' for p in policies[arm].values())
        counts={s:tally(subset(cal,s),arm,policies) for s in ('contact0-2','contact2-5','contact','clear')}
        rates=[counts[s]['rate'] for s in ('contact0-2','contact2-5','contact')]
        key=tuple(-1. if rate is None else rate for rate in rates)+(-counts['clear']['rate'] if counts['clear']['rate'] is not None else -np.inf,-index)
        candidates.append(dict(arm=arm,calibrated=ready,cal_counts=counts,selection_key=key))
    eligible=[c for c in candidates if c['calibrated']]
    return dict(status='READY' if eligible else 'NOT_EVALUABLE_NO_CAL_CLEAR',
                selected_arm=max(eligible,key=lambda c:c['selection_key'])['arm'] if eligible else None,candidates=candidates)


def pairing(rows,baseline,policies):
    groups={}
    for stratum in STRATA:
        selected=subset(rows,stratum); pairs=[(fired(r,'rgb_plane',policies),fired(r,baseline,policies)) for r in selected]
        valid=[(a,b) for a,b in pairs if a is not None and b is not None]
        groups[stratum]=dict(n=len(selected),paired_n=len(valid),uncalibrated_rows=len(pairs)-len(valid),
            rgb_only=sum(a and not b for a,b in valid),baseline_only=sum(not a and b for a,b in valid),
            both=sum(a and b for a,b in valid),neither=sum(not a and not b for a,b in valid))
    return groups


def summarize(rows,policies,selection,budget):
    output=dict(queries=len(rows),arms={arm:{s:tally(subset(rows,s),arm,policies) for s in STRATA} for arm in ARMS})
    output['shallow_support']={s:dict(n=len(subset(rows,s)),families=len({r['family'] for r in subset(rows,s)})) for s in ('contact0-2','contact2-5')}
    low=any(v['n']<20 or v['families']<3 for v in output['shallow_support'].values())
    output['status']='DESCRIPTIVE_LOW_SHALLOW_SUPPORT' if low else 'DESCRIPTIVE_CACHED_DEVELOPMENT'
    output['clear_budget']={arm:dict(n=output['arms'][arm]['clear']['n'],alarms=output['arms'][arm]['clear']['alarms'],
        rate=output['arms'][arm]['clear']['rate'],within_budget=(output['arms'][arm]['clear']['rate']<=budget+1e-12)
        if output['arms'][arm]['clear']['rate'] is not None else None) for arm in ARMS}
    output['rgb_vs_block']=pairing(rows,'tof_block_plane',policies)
    best=selection['selected_arm']
    output['selected_tof_arm']=best
    output['rgb_vs_selected_tof']=pairing(rows,best,policies) if best is not None else None
    if best is not None:
        a,b=output['clear_budget']['rgb_plane'],output['clear_budget'][best]
        ready=a['within_budget'] is not None and b['within_budget'] is not None
        output['cost_comparison']=dict(both_actual_clear_rates_available=ready,
            both_within_budget=bool(a['within_budget'] and b['within_budget']) if ready else None,
            equal_observed_clear_alarms=a['alarms']==b['alarms'] if ready else None,
            clear_alarm_delta_rgb_minus_tof=a['alarms']-b['alarms'] if ready else None,
            interpretation='Contact RGB-only recovers contact alerts; clear RGB-only adds clear alarms. Calibration budget alone is not matched evaluation cost.')
    return output


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan=P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():assert P.sha(ROOT/path)==digest,path
    for item in plan['artifacts'].values():assert P.sha(item['path'])==item['sha256']
    old={f['id']:f for f in P.read(T.OUT/'frame-ledger.json')}
    audit={f['id']:f for f in P.read(R.OUT/'inference-audit.json')['frames']}
    ids=[r['id'] for r in plan['inputs']];assert len(ids)==len(set(ids))==343 and set(ids)==set(old)==set(audit)
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames=list(pool.map(evaluate_frame,[(r,old[r['id']],audit[r['id']]) for r in plan['inputs']]))
    rows=[q for frame in frames for q in frame['rows']];assert len(rows)==686
    cal=[r for r in rows if r['split']=='cal']; queries=sorted({r['query'] for r in rows});assert len(queries)==2
    result=dict(run_id=RUN_ID,role=plan['role'],whole_range_m=[.6,2.1],queries=686,budgets={},limits=plan['limits'],bootstrap_performed=False,
                original_truth_and_object_metric_reconstruction_verified=True)
    for budget in BUDGETS:
        key=str(int(100*budget))
        policies={arm:{q:P.calibrate(cal,arm,q,budget) for q in queries} for arm in ARMS}
        selection=select_tof(cal,policies)
        groups={split:summarize([r for r in rows if split=='all' or r['split']==split],policies,selection,budget) for split in ('cal','eval','all')}
        by_query={split:{q:summarize([r for r in rows if r['query']==q and (split=='all' or r['split']==split)],policies,selection,budget) for q in queries} for split in ('cal','eval','all')}
        by_family={split:{family:summarize([r for r in rows if r['family']==family and (split=='all' or r['split']==split)],policies,selection,budget)
                         for family in sorted({r['family'] for r in rows if split=='all' or r['split']==split})} for split in ('cal','eval','all')}
        result['budgets'][key]=dict(budget=budget,policies=policies,cal_selected_tof=selection,groups=groups,by_query=by_query,by_family=by_family)
        for row in rows:
            row.setdefault('alarms',{})[key]={arm:fired(row,arm,policies) for arm in ARMS}
    result['status']='DESCRIPTIVE_LOW_SHALLOW_SUPPORT' if any(b['groups']['eval']['status']=='DESCRIPTIVE_LOW_SHALLOW_SUPPORT' for b in result['budgets'].values()) else 'DESCRIPTIVE_CACHED_DEVELOPMENT'
    P.save(OUT/'frame-ledger.json',frames);P.save(OUT/'result.json',result)
    print(json.dumps(dict(status=result['status'],queries=len(rows))))


def selftest():
    def row(category,score,abstain=False):
        return dict(query='HEAD',category=category,contact_bin='contact0-2' if category=='contact' else '',family='toy',
                    scores={arm:dict(score=score,abstain=abstain) for arm in ARMS})
    rows=[row('clear',.4),row('clear',.4),row('clear',.1),row('clear',.1),row('contact',.6),row('pass',.2)]
    policy=P.calibrate(rows,'coarse_q10','HEAD',.25)
    assert policy['threshold']==.6 and policy['clear_alarms']==0  # Cannot split tied clear scores.
    equal=P.calibrate([row('clear',.4),row('contact',.4)],'coarse_q10','HEAD',.1)
    assert equal['threshold'] is None and equal['all_off']
    no_clear=P.calibrate([row('contact',.1)],'coarse_q10','HEAD',.1)
    assert no_clear['status']!='READY' and no_clear['threshold'] is None
    policies={arm:{'HEAD':dict(status='READY',threshold=.3)} for arm in ARMS}
    assert fired(row('contact',None,True),'coarse_q10',policies) is False
    assert select_tof(rows,policies)['selected_arm']=='coarse_q10'  # Global fixed-order tie.
    g=A.whole_geometry(V.ray_geometry((32,32),np.diag([.4,.4,-1.])))
    labels=np.where(g['fx']<0,0,1).astype(np.int32)
    values=np.full(64,1.5)
    rgb=[dict(region=0,status='accepted',parameters=[.1,-.05,.8]),dict(region=1,status='rejected',parameters=[0,0,.1])]
    block=[dict(region=0,status='accepted',parameters=[0,0,1.])]
    maps,accepted=reconstruct(g,values,labels,rgb,block)
    base,constant=coarse_maps(g,values)
    take=g['fov_mask']&(labels==0)
    np.testing.assert_allclose(maps['rgb_plane'][take],1/(.1*g['fx'][take]-.05*g['fy'][take]+.8))
    np.testing.assert_array_equal(maps['rgb_plane'][~take],base[~take])
    np.testing.assert_array_equal(maps['coarse_q10'],base);np.testing.assert_array_equal(maps['coarse_constant_z'],constant)
    bmask=g['fov_mask']&(T.block_labels(g)==0)
    np.testing.assert_array_equal(maps['tof_block_plane'][bmask],np.ones(int(bmask.sum())))
    np.testing.assert_array_equal(maps['tof_block_plane'][~bmask],base[~bmask])
    assert np.array_equal(accepted['rgb_plane'],take) and np.array_equal(accepted['tof_block_plane'],bmask)
    assert len(A.query_scores(maps['rgb_plane'],g)['score'])==2
    print('PASS unchanged calibration ties/all-off/no-clear, abstain, global ToF tie, complete cached reconstruction/fallback')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
