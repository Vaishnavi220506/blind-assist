"""One explicit nonnegative source-feasibility diagnostic; never probabilities.

Prediction functions accept observable packets and anonymous 2D components only.
With free per-return unresolved signal, visible source weights have zero lower
bounds. This weak-model limitation is reported, not hidden by regularization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz126_central_tof import read, write, sha
from mz128_zone_weighting import FOUR, THRESHOLD, column_weights
from mz124_measurement_geometry import metrics
from mz132_contour_adapter import clipped, outside_image
import mz115_spatial_allocation as tof
from research_backend import BackendCandidate, DeviceObservation, select_backend

MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
MZ132 = ROOT/'artifacts.local/work/mz132-visible-contour-20260914/replay-v1'
BRIEF = Path(__file__).with_name('MZ134_JOINT_ATTRIBUTION_BRIEF_20260914.md')


def feasible_intervals(design, signal):
    """Exact single-coordinate bounds for A alpha + u = y; alpha,u >= 0.

Bounds are marginal, not a box of jointly attainable simultaneous maxima.
An all-zero column has unbounded coefficient, no observable contribution.
"""
    a = np.asarray(design, float); y = np.asarray(signal, float)
    assert a.ndim == 2 and a.shape[0] == len(y)
    assert np.isfinite(a).all() and np.isfinite(y).all()
    assert (a >= 0).all() and (y >= 0).all()
    result = []
    for column in a.T:
        selected = column > 0
        upper = float(np.min(y[selected]/column[selected])) if selected.any() else None
        coefficient = upper/2 if upper is not None else 0.
        residual = y-column*coefficient
        assert residual.min(initial=0) >= -1e-10
        assert np.allclose(column*coefficient+residual, y, rtol=0, atol=1e-10)
        result.append(dict(lower=0., upper=upper, observable=bool(selected.any()),
            alternative_coefficient=coefficient, unresolved_residual=residual.tolist()))
    return result


def falsifier():
    # Both source columns have positive area in the same coarse range bin.
    # Source location is deliberately unavailable to the inverse observation.
    design = np.array([[.2]])
    signal = np.array([.4])
    bounds = feasible_intervals(design, signal)[0]
    visible_world = dict(visible_off_corridor_coefficient=2., hidden_in_corridor_signal=0.)
    hidden_world = dict(visible_off_corridor_coefficient=0., hidden_in_corridor_signal=.4)
    signals = [design[0,0]*w['visible_off_corridor_coefficient']+w['hidden_in_corridor_signal']
               for w in (visible_world, hidden_world)]
    assert signals == [.4,.4] and bounds['lower'] == 0 and bounds['upper'] == 2
    return dict(status='PASS_HIDDEN_CORRIDOR_EXPLANATION_UNRESOLVED', range_m=2.,
        worlds=[visible_world,hidden_world], identical_aggregate_signal=signals,
        bounds=bounds, limit='Algebraic inverse-model falsifier, not two rendered physical worlds')


def predict_frame(row, yaw, components):
    evidence = tof.allocate(row,dict(integrated_yaw_deg=yaw,proposals=[]))
    valid = sorted((e for e in evidence if e['status']=='SIM_VALID'),
                   key=lambda e:(e['forward_depth'],e['zone_id'],e['target_slot']))
    groups = []
    for e in valid:
        if not groups or e['forward_depth']-groups[-1][0]['forward_depth']>tof.DEPTH_SPAN_M:
            groups.append([])
        groups[-1].append(e)
    diagnostics = []
    intr = row['rgb_intrinsics']
    for items in groups:
        y = np.array([e['signal']*e['range_m']**2 for e in items])
        visible = [[sum(tof.area(t) for t in clipped(c['tiles'],e['zone_box']))/tof.area(e['zone_box'])
                    for c in components] for e in items]
        outside = [sum(tof.area(t) for t in outside_image(e['zone_box'],intr['width'],intr['height']))/
                   tof.area(e['zone_box']) for e in items]
        a = np.column_stack([np.asarray(visible).reshape(len(items),len(components)),
                             np.ones(len(items)),outside])
        intervals = feasible_intervals(a,y)
        names = ['anonymous_visible_'+str(j) for j in range(len(components))]
        names += ['diffuse_background_nuisance','outside_rgb_nuisance']
        diagnostics.append(dict(returns=[[e['zone_id'],e['target_slot']] for e in items],
            signal=y.tolist(),design=a.tolist(),hypotheses=[dict(name=n,**v) for n,v in zip(names,intervals)],
            all_unresolved_solution=y.tolist(),mandatory_visible_sources=0,
            state='SOURCE_NOT_IDENTIFIABLE_UNDER_FREE_UNRESOLVED_MODEL'))
    weights = column_weights(row,FOUR)
    returns = [dict(zone_id=e['zone_id'],slot=e['target_slot'],status=e['status'],
        range_m=e['range_m'],range_bounds=e['range_bounds'],signal=e['signal'],
        tiles=[e['zone_box']],possible=tof.possible(e['coarse_xyz']),
        weight=weights[e['zone_id']],state='UNRESOLVED_NATIVE_SUPPORT_RETAINED') for e in evidence]
    active = {r['zone_id'] for r in returns if r['possible']}
    score = sum(weights[z] for z in active)
    certain = any(tof.certain(e['coarse_xyz']) for e in evidence)
    flag = bool(score>=THRESHOLD or certain)
    return dict(id=row['id'],candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',
                score=score,certain_coarse=certain),dict(id=row['id'],groups=diagnostics,returns=returns)


def score(output,rows,arms,details):
    # Evaluator-only access starts after predictions and their seal were saved.
    from mz115_allocation_audit import native_truth, project_point, contains
    evaluator_path=SOURCE/'capture-v1/evaluator.jsonl'
    evaluator=[json.loads(s) for s in evaluator_path.read_text().splitlines()]
    report=read(SOURCE/'analysis-v1/frame-report.json')
    assert [r['id'] for r in rows]==[r['id'] for r in evaluator]==[r['id'] for r in report]
    truth=[native_truth(e) for e in evaluator]
    assert truth==[r['truth'] for r in report]
    result=dict(status='MZ134_SOURCE_ATTRIBUTION_NOT_EVALUABLE_WEAK_MODEL',frames=len(rows),arms={})
    for name,pred in arms.items():
        flags=[p['candidate'] for p in pred]
        m=metrics(rows,truth,flags);m['events']=events(rows,truth,flags)
        baseline=arms['mz129_tof' if name.endswith('tof') else 'mz129_full']
        m['lost_TP']=sum(t and b['candidate'] and not a for t,b,a in zip(truth,baseline,flags))
        m['added_FP']=sum(not t and not b['candidate'] and a for t,b,a in zip(truth,baseline,flags))
        m['families']={}
        for family in sorted({r['family'] for r in report}):
            ix=[i for i,r in enumerate(report) if r['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        result['arms'][name]=m
    native=[]
    for row,ev,detail in zip(rows,evaluator,details):
        zones={z['zone_id']:z for z in ev['zonal_tof_native']}
        for ret in detail['returns']:
            zone=zones[ret['zone_id']]
            lineage=next(l for l in zone['returned_lineage'] if l['target_index']==ret['slot'])
            assert lineage['hit_indices']
            for index in lineage['hit_indices']:
                hit=zone['private_rays'][index];point=hit['hit_point_m']
                pixel=project_point(point,ev['camera'],row['rgb_intrinsics'])
                assert pixel is not None
                body=[v-o for v,o in zip(point,ev['body_origin_m'])]
                native.append(dict(id=row['id'],zone_id=ret['zone_id'],slot=ret['slot'],sample=index,
                    retained=any(contains(t,pixel) for t in ret['tiles']),
                    corridor=tof.possible([(v,v) for v in body]),
                    outside_rgb=not(0<=pixel[0]<=640 and 0<=pixel[1]<=360)))
    result['native']=dict(samples=len(native),dropped=sum(not s['retained'] for s in native),
        corridor_samples=sum(s['corridor'] for s in native),
        dropped_corridor=sum(s['corridor'] and not s['retained'] for s in native),
        outside_rgb_samples=sum(s['outside_rgb'] for s in native),
        dropped_outside_rgb=sum(s['outside_rgb'] and not s['retained'] for s in native))
    groups=[g for d in details for g in d['groups']]
    result['attribution']=dict(cohorts=len(groups),mandatory_visible_sources=0,
        cohorts_with_visible_alternative=sum(any(h['name'].startswith('anonymous_visible_') and
            h['upper'] is not None and h['upper']>0 for h in g['hypotheses']) for g in groups),
        cohorts_with_outside_rgb_alternative=sum(g['hypotheses'][-1]['upper'] is not None and
            g['hypotheses'][-1]['upper']>0 for g in groups),
        retained_returns=sum(len(d['returns']) for d in details),
        return_statuses=dict(Counter(r['status'] for d in details for r in d['returns'])),
        limits='Zero lower bounds follow from the explicitly free unresolved model; not a universal identifiability theorem for physical sensors')
    write(output/'native-retention.json',native)
    write(output/'summary.json',result)
    seal=read(output/'prediction-seal.json')
    for path,digest in seal['input_hashes'].items():assert sha(Path(path))==digest,path
    assert sha(output/'predictions.json')==seal['predictions_sha256']
    assert sha(output/'support-details.json')==seal['support_sha256']
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        evaluator_sha256=sha(evaluator_path),resources='No persistent process or allocation'))
    print(json.dumps(result,indent=2))


def run(output):
    assert BRIEF.exists()
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw=SOURCE/'capture-v1/raw.jsonl';anonymous=MZ132/'anonymous-candidates.json'
    correction=CORRECTION/'predictions.json';baseline_path=MZ129/'replay-v1r2/predictions.json'
    radar_path=MZ129/'radar-v1/predictions.json'
    seal132=read(MZ132/'prediction-seal.json');seal125=read(CORRECTION/'prediction-seal.json')
    assert sha(anonymous)==seal132['anonymous_sha256']
    assert sha(raw)==seal125['raw_sha256'] and sha(correction)==seal125['predictions_sha256']
    for directory in (MZ129/'radar-v1',MZ129/'replay-v1r2'):
        previous=read(directory/'prediction-seal.json')
        assert sha(directory/'predictions.json')==previous['predictions_sha256']
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    components=read(anonymous);cache=read(correction);radar=read(radar_path)
    baseline=read(baseline_path)['radar']
    assert len(rows)==len(components)==len(cache)==len(radar)==len(baseline)==288
    assert [r['id'] for r in rows]==[r['id'] for r in components]==[r['id'] for r in radar]==[r['id'] for r in baseline]
    # No cached proposals/supports are supplied to the new predictor.
    yaw=[r['integrated_yaw_deg'] for r in cache]
    output.mkdir(parents=True)
    select_backend('scalar-scoring',cpu=BackendCandidate('nonnegative-feasible-intervals','cpu',
        lambda:bool(rows),lambda _:DeviceObservation('cpu','host CPU','Python/NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=output/'backend.json')
    write(output/'falsifier.json',falsifier())
    start=time.perf_counter()
    pairs=[predict_frame(r,y,c['components']) for r,y,c in zip(rows,yaw,components)]
    seconds=time.perf_counter()-start
    tof_predictions=[p for p,d in pairs];details=[d for p,d in pairs]
    arms=dict(mz129_tof=[dict(id=r['id'],candidate=bool(p['score']>=THRESHOLD or p['certain_coarse']))
        for r,p in zip(rows,baseline)],mz129_full=baseline,joint_tof=tof_predictions,
        joint_full=[dict(p,candidate=bool(p['candidate'] or rb['candidate']),
            candidate_state='ALERT' if p['candidate'] or rb['candidate'] else 'UNKNOWN',
            radar=rb['candidate']) for p,rb in zip(tof_predictions,radar)])
    for p,rb in zip(baseline,radar):
        assert p['candidate']==bool(p['score']>=THRESHOLD or p['certain_coarse'] or rb['candidate'])
    write(output/'predictions.json',arms);write(output/'support-details.json',details)
    paths=[raw,anonymous,correction,baseline_path,radar_path,BRIEF,Path(__file__),
        MZ132/'prediction-seal.json',CORRECTION/'prediction-seal.json',
        MZ129/'radar-v1/prediction-seal.json',MZ129/'replay-v1r2/prediction-seal.json']
    paths += [Path(__file__).with_name(n) for n in ['mz115_spatial_allocation.py',
        'mz132_contour_adapter.py','mz128_zone_weighting.py','mz124_measurement_geometry.py']]
    write(output/'prediction-seal.json',dict(input_hashes={str(p):sha(p) for p in paths},
        predictions_sha256=sha(output/'predictions.json'),support_sha256=sha(output/'support-details.json'),
        seconds=seconds,authority='OBSERVABLE_CONSUMED_PREDICTIONS_SAVED_BEFORE_EVALUATOR_ACCESS',
        backend='CPU TASK_NOT_GPU_SUITABLE'))
    score(output,rows,arms,details)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
