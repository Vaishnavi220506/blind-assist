"""Fixed 2x2-zone ToF-only plane competitor, without RGB-dependent geometry.

This is a stronger fixed ToF-only method, not a count-matched pure ablation.
All plane fitting, validation, failure handling and tolerances remain in frozen M.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time

import numpy as np

import cnh_rgb_region_plane_gate as H

M,C,P,R,S,A,V,G,ROOT = H.M,H.C,H.P,H.R,H.S,H.A,H.V,H.G,H.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-tof-plane-control-20261002'
RUN_ID = 'CNH_RGB_TOF_PLANE_CONTROL_20261002'


def block_labels(geometry):
    zone = np.asarray(geometry['zone_id'])
    if not np.issubdtype(zone.dtype, np.integer) or np.any((zone < -1) | (zone > 63)):
        raise ValueError('Public zone IDs must be -1 or 0..63')
    inside = zone >= 0
    if not np.array_equal(inside, geometry['fov_mask']):
        raise ValueError('Public zone and FOV membership disagree')
    labels = np.full(zone.shape, -1, np.int32)
    zy, zx = zone[inside]//8, zone[inside]%8
    labels[inside] = (zy//2)*4+(zx//2)
    return labels


def predict(geometry, returns):
    """Two public inputs only: native ray geometry and 64 cached q10 returns."""
    return M.predict(block_labels(geometry), geometry, returns)


def artifact_paths():
    return dict(plane_plan=H.OUT/'PLAN.json', plane_result=H.OUT/'result.json',
                plane_ledger=H.OUT/'frame-ledger.json', support_ledger=S.OUT/'frame-ledger.json')


def prepare():
    assert not (OUT/'PLAN.json').exists(), 'Preserve frozen control'
    previous = P.read(H.OUT/'PLAN.json')
    for path,digest in previous['source_sha256'].items():
        assert P.sha(ROOT/path) == digest, path
    for item in previous['artifacts'].values():
        assert P.sha(item['path']) == item['sha256']
    paths = [Path(__file__),Path(M.__file__),Path(H.__file__),Path(C.__file__),Path(R.__file__),
             Path(S.__file__),Path(A.__file__),Path(V.__file__),Path(G.__file__),Path(P.__file__)]
    row = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same343 consumed Development and fixed135 eval/108 cal contact views; ToF-only public 16 fixed2x2-zone blocks; unchanged plane >=4anchors/highest-zone holdout/trainrank3/train-only flat init/single SLSQP100 ftol1e-12/bbox inverseZ>=1e-6/train+heldout<=.05m; cached64 q10 only; preserve all support/missing/UNKNOWN and prior RGB-plane/four-range ledgers; fixed5/10cm and all-support50% hits | NOT_RUN | Stronger fixed ToF competitor, not same-count ablation or new confirmation; new-arm timing only | `artifacts.local/work/cnh-rgb-tof-plane-control-20261002/result.json` |')
    body = P.RUNS.read_text(encoding='utf8'); assert RUN_ID not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+row+'\n',encoding='utf8')
    (OUT/'prerun-row.txt').write_text(row+'\n',encoding='utf8')
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),
        role='Same consumed Development; no new confirmation, hardware or alarm claim',inputs=previous['inputs'],
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in paths},
        artifacts={k:dict(path=str(p),sha256=P.sha(p)) for k,p in artifact_paths().items()},
        preregistration=row,workers=3,thresholds_m=[.05,.10],native_shape=[768,1024],
        predictor='labels=(zone_y//2)*4+(zone_x//2), outsideFOV=-1; M.predict unchanged; no RGB/SAM cache read',
        limits=previous['limits']+[
            'Fixed block control is not count matched to RGB masks; it is a stronger ToF-only competitor.',
            'All numeric failures and original strict floating-point constraints are preserved; no retries or tolerance changes.',
            'New-arm wall time is descriptive and measured under three CPU workers; old RGB arm has no comparable timing.',
            'No GT best-arm/subset selection. All original contact/pass/UNKNOWN and denominators remain.']))
    print('Prepared fixed ToF-only block plane control')


def evaluate_frame(args):
    row,old,support = args
    # Native dimensions are public fixed metadata, never inferred by opening SAM/RGB.
    geometry = A.whole_geometry(V.ray_geometry((768,1024),row['camera_matrix']))
    returns = np.asarray([np.nan if v is None else v for v in old['range_prediction']['sensor_q10_radial']],float)
    assert returns.shape == (64,)
    start = time.perf_counter()
    prediction = predict(geometry,returns)
    elapsed = time.perf_counter()-start
    # Only after prediction: evaluator-only reference arrays and identities.
    for name in ('depth','instance'):
        assert P.sha(row[name+'_path']) == row[name+'_sha256']
    radial = S.hdf(row['depth_path']); instance = S.hdf(row['instance_path'])
    z = radial/geometry['radial_factor']; y = z*geometry['fy']; x = np.abs(z*geometry['fx'])
    contacts,expanded = [],[]
    for query in geometry['queries']:
        take = geometry['fov_mask'] & np.isfinite(z) & (z >= .6) & (z < 2.1) & (y >= query['y_low']) & (y <= query['y_high'])
        contacts.append(take & (x < .3)); expanded.append(take & (x <= .4))
    for obj in support['objects']:
        target = instance == obj['instance_id']
        for index, q in enumerate(obj['query_support']):
            assert int((target & contacts[index]).sum()) == q['contact_pixels']
            assert int((target & expanded[index]).sum()) == q['expanded_pixels']
    truth = A.visible_truth(z,geometry)
    assert old['queries'] == [dict(name=q['name'],category=str(truth['category'][i]),abstain=bool(truth['abstain'][i]),
        contact_pixels=int(contacts[i].sum()),expanded_pixels=int(expanded[i].sum())) for i,q in enumerate(geometry['queries'])]
    contact = np.logical_or.reduce(contacts); expansion = np.logical_or.reduce(expanded)
    frame = deepcopy(old)
    frame['tof_block_diagnostics'] = prediction['diagnostics']
    frame['tof_block_predict_seconds'] = elapsed
    for obj in frame['objects']:
        target = (instance == obj['instance_id']) & (contact if obj['stratum'] == 'contact' else expansion)
        assert int(target.sum()) == obj['support_pixels']
        obj['tof_block_range'] = C.error_stats(prediction['predicted_z'],z,target)
        obj['tof_block_range']['accepted_pixels'] = int((target & (prediction['region_map'] >= 0)).sum())
    return frame


def summarize(frames):
    out = dict(frames=len(frames),queries=dict(Counter(q['category'] for f in frames for q in f['queries'])),
        block_statuses=dict(Counter(d['status'] for f in frames for d in f['tof_block_diagnostics'])),
        failure_reasons=dict(Counter(reason for f in frames for d in f['tof_block_diagnostics'] for reason in d['reasons'])),
        predict_seconds_sum=sum(f['tof_block_predict_seconds'] for f in frames),strata={})
    for stratum in ('contact','pass_adjacent'):
        objects = [o for f in frames for o in f['objects'] if o['stratum'] == stratum]
        metrics = [o['tof_block_range'] for o in objects]
        n = sum(o['support_pixels'] for o in objects); finite = sum(m['finite_pixels'] for m in metrics)
        error = sum(m['abs_z_error_sum_m'] for m in metrics)
        group = dict(n=len(objects),support_pixels=n,finite_pixels=finite,missing_pixels=n-finite,
            error_sum_m=error,pooled_mae_m=error/finite if finite else None,
            accepted_any=sum(m['accepted_pixels'] > 0 for m in metrics),
            accepted_half=sum(m['accepted_pixels'] >= .5*m['support_pixels'] for m in metrics),
            accepted_pixels=sum(m['accepted_pixels'] for m in metrics),paired={})
        for tag in ('05','10'):
            group['hits_'+tag] = sum(m['hit_'+tag] for m in metrics)
            group['within_'+tag+'_pixels'] = sum(m['within_'+tag+'_pixels'] for m in metrics)
        for baseline in ('rgb_plane','coarse_q10','coarse_constant_z'):
            old = [o['plane_range'] if baseline == 'rgb_plane' else o['range_arms'][baseline] for o in objects]
            pair = dict(n=len(objects))
            for tag in ('05','10'):
                pair[tag] = dict(block_only=sum(m['hit_'+tag] and not b['hit_'+tag] for m,b in zip(metrics,old)),
                    baseline_only=sum(not m['hit_'+tag] and b['hit_'+tag] for m,b in zip(metrics,old)),
                    both=sum(m['hit_'+tag] and b['hit_'+tag] for m,b in zip(metrics,old)),
                    neither=sum(not m['hit_'+tag] and not b['hit_'+tag] for m,b in zip(metrics,old)))
            differences = [m['abs_z_mae_m']-b['abs_z_mae_m'] for m,b in zip(metrics,old)
                           if m['abs_z_mae_m'] is not None and b['abs_z_mae_m'] is not None]
            pair.update(mae_pairs_with_both_finite=len(differences),
                        mean_object_mae_delta_m=float(np.mean(differences)) if differences else None)
            group['paired'][baseline] = pair
        out['strata'][stratum] = group
    return out


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan = P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():
        assert P.sha(ROOT/path) == digest,path
    for item in plan['artifacts'].values():
        assert P.sha(item['path']) == item['sha256'],item['path']
    old = {f['id']:f for f in P.read(H.OUT/'frame-ledger.json')}
    support = {f['id']:f for f in P.read(S.OUT/'frame-ledger.json')}
    ids = [r['id'] for r in plan['inputs']]
    assert len(ids) == len(set(ids)) == len(old) == 343 and set(ids) == set(old)
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = list(pool.map(evaluate_frame,[(r,old[r['id']],support[r['id']]) for r in plan['inputs']]))
    for frame in frames:
        original = deepcopy(frame)
        del original['tof_block_diagnostics'];del original['tof_block_predict_seconds']
        for obj in original['objects']:
            del obj['tof_block_range']
        assert original == old[frame['id']], 'Frozen RGB plane/range ledger changed'
    groups = {split:summarize([f for f in frames if split == 'all' or f['split'] == split]) for split in ('cal','eval','all')}
    families = {split:{family:summarize([f for f in frames if f['family'] == family and (split == 'all' or f['split'] == split)])
        for family in sorted({f['family'] for f in frames if split == 'all' or f['split'] == split})} for split in ('cal','eval','all')}
    assert groups['eval']['strata']['contact']['n'] == 135 and groups['cal']['strata']['contact']['n'] == 108
    P.save(OUT/'frame-ledger.json',frames)
    P.save(OUT/'result.json',dict(run_id=RUN_ID,role=plan['role'],groups=groups,families=families,
        previous_ledger_unchanged=True,timing_scope='Only new ToF block predict under3CPU workers, not a speed comparison',limits=plan['limits']))
    print(json.dumps(P.plain(groups)))


def selftest():
    geometry = V.ray_geometry((32,32),np.diag([.5,.5,-1.]))
    labels = block_labels(geometry); zone = geometry['zone_id']; fov = geometry['fov_mask']
    assert set(np.unique(labels[fov])) == set(range(16))
    assert np.all(labels[~fov] == -1)
    for block in range(16):
        actual = set(np.unique(zone[labels == block]))
        by,bx = divmod(block,4)
        expected = {8*(2*by+dy)+(2*bx+dx) for dy in (0,1) for dx in (0,1)}
        assert actual == expected
    assert list(inspect.signature(predict).parameters) == ['geometry','returns']
    returns = np.full(64,np.nan)  # Exercise wrapper equivalence without running SLSQP.
    a = predict(geometry,returns); b = M.predict(labels,geometry,returns)
    np.testing.assert_array_equal(a['predicted_z'],b['predicted_z'])
    np.testing.assert_array_equal(a['region_map'],b['region_map'])
    assert a['diagnostics'] == b['diagnostics']
    print('PASS16 fixed blocks x4 zones, outsideFOV, two-input API, identical M call with no optimizer execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
