"""Frozen metric prior contrast; public predictions are sealed before evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from run_mz161_dense_task import ROOT, CODE, CAP, BASE, read, write, sha
from run_mz139_surface_fit import selected_jsonl, local_dependencies
from mz136_incumbent import public_observations
from mz136_boundary_geometry import camera_to_body
from mz164_metric_prior import load_model, predict

WORK = ROOT/'artifacts.local/work/mz164-camera-metric-prior-20260916'
ARMS = ('known_camera', 'estimated_camera')
LIMIT = 1200.


def corridor_mask(row, depth, yaw):
    intr = row['rgb_intrinsics']
    yy, xx = np.mgrid[:intr['height'], :intr['width']]
    rays = np.stack([np.ones_like(xx), (xx-intr['cx'])/intr['fx'], (intr['cy']-yy)/intr['fy']], -1)
    points = depth[..., None]*(rays @ camera_to_body(row, yaw).T)+np.asarray(row['camera_in_body_m'])
    return (np.isfinite(depth) & (depth > 0) &
            np.all((points >= [.2,-.3,.4]) & (points <= [3.6,.3,2.05]), axis=-1))


def alert(row, depth, yaw):
    mask = corridor_mask(row, depth, yaw)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return dict(candidate=bool(row['imu_valid'] and largest >= 9), largest_patch_pixels=largest,
                corridor_pixels=int(mask.sum()), imu_available=bool(row['imu_valid']))


def correct_target_patch(mask, target_corridor, depth, reference_depth):
    correct = mask & target_corridor & np.isfinite(depth) & np.isfinite(reference_depth)
    correct &= np.abs(depth-reference_depth) <= .12
    _, _, stats, _ = cv2.connectedComponentsWithStats(correct.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return dict(correct_target_corridor_pixels=int(correct.sum()), largest_correct_target_patch_pixels=largest)


def target_stratum(row, evaluation):
    statuses = []
    zones = {z['zone_id']: z for z in row['tof_zones']}
    for z in evaluation['zonal_tof_native']:
        rays = {r['subray']: r for r in z['private_rays']}
        for line in z['returned_lineage']:
            if any(str(rays[i].get('actor_id', '')).endswith('/shape0') for i in line['hit_indices']):
                statuses.append(zones[z['zone_id']]['targets'][line['target_index']]['status'])
    return 'valid_supported' if 'SIM_VALID' in statuses else 'merged_only' if statuses else 'no_target_return'


def geometry_summary(cases, arm):
    groups = {}
    for family in ['ALL']+sorted({c['family'] for c in cases}):
        groups[family] = {}
        for stratum in ('all', 'no_target_return', 'merged_only', 'valid_supported'):
            cc = [c for c in cases if (family == 'ALL' or c['family'] == family) and
                  (stratum == 'all' or c['tof_stratum'] == stratum)]
            rr = [c['arms'][arm]['geometry'] for c in cc]
            cols = sum(r['target']['transverse_columns']['visible'] for r in rr)
            good = sum(r['target']['transverse_columns']['with_at_least_half_pixels_agreeing'] for r in rr)
            px = sum(r['target']['visible_pixels'] for r in rr)
            agreeing = sum(r['target']['range_agreeing_pixels'] for r in rr)
            comparable = sum(r['target']['comparable_pixels'] for r in rr)
            ring = sum(r['silhouette_neighborhood']['referenced_pixels'] for r in rr)
            over = sum(r['silhouette_neighborhood']['closer_or_target_like_error_pixels'] for r in rr)
            spurious = sum(r['silhouette_neighborhood']['target_depth_band_spurious_pixels'] for r in rr)
            error = sum((r['target']['absolute_error_m']['mean'] or 0)*r['target']['absolute_error_m']['count'] for r in rr)
            groups[family][stratum] = dict(frames=len(cc), target_columns=cols, recovered_columns=good,
                column_coverage=good/cols if cols else None, target_pixels=px, agreeing_pixels=agreeing,
                comparable_pixels=comparable, target_mae_m=error/comparable if comparable else None,
                ring_pixels=ring, too_near_ring_pixels=over, too_near_ring_rate=over/ring if ring else None,
                local_precision=agreeing/(agreeing+spurious) if agreeing+spurious else None,
                native={part: dict(in_rgb=sum(r['native_returned_contributors']['summary'][part]['inside_rgb'] for r in rr),
                    agreeing=sum(r['native_returned_contributors']['summary'][part]['diagnostic_012m_agreement'] for r in rr))
                    for part in ('all','corridor')})
    return groups


def run(output):
    start = time.perf_counter()
    assert not output.exists() and output.resolve().is_relative_to(WORK.resolve()) and torch.cuda.is_available()
    output.mkdir(parents=True)
    (output/'predictions').mkdir()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.manual_seed(164016); np.random.seed(164016)
    def within():
        if time.perf_counter()-start >= LIMIT: raise TimeoutError('MZ164 allocation exhausted')
    spec, receipt = read(CAP/'spec.json'), read(CAP/'receipt.json')
    assert receipt['status'] == 'PASS' and sha(CAP/'spec.json') == receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'): assert sha(CAP/n) == receipt['hashes'][n]
    ids = {f['id'] for f in spec['frames'] if f['split'] == 'train'}
    rows = public_observations(selected_jsonl(CAP/'raw.jsonl', ids))
    assert len(ids) == len(rows) == 192
    assert [f['id'] for f in read(BASE/'folds.json')[:192]] == [r['id'] for r in rows]
    assert sha(BASE/'oof.npz') == read(BASE/'model-seal.json')['oof_sha256']
    with np.load(BASE/'oof.npz', allow_pickle=False) as saved:
        baseline = saved['baseline'][:192].astype(bool)
    sources = local_dependencies(__file__)
    sources[str(CODE/'MZ164_PROTOCOL_20260916.md')] = sha(CODE/'MZ164_PROTOCOL_20260916.md')
    for p in (WORK/'upstream/unidepth').rglob('*.py'): sources[str(p)] = sha(p)
    inputs = {str(p):sha(p) for p in [CAP/n for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json')]+
              [BASE/n for n in ('oof.npz','folds.json','model-seal.json')]+
              [WORK/n for n in ('model.safetensors','config.json','download-receipt.json',
                               'adapter-preflight/benchmark.json','adapter-preflight/tests.log',
                               'backend-selection.json','backend-equivalence.json','readout-tests.log')]}
    for row in rows:
        p = (CAP/row['rgb_path']).resolve()
        assert p.is_relative_to(CAP.resolve()) and sha(p) == receipt['hashes'][row['rgb_path']]
        inputs[str(p)] = sha(p)
    write(output/'freeze.json', dict(time_utc=datetime.now(timezone.utc).isoformat(), sources=sources, inputs=inputs,
        ids=[r['id'] for r in rows], arms=ARMS, compute_cap_s=LIMIT, fit=False, original_dev_test_access=False,
        access_detail='Full source spec parsed for TRAIN selection; no dev/test raw/evaluator/RGB decode',
        baseline_cache_scope='Only TRAIN boolean member; first192 selected by exact fold ID order',
        real_native_parse_before_predictions=False, authority='CONSUMED_TRAIN192_FIXED_METRIC_PRIOR'))
    model = None
    try:
        model, binding = load_model(WORK, 'cuda')
        write(output/'model-binding.json', binding)
        torch.cuda.reset_peak_memory_stats()
        manifest = {}; predictions = []; yaws = []
        yaw = 0.; previous = None; gpu_s = {a:0. for a in ARMS}; peak_bytes = 0
        for i, row in enumerate(rows):
            if row['episode_id'] != previous: yaw = 0.
            if row['imu_valid']: yaw += row['delta_yaw']
            previous = row['episode_id']; yaws.append(yaw)
            image = cv2.cvtColor(cv2.imread(str(CAP/row['rgb_path'])), cv2.COLOR_BGR2RGB)
            maps = {}; report = dict(id=row['id'], yaw_deg=yaw, baseline=bool(baseline[i]), arms={})
            for arm in ARMS:
                torch.cuda.synchronize(); tick = time.perf_counter()
                depth, audit = predict(model, image, row['rgb_intrinsics'], arm)
                peak_bytes = max(peak_bytes, audit['peak_allocated_bytes'] or 0)
                torch.cuda.synchronize(); gpu_s[arm] += time.perf_counter()-tick
                assert depth.shape == (360,640) and depth.dtype == np.float32
                maps[arm] = depth
                report['arms'][arm] = dict(**alert(row, depth, yaw), adapter=audit,
                    finite_positive_pixels=int((np.isfinite(depth)&(depth>0)).sum()))
            dest = output/'predictions'/(row['id']+'.npz')
            np.savez_compressed(dest, **maps)
            manifest[str(dest.relative_to(output))] = sha(dest)
            predictions.append(report); within()
            if (i+1)%24 == 0: print(json.dumps(dict(stage='public_prediction', frames=i+1, seconds=time.perf_counter()-start)), flush=True)
        write(output/'predictions.json', predictions)
        assert all(sha(p) == h for p,h in (sources|inputs).items())
        write(output/'prediction-seal.json', dict(freeze_sha256=sha(output/'freeze.json'), maps=manifest,
            predictions_sha256=sha(output/'predictions.json'), model_binding_sha256=sha(output/'model-binding.json'),
            gpu_seconds=gpu_s, peak_allocated_bytes=peak_bytes,
            authority='ALL_MAPS_AND_FLAGS_SEALED_BEFORE_SELECTED_NATIVE_PARSE'))
        del model; model = None; torch.cuda.empty_cache()
        print(json.dumps(dict(stage='SEALED', gpu_seconds=gpu_s)), flush=True)
        # Evaluator-only imports and records enter after the predictor seal.
        from evaluate_mz140_depthor import evaluate_frame, rasterize_native_bounds
        from mz161_dense_labels import make_labels
        from evaluate_mz136_corridor_pair import score, pair_metrics
        from run_mz143_corridor_evidence import native_account
        from run_mz107_four_sensor import truth
        within()
        es = selected_jsonl(CAP/'evaluator.jsonl', ids)
        assert [e['id'] for e in es] == [r['id'] for r in rows]
        write(output/'evaluation-start.json', dict(prediction_seal_sha256=sha(output/'prediction-seal.json')))
        cases = []
        for i, (row, e, p) in enumerate(zip(rows, es, predictions)):
            label = make_labels(row, e, yaws[i])
            reference = rasterize_native_bounds(row, e, dict(integrated_yaw_deg=yaws[i]))
            case = dict(id=row['id'], family=e['family'], truth=bool(truth(e)),
                        tof_stratum=target_stratum(row,e), arms={})
            with np.load(output/'predictions'/(row['id']+'.npz'), allow_pickle=False) as maps:
                for arm in ARMS:
                    depth = maps[arm]; mask = corridor_mask(row, depth, yaws[i])
                    geometry = evaluate_frame(row, e, depth, dict(integrated_yaw_deg=yaws[i]))
                    case['arms'][arm] = dict(geometry=geometry, candidate=p['arms'][arm]['candidate'],
                        overlapping_known_corridor_pixels=int((mask & label['target']).sum()),
                        **correct_target_patch(mask, label['target_corridor'], depth, reference['depth']),
                        known_false_corridor_pixels=int((mask & label['known'] & ~label['target']).sum()),
                        unknown_predicted_corridor_pixels=int((mask & ~label['known']).sum()))
            cases.append(case); within()
            if (i+1)%48 == 0: print(json.dumps(dict(stage='native_evaluation', frames=i+1)), flush=True)
        write(output/'geometry-cases.json', cases)
        within()
        gt = np.array([c['truth'] for c in cases], bool)
        flags = dict(baseline=baseline)
        for arm in ARMS:
            flags[arm] = np.array([p['arms'][arm]['candidate'] for p in predictions], bool)
            flags[arm+'_or'] = baseline | flags[arm]
        episode_indices = {}
        for i,r in enumerate(rows): episode_indices.setdefault(r['episode_id'], []).append(i)
        pairs = []
        for pair in spec['pairs']:
            a,b = pair['episodes']
            if a in episode_indices and b in episode_indices:
                pairs += [dict(a=i,b=j) for i,j in zip(episode_indices[a],episode_indices[b])]
        reports = {}
        for name, ff in flags.items():
            result = score(rows, es, gt, ff, list(range(192)), baseline)
            m = result['metrics']
            result.update(precision=m['TP']/(m['TP']+m['FP']) if m['TP']+m['FP'] else None,
                recall=m['TP']/(m['TP']+m['FN']), pairs=pair_metrics(gt, ff.astype(float), ff, pairs),
                native=native_account(rows,es,ff,baseline), strata={})
            result['gained_true'] = [r['id'] for r,t,a,b in zip(rows,gt,ff,baseline) if t and a and not b]
            for stratum, ix in dict(ordinary=[i for i,e in enumerate(es) if e['family']!='shallow_boundary_stress'],
                                   pressure=[i for i,e in enumerate(es) if e['family']=='shallow_boundary_stress']).items():
                result['strata'][stratum] = score(rows,es,gt,ff,ix,baseline)
            reports[name] = result
        candidate = reports['known_camera_or']
        gained = set(candidate['gained_true'])
        unanchored = [c['id'] for c in cases if c['id'] in gained and c['tof_stratum']=='no_target_return'
                      and c['arms']['known_camera']['largest_correct_target_patch_pixels'] >= 9]
        native_before = {c['id'] for c in reports['baseline']['native']['nonalert_with_native_corridor_contributors']}
        native_after = {c['id'] for c in candidate['native']['nonalert_with_native_corridor_contributors']}
        event_retained = all(a['first_alert_s'] is not None and a['first_alert_s']<=b['first_alert_s']
            for a,b in zip(candidate['event_details'], reports['baseline']['event_details']) if b['first_alert_s'] is not None)
        checks = dict(new_true_frame=bool(gained), no_new_fp=not candidate['new_fp'],
            true_retention=not candidate['lost_baseline_tp'], event_time_retention=event_retained,
            no_new_native_suppression=not(native_after-native_before),
            localized_no_target_return_gain=bool(unanchored))
        passed = all(checks.values())
        summary = dict(authority='CONSUMED_TRAIN192_METRIC_PRIOR_AND_FIXED_ALERT_READOUT',
            arms=reports, geometry={arm:geometry_summary(cases,arm) for arm in ARMS}, checks=checks,
            no_target_return_gained_true_ids=unanchored, mechanism_passed=passed,
            decision='MZ164_RETAIN_SCOPED_UNANCHORED_ALERT_COMPONENT' if passed else 'MZ164_METRIC_PRIOR_NO_UNANCHORED_ALERT_GAIN',
            seconds=time.perf_counter()-start, gpu_seconds=gpu_s, training=False, original_dev_test_access=False)
        write(output/'summary.json', summary)
        within()
        assert all(sha(p) == h for p,h in (sources|inputs).items())
        write(output/'completion.json', dict(status='PASS', seconds=summary['seconds'], decision=summary['decision'],
            summary_sha256=sha(output/'summary.json'), cases_sha256=sha(output/'geometry-cases.json'),
            prediction_seal_sha256=sha(output/'prediction-seal.json'), resources='CUDA released after prediction; no persistent worker'))
        print(json.dumps(dict(decision=summary['decision'], checks=checks,
            alerts={n:r['metrics'] for n,r in reports.items()}, seconds=summary['seconds'])), flush=True)
    except Exception as exc:
        write(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), seconds=time.perf_counter()-start,
            predictions_sealed=(output/'prediction-seal.json').exists()))
        raise
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=WORK/'run-v1')
    run(parser.parse_args().output)
