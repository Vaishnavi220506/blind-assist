"""Post-seal evaluator for conditional dense parallax; never predictor input.

MZ119 evaluator.camera is the source-commanded reference pose, not independently
read-back native camera pose. Actor AABBs are native; first hits are analytic
references including frozen background/floor, not rendered depth/segmentation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from mz119_parallax_audit import (
    context_objects, first_hit, interval_contains, object_hazard, reference_issues,
)

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work/mz160-dense-parallax-20260916'
CAPTURE = ROOT/'artifacts.local/work/mz119-tof-scaled-parallax-20260913/capture-v1'
NEAR_M = 4.
CONTROL_MODES = {'rotation_only', 'stationary', 'low_translation_rotation'}
AUTHORITY = 'POST_SEAL_CONSUMED_MZ119_ANALYTIC_FIRST_HIT_AND_NATIVE_TOF_EVALUATION_ONLY'


def _metrics(records, arm):
    counts = dict(TP=0, FP=0, FN=0, TN=0, UNKNOWN=0)
    for r in records:
        truth, alert = r['truth'], r[arm]
        counts[('TP' if truth else 'FP') if alert else ('FN' if truth else 'TN')] += 1
        counts['UNKNOWN'] += int(not alert)
    return counts


def _events(records, arm):
    details = []; false_segments = false_bins = 0
    episode = None; previous_truth = previous_false = False; previous_time = None
    for r in records:
        if r['episode_id'] != episode:
            previous_truth = previous_false = False; previous_time = None
        if previous_time is not None and abs(r['time_s']-previous_time-.25) > 1e-6:
            raise ValueError('Expected unchanged MZ119 .25 s sampling')
        truth, alert = r['truth'], r[arm]
        if truth and not previous_truth:
            details.append(dict(episode=r['episode_id'], onset_s=r['time_s'], first_alert_s=None))
        if truth and alert and details[-1]['first_alert_s'] is None:
            details[-1]['first_alert_s'] = r['time_s']
        false = alert and not truth
        false_segments += int(false and not previous_false); false_bins += int(false)
        episode, previous_time = r['episode_id'], r['time_s']
        previous_truth, previous_false = truth, false
    delays = [r['first_alert_s']-r['onset_s'] for r in details if r['first_alert_s'] is not None]
    return dict(positive_segments=len(details), missed_positive_segments=len(details)-len(delays),
        first_alert_delay_s=delays, max_detected_delay_s=max(delays, default=None),
        mean_detected_delay_s=sum(delays)/len(delays) if delays else None,
        false_alert_segments=false_segments, false_alert_bin_duration_s=false_bins*.25,
        event_details=details)


def _report(records, arm):
    m = _metrics(records, arm)
    return dict(metrics=m, precision=m['TP']/max(1, m['TP']+m['FP']),
        recall=m['TP']/max(1, m['TP']+m['FN']), events=_events(records, arm),
        families={f:_metrics([r for r in records if r['family'] == f], arm) for f in sorted({r['family'] for r in records})},
        by_camera_motion={f:_metrics([r for r in records if r['camera_motion'] == f], arm) for f in sorted({r['camera_motion'] for r in records})},
        by_object_motion={f:_metrics([r for r in records if r['object_motion'] == f], arm) for f in sorted({r['object_motion'] for r in records})},
        unknown_semantics='NONALERT_NOT_CERTIFIED_FREE_SPACE')


def _valid_interval(bounds):
    return isinstance(bounds, (list, tuple)) and len(bounds) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in bounds) and .2 < bounds[0] <= bounds[1] <= NEAR_M


def _native_counts(row, evaluation):
    zones = {z['zone_id']:z for z in row['tof_zones']}
    total = corridor = 0
    for zone in evaluation.get('zonal_tof_native', []):
        rays = {r['subray']:r for r in zone['private_rays']}
        for line in zone['returned_lineage']:
            if not 0 <= line['target_index'] < len(zones[zone['zone_id']]['targets']):
                raise ValueError('Returned native lineage does not match a public slot')
            for i in line['hit_indices']:
                point = rays[i]['hit_point_m']
                if len(point) != 3 or not all(math.isfinite(v) for v in point):
                    raise ValueError('Unresolved native returned hit')
                body = [x-o for x,o in zip(point, evaluation['body_origin_m'])]
                total += 1
                corridor += all(lo <= x <= hi for x,lo,hi in zip(body, (.2,-.3,.4), (3.6,.3,2.05)))
    return dict(returned_contributor_samples=total, corridor_contributor_samples=corridor)


def evaluate_records(rows, evaluations, geometry, baseline, candidate, spec, receipt):
    """Pure evaluator API. Caller must authenticate all prediction seals first."""
    if not len(rows) == len(evaluations) == len(geometry) == len(baseline) == len(candidate):
        raise ValueError('Unmatched cohort lengths')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or any([x['id'] for x in values] != ids for values in (evaluations, geometry, baseline, candidate)):
        raise ValueError('Unmatched/duplicate frame identities')
    contexts = context_objects(spec, receipt)
    source_frames = {f['id']:f for f in spec['frames']}
    source_audits = {a['episode']:a for a in spec['source_audit']}
    roles = {f['episode']+'/'+o['name']:o.get('source_role') for f in spec['frames'] for o in f['objects']}
    records = []; point_records = []; pair_records = []
    counts = Counter(); by_camera = defaultdict(Counter); by_motion = defaultdict(Counter); by_family = defaultdict(Counter); by_role = Counter()
    good_episodes = set(); rod_frames = set(); rod_episodes = set(); geometry_error = 0.
    for i,(row,e,g,b,p) in enumerate(zip(rows,evaluations,geometry,baseline,candidate)):
        frame = source_frames[row['id']]; scene = source_audits[row['episode_id']]
        mode = scene['camera_motion_mode']
        motion = 'moving' if any(any(abs(v)>0 for v in o['velocity_mps']) for o in scene['objects']) else 'static'
        if not all(e['camera'].get(k,0) == frame['camera'].get(k,0) for k in ('x','y','z','yaw','pitch','roll')):
            raise ValueError('Commanded reference camera differs from frozen source')
        native = {o['name']:o for o in e['native_bounds']}
        if set(native) != {o['name'] for o in frame['objects']}:
            raise ValueError('Native actor inventory mismatch')
        for x,y in zip(e['body_origin_m'],frame['body_origin_m']): geometry_error=max(geometry_error,abs(x-y))
        for o in frame['objects']:
            for x,y in zip(native[o['name']]['center_m'],o['center_m']): geometry_error=max(geometry_error,abs(x-y))
            for x,y in zip(native[o['name']]['extent_m'],o['size_m']): geometry_error=max(geometry_error,abs(2*x-y))
        truth = any(object_hazard(o,e['body_origin_m']) for o in e['native_bounds'])
        source_truth = any(object_hazard(dict(center_m=o['center_m'],extent_m=[v/2 for v in o['size_m']]),frame['body_origin_m']) for o in frame['objects'])
        if source_truth != truth:
            raise ValueError('Source/native corridor-label mismatch')
        if bool(p['baseline']) != bool(b['candidate']):
            raise ValueError('Candidate baseline flag differs from sealed MZ129')
        support = any(x['alert_support'] for x in g['points'])
        if bool(p['new_geometry_support']) != support or bool(p['candidate']) != bool(b['candidate'] or support):
            raise ValueError('Expected unchanged MZ129 OR new geometry support')
        record = dict(id=row['id'],episode_id=row['episode_id'],time_s=row['time_s'],family=e['family'],
            camera_motion=mode,object_motion=motion,truth=truth,baseline=bool(b['candidate']),candidate=bool(p['candidate']),
            new_geometry_support=support,**_native_counts(row,e))
        records.append(record)
        frame_counts = Counter(frames=1,accepted_points=0,contained_points=0,known_reference_points=0,
                               actor_correct_near_points=0,rod_correct_near_points=0,false_near_context_points=0,finite_near_points=0)
        for pair in g.get('pairs',[]):
            ref = pair['reference_index']; issues=reference_issues(rows,i,ref)
            if issues: raise ValueError('Invalid accepted pair reference: '+str(issues))
            pair_records.append(dict(id=row['id'],current_index=i,reference_index=ref,
                reference_age_s=row['time_s']-rows[ref]['time_s'],camera_motion=mode,object_motion=motion))
        for j,point in enumerate(g['points']):
            bounds = point['range_bounds_m']
            if not _valid_interval(bounds): raise ValueError('Accepted point needs a positive finite ordered interval')
            refs = point['reference_indices']
            if refs != [i-2,i-3] or any(reference_issues(rows,i,ref) for ref in refs):
                raise ValueError('Accepted point must have causal .5/.75 s references in its episode')
            if not isinstance(point['excluded_zone'],int) or not -1 <= point['excluded_zone'] <= 63:
                raise ValueError('Invalid public excluded-zone marker')
            hit = first_hit(row,e,point['pixel'],contexts,roles)
            contained = 'range_m' in hit and interval_contains(hit['range_m'],bounds)
            near = bounds[0] <= NEAR_M
            native_actor = hit.get('status') == 'FIRST_AABB_HIT' and hit.get('foreground_native_actor',False)
            actor_good = bool(native_actor and hit['range_m'] <= NEAR_M and contained)
            rod = actor_good and 'near_rod' in hit.get('source_roles',[])
            static_lateral = motion == 'static' and mode == 'lateral'
            context = 'CONTEXT' in hit.get('kinds',[]) and not hit.get('foreground_native_actor',False)
            false_context = bool(context and near and not contained)
            frame_counts['accepted_points'] += 1
            frame_counts['known_reference_points'] += 'range_m' in hit
            frame_counts['contained_points'] += contained
            frame_counts['actor_correct_near_points'] += actor_good
            frame_counts['rod_correct_near_points'] += rod
            frame_counts['false_near_context_points'] += false_context
            frame_counts['finite_near_points'] += near
            if actor_good and static_lateral: good_episodes.add(row['episode_id'])
            if rod and static_lateral:
                rod_frames.add(row['id']);rod_episodes.add(row['episode_id']);counts['static_lateral_rod_points'] += 1
            if mode in CONTROL_MODES and near: counts['control_finite_near_points'] += 1
            if actor_good:
                for role in hit.get('source_roles',[]): by_role[str(role)] += 1
            point_records.append(dict(id=row['id'],point_index=j,pixel=point['pixel'],range_bounds_m=bounds,
                reference_indices=refs,excluded_zone=point['excluded_zone'],alert_support=point['alert_support'],
                camera_motion=mode,object_motion=motion,family=e['family'],first_hit=hit,
                analytic_range_contained=contained,finite_near=near,actor_correct_near=actor_good,
                static_lateral_rod_correct=bool(rod and static_lateral),false_near_context=false_context,
                temporal_actor_correspondence='NOT_EVALUABLE_NO_REFERENCE_PIXELS'))
        frame_counts['frames_with_points'] += bool(g['points'])
        counts.update(frame_counts);by_camera[mode].update(frame_counts);by_motion[motion].update(frame_counts);by_family[e['family']].update(frame_counts)
    if geometry_error > 1e-5: raise ValueError('Native geometry differs from frozen source')
    baseline_report,candidate_report = (_report(records,a) for a in ('baseline','candidate'))
    changes = {key:[] for key in ('gained_true','lost_true','new_fp','removed_fp')}
    for r in records:
        if r['baseline'] != r['candidate']:
            key = ('gained_true' if r['truth'] else 'new_fp') if r['candidate'] else ('lost_true' if r['truth'] else 'removed_fp')
            changes[key].append(r['id'])
    event_deltas = []
    for b,c in zip(baseline_report['events']['event_details'],candidate_report['events']['event_details']):
        if (b['episode'],b['onset_s']) != (c['episode'],c['onset_s']): raise ValueError('Unmatched events')
        if b['first_alert_s'] is not None:
            event_deltas.append(dict(episode=b['episode'],onset_s=b['onset_s'],extra_delay_s=None if c['first_alert_s'] is None else c['first_alert_s']-b['first_alert_s']))
    native_losses = [r for r in records if r['baseline'] and not r['candidate'] and r['corridor_contributor_samples']>0]
    coverage = counts['contained_points']/counts['accepted_points'] if counts['accepted_points'] else 0.
    component_checks = dict(static_lateral_correct_actor_episodes_at_least_4=len(good_episodes)>=4,
        static_lateral_correct_rod_frames_at_least_4=len(rod_frames)>=4,
        static_lateral_correct_rod_points_at_least_10=counts['static_lateral_rod_points']>=10,
        accepted_point_analytic_containment_at_least_90pct=coverage>=.9,
        zero_false_near_context_points=counts['false_near_context_points']==0,
        zero_finite_near_control_points=counts['control_finite_near_points']==0)
    alert_checks = dict(at_least_one_extra_true_frame=len(changes['gained_true'])>=1,
        zero_lost_true_frames=not changes['lost_true'],zero_lost_or_delayed_events=all(d['extra_delay_s'] is not None and d['extra_delay_s']<=1e-9 for d in event_deltas),
        zero_new_false_positive_frames=not changes['new_fp'],
        zero_family_fp_increase=all(candidate_report['families'][f]['FP']<=baseline_report['families'][f]['FP'] for f in baseline_report['families']),
        zero_native_contributor_supported_alert_losses=not native_losses)
    component_pass,alert_pass = all(component_checks.values()),all(alert_checks.values())
    return dict(authority=AUTHORITY,frames=len(rows),primary_corridor_distance_m=3.6,
        baseline=baseline_report,candidate=candidate_report,changes=changes,event_deltas=event_deltas,
        pressure_strata=dict(status='NOT_AVAILABLE',reason='MZ119 source defines no ordinary/pressure partition; report all families and camera/object-motion strata'),
        geometry=dict(counts=dict(counts),accepted_point_containment_fraction=coverage,
            static_lateral_correct_actor_episodes=sorted(good_episodes),static_lateral_correct_rod_frames=sorted(rod_frames),static_lateral_correct_rod_episodes=sorted(rod_episodes),
            by_camera_motion=dict(by_camera),by_object_motion=dict(by_motion),by_family=dict(by_family),correct_near_actor_points_by_role=dict(by_role),
            points=point_records,pairs=pair_records),
        native_tof=dict(totals={k:sum(r[k] for r in records) for k in ('returned_contributor_samples','corridor_contributor_samples')},
            lost_contributor_supported_alerts=native_losses,
            candidate_nonalert_with_corridor_contributors=[r for r in records if not r['candidate'] and r['corridor_contributor_samples']>0],
            radar_native_lineage='NOT_EVALUABLE'),
        native_admission=dict(max_source_native_geometry_error_m=geometry_error,source_native_truth_agree=True),
        component_checks=component_checks,component_pass=component_pass,alert_checks=alert_checks,alert_pass=alert_pass,
        overall_pass=component_pass and alert_pass,
        decision='DENSE_PARALLAX_COMPONENT_AND_ALERT_PASS' if component_pass and alert_pass else 'COMPONENT_ONLY_NOT_ALERT_GAIN' if component_pass else 'FIXED_DENSE_PARALLAX_GATE_NOT_MET',
        records=records,limits=[
            'Evaluator camera equals source-commanded reference, not an independent native camera readback.',
            'First-hit native actor AABBs plus frozen context are analytic geometry, not rendered depth or segmentation.',
            'Every accepted point is in the containment denominator; missing analytic reference counts as not contained.',
            'Near means finite positive interval intersects (0,4m]; false-near context means such interval excludes its analytic context first hit.',
            'Correct actor point requires unique current first-hit native actor, true slant range<=4m and interval containment. Temporal actor identity is not evaluable without reference pixels.',
            'Source motion and roles are evaluator strata only; never inference inputs.',
            'Repeated points/frames are not independent obstacles; ToF sample retention does not certify unsampled surfaces.',
            'All MZ119 frames are already consumed simulated Development; no hardware, natural-domain or safety claim.'
        ])


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def run(capture=CAPTURE, work=WORK, output=None):
    """Authenticate public prediction products before parsing native evaluation."""
    start = time.perf_counter()
    capture, work = Path(capture).resolve(), Path(work).resolve()
    output = Path(output).resolve() if output else work/'evaluation-v1'
    if not output.is_relative_to(work) or output == work or output.exists():
        raise ValueError('Evaluation output must be a fresh directory inside the work root')
    checked = {}

    def verify(path, expected):
        path = Path(path).resolve()
        actual = _sha(path)
        if actual != expected: raise ValueError('Hash mismatch: '+str(path))
        checked[str(path)] = actual
        return actual

    def write(path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    recipe_path = work/'recipe-freeze.json'
    recipe_hash = _sha(recipe_path); recipe = _read(recipe_path)
    checked[str(recipe_path)] = recipe_hash
    if recipe['frames'] != 240: raise ValueError('Expected all 240 consumed MZ119 frames')
    for group in ('sources', 'inputs'):
        for path, expected in recipe[group].items(): verify(path, expected)
    if str(Path(__file__).resolve()) not in recipe['sources']:
        raise ValueError('Evaluator source is not bound by the recipe freeze')
    for name in ('raw.jsonl', 'evaluator.jsonl', 'receipt.json', 'spec.json'):
        if str((capture/name).resolve()) not in checked:
            raise ValueError('Capture file is absent from the recipe input freeze: '+name)
    raw_hash, receipt_hash = _sha(capture/'raw.jsonl'), _sha(capture/'receipt.json')
    seals = {}
    for arm, required in (('baseline-v1', {'predictions.json', 'incumbent-details.json'}),
                          ('inference-v1', {'predictions.json', 'geometry.json', 'geometry-seal.json'})):
        directory = work/arm
        completion = _read(directory/'completion.json')
        checked[str(directory/'completion.json')] = _sha(directory/'completion.json')
        if completion['status'] != 'PASS': raise ValueError('Public prediction process did not complete')
        verify(directory/'prediction-seal.json', completion['prediction_seal_sha256'])
        seal = _read(directory/'prediction-seal.json'); seals[arm] = seal
        if (seal['status'] != 'SEALED_PUBLIC_BEFORE_EVALUATOR_PARSE'
                or Path(seal['capture']).resolve() != capture
                or seal['raw_sha256'] != raw_hash or seal['receipt_sha256'] != receipt_hash
                or seal['recipe_freeze_sha256'] != recipe_hash
                or set(seal['outputs']) != required):
            raise ValueError('Public prediction seal binding mismatch: '+arm)
        for path, expected in seal['inputs'].items(): verify(path, expected)
        if {Path(p).resolve() for p in seal['inputs']} != {capture/'raw.jsonl', capture/'receipt.json'}:
            raise ValueError('Unexpected prediction input inventory')
        for name, expected in seal['outputs'].items(): verify(directory/name, expected)
    baseline_seal_hash = _sha(work/'baseline-v1/prediction-seal.json')
    inference_seal_hash = _sha(work/'inference-v1/prediction-seal.json')
    if seals['inference-v1']['context']['baseline_prediction_seal_sha256'] != baseline_seal_hash:
        raise ValueError('Candidate does not bind the unchanged baseline prediction seal')
    prior = ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/incumbent-v1/prediction-seal.json'
    verify(prior, seals['baseline-v1']['context']['prior_source_seal_sha256'])
    for path, expected in _read(prior)['source_hashes'].items(): verify(path, expected)
    geometry_seal = _read(work/'inference-v1/geometry-seal.json')
    if (geometry_seal['recipe_freeze_sha256'] != recipe_hash or
            geometry_seal['authority'] != 'PUBLIC_GEOMETRY_SEALED_BEFORE_EVALUATOR_PARSE'):
        raise ValueError('Geometry seal binding mismatch')
    verify(work/'inference-v1/geometry.json', geometry_seal['geometry_sha256'])
    for name, expected in geometry_seal['disparities'].items():
        path = (work/'inference-v1'/name).resolve()
        if not path.is_relative_to(work/'inference-v1/disparities'):
            raise ValueError('Disparity path escapes inference directory')
        verify(path, expected)
    receipt = _read(capture/'receipt.json')
    if receipt['status'] != 'PASS': raise ValueError('Capture receipt is not PASS')
    verify(capture/'spec.json', receipt['spec_sha256'])
    for name, expected in receipt['hashes'].items():
        path = (capture/name).resolve()
        if not path.is_relative_to(capture): raise ValueError('Capture receipt path escapes capture')
        verify(path, expected)
    rows = [json.loads(line) for line in (capture/'raw.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows) != 240 or len({r['id'] for r in rows}) != 240:
        raise ValueError('Expected exact complete 240-frame cohort')
    expected_rgb = {(capture/r['rgb_path']).resolve() for r in rows}
    if {Path(p).resolve() for p in geometry_seal['rgb_sha256']} != expected_rgb:
        raise ValueError('Geometry RGB seal does not cover exactly the public cohort')
    for path, expected in geometry_seal['rgb_sha256'].items(): verify(path, expected)
    baseline = _read(work/'baseline-v1/predictions.json')
    candidate = _read(work/'inference-v1/predictions.json')
    geometry = _read(work/'inference-v1/geometry.json')
    ids = [r['id'] for r in rows]
    if any([r['id'] for r in values] != ids for values in (baseline, candidate, geometry)):
        raise ValueError('Public prediction cohort/order mismatch')
    output.mkdir(parents=True)
    write(output/'evaluation-start.json', dict(status='PUBLIC_SEALS_AUTHENTICATED_BEFORE_NATIVE_PARSE',
        recipe_freeze_sha256=recipe_hash, baseline_prediction_seal_sha256=baseline_seal_hash,
        inference_prediction_seal_sha256=inference_seal_hash, authenticated_files=checked.copy()))
    # This is the first real evaluator-record parse in this entry point.
    evaluations = [json.loads(line) for line in (capture/'evaluator.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    result = evaluate_records(rows, evaluations, geometry, baseline, candidate, _read(capture/'spec.json'), receipt)
    write(output/'cases.json', result.pop('records'))
    geom = result['geometry']
    write(output/'geometry-points.json', geom.pop('points'))
    write(output/'geometry-pairs.json', geom.pop('pairs'))
    result['provenance'] = dict(recipe_freeze_sha256=recipe_hash,
        baseline_prediction_seal_sha256=baseline_seal_hash, inference_prediction_seal_sha256=inference_seal_hash,
        evaluation_start_sha256=_sha(output/'evaluation-start.json'), authenticated_file_count=len(checked))
    write(output/'summary.json', result)
    for path, expected in list(checked.items()): verify(path, expected)
    products = ('evaluation-start.json', 'cases.json', 'geometry-points.json', 'geometry-pairs.json', 'summary.json')
    write(output/'completion.json', dict(status='PASS', seconds=time.perf_counter()-start,
        recipe_freeze_sha256=recipe_hash, baseline_prediction_seal_sha256=baseline_seal_hash,
        inference_prediction_seal_sha256=inference_seal_hash, outputs={name:_sha(output/name) for name in products},
        resources='TASK_NOT_GPU_SUITABLE: CPU scalar saved-output audit; process exit releases local memory'))
    print(json.dumps(dict(output=str(output), frames=result['frames'], baseline=result['baseline']['metrics'],
        candidate=result['candidate']['metrics'], component_checks=result['component_checks'],
        alert_checks=result['alert_checks'], decision=result['decision'])), flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,default=CAPTURE)
    parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.capture, args.work, args.output)


if __name__=='__main__': main()
