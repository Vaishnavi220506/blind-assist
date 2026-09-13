"""Consumed measurement-centered interval geometry and accountable correction.

Prediction uses current observation packets and sealed observable-derived boxes.
Full angular support is retained; RGB alternatives never erase the null branch.
No evaluator identity, score threshold fit, or physical clearance claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from mz109_interval_extent import add, sub, mul, div, square, trig, projection_envelope

CORRIDOR = ((.2, 3.6), (-.3, .3), (.4, 2.05))
MAX_DEPTH = 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def intersects(xyz, corridor=CORRIDOR):
    return all(hi >= c0 and lo <= c1 for (lo, hi), (c0, c1) in zip(xyz, corridor))


def slant_box(box, ranges, intr, pitches, yaws, height):
    """Same enclosing interval arithmetic as MZ115, without vision imports."""
    a = ((box[0]-intr['cx'])/intr['fx'], (box[2]-intr['cx'])/intr['fx'])
    b = ((intr['cy']-box[3])/intr['fy'], (intr['cy']-box[1])/intr['fy'])
    norm = tuple(math.sqrt(v) for v in add((1., 1.), add(square(a), square(b))))
    p, y = tuple(map(math.radians, pitches)), tuple(map(math.radians, yaws))
    cp, sp, cy, sy = trig(p, True), trig(p), trig(y, True), trig(y)
    f, up = sub(cp, mul(b, sp)), add(sp, mul(b, cp))
    return [mul(ranges, div(sub(mul(f, cy), mul(a, sy)), norm)),
            mul(ranges, div(add(mul(f, sy), mul(a, cy)), norm)),
            add((height, height), mul(ranges, div(up, norm)))]


def zone_box(zone, intr):
    a, b = zone['theta_bounds_deg']; c, d = zone['phi_bounds_deg']
    return [intr['cx']+intr['fx']*math.tan(math.radians(a)),
            intr['cy']-intr['fy']*math.tan(math.radians(d)),
            intr['cx']+intr['fx']*math.tan(math.radians(b)),
            intr['cy']-intr['fy']*math.tan(math.radians(c))]


def refine_support(box, ranges, row, yaw, corridor=CORRIDOR):
    """OUT only if every enclosure is disjoint; budget exhaustion is POSSIBLE.

    Subdivide pixels in the longest angular dimension. No center sample can
    certify exclusion. Range/pitch/yaw bounds remain complete in every child.
    """
    intr = row['rgb_intrinsics']; p = row['camera_pitch_deg']
    dy = .5+.2*row['time_s']; height = row['camera_in_body_m'][2]
    stack = [(list(box), 0)]; nodes = 0
    while stack:
        current, depth = stack.pop(); nodes += 1
        xyz = slant_box(current, ranges, intr, (p-.5, p+.5), (yaw-dy, yaw+dy), height)
        if not intersects(xyz, corridor):
            continue
        if depth == MAX_DEPTH or all(lo >= c0 and hi <= c1 for (lo, hi), (c0, c1) in zip(xyz, corridor)):
            return dict(possible=True, nodes=nodes, reason='INTERSECTING_OR_UNRESOLVED_ENCLOSURE')
        axis = 0 if (current[2]-current[0])/intr['fx'] >= (current[3]-current[1])/intr['fy'] else 1
        middle = (current[axis]+current[axis+2])/2
        left, right = current.copy(), current.copy()
        left[axis+2] = middle; right[axis] = middle
        stack.extend([(left, depth+1), (right, depth+1)])
    return dict(possible=False, nodes=nodes, reason='ALL_SUBDIVISIONS_DISJOINT')


def radar_support(row, yaw):
    out = []
    if not row['radar_packet_received']:
        return out
    for slot, (r, a, valid) in enumerate(zip(row['radar_range_m'], row['radar_angle'], row['radar_valid'])):
        if not valid or r is None or a is None or not math.isfinite(r+a) or r <= 0:
            continue
        # Inherited 12deg association working bound; not a calibrated sensor CI.
        delta = 12+.5+.2*row['time_s']
        angle = tuple(math.radians(a+yaw+x) for x in (-delta, delta))
        bounds = (max(.02, r-.15), r+.15)
        xyz = [mul(bounds, trig(angle, True)), mul(bounds, trig(angle)), (-1e6, 1e6)]
        out.append(dict(slot=slot, range_m=r, bearing_deg=a, possible=intersects(xyz), xyz=xyz,
                        height_state='HEIGHT_UNKNOWN', null_explanation_retained=True))
    return out


def predict_frame(row, cached):
    yaw = cached['integrated_yaw_deg']; intr = row['rgb_intrinsics']
    records = []; malformed = False
    if row['tof_packet_received']:
        for zone in row['tof_zones']:
            box = zone_box(zone, intr)
            for slot, target in enumerate(zone['targets']):
                r = target['distance_m']; sigma = target['range_noise_sigma_m']; status = target['status']
                if status not in ('SIM_VALID', 'SIM_MERGED') or not math.isfinite(r+sigma) or r <= 0 or sigma < 0:
                    malformed = True
                    continue
                ranges = (.02, 4.) if status == 'SIM_MERGED' else (max(.02, r-3*sigma), r+3*sigma)
                dy = .5+.2*row['time_s']; p = row['camera_pitch_deg']
                outer = slant_box(box, ranges, intr, (p-.5, p+.5), (yaw-dy, yaw+dy), row['camera_in_body_m'][2])
                result = refine_support(box, ranges, row, yaw)
                assert not result['possible'] or intersects(outer)
                records.append(dict(zone_id=zone['zone_id'], slot=slot, status=status,
                    coarse_possible=intersects(outer), **result))
    radar = radar_support(row, yaw)
    rgb = []
    for ret in radar:
        for index, box in enumerate(cached['proposals']):
            # Preserve ALL compatible boxes, including competing associations.
            low = math.degrees(math.atan((box[0]-intr['cx'])/intr['fx']))
            high = math.degrees(math.atan((box[2]-intr['cx'])/intr['fx']))
            if low-12 <= ret['bearing_deg'] <= high+12:
                dy = .5+.2*row['time_s']; p = row['camera_pitch_deg']
                xyz = projection_envelope((box[0]-2, box[2]+2), (box[1]-2, box[3]+2), intr,
                    (max(.02, ret['range_m']-.15), ret['range_m']+.15),
                    (p-.5, p+.5), (yaw-dy, yaw+dy), row['camera_in_body_m'][2])
                rgb.append(dict(slot=ret['slot'], proposal=index, possible=xyz is None or intersects(xyz),
                    reason='ALL_COMPATIBLE_RGB_SHELL_HYPOTHESES_PLUS_NULL'))
    tof = any(x['possible'] for x in records)
    sensor = tof or any(x['possible'] for x in radar)
    with_rgb = sensor or any(x['possible'] for x in rgb)
    # Corrections concern a positive backed solely by current ToF support.
    # Lack of a packet/IMU/return, or a merged return, cannot certify exclusion.
    eligible = bool(cached['candidate'] and not cached['common_radar'] and not cached['guard_events']
        and row['imu_valid'] and row['tof_packet_received'] and row['radar_packet_received']
        and records and not malformed and all(x['status'] == 'SIM_VALID' for x in records)
        and any(x['coarse_possible'] for x in records))
    excluded = eligible and not tof
    # The incumbent can already have rejected Radar through its association
    # logic. Its common_radar=False is not absence of our new independent raw
    # support. A strict correction must preserve that possible/null branch too.
    strict_excluded = excluded and not any(x['possible'] for x in radar)
    return dict(tof=records, radar=radar, rgb_hypotheses=rgb,
        correction_eligible=eligible, inherited_radar_exclusion=excluded,
        correction_exclusion=strict_excluded,
        flags=dict(sensor_measurement=bool(sensor), rgb_measurement=bool(with_rgb),
            incumbent_radar_correction=bool(cached['candidate'] and not excluded),
            accountable_correction=bool(cached['candidate'] and not strict_excluded),
            correction_plus_sensor_recovery=bool((cached['candidate'] and not excluded) or sensor)))


def metrics(rows, truth, predictions):
    tp = sum(t and p for t, p in zip(truth, predictions)); fp = sum(not t and p for t, p in zip(truth, predictions))
    fn = sum(t and not p for t, p in zip(truth, predictions)); tn = len(truth)-tp-fp-fn
    events = []; false_segments = 0; last = None; active = None; false_active = False
    for r, t, p in zip(rows, truth, predictions):
        if r['episode_id'] != last:
            active = None; false_active = False
        last = r['episode_id']
        if t and active is None:
            active = dict(episode=last, onset=r['time_s'], delay=None); events.append(active)
        if t and p and active['delay'] is None:
            active['delay'] = r['time_s']-active['onset']
        if not t:
            active = None
        false_now = not t and p
        false_segments += int(false_now and not false_active); false_active = false_now
    delays = [e['delay'] for e in events if e['delay'] is not None]
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, UNKNOWN=fn+tn,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        event_hits=len(delays), positive_events=len(events), missed_events=[e for e in events if e['delay'] is None],
        max_detected_delay_s=max(delays) if delays else None,
        false_bin_duration_s=fp*.25, false_segments=false_segments)


def run(base, out):
    assert not out.exists(); out.mkdir(parents=True)
    cap, analysis = base/'capture-v1', base/'analysis-v1'
    raw = cap/'raw.jsonl'; cache = analysis/'baseline-predictions.json'
    seal = json.loads((analysis/'prediction-seal.json').read_text())
    assert sha(raw) == seal['raw_sha256'] and sha(cache) == seal['baseline_sha256']
    rows = [json.loads(x) for x in raw.read_text().splitlines()]
    cached = json.loads(cache.read_text()); assert len(rows) == len(cached) == 288
    start = time.perf_counter(); predictions = [predict_frame(r, p) for r, p in zip(rows, cached)]
    elapsed = time.perf_counter()-start
    write(out/'predictions.json', predictions)
    write(out/'prediction-seal.json', dict(raw_sha256=sha(raw), baseline_sha256=sha(cache),
        predictions_sha256=sha(out/'predictions.json'), code_sha256=sha(__file__),
        interval_code_sha256=sha(Path(__file__).with_name('mz109_interval_extent.py')),
        cpu_reason='TASK_NOT_GPU_SUITABLE', seconds=elapsed,
        authority='OBSERVABLE_DERIVED_CONSUMED_DIAGNOSTIC', max_depth=MAX_DEPTH))
    # Evaluator/frame truth is read only after output persistence.
    report = json.loads((analysis/'frame-report.json').read_text())
    assert [r['id'] for r in rows] == [r['id'] for r in report]
    gt = [r['truth'] for r in report]; baseline = [p['candidate'] for p in cached]
    assert baseline == [r['baseline'] for r in report]
    arms = dict(baseline=baseline, frozen_early=[r['early'] for r in report])
    arms.update({k: [p['flags'][k] for p in predictions] for k in predictions[0]['flags']})
    results = {}
    for arm, flags in arms.items():
        result = metrics(rows, gt, flags)
        result.update(removed_FP=sum(not t and b and not p for t, b, p in zip(gt, baseline, flags)),
            added_FP=sum(not t and not b and p for t, b, p in zip(gt, baseline, flags)),
            lost_baseline_TP=sum(t and b and not p for t, b, p in zip(gt, baseline, flags)),
            gained_TP=sum(t and not b and p for t, b, p in zip(gt, baseline, flags)))
        result['families'] = {}
        for family in sorted({r['family'] for r in report}):
            idx = [i for i, r in enumerate(report) if r['family'] == family]
            result['families'][family] = metrics([rows[i] for i in idx], [gt[i] for i in idx], [flags[i] for i in idx])
        results[arm] = result
    summary = dict(status='COMPLETE_CONSUMED_MEASUREMENT_DEVELOPMENT', frames=len(rows), arms=results,
        correction_eligible=sum(p['correction_eligible'] for p in predictions),
        inherited_radar_excluded=sum(p['inherited_radar_exclusion'] for p in predictions),
        correction_excluded=sum(p['correction_exclusion'] for p in predictions),
        tof_returns=sum(len(p['tof']) for p in predictions),
        coarse_possible_returns=sum(t['coarse_possible'] for p in predictions for t in p['tof']),
        proven_disjoint_after_subdivision=sum(t['coarse_possible'] and not t['possible'] for p in predictions for t in p['tof']),
        rgb_hypotheses=sum(len(p['rgb_hypotheses']) for p in predictions),
        interval_nodes=sum(t['nodes'] for p in predictions for t in p['tof']), seconds=elapsed,
        limitation='Working uncertainty and surface-support proxies, not physical volume or safety; RGB null branches retained')
    write(out/'summary.json', summary)
    write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'), owned_process='foreground exits after completion'))
    print(json.dumps({k:{f:v[f] for f in ('TP','FP','FN','precision','event_hits')} for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]/'artifacts.local'
    assert args.output.resolve().is_relative_to(root.resolve())
    run(args.source, args.output)
