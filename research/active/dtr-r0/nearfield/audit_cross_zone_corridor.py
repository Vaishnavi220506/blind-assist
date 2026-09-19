"""Same-frame simulated return lineage ceiling; no new predictions or model calls.

Target ownership means the existing native point-ray +/-2cm authenticated bounds
proxy, not exact actor segmentation or real photon composition. A cross-zone
witness never propagates distance, shape, or a decision to another pixel/frame.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from evaluate_ba_camera_corridor import read, require, sha, summarize, validate_predictions

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'artifacts.local/work/ba-camera-corridor-20260919'
VIEWS = ('all_ranges', 'observed_range_lt_3m')


def target_footprint(native, source):
    """Exactly the old materializer's full-native bounds footprint, before sampling."""
    require(native.shape == (360, 640), 'Native shape changed')
    case, geo = source['case'], source['geometry']
    target = next(o for o in geo['objects'] if o['name'] == case['target_name'])
    require(target['trace']['hit_expected_actor'], 'Target authentication absent')
    require(target['trace']['hit_actor_path'] == target['actor_path'], 'Target actor differs')
    center, extent = np.asarray(target['render_bounds_center_m']), np.asarray(target['render_bounds_extent_m'])
    camera = case['camera']
    require(all(camera[k] == 0 for k in ('pitch', 'yaw', 'roll')), 'Frozen camera axes changed')
    yy, xx = np.mgrid[:360, :640]
    focal = 640 / (2 * np.tan(np.deg2rad(100 / 2)))
    a, b = (xx + .5 - 320) / focal, (yy + .5 - 180) / focal
    world = np.stack([native + camera['x'], a * native + camera['y'], camera['z'] - b * native], -1)
    known = np.isfinite(native) & (native > 0)
    footprint = known & np.all((world >= center - extent - .02) & (world <= center + extent + .02), axis=-1)
    inside = known & (native >= .3) & (native <= 3) & (np.abs(a * native) <= .3) & (b * native >= -.2) & (b * native <= .9)
    admission = dict(native_valid_pixels=int(known.sum()), visible_target_pixels=int(footprint.sum()),
                     competing_corridor_pixels=int((inside & ~footprint).sum()))
    require(all(source['check'][k] == value for k, value in admission.items()), 'Original source admission differs')
    return footprint, admission


def frame_lineage(target_mask, zones, traced):
    """Classify exact observed winning-bin contributors against a fixed footprint.

Local absence uses ANY observed target-owned contributor at ANY saved range.
The strict 3m view filters candidate anchors only; mixed local ownership is not
counted as local absence. Pixels outside all sensor zones have a separate count.
"""
    target = np.asarray(target_mask, dtype=bool).ravel()
    zones = np.asarray(zones).ravel()
    require(target.shape == zones.shape, 'Footprint/zone shape mismatch')
    ids = [int(t['zone_id']) for t in traced]
    require(ids == list(range(len(traced))), 'Zone order/identity changed')
    require(np.all((zones >= -1) & (zones < len(traced))), 'Invalid zone map')
    ownership, observed, records = [], [], []
    for trace in traced:
        zid = int(trace['zone_id'])
        pixels = np.asarray(trace['pixel_indices'], dtype=int)
        weights = np.asarray(trace['weights'])
        require(pixels.shape == weights.shape and pixels.ndim == 1, 'Contributor arrays differ')
        require(np.all((pixels >= 0) & (pixels < target.size)), 'Contributor index out of bounds')
        require(np.unique(pixels).size == pixels.size and np.all(zones[pixels] == zid), 'Contributor zone mismatch')
        if trace['observed']:
            require(pixels.size > 0 and np.isfinite(trace['distance_m']), 'Observed return lacks contributors')
            require(np.isfinite(weights).all() and (weights > 0).all(), 'Invalid contributor weights')
        else:
            require(pixels.size == 0 and trace['distance_m'] is None, 'Unobserved data used as anchor')
        count = int(target[pixels].sum())
        kind = 'pure' if count and count == pixels.size else 'mixed' if count else 'non_target'
        ownership.append(count > 0)
        observed.append(bool(trace['observed']))
        records.append(dict(zone_id=zid, observed=bool(trace['observed']), distance_m=trace['distance_m'],
            winner_bin=trace['winner_bin'], reason=trace['reason'], ownership=kind,
            contributor_pixels=int(pixels.size), target_contributor_pixels=count,
            target_weight_fraction=float(weights[target[pixels]].sum() / weights.sum()) if pixels.size else None))
    owned, observed = np.asarray(ownership), np.asarray(observed)
    local = zones[target & (zones >= 0)]
    missing = local[~owned[local]]
    result = dict(target_sampled_pixels=int(target.sum()), target_pixels_in_sensor_zones=int(local.size),
        target_pixels_outside_sensor_zones=int((target & (zones < 0)).sum()),
        target_pixels_without_local_target_return=int(missing.size),
        local_absence_unobserved_pixels=int((~observed[missing]).sum()),
        local_absence_observed_non_target_pixels=int(observed[missing].sum()), zones=records)
    for view in VIEWS:
        anchors = [r for r in records if r['observed'] and (view == 'all_ranges' or r['distance_m'] < 3)]
        counts = {}
        for kind in ('pure', 'mixed'):
            anchor_ids = [r['zone_id'] for r in anchors if r['ownership'] == kind]
            cross = sum(any(zid != int(local_id) for zid in anchor_ids) for local_id in missing)
            counts.update({kind + '_anchor_zones': len(anchor_ids),
                           kind + '_cross_zone_target_pixels': int(cross),
                           kind + '_anchor_available': bool(anchor_ids),
                           kind + '_cross_zone_available': cross > 0})
        result[view] = counts
    return result


def aggregate(rows):
    """Sum per-frame witnesses; never combine an anchor and a pixel across frames."""
    result = dict(frames=len(rows))
    for key in ('target_sampled_pixels', 'target_pixels_in_sensor_zones', 'target_pixels_outside_sensor_zones',
                'target_pixels_without_local_target_return', 'local_absence_unobserved_pixels',
                'local_absence_observed_non_target_pixels'):
        result[key] = sum(r['lineage'][key] for r in rows)
    for view in VIEWS:
        result[view] = {key: sum(r['lineage'][view][key] for r in rows)
                       for key in ('pure_anchor_zones', 'mixed_anchor_zones', 'pure_cross_zone_target_pixels',
                                   'mixed_cross_zone_target_pixels')}
        for kind in ('pure', 'mixed'):
            for suffix in ('anchor_available', 'cross_zone_available'):
                key = kind + '_' + suffix
                result[view]['frames_with_' + key] = sum(r['lineage'][view][key] for r in rows)
    return result


def group_report(rows, events):
    entry_ids = {(e['clip_id'], e['start_frame']) for e in events}
    positive = [r for r in rows if r['truth'] is True]
    missed = [r for r in positive if not r['predictions']['nfo']['alert']]
    return dict(all_frames=aggregate(rows), positive_frames=aggregate(positive),
        nfo_positive_misses=aggregate(missed),
        nfo_interior_positive_misses=aggregate([r for r in missed if not r['boundary']]),
        nfo_boundary_positive_misses=aggregate([r for r in missed if r['boundary']]),
        nfo_entry_misses=aggregate([r for r in missed if (r['clip_id'], r['frame_in_clip']) in entry_ids]))


def event_report(rows, events):
    results = []
    for event in events:
        members = [r for r in rows if r['clip_id'] == event['clip_id'] and
                   event['start_frame'] <= r['frame_in_clip'] <= event['end_frame']]
        require(len(members) == event['end_frame'] - event['start_frame'] + 1, 'Event denominator changed')
        interior = [r for r in members if not r['boundary']]
        result = dict(original_nfo_event=event, nfo_missed_event=not event['detected'],
            nfo_missed_interior_event=bool(event['interior_frames'] and not event['interior_detected']),
            entry=aggregate(members[:1]), all_event_frames=aggregate(members), interior=aggregate(interior),
            boundary=aggregate([r for r in members if r['boundary']]))
        results.append(result)
    return results


def baseline_parity(rows, old_result, dt_s):
    replay = summarize(rows, dt_s)
    require(replay == old_result['metrics'], 'Exact old baseline metric parity failed')
    for clip in sorted({r['clip_id'] for r in rows}):
        require(summarize([r for r in rows if r['clip_id'] == clip], dt_s) == old_result['by_clip'][clip],
                'Old per-clip metric parity failed')
    for pair in sorted({r['pair_id'] for r in rows}):
        require(summarize([r for r in rows if r['pair_id'] == pair], dt_s) == old_result['by_pair'][pair],
                'Old per-pair metric parity failed')
    return replay


def audit_corridor(source_root=OLD):
    """Root calls only after the new audit protocol/code seal is frozen."""
    import ba_camera_corridor as corridor
    from cross_zone_anchor_core import trace_sensor, zone_map

    began = time.perf_counter()
    source_root = Path(source_root)
    protocol, _, _, observations, predictions = validate_predictions(source_root)
    old_result = read(source_root / 'results.json')
    require(old_result['status'] == 'COMPLETE', 'Original result incomplete')
    require(old_result['frame_results_sha256'] == sha(source_root / 'frame-results.json'), 'Frame results changed')
    for name, digest in old_result['evidence_hashes'].items():
        require(sha(source_root / name) == digest, 'Old evidence changed: ' + name)
    rows = read(source_root / 'frame-results.json')
    sources = read(source_root / 'evaluator-source.json')
    require(len(rows) == len(sources) == len(observations) == len(predictions) == 96, 'Fixed 96 cohort changed')
    require(len({r['clip_id'] for r in rows}) == 8, 'Fixed 8 clips changed')
    baseline = baseline_parity(rows, old_result, protocol['budget']['nominal_dt_s'])
    payload_hashes = []
    for row, source, observation, prediction in zip(rows, sources, observations, predictions):
        require(row['id'] == source['id'] == observation['id'] == prediction['id'], 'Frame identity differs')
        require(row['predictions'] == prediction['predictions'], 'Sealed decisions changed')
        for key in ('clip_id', 'frame_in_clip', 'time_s'):
            require(row[key] == observation[key] == prediction[key], 'Temporal identity differs')
        case = source['case']
        require(case['clip_id'] == row['source_clip_id'] and case['pair_id'] == row['pair_id'], 'Target identity differs')
        require(case['frame_in_clip'] == row['frame_in_clip'] and case['time_s'] == row['time_s'], 'Source time differs')
        native_path = source_root / 'capture/evaluator' / source['geometry']['native_path']
        require(sha(native_path) == source['sensor_native_sha256'] == source['geometry']['native_sha256'], 'Native changed')
        native = np.load(native_path, allow_pickle=False)
        mask, admission = target_footprint(native, source)
        require(admission == row['native_admission'], 'Frozen native admission differs')
        seed = case['pair_id'] + '/' + str(case['frame_in_clip'])
        require(seed == source['sensor_seed_identity'], 'Sensor seed changed')
        with np.load(source_root / observation['prepared'], allow_pickle=False) as prepared:
            traces = trace_sensor(corridor.sample_native(native), seed, prepared['boxes'], prepared['values'])
            zones = zone_map(prepared['boxes'], (192, 256))
        row['lineage'] = frame_lineage(corridor.sample_native(mask), zones, traces)
        row['native_target_pixels'] = admission['visible_target_pixels']
        row['sensor_seed_identity'] = seed
        payload_hashes.append(dict(id=row['id'], native_sha256=sha(native_path),
            prepared_sha256=observation['prepared_sha256'], prediction_sha256=prediction['sha256']))
    events = baseline['arms']['nfo']['events']
    event_results = event_report(rows, events)
    fixed_missed = [r for r in event_results if r['nfo_missed_interior_event']]
    event_ceiling = dict(fixed_interior_events=baseline['arms']['nfo']['interior_event_count'],
                        fixed_nfo_missed_interior_events=len(fixed_missed))
    for view in VIEWS:
        event_ceiling[view] = {kind + '_same_frame_cross_zone_witness_events':
            sum(r['interior'][view]['frames_with_' + kind + '_cross_zone_available'] > 0 for r in fixed_missed)
            for kind in ('pure', 'mixed')}
    names = ('protocol.json', 'observation-seal.json', 'observations.json', 'prediction-seal.json',
             'evaluator-source.json', 'source-admission.json', 'frame-results.json', 'results.json')
    return dict(status='COMPLETE', scope='CONSUMED_CONTROLLED_SIMULATION_INFORMATION_CEILING',
        backend=dict(device='CPU', reason='TASK_NOT_GPU_SUITABLE', model_calls=0, training_updates=0, new_captures=0),
        baseline_exact_parity=True, original_metrics=baseline,
        definitions=dict(target='Original native point-ray bounds footprint with +/-2cm tolerance, sampled by original native ray indices; not actor segmentation.',
            pure='Every exact observed winning-bin contributor lies in target footprint.',
            mixed='Some, but not all, exact observed winning-bin contributors lie in target footprint.',
            cross_zone='Same-frame target pixel inside sensor FOV, local zone has zero target-owned observed contributors at any range, another zone has a pure/mixed observed anchor.',
            near='Saved noisy returned distance strictly <3m; all-ranges diagnostic reported separately.',
            events='Original NFO events and interior-missed subset; event count means at least one qualifying same-frame interior witness, not new detection.',
            limitations='BBox-proxy lineage of simulated winning bins, not measured photons or a feasible public algorithm. No temporal transfer, distance substitution, plane or constant-depth extrapolation.'),
        summary=group_report(rows, events), fixed_missed_interior_event_ceiling=event_ceiling,
        by_clip={clip: group_report([r for r in rows if r['clip_id'] == clip], events) for clip in sorted({r['clip_id'] for r in rows})},
        events=event_results, frames=rows, evidence_hashes={name: sha(source_root / name) for name in names},
        payload_hashes=payload_hashes, audit_code_sha256=sha(__file__), elapsed_seconds=time.perf_counter() - began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=OLD)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-protocol', type=Path, required=True)
    args = parser.parse_args()
    # Root's combined runner is preferred. CLI must bind this exact file in its protocol.
    frozen = read(args.frozen_protocol)
    require(sha(__file__) in json.dumps(frozen), 'Audit source not bound by frozen protocol')
    require(not args.output.exists(), 'Audit output already exists')
    result = audit_corridor(args.source_root)
    result['audit_protocol_sha256'] = sha(args.frozen_protocol)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
