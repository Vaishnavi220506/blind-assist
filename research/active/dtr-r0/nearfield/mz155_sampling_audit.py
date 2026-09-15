"""Evaluator-only MZ155 sampling audit; pure inputs/outputs, no predictor or I/O.

Native ownership and commanded yaw are diagnostic authorities only. Call after
prediction sealing. Ever-observed admission uses all valid target returns;
strongest-slot and geometric-hit retention remain separate reported contrasts.
"""
from collections import Counter
import math

from mz155_active_sampling import check_source

SIGNALS = ('geometric_hit', 'valid_target_return', 'strongest_target_return')
ARMS = ('passive', 'scan')
ROD = 'near_rod_farwall'
NATIVE_TOLERANCE_M = 1e-5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _positive(value):
    return _finite(value) and value > 0


def _close(actual, expected, tolerance=NATIVE_TOLERANCE_M):
    return len(actual) == len(expected) and all(
        _finite(a) and _finite(b) and abs(a-b) <= tolerance
        for a, b in zip(actual, expected))


def _inside(body, center, extent):
    lo = [c-e-b for c, e, b in zip(center, extent, body)]
    hi = [c+e-b for c, e, b in zip(center, extent, body)]
    return hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3 and hi[2] >= .4 and lo[2] <= 2.05


def _index(rows, label):
    result = {r['id']: r for r in rows}
    _require(len(result) == len(rows), label+' contains duplicate IDs')
    return result


def _native_admission(spec, rows, es):
    try:
        source_admission = check_source(spec)
    except (AssertionError, KeyError) as error:
        raise ValueError('MZ155 source admission failed') from error
    frames = _index(spec['frames'], 'Source')
    _require(set(frames) == set(rows) == set(es), 'Source/public/evaluator IDs differ')
    label_counts = Counter()
    first_native = {}
    for ident, frame in frames.items():
        e, row = es[ident], rows[ident]
        _require(e['episode_id'] == row['episode_id'] == frame['episode'], 'Episode mismatch: '+ident)
        _require(e['time_s'] == row['time_s'] == frame['time_s'], 'Time mismatch: '+ident)
        _require(e['family'] == frame['family'], 'Family mismatch: '+ident)
        _require(_close(e['body_origin_m'], frame['body_origin_m'], 1e-9), 'Body mismatch: '+ident)
        _require(all(_finite(e['camera'][k]) and abs(e['camera'][k]-frame['camera'][k]) <= 1e-9
                     for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')), 'Camera mismatch: '+ident)
        _require(_close(row['camera_in_body_m'], [0., 0., 1.7], 1e-9)
                 and row['camera_pitch_deg'] == -3., 'Public camera calibration mismatch: '+ident)
        for name in ('background', 'floor'):
            context = spec.get(name)
            _require(not context or not _inside(frame['body_origin_m'], context['center_m'],
                         [v/2 for v in context['size_m']]), 'Source global context enters corridor: '+name)
        objects = {o['name']: o for o in frame['objects']}
        native = {o['name']: o for o in e['native_bounds']}
        _require(len(native) == len(e['native_bounds']) and set(native) == set(objects), 'Native object set mismatch: '+ident)
        for name, obj in objects.items():
            actual = native[name]
            _require(_close(actual['center_m'], obj['center_m'])
                     and _close(actual['extent_m'], [v/2 for v in obj['size_m']]),
                     'Native center/extent mismatch: '+ident+'/'+name)
        old = first_native.setdefault(frame['episode'], native)
        _require(all(_close(native[n]['center_m'], old[n]['center_m'])
                     and _close(native[n]['extent_m'], old[n]['extent_m']) for n in native),
                 'Native world geometry changed: '+ident)
        truth = any(_inside(e['body_origin_m'], o['center_m'], o['extent_m']) for o in native.values())
        _require(truth == (frame['pair_member'] == 'in'), 'Native corridor label differs: '+ident)
        _require(not any(_inside(e['body_origin_m'], o['center_m'], o['extent_m'])
                         for name, o in native.items() if name != 'shape0'), 'Native context enters corridor: '+ident)
        label_counts[frame['observation_arm']+('_positive' if truth else '_negative')] += 1
    return dict(status='PASS', source=source_admission, native_frames=len(frames),
                native_center_extent_tolerance_m=NATIVE_TOLERANCE_M,
                world_static=True, labels=dict(label_counts),
                global_background_floor='SOURCE_GEOMETRY_OUTSIDE_CORRIDOR_NATIVE_BOUNDS_NOT_EXPOSED',
                authority='SOURCE_AND_NATIVE_GEOMETRY_ADMISSION_NOT_PREDICTOR_INPUT')


def _target_ray(ray):
    return _positive(ray.get('range_m')) and str(ray.get('actor_id', '')).endswith('/shape0')


def _frame_sampling(row, e):
    public = {z['zone_id']: z for z in row['tof_zones']}
    native = {z['zone_id']: z for z in e['zonal_tof_native']}
    _require(len(row['tof_zones']) == len(e['zonal_tof_native']) == 64
             and set(public) == set(native) == set(range(64)), 'Expected all 64 ToF zones: '+row['id'])
    geometric = valid_hits = strongest_hits = target_slots = strongest_slots = valid_slots = 0
    choices = []
    for zid in range(64):
        zone, observed = native[zid], public[zid]
        rays = {p['subray']: p for p in zone['private_rays']}
        _require(len(rays) == len(zone['private_rays']), 'Duplicate native subray')
        geometric += sum(_target_ray(p) for p in rays.values())
        targets = observed['targets']
        _require(observed['target_count'] == len(targets) <= 2, 'Invalid target count')
        lineage = {r['target_index']: r for r in zone['returned_lineage']}
        _require(len(lineage) == len(zone['returned_lineage']), 'Duplicate returned lineage')
        _require(set(lineage) == set(range(len(targets))), 'Public return/native lineage mismatch')
        _require(bool(zone['packet_received']) == bool(row['tof_packet_received']), 'Packet/native mismatch')
        selected = []
        if row['tof_packet_received']:
            for slot, target in enumerate(targets):
                if target['status'] == 'SIM_VALID':
                    _require(_positive(target['distance_m']) and _finite(target['signal_strength_proxy']), 'Invalid SIM_VALID numeric fields')
                    selected.append((slot, target))
        else:
            _require(not targets, 'Returned target despite missing packet')
        strongest = max(selected, key=lambda p: (p[1]['signal_strength_proxy'], -p[0]))[0] if selected else None
        valid_slots += len(selected)
        for slot, target in selected:
            hit_indices = lineage[slot]['hit_indices']
            _require(len(hit_indices) == len(set(hit_indices)) and all(k in rays for k in hit_indices), 'Invalid lineage hit indices')
            count = sum(_target_ray(rays[k]) for k in hit_indices)
            valid_hits += count
            target_slots += int(count > 0)
            if slot == strongest:
                strongest_hits += count
                strongest_slots += int(count > 0)
        if strongest is not None:
            choices.append(dict(zone=zid, slot=strongest))
    _require(len(row['radar_valid']) == len(row['radar_range_m']) == len(row['radar_angle']) == 4,
             'Expected four public Radar slots')
    return dict(geometric_hit=geometric > 0, valid_target_return=valid_hits > 0,
        strongest_target_return=strongest_hits > 0, geometric_target_ray_hits=geometric,
        valid_target_lineage_contributors=valid_hits, strongest_target_lineage_contributors=strongest_hits,
        valid_target_slots=target_slots, strongest_target_slots=strongest_slots,
        all_public_valid_tof_slots=valid_slots, strongest_choices=choices,
        tof_packet_received=bool(row['tof_packet_received']),
        radar_packet_received=bool(row['radar_packet_received']),
        public_radar_valid_slots=sum(bool(v) for v in row['radar_valid']))


def _aggregate(episodes):
    result = dict(episodes=len(episodes), frames=sum(len(e['frames']) for e in episodes), signals={})
    for signal in SIGNALS:
        result['signals'][signal] = dict(
            current_frames=sum(sum(e['current'][signal]) for e in episodes),
            prefix_available_frames=sum(sum(e['prefix'][signal]) for e in episodes),
            ever_observed_episodes=sum(e['first_time_s'][signal] is not None for e in episodes),
            first_time_s_by_episode={e['episode']: e['first_time_s'][signal] for e in episodes})
    frames = [f for e in episodes for f in e['frames']]
    result['counts'] = {key: sum(f[key] for f in frames) for key in (
        'geometric_target_ray_hits', 'valid_target_lineage_contributors',
        'strongest_target_lineage_contributors', 'valid_target_slots', 'strongest_target_slots',
        'all_public_valid_tof_slots', 'public_radar_valid_slots')}
    result['packets'] = {sensor: dict(received=sum(f[sensor+'_packet_received'] for f in frames),
                                     lost=sum(not f[sensor+'_packet_received'] for f in frames))
                         for sensor in ('tof', 'radar')}
    return result


def audit_sampling(spec, rows, es):
    """Return evaluator-only JSON diagnostics, requiring an admitted MZ155 spec.

The caller seals predictor outputs before supplying evaluator rows. This pure
function cannot inspect the seal or any file. Invalid admission raises ValueError.
"""
    rows, es = _index(rows, 'Public rows'), _index(es, 'Evaluator rows')
    admission = _native_admission(spec, rows, es)
    groups = {}
    for frame in spec['frames']:
        groups.setdefault(frame['episode'], []).append(frame)
    episodes = []
    for episode, source_frames in groups.items():
        source_frames = sorted(source_frames, key=lambda f: f['time_s'])
        frame_records = []
        previous_yaw = None
        for frame in source_frames:
            ident = frame['id']; row, e = rows[ident], es[ident]
            native_delta = 0. if previous_yaw is None else e['camera']['yaw']-previous_yaw
            previous_yaw = e['camera']['yaw']
            residual = row['delta_yaw']-native_delta if row['imu_valid'] and _finite(row['delta_yaw']) else None
            frame_records.append(dict(id=ident, time_s=frame['time_s'], **_frame_sampling(row, e),
                evaluator_commanded_delta_yaw_deg=native_delta,
                public_delta_yaw_deg=row['delta_yaw'] if _finite(row['delta_yaw']) else None, imu_residual_deg=residual))
        current = {signal: [f[signal] for f in frame_records] for signal in SIGNALS}
        prefix = {signal: [any(values[:i+1]) for i in range(len(values))] for signal, values in current.items()}
        first = {signal: next((f['time_s'] for f in frame_records if f[signal]), None) for signal in SIGNALS}
        initial = source_frames[0]
        episodes.append(dict(episode=episode, arm=initial['observation_arm'], family=initial['family'],
            member=initial['pair_member'], matched_case=initial['matched_case'],
            current=current, prefix=prefix, first_time_s=first, frames=frame_records))
    episode_map = {e['episode']: e for e in episodes}
    families = sorted({e['family'] for e in episodes})
    by_arm = {}
    for arm in ARMS:
        group = [e for e in episodes if e['arm'] == arm]
        by_arm[arm] = dict(all=_aggregate(group),
            in_out={member: _aggregate([e for e in group if e['member'] == member]) for member in ('in', 'out')},
            families={family: dict(all=_aggregate([e for e in group if e['family'] == family]),
                in_out={member: _aggregate([e for e in group if e['family'] == family and e['member'] == member])
                        for member in ('in', 'out')}) for family in families})
    matched = []; parity = []; stochastic = []
    for pair in spec['observation_pairs']:
        passive, scan = (episode_map[pair[arm]] for arm in ARMS)
        _require(passive['family'] == scan['family'] and passive['member'] == scan['member'], 'Invalid matched case')
        changes = {}
        for signal in SIGNALS:
            p, s = passive['current'][signal], scan['current'][signal]
            pt, st = passive['first_time_s'][signal], scan['first_time_s'][signal]
            changes[signal] = dict(passive_current_frames=sum(p), scan_current_frames=sum(s),
                net_frames=sum(s)-sum(p), gained_frame_indices=[i for i, (a, b) in enumerate(zip(p, s)) if b and not a],
                lost_frame_indices=[i for i, (a, b) in enumerate(zip(p, s)) if a and not b],
                passive_prefix_available_frames=sum(passive['prefix'][signal]),
                scan_prefix_available_frames=sum(scan['prefix'][signal]),
                prefix_available_frame_change=sum(scan['prefix'][signal])-sum(passive['prefix'][signal]),
                passive_first_time_s=pt, scan_first_time_s=st,
                first_time_change_s=st-pt if st is not None and pt is not None else None,
                gained_ever=pt is None and st is not None, lost_ever=pt is not None and st is None)
        matched.append(dict(matched_case=pair['matched_case'], family=passive['family'], member=passive['member'], changes=changes))
        pfirst, sfirst = (rows[e['frames'][0]['id']] for e in (passive, scan))
        excluded = {'id', 'episode_id', 'rgb_path'}
        keys = (set(pfirst) | set(sfirst))-excluded
        different = sorted(k for k in keys if k not in pfirst or k not in sfirst or pfirst[k] != sfirst[k])
        parity.append(dict(matched_case=pair['matched_case'], equal=not different, differing_fields=different))
        for p, s in zip(passive['frames'], scan['frames']):
            pres, sres = p['imu_residual_deg'], s['imu_residual_deg']
            stochastic.append(dict(matched_case=pair['matched_case'], time_s=p['time_s'],
                passive_id=p['id'], scan_id=s['id'],
                tof_packet_differs=p['tof_packet_received'] != s['tof_packet_received'],
                radar_packet_differs=p['radar_packet_received'] != s['radar_packet_received'],
                passive_imu_residual_deg=pres, scan_imu_residual_deg=sres,
                residual_difference_deg=sres-pres if sres is not None and pres is not None else None))
    rod = {arm: by_arm[arm]['families'][ROD]['all']['signals'] for arm in ARMS}
    lost_rod = [m['matched_case'] for m in matched if m['family'] == ROD and m['changes']['valid_target_return']['lost_ever']]
    family_ever_differences = {family:
        by_arm['scan']['families'][family]['all']['signals']['valid_target_return']['ever_observed_episodes']-
        by_arm['passive']['families'][family]['all']['signals']['valid_target_return']['ever_observed_episodes'] for family in families}
    geometric_gain = rod['scan']['geometric_hit']['current_frames']-rod['passive']['geometric_hit']['current_frames']
    return_gain = rod['scan']['valid_target_return']['current_frames']-rod['passive']['valid_target_return']['current_frames']
    _require(all(by_arm[a]['families'][ROD]['all']['frames'] == 36 for a in ARMS), 'Rod denominator must be36 per arm')
    conditions = dict(rod_geometric_frame_gain_at_least6=geometric_gain >= 6,
        rod_valid_return_frame_gain_at_least6=return_gain >= 6,
        no_lost_passive_rod_ever_observed_cases=not lost_rod,
        no_family_valid_return_ever_count_reduction=all(v >= 0 for v in family_ever_differences.values()))
    residual_differences = [s['residual_difference_deg'] for s in stochastic if s['residual_difference_deg'] is not None]
    return dict(authority='EVALUATOR_ONLY_NATIVE_SAMPLING_COMPONENT_NOT_ALERT_OR_HARDWARE_EVIDENCE',
        radar_native_lineage_authority='NOT_EVALUABLE',
        native_admission=admission, by_arm=by_arm, episodes=episodes, matched_cases=matched,
        first_frame_public_sensor_parity=dict(equal_pairs=sum(p['equal'] for p in parity), total_pairs=len(parity),
            all_equal=all(p['equal'] for p in parity), excluded_fields=sorted(excluded), pairs=parity,
            rgb_pixels_compared=False),
        paired_stochastic_differences=dict(frames=len(stochastic),
            tof_packet_disagreements=sum(s['tof_packet_differs'] for s in stochastic),
            radar_packet_disagreements=sum(s['radar_packet_differs'] for s in stochastic),
            imu_residual_paired_frames=len(residual_differences),
            maximum_abs_imu_residual_difference_deg=max(map(abs, residual_differences), default=None),
            pairs=stochastic, predictor_imu_modified=False),
        component_gate=dict(passed=all(conditions.values()), conditions=conditions, rod_frames_per_arm=36,
            required_additional_frames=6, rod_geometric_frame_gain=geometric_gain,
            rod_valid_return_frame_gain=return_gain, lost_passive_rod_valid_return_cases=lost_rod,
            family_valid_return_ever_count_changes=family_ever_differences,
            ever_observed_definition='AT_LEAST_ONE_PUBLIC_SIM_VALID_TARGET_LINEAGE_RETURN_IN_EPISODE',
            lost_passive_rod_geometric_cases=[m['matched_case'] for m in matched if m['family'] == ROD and m['changes']['geometric_hit']['lost_ever']]),
        limits=['The caller must seal predictions before providing evaluator rows; this pure function performs no I/O or seal inspection.',
            'Target ownership and camera/body geometry are native diagnostic truth, never model inputs.',
            'Geometric ray hits are hypothetical sample coverage, not full object-surface coverage.',
            'Public Radar valid slots are counted; native per-return Radar lineage is NOT_EVALUABLE.',
            'Shared seeds do not imply identical later stochastic draws; IMU residuals are diagnostic only.',
            'No-return and unobserved prefixes are UNKNOWN, not clear-space evidence.'])
