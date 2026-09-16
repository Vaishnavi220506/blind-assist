"""Consumed evaluator-only scan/source diagnostic; never an alert predictor."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from mz133_angular_resolution import ROOT, admission, geometry, trace, reduce_zone, read, write, sha, jsonl
from mz155_sampling_audit import _frame_sampling, _target_ray
from research_backend import BackendCandidate, DeviceObservation, select_backend

SOURCE = ROOT/'artifacts.local/work/mz155-active-sampling-20260916/source/returned-v1/capture-v1'
WORK = ROOT/'artifacts.local/work/mz163-integration-robustness-20260916'
SIGNALS = ('geometric_hit', 'returned_target', 'valid_target_return', 'strongest_target_return')


def admit_target_identity(spec, rows, es):
    _, rays = geometry(8, 3)
    checked = slots = 0
    failures = []
    states = {}
    for frame, row, native in zip(spec['frames'], rows, es):
        ds, rs, owners, _, _ = trace(frame, spec, rays)
        rng = states.setdefault(frame['episode'], random.Random(frame.get('tof_sensor_seed', frame['sensor_seed']+1150003)))
        for zid, z in enumerate(native['zonal_tof_native']):
            expected = [_target_ray(r) for r in z['private_rays']]
            actual = (np.isfinite(ds[zid]) & (owners[zid] == 0)).tolist()
            checked += len(actual)
            if actual != expected:
                failures.append([row['id'], zid, 'TARGET_IDENTITY'])
            _, lineages, _ = reduce_zone(ds[zid], rs[zid], rng, row['tof_packet_received'], 1., .04)
            previous = [r['hit_indices'] for r in z['returned_lineage']]
            slots += len(lineages)
            if lineages != previous:
                failures.append([row['id'], zid, 'RETURNED_LINEAGE'])
    return dict(status='PASS' if not failures else 'FAIL', target_ray_bits=checked,
                returned_slots=slots, failures=failures)


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['episode']].append(r)
    episodes = []
    for name, rows in groups.items():
        rows = sorted(rows, key=lambda r: r['time_s'])
        assert len(rows) == 6 and [r['time_s'] for r in rows] == [i*.25 for i in range(6)]
        first = {s: next((r['time_s'] for r in rows if r[s]), None) for s in SIGNALS}
        episodes.append(dict(episode=name, family=rows[0]['family'], arm=rows[0]['arm'],
            member=rows[0]['member'], matched_case=rows[0]['matched_case'], first=first,
            current={s: sum(r[s] for r in rows) for s in SIGNALS},
            prefix={s: sum(any(r[s] for r in rows[:i+1]) for i in range(6)) for s in SIGNALS},
            flags={s: [r[s] for r in rows] for s in SIGNALS}))
    totals = {}
    for family in ['ALL'] + sorted({e['family'] for e in episodes}):
        totals[family] = {}
        for member in ('all', 'in', 'out'):
            totals[family][member] = {}
            for arm in ('passive', 'scan'):
                es = [e for e in episodes if e['arm'] == arm and
                      (family == 'ALL' or e['family'] == family) and (member == 'all' or e['member'] == member)]
                totals[family][member][arm] = dict(episodes=len(es), frames=6*len(es),
                    signals={s: dict(current=sum(e['current'][s] for e in es),
                        prefix=sum(e['prefix'][s] for e in es), ever=sum(e['first'][s] is not None for e in es)) for s in SIGNALS})
    paired = []
    lookup = {e['episode']: e for e in episodes}
    for p in episodes:
        if p['arm'] != 'passive':
            continue
        s = lookup[p['matched_case']+'_scan']
        changes = {}
        for key in SIGNALS:
            a, b = p['first'][key], s['first'][key]
            changes[key] = dict(current_delta=s['current'][key]-p['current'][key],
                prefix_delta=s['prefix'][key]-p['prefix'][key], passive_first=a, scan_first=b,
                gained_ever=a is None and b is not None, lost_ever=a is not None and b is None,
                delay_s=b-a if a is not None and b is not None else None,
                gained_frames=[i for i, (x,y) in enumerate(zip(p['flags'][key],s['flags'][key])) if y and not x],
                lost_frames=[i for i, (x,y) in enumerate(zip(p['flags'][key],s['flags'][key])) if x and not y])
        paired.append(dict(matched_case=p['matched_case'], family=p['family'], member=p['member'], changes=changes))
    return dict(totals=totals, episodes=episodes, paired=paired)


def run(output):
    start = time.perf_counter()
    assert not output.exists() and output.resolve().is_relative_to(WORK.resolve())
    output.mkdir(parents=True)
    paths = [SOURCE/n for n in ('spec.json', 'raw.jsonl', 'evaluator.jsonl', 'receipt.json')]
    near = Path(__file__).parent
    paths += [near/n for n in ('run_mz163_integration_robustness.py', 'MZ163_PROTOCOL_20260916.md',
        'mz133_angular_resolution.py', 'mz115_zonal_tof.py', 'mz113_dynamic_sensors.py', 'mz155_sampling_audit.py')]
    hashes = {str(p): sha(p) for p in paths}
    write(output/'freeze.json', dict(time_utc=datetime.now(timezone.utc).isoformat(), input_hashes=hashes,
        authority='CONSUMED_EVALUATOR_ONLY_SIMULATOR_INTEGRATION_DIAGNOSTIC',
        densities=[3,12], packet_conditions=['recorded','all_available'], budget_s=600,
        previous_goal_turn='NO_PROGRESS_STATUS_ONLY_NOW_TAKING_SOURCE_FALSIFIER_ACTION'))
    spec, rows, es, receipt = read(paths[0]), jsonl(paths[1]), jsonl(paths[2]), read(paths[3])
    assert receipt['status'] == 'PASS' and sha(paths[0]) == receipt['spec_sha256']
    assert all(sha(SOURCE/n) == receipt['hashes'][n] for n in ('raw.jsonl', 'evaluator.jsonl'))
    assert len(rows) == len(es) == len(spec['frames']) == 288
    assert [f['id'] for f in spec['frames']] == [r['id'] for r in rows] == [e['id'] for e in es]
    admit = admission(spec, rows, es)
    write(output/'admission.json', admit)
    assert admit['status'] == 'PASS', 'Analytic parity failed; no dense comparison admitted'
    ownership = admit_target_identity(spec, rows, es)
    write(output/'ownership-admission.json', ownership)
    assert ownership['status'] == 'PASS', 'Target identity/lineage mismatch; no dense comparison admitted'
    print(json.dumps({'admission': admit}), flush=True)
    select_backend('scalar-scoring', cpu=BackendCandidate('numpy-and-scalar-histograms', 'cpu',
        lambda: reduce_zone(np.array([2.]*144), np.array([.5]*144), random.Random(1), True, 1., .04),
        lambda _: DeviceObservation('cpu', 'host CPU', 'NumPy '+np.__version__)),
        record_path=output/'backend.json', capabilities=dict(python=sys.executable, numpy=np.__version__,
            cpu_reason='TASK_NOT_GPU_SUITABLE', note='Bounded irregular scalar histogram and lineage reduction; no model work'))
    summary = dict(arms={}, timing_s={})
    for sub in (3, 12):
        begin = time.perf_counter()
        _, rays = geometry(8, sub)
        records = {mode: [] for mode in ('recorded', 'all_available')}
        states = {}
        for index, (frame, row, native) in enumerate(zip(spec['frames'], rows, es)):
            assert time.perf_counter()-start < 600, 'Declared budget exhausted'
            assert frame['objects'][0]['name'] == 'shape0'
            ds, rs, owners, _, _ = trace(frame, spec, rays)
            target = np.isfinite(ds) & (owners == 0)
            count = {mode: dict(returned_target=0, valid_target_return=0, strongest_target_return=0,
                               target_slots=0, valid_slots=0, merged_target_slots=0) for mode in records}
            rng = states.setdefault(frame['episode'], random.Random(frame.get('tof_sensor_seed', frame['sensor_seed']+1150003)))
            for zid in range(64):
                # All-available status/lineage is deterministic before range noise.
                targets, lines, _ = reduce_zone(ds[zid], rs[zid], rng, True, 1., .04)
                valid = [i for i, t in enumerate(targets) if t['status'] == 'SIM_VALID']
                strongest = max(valid, key=lambda i: (targets[i]['signal_strength_proxy'], -i)) if valid else None
                for mode in records:
                    if mode == 'recorded' and not row['tof_packet_received']:
                        continue
                    count[mode]['valid_slots'] += len(valid)
                    for slot, (t, ids) in enumerate(zip(targets, lines)):
                        if not target[zid, ids].any():
                            continue
                        count[mode]['returned_target'] += 1
                        count[mode]['target_slots'] += 1
                        if t['status'] == 'SIM_VALID':
                            count[mode]['valid_target_return'] += 1
                            count[mode]['strongest_target_return'] += int(slot == strongest)
                        else:
                            count[mode]['merged_target_slots'] += 1
            finite_weights = np.zeros_like(ds)
            finite_weights[target] = rs[target]/(sub*sub*np.maximum(ds[target], .2)**2)
            for mode in records:
                c = count[mode]
                record = dict(id=frame['id'], episode=frame['episode'], matched_case=frame['matched_case'],
                    family=frame['family'], member=frame['pair_member'], arm=frame['observation_arm'],
                    time_s=frame['time_s'], geometric_hit=bool(target.any()),
                    **{s: bool(c[s]) for s in SIGNALS if s != 'geometric_hit'}, counts=c,
                    geometric_target_zone_equivalents=float(target.sum()/(sub*sub)),
                    geometric_target_signal=float(finite_weights.sum()), packet_received=bool(row['tof_packet_received']))
                if sub == 3 and mode == 'recorded':
                    previous = _frame_sampling(row, native)
                    assert all(record[s] == previous[s] for s in SIGNALS if s != 'returned_target'), row['id']
                records[mode].append(record)
            if (index+1) % 72 == 0:
                print(json.dumps(dict(sub=sub, frames=index+1, elapsed_s=time.perf_counter()-start)), flush=True)
        for mode, frames in records.items():
            name = f'q{sub}_{mode}'
            write(output/(name+'-frames.json'), frames)
            summary['arms'][name] = aggregate(frames)
        summary['timing_s'][str(sub)] = time.perf_counter()-begin
    checks = {}
    rod = 'near_rod_farwall'
    for mode in ('recorded', 'all_available'):
        dense = summary['arms']['q12_'+mode]
        pair = dense['totals'][rod]['all']
        p, s = (pair[a]['signals']['valid_target_return'] for a in ('passive', 'scan'))
        checks[mode] = dict(rod_ever_delta=s['ever']-p['ever'], rod_prefix_delta=s['prefix']-p['prefix'],
            lost_ever_cases=[c['matched_case'] for c in dense['paired'] if c['changes']['valid_target_return']['lost_ever']])
    robust = all(v['rod_ever_delta'] > 0 and v['rod_prefix_delta'] > 0 and not v['lost_ever_cases'] for v in checks.values())
    summary.update(checks=checks, robustness_condition_met=robust,
        decision='MZ163_SCAN_COVERAGE_SURVIVES_INTEGRATION_CONTROL' if robust else 'MZ163_SCAN_COVERAGE_INTEGRATION_SENSITIVE_OR_CONDITIONAL',
        seconds=time.perf_counter()-start, alerts_evaluated=False, hardware_inference=False)
    write(output/'summary.json', summary)
    assert all(sha(p) == digest for p, digest in hashes.items())
    write(output/'completion.json', dict(status='PASS', summary_sha256=sha(output/'summary.json'),
        artifact_hashes={p.name: sha(p) for p in output.glob('*.json')}, decision=summary['decision'],
        seconds=summary['seconds'], resources='No persistent workers or allocations'))
    print(json.dumps(dict(decision=summary['decision'], checks=checks, seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
