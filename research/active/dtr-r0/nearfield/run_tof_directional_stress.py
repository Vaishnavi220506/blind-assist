"""Fixed synthetic packet falsifier; scene truth never enters the readout."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from mz115_zonal_tof import zone_geometry
from tof_directional_readout import readout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cases(protocol):
    def blank():
        return dict(tof_packet_received=True, tof_zones=[
            dict(zone_geometry(i)[0], targets=[]) for i in range(64)])

    def put(row, ids, distance, status='SIM_VALID'):
        for i in ids:
            row['tof_zones'][i]['targets'].append(dict(status=status, distance_m=distance))

    def mirrored(row):
        out = copy.deepcopy(row)
        for i in range(64):
            out['tof_zones'][i]['targets'] = copy.deepcopy(row['tof_zones'][i//8*8+7-i%8]['targets'])
        return out

    cases = []

    def append(family, row, footprints, variant, mirror=False, rgb=None):
        if mirror:
            row = mirrored(row)
            footprints = {('RIGHT' if k == 'LEFT' else 'LEFT'): [i//8*8+7-i%8 for i in ids]
                          for k, ids in footprints.items()}
        for zone in row['tof_zones']:
            zone['target_count'] = len(zone['targets'])
        cases.append(dict(id=f'{family}-{variant}-{int(mirror)}', family=family,
                          observable=row, evaluator=dict(near_footprints=footprints),
                          synthetic_rgb_direction=rgb))

    near, far, merged = (protocol[k] for k in ('near_m', 'far_m', 'merged_nominal_m'))
    for variant in range(5):
        # A valid background progressively connects to and outnumbers the near zone.
        row = blank(); put(row, [25], near)
        put(row, list(range(26, min(32, 29+variant))) + ([18, 19, 20, 21, 22, 23] if variant == 4 else []), far)
        for mirror in (False, True):
            append('near_far_background', row, {'LEFT': [25]}, variant, mirror)

        # Both sides truly near, but one has only 1-3 reliable zones. A weak
        # merged-return bridge belongs to background; status is not actor identity.
        row = blank(); left = [17, 25, 33][:1+variant%3]; right = [21, 29, 37]
        put(row, left, near); put(row, right, merged, 'SIM_MERGED')
        put(row, [18, 19, 20, 26, 27, 28, 34, 35, 36], merged, 'SIM_MERGED')
        for mirror in (False, True):
            append('sparse_reliable_side', row, {'LEFT': left, 'RIGHT': right}, variant, mirror)

        # Two resolved slots vs one far-biased merged tuple in the very same zone.
        row = blank()
        if variant % 2 == 0:
            put(row, [25], near); put(row, [25], far)
        else:
            put(row, [25], merged, 'SIM_MERGED')
        put(row, list(range(26, min(32, 29+variant))) + ([18, 19, 20, 21, 22, 23] if variant == 4 else []), far)
        for mirror in (False, True):
            append('same_zone_pole_wall', row, {'LEFT': [25]}, variant, mirror)

        # Two real near modes, with an absent, incomplete, or complete far bridge.
        row = blank(); put(row, [25, 30], near)
        bridges = [[], [27], [26, 27], [26, 27, 28], [26, 27, 28, 29]]
        put(row, bridges[variant], far)
        for mirror in (False, True):
            append('bilateral_near', row, {'LEFT': [25], 'RIGHT': [30]}, variant, mirror)

        row = blank(); put(row, [30, 38][:1+variant%2], near)
        for mirror in (False, True):
            append('rgb_tof_conflict', row, {'RIGHT': [30, 38][:1+variant%2]}, variant,
                   mirror, rgb='RIGHT' if mirror else 'LEFT')

    for frame in range(10):
        row = blank(); put(row, [25], near)
        if frame % 2 == 0:
            put(row, [30], near)
        append('one_side_dropout', row, {'LEFT': [25], 'RIGHT': [30]}, frame)
    return sorted(cases, key=lambda c: (protocol['cases'].index(c['family']), c['id']))


def score(result, truth):
    actual = sorted({r['horizontal'] for r in result['regions']})
    expected = sorted(truth['near_footprints'])
    footprints = truth['near_footprints']
    bilateral = 'LEFT' in footprints and 'RIGHT' in footprints
    merge = bilateral and any(set(r['zone_ids']) & set(footprints['LEFT']) and
                              set(r['zone_ids']) & set(footprints['RIGHT']) for r in result['regions'])
    return dict(direction_correct=int(actual == expected),
                false_center=int('CENTER' in actual and 'CENTER' not in expected),
                bilateral_merge=int(bool(merge)), unknown=int(result['state'].startswith('UNKNOWN')),
                missing_modes=sorted(set(expected)-set(actual)),
                extra_modes=sorted(set(actual)-set(expected)), actual=actual, expected=expected)


def summarize(rows):
    keys = ('direction_correct', 'false_center', 'bilateral_merge', 'unknown')
    return dict(frames=len(rows), **{k: sum(r[k] for r in rows) for k in keys},
                direction_accuracy=sum(r['direction_correct'] for r in rows)/len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol_path = Path(__file__).with_name('tof_directional_stress_protocol_20260914.json')
    protocol = json.loads(protocol_path.read_text())
    cases = build_cases(protocol)
    assert len(cases) == 60 and len({c['id'] for c in cases}) == 60
    assert all(sum(c['family'] == family for c in cases) == 10 for family in protocol['cases'])
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze packets and evaluator labels before either arm is executed.
    packet_path = args.output/'packets.jsonl'
    truth_path = args.output/'evaluator.jsonl'
    packet_path.write_text(''.join(json.dumps(dict(id=c['id'], **c['observable']))+'\n' for c in cases), encoding='utf-8')
    truth_path.write_text(''.join(json.dumps({k:v for k,v in c.items() if k != 'observable'})+'\n' for c in cases), encoding='utf-8')
    results = []
    start = time.perf_counter()
    for case in cases:
        arms = {}
        for arm, equal in protocol['arms'].items():
            output = readout(copy.deepcopy(case['observable']), equal_weights=equal)
            arms[arm] = dict(output=output, score=score(output, case['evaluator']))
        results.append(dict(id=case['id'], family=case['family'], arms=arms))
    seconds = time.perf_counter()-start
    totals = {a: summarize([r['arms'][a]['score'] for r in results]) for a in protocol['arms']}
    families = {f: {a: summarize([r['arms'][a]['score'] for r in results if r['family'] == f])
                    for a in protocol['arms']} for f in protocol['cases']}
    mixed = ('near_far_background', 'same_zone_pole_wall')
    gain = sum(families[f]['status_1_0.5']['direction_correct'] for f in mixed) > sum(families[f]['equal']['direction_correct'] for f in mixed)
    eq, wt = totals['equal'], totals['status_1_0.5']
    retain = gain and wt['direction_correct'] >= eq['direction_correct'] and all(wt[k] <= eq[k] for k in ('false_center', 'bilateral_merge'))
    dropout_changes = {}
    for arm in protocol['arms']:
        sequence = [r['arms'][arm]['score']['actual'] for r in results if r['family'] == 'one_side_dropout']
        dropout_changes[arm] = sum(a != b for a, b in zip(sequence, sequence[1:]))
    summary = dict(protocol_id=protocol['id'], authority=protocol['authority'], totals=totals,
                   families=families, dropout_direction_set_changes=dropout_changes,
                   decision='RETAIN_WEIGHT_CHALLENGER' if retain else 'FREEZE_EQUAL_NO_STATUS_WEIGHT_GAIN',
                   backend='CPU: TASK_NOT_GPU_SUITABLE', paired_readout_seconds=seconds,
                   hashes={str(p): sha(p) for p in (protocol_path, Path(__file__),
                       Path(__file__).with_name('tof_directional_readout.py'),
                       Path(__file__).with_name('mz115_zonal_tof.py'), packet_path, truth_path)})
    (args.output/'predictions.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    summary['hashes'][str(args.output/'predictions.json')] = sha(args.output/'predictions.json')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
