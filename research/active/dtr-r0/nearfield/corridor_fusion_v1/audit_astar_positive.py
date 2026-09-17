"""Independent scalar counts/native strata/contiguous events for saved OR."""
from itertools import groupby
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT/'artifacts.local/work/corridor-public-single-20260917'
OUT = ROOT/'artifacts.local/work/corridor-astar-positive-posthoc-20260917'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    seal, done, result = [read(OUT/name) for name in ['analysis-seal.json', 'completion.json', 'summary.json']]
    for p, h in seal['inputs'].items():
        assert sha(p) == h, p
    for name in ['summary', 'predictions', 'analysis-seal']:
        assert sha(OUT/(name+'.json')) == done[name.replace('-', '_')+'_sha256']
    original = read(SOURCE/'confirmation/predictions.json')
    cases = read(SOURCE/'confirmation/cases.json')
    combined = read(OUT/'predictions.json')
    native = [json.loads(s) for s in (SOURCE/'source/returned-v1/capture-v1/evaluator.jsonl').read_text().splitlines()]
    assert len(original) == len(cases) == len(combined) == len(native) == 288
    for p, c, q, e in zip(original, cases, combined, native):
        assert p['id'] == c['id'] == q['id'] == e['id']
        assert q['combined'] == (p['control'] or p['positive'])
        assert not p['control'] or q['combined']
        if p['usable_tof_returns'] == 0:
            assert q['combined'] == p['control']
        margins = []
        for o in e['native_bounds']:
            lo = [o['center_m'][k]-o['extent_m'][k]-e['body_origin_m'][k] for k in range(3)]
            hi = [o['center_m'][k]+o['extent_m'][k]-e['body_origin_m'][k] for k in range(3)]
            if hi[0] >= .2 and lo[0] <= 3.6 and hi[2] >= .4 and lo[2] <= 2.05:
                margins.append(min(hi[1]+.3, .3-lo[1]))
        m = max(margins) if margins else float('-inf')
        assert c['truth'] == (m >= 0)
        assert c['stratum'] == ('positive' if m >= .05 else 'negative' if m < -.05 else 'boundary')
    audited = {}
    for name, key in [('A', 'A'), ('A_plus_public', 'alert'), ('A_retrained', 'control'), ('A_star_plus_public', None)]:
        flags = [p[key] if key else q['combined'] for p, q in zip(original, combined)]
        report = result['methods'][name]
        for tag in ['clear', 'strict', 'boundary']:
            selected = [i for i, c in enumerate(cases) if tag == 'strict' or ((c['stratum'] != 'boundary') == (tag == 'clear'))]
            counts = {k: 0 for k in ['TP', 'FP', 'FN', 'TN']}
            for i in selected:
                counts[('T' if cases[i]['truth'] else 'F')+'P' if flags[i] else ('F' if cases[i]['truth'] else 'T')+'N'] += 1
            assert all(report[tag][k] == v for k, v in counts.items())
        for key in ['temporal', 'strict_temporal']:
            events = []
            for episode, part in groupby(range(288), key=lambda i: cases[i]['episode_id']):
                ii = list(part)
                inside = [cases[i]['truth'] if key == 'strict_temporal' else cases[i]['stratum'] == 'positive' for i in ii]
                start = 0
                while start < len(ii):
                    if not inside[start]:
                        start += 1
                        continue
                    stop = start+1
                    while stop < len(ii) and inside[stop]:
                        stop += 1
                    hits = [i for i in ii[start:stop] if flags[i]]
                    onset = cases[hits[0]]['time_s']-cases[ii[start]]['time_s'] if hits else None
                    events.append((episode, cases[ii[start]]['time_s'], onset))
                    start = stop
            expected = [(e['episode'], e['start_s'], e['first_in_core_delay_s']) for e in report[key]['events']]
            assert events == expected
            assert report[key]['core_events_detected'] == sum(e[2] is not None for e in events)
        audited[name] = dict(core=report['temporal']['core_events_detected'], strict=report['strict_temporal']['core_events_detected'])
    recovered = [v for v in result['old_A_clear_TP_lost_by_A_star'] if v['recovered']]
    assert len(result['old_A_clear_TP_lost_by_A_star']) == 4 and not recovered
    audit = dict(status='PASS', frames=288, original_inputs_unchanged=True,
        exact_fixed_OR=True, A_star_retained=True, unknown_fallback=True,
        scalar_counts_and_native_geometry=True, independent_event_onsets=True,
        original_four_lost_clear_TP_recovered=0, events=audited,
        summary_sha256=sha(OUT/'summary.json'), audit_source_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
