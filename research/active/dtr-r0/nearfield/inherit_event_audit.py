"""Independent state/metric audit; never imports the candidate or run module."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

# A's original geometric scorer is reused, while inheritance and metrics below
# are independently reconstructed. This is not an independent audit of A itself.
from tof_corridor_calibration import score_frame, decide
from tof_fov45_core import boxes45

ARMS = ('A_current', 'A_hold', 'support_inherit')
REPO = Path(__file__).resolve().parents[4]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def runs(flags):
    start = None
    result = []
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            result.append((start, i))
            start = None
    return result


def counts_and_events(rows, arm):
    counts = dict(TP=0, FP=0, FN=0, TN=0, prediction_unknown=0,
                  abstained_positive=0, abstained_negative=0)
    clips = defaultdict(list)
    for r in rows:
        p, truth = r['predictions'][arm], r['truth']
        assert type(truth) is bool
        counts['prediction_unknown'] += p['unknown']
        if p['alert']:
            counts['TP' if truth else 'FP'] += 1
        elif truth:
            counts['FN'] += 1
        elif not p['unknown']:
            counts['TN'] += 1
        if p['unknown'] and not p['alert']:
            counts['abstained_positive' if truth else 'abstained_negative'] += 1
        clips[r['clip_id']].append(r)
    events, segments, details = [], [], []
    for clip_id, seq in sorted(clips.items()):
        seq.sort(key=lambda r: r['frame_in_clip'])
        flags = [r['predictions'][arm]['alert'] for r in seq]
        positives = [r['truth'] for r in seq]
        assert len(seq) == 12
        for lo, hi in runs(positives):
            hit = [i for i in range(lo, hi) if flags[i]]
            event = (clip_id, lo, hi-1, bool(hit), seq[hit[0]]['time_s'] if hit else None)
            events.append(event)
            gaps = sum(not flags[i] for i in range(hit[0]+1, hit[-1])) if hit else 0
            first = next((i for i in range(hi, len(seq)) if not flags[i]), None)
            tail = (first-hi if first is not None else len(seq)-hi) if hi < len(seq) else None
            detail = dict(clip_id=clip_id, positive_alert_frames=sum(flags[lo:hi]),
                          initial_silent_frames=hit[0]-lo if hit else hi-lo,
                          internal_silent_frames=gaps,
                          terminal_silent_frames=hi-1-hit[-1] if hit else 0,
                          preentry_false_alert_frames=sum(flags[:lo]),
                          postexit_false_alert_frames=sum(flags[hi:]),
                          postexit_leading_alert_frames=tail,
                          postexit_carryover_alert_frames=(tail if flags[hi-1] else 0) if tail is not None else None,
                          first_silent_relative_to_exit_s=(first-hi)*.2 if first is not None else None,
                          release_right_censored=first is None)
            details.append(detail)
        for lo, hi in runs([f and not y for f, y in zip(flags, positives)]):
            segments.append((clip_id, lo, hi-1))
    return counts, events, segments, details


def paired(rows, left, right):
    result = {k: [] for k in ('TP_gained','TP_lost','FP_added','FP_removed')}
    for r in rows:
        a, b = r['predictions'][left]['alert'], r['predictions'][right]['alert']
        if a != b:
            name = ('TP_gained' if b else 'TP_lost') if r['truth'] else ('FP_added' if b else 'FP_removed')
            result[name].append(r['id'])
    return result


def run(args):
    folder = args.run
    seal = read(folder/'evaluation-seal.json')
    for name, digest in seal['hashes'].items():
        assert sha(folder/name) == digest, name
    for name, digest in seal['source_hashes'].items():
        assert sha(REPO/name) == digest, name
    preds, frame_rows = read(folder/'predictions.json'), read(folder/'frame-results.json')
    report = read(folder/'metrics.json')
    identities = read(args.identities)
    tokens = np.load(args.tof, allow_pickle=False, mmap_mode='r')
    previous = None
    for i, (p, m) in enumerate(zip(preds, identities)):
        assert p['id'] == m['id'] and p['index'] == i
        raw = tokens[i]
        assert np.array_equal(np.rint(raw[:,2:]*[192,256,192,256]).astype(int), boxes45())
        values = np.where(raw[:,1] == 1, raw[:,0]*8, np.nan).astype(np.float32)
        score = score_frame(boxes45(), values)
        base = decide(score, .4071309640537889)
        assert p['base'] == base == m['baseline']
        joint = {s['zone']: s['joint'] for s in score['zone_scores']}
        supports = {a['zone']: dict(zone=a['zone'], range_m=float(values[a['zone']]),
                      low_m=a['interval_m'][0], high_m=a['interval_m'][1],
                      possible=a['possible'], definite=a['definite'], joint=joint[a['zone']])
                    for a in score['anchors']}
        assert list(supports.values()) == p['supports']
        decisive = [s for s in supports.values() if s['definite'] or s['possible'] and s['joint'] >= .4071309640537889]
        same = previous is not None and previous['clip_id'] == p['clip_id']
        adjacent = same and previous['frame_in_clip']+1 == p['frame_in_clip']
        age = p['captured_at_ns']-previous['captured_at_ns'] if same else None
        eligible = adjacent and previous['flags']['A_current'] and 0 < age <= 200_000_000
        matches = []
        if eligible:
            for old in previous['supports']:
                if not (old['definite'] or old['possible'] and old['joint'] >= .4071309640537889):
                    continue
                new = supports.get(old['zone'])
                if new and new['possible'] and .3 <= new['range_m'] <= 3. and max(old['low_m'],new['low_m']) < min(old['high_m'],new['high_m']):
                    matches.append(old['zone'])
        expected = dict(A_current=base['alert'],
                        A_hold=bool(base['alert'] or adjacent and previous['flags']['A_current']),
                        support_inherit=bool(base['alert'] or matches))
        assert p['flags'] == expected
        assert p['unknown'] == base['unknown']
        assert p['decisive_zones'] == [s['zone'] for s in decisive]
        assert p['matched_zones'] == matches
        assert not p['flags']['support_inherit'] or p['flags']['A_hold']
        previous = p
    by_index = {r['index']: r for r in frame_rows}
    audits = {}
    for role in ('train','dev','evaluation'):
        path = getattr(args, role+'_labels')
        receipt = next(r for r in report['evaluator_inputs'] if r['role'] == role)
        assert sha(path) == receipt['sha256']
        labels = np.load(path, allow_pickle=False)
        rows = [by_index[int(i)] for i in labels['indices']]
        for k, row in enumerate(rows):
            assert labels['valid'][k,[1,4]].all()
            assert row['truth'] == bool((labels['classes'][k,[1,4]] < 6).any())
            p = preds[row['index']]
            for arm in ARMS:
                assert row['predictions'][arm] == dict(alert=p['flags'][arm], unknown=p['unknown'],
                    ambiguous=bool(p['unknown'] and p['flags'][arm]))
        for arm in ARMS:
            counts, events, segments, detail = counts_and_events(rows, arm)
            stored = report['roles'][role]['metrics']['arms'][arm]
            for key, value in counts.items():
                assert stored['frames']['all_known'][key] == value, (role,arm,key)
            assert events == [(e['clip_id'],e['start_frame'],e['end_frame'],e['detected'],e['first_alert_time_s']) for e in stored['events']]
            assert segments == [(e['clip_id'],e['start_frame'],e['end_frame']) for e in stored['false_alert_segments']]
            assert stored['false_alert_segment_count'] == len(segments)
            sd = {d['clip_id']: d['event'] for d in report['roles'][role]['event_details'][arm]}
            for d in detail:
                for key, value in d.items():
                    if key != 'clip_id':
                        assert sd[d['clip_id']][key] == value, (role,arm,d['clip_id'],key)
        comp = report['roles'][role]['comparison']
        for left, right, key in (('A_current','A_hold','paired_current_to_hold'),
                                 ('A_current','support_inherit','paired_current_to_inherit'),
                                 ('A_hold','support_inherit','paired_hold_to_inherit')):
            assert comp[key] == paired(rows,left,right)
        ch, ci, hi = (comp[k] for k in ('paired_current_to_hold','paired_current_to_inherit','paired_hold_to_inherit'))
        layout_count = len({r['base_group_id'] for r in rows if r['id'] in ci['TP_gained']})
        current_events = {e[:3]: e for e in counts_and_events(rows,'A_current')[1]}
        candidate_events = {e[:3]: e for e in counts_and_events(rows,'support_inherit')[1]}
        no_lost = all(not e[3] or candidate_events[k][3] for k,e in current_events.items())
        no_delayed = all(not e[3] or candidate_events[k][4] <= e[4] for k,e in current_events.items())
        gates = dict(hold_TP_gain_nonzero=bool(ch['TP_gained']),hold_FP_cost_nonzero=bool(ch['FP_added']),
            retain_75_percent_hold_TP_gain=bool(ch['TP_gained']) and len(ci['TP_gained']) >= .75*len(ch['TP_gained']),
            remove_50_percent_hold_added_FP=bool(ch['FP_added']) and len(hi['FP_removed']) >= .5*len(ch['FP_added']),
            at_least_8_added_TP=len(ci['TP_gained']) >= 8,at_least_4_gain_layouts=layout_count >= 4,
            preserve_A_current_TP=not ci['TP_lost'],no_lost_current_events=no_lost,no_delayed_current_events=no_delayed,
            no_new_false_segments_vs_hold=len(counts_and_events(rows,'support_inherit')[2]) <= len(counts_and_events(rows,'A_hold')[2]))
        assert gates == comp['gates'] and comp['retain'] == all(gates.values())
        audits[role] = dict(frames=len(rows),gates=gates,retain=all(gates.values()))
    assert report['terminal'] == ('RETAIN_SCOPED_CHALLENGER' if audits['evaluation']['retain'] else 'EXACT_RECIPE_NEGATIVE_CONTROL')
    result = dict(status='PASS',frames=len(preds),roles=audits,
        scope='Independent inheritance equations, public-support replay, per-frame counts, events, gaps, exit tails and gates; frozen A scorer reused',
        prediction_seal_sha256=sha(folder/'prediction-seal.json'),metrics_sha256=sha(folder/'metrics.json'),
        candidate_module_imported=False, runner_module_imported=False,
        new_candidate_or_fit=False, unknown_parity_frames=1728)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','identities','tof','train-labels','dev-labels','evaluation-labels','result'):
        parser.add_argument('--'+name,required=True,type=Path)
    run(parser.parse_args())
