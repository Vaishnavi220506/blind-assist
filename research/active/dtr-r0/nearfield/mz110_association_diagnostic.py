"""Consumed evaluator-only opportunity audit; never import into prediction."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import mz108_competitive_association as model
import mz109_interval_extent as interval
from run_mz107_four_sensor import ROOT, readrows, sha, truth, metrics, write
from run_mz108_competitive_association import projected_objects, iou


def observable_trace(row, pred, require_tof=True):
    """No evaluator inputs; reconstruct gates on cached pixel-derived boxes."""
    intr = row['rgb_intrinsics']; yaw = pred['integrated_yaw_deg']
    tof = []
    if row['tof_packet_received']:
        for k, (r, s, a, e) in enumerate(zip(row['tof64_range_m'], row['tof64_status'],
                                           row['tof64_theta_deg'], row['tof64_phi_deg'])):
            if s != 5 or r is None or not math.isfinite(r) or r <= 0:
                continue
            d = model.ray(a, e, row['camera_pitch_deg'], yaw)
            tof.append((k, intr['cx']+intr['fx']*math.tan(math.radians(a)),
                        intr['cy']-intr['fy']*math.tan(math.radians(e)),
                        r*math.hypot(d[0], d[1])))
    returns = []
    if row['radar_packet_received']:
        for k, (r, a, valid) in enumerate(zip(row['radar_range_m'], row['radar_angle'], row['radar_valid'])):
            if not valid or r is None or a is None or not math.isfinite(r+a) or r <= 0:
                continue
            angle = math.radians(a+yaw)
            current = .2 <= r*math.cos(angle) <= 3.6 and abs(r*math.sin(angle)) <= .3
            edges = []
            for j, box in enumerate(pred['proposals']):
                angles = [math.degrees(math.atan((u-intr['cx'])/intr['fx'])) for u in (box[0], box[2])]
                angular = min(angles)-12 <= a <= max(angles)+12
                inside = [t for t in tof if box[0] <= t[1] <= box[2] and box[1] <= t[2] <= box[3]]
                agreeing = [t for t in inside if abs(t[3]-r) <= .35]
                tof_gate = bool(agreeing) and len(inside) == len(agreeing)
                reason = ('PASS' if tof_gate else 'TOF_PACKET_MISSING' if not row['tof_packet_received']
                          else 'NO_VALID_TOF_IN_BOX' if not inside
                          else 'NO_AGREEING_RANGE' if not agreeing else 'COMPETING_RANGE')
                nominal = model.extent_inside(box, r, intr, row['camera_pitch_deg'], yaw, row['camera_in_body_m'][2])
                state = interval.classify(box, r, row, yaw)['state']
                edges.append(dict(proposal=j, angular=angular, tof_gate=tof_gate,
                                  tof_reason=reason, tof_zones=[t[0] for t in agreeing],
                                  eligible=angular and (tof_gate or not require_tof),
                                  nominal=nominal, interval=state == 'IN' or (state == 'AMBIGUOUS' and current)))
            returns.append(dict(slot=k, baseline=current, edges=edges))
    counts = Counter(e['proposal'] for ret in returns for e in ret['edges'] if e['eligible'])
    candidate = bool(pred['tof_support']); baseline = candidate
    for ret in returns:
        eligible = [e for e in ret['edges'] if e['eligible']]
        selected = eligible[0] if len(eligible) == 1 and counts[eligible[0]['proposal']] == 1 else None
        ret['selected'] = selected['proposal'] if selected else None
        ret['support'] = selected['nominal'] if selected else ret['baseline']
        candidate |= ret['support']; baseline |= ret['baseline']
    return dict(candidate=bool(candidate), baseline=bool(baseline), returns=returns)


def identity_pairs(row, pred, evaluation, provenance, trace, hazard_only=True):
    """Evaluator-only projected-box correspondence, never inferred identity."""
    objects = {row['episode_id']+'/'+o['name']: o for o in evaluation['native_bounds']}
    owners = {}
    for j, box in enumerate(pred['proposals']):
        owners[j] = []
        for name, obj in objects.items():
            projected = projected_objects(row, dict(evaluation, native_bounds=[obj]))
            if projected and iou(box, projected[0]) >= .5:
                owners[j].append(name)
    pairs = []; hazard_returns = 0
    for ret in trace['returns']:
        source = provenance['radar_slots'][ret['slot']]
        name = source.get('actor_id') if source and source['kind'] == 'real_actor' else None
        if name not in objects or (hazard_only and not truth(dict(evaluation, native_bounds=[objects[name]]))):
            continue
        hazard_returns += 1
        for edge in ret['edges']:
            if owners[edge['proposal']] == [name]:
                pairs.append(dict(edge, slot=ret['slot'], actor_id=name,
                                  selected=ret['selected'] == edge['proposal']))
    return hazard_returns, pairs


def diagnose(row, pred, evaluation, provenance, trace):
    hazard_returns, pairs = identity_pairs(row, pred, evaluation, provenance, trace)
    angular = [e for e in pairs if e['angular']]
    gated = [e for e in angular if e['tof_gate']]
    selected = [e for e in gated if e['selected']]
    stage = ('NO_HAZARD_RADAR' if not hazard_returns else 'NO_IDENTITY_SUPPORTED_PROPOSAL' if not pairs
             else 'ANGULAR_GATE' if not angular else 'TOF_GATE' if not gated
             else 'RECIPROCAL_SELECTION' if not selected else 'GEOMETRY_OR_READOUT')
    return dict(stage=stage, hazard_returns=hazard_returns, pairs=pairs,
                rgb_radar_available=bool(pairs), angular_available=bool(angular),
                tof_blocked_recoverable=any(e['nominal'] and not e['tof_gate'] for e in angular),
                nominal_ceiling=any(e['nominal'] for e in pairs),
                angular_nominal_ceiling=any(e['nominal'] for e in angular),
                angular_interval_ceiling=any(e['interval'] for e in angular))


def summarize(records):
    t = np.array([r['truth'] for r in records], bool)
    result = {}
    for arm in ('baseline', 'nominal', 'interval', 'no_tof_gate'):
        p = np.array([r[arm] for r in records], bool)
        misses = [r for r in records if r['truth'] and not r[arm]]
        result[arm] = dict(metrics=metrics(t, p), original_nominal_gate_stages_on_FN=dict(Counter(r['diagnostic']['stage'] for r in misses)),
            fn_opportunities={key: sum(r['diagnostic'][key] for r in misses) for key in
                ('rgb_radar_available', 'angular_available', 'tof_blocked_recoverable',
                 'nominal_ceiling', 'angular_nominal_ceiling', 'angular_interval_ceiling')})
    n = np.array([r['nominal'] for r in records], bool)
    c = np.array([r['no_tof_gate'] for r in records], bool)
    result['gate_removal_vs_nominal'] = dict(gained_TP=int((t & ~n & c).sum()), lost_TP=int((t & n & ~c).sum()),
        removed_FP=int((~t & n & ~c).sum()), added_FP=int((~t & ~n & c).sum()))
    result['nominal_interval_positive_disagreement'] = {
        'nominal_only': [r['panel']+'/'+r['id'] for r in records if r['truth'] and r['nominal'] and not r['interval']],
        'interval_only': [r['panel']+'/'+r['id'] for r in records if r['truth'] and r['interval'] and not r['nominal']]}
    return result


def run(out):
    started = time.perf_counter()
    out = out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    work = ROOT/'artifacts.local/work'; study = work/'mz109-interval-extent-20260913/analysis-v1'
    sources = dict(consumed_mz107=work/'mz107-rgb-tof-radar-imu-20260913/capture-v1',
                   consumed_mz108=work/'mz108-competitive-association-20260913/capture-v2',
                   new_mz109=work/'mz109-interval-extent-20260913/capture-v1')
    seal = json.loads((study/'prediction-seal.json').read_text())
    for name, digest in seal['code_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest, name
    packets = {}; evidence = {}
    for panel, capture in sources.items():
        old = seal['panels'][panel]; pp = study/panel/'predictions.json'
        assert sha(pp) == old['prediction_sha256']
        assert sha(capture/'raw.jsonl') == old['raw_sha256']
        assert sha(capture/'receipt.json') == old['capture_receipt_sha256']
        receipt = json.loads((capture/'receipt.json').read_text())
        assert receipt['status'] == 'PASS' and receipt['frames'] == 96
        for name, digest in receipt['hashes'].items():
            assert sha(capture/name) == digest, name
        rows = readrows(capture/'raw.jsonl'); cached = json.loads(pp.read_text())
        assert len(rows) == 96 and len({r['id'] for r in rows}) == 96
        assert [r['id'] for r in rows] == [p['id'] for p in cached['nominal']] == [p['id'] for p in cached['interval']]
        traces = []; counterfactuals = []
        for row, pred in zip(rows, cached['nominal']):
            trace = observable_trace(row, pred)
            assert trace['candidate'] == pred['candidate'] and trace['baseline'] == pred['baseline']
            previous = {a['slot']: a for a in pred['associations']}
            assert len(previous) == len(trace['returns'])
            for ret in trace['returns']:
                a = previous[ret['slot']]
                assert ret['selected'] == (a['proposal'] if a['state'] == 'ASSOCIATED' else None)
                assert ret['support'] == a['refined_support']
            counter = observable_trace(row, pred, require_tof=False)
            assert not pred['tof_support'] or counter['candidate']
            traces.append(trace); counterfactuals.append(counter)
        target = out/panel; target.mkdir()
        write(target/'observable-traces.json', dict(traces=traces, no_tof_gate=counterfactuals))
        evidence[panel] = dict(capture=str(capture), source_seal=old,
                              observable_trace_sha256=sha(target/'observable-traces.json'))
        packets[panel] = (rows, cached, traces, counterfactuals)
    shutil.copyfile(__file__, out/Path(__file__).name)
    protocol = Path(__file__).with_name('MZ110_DIAGNOSTIC_PROTOCOL_20260913.md')
    shutil.copyfile(protocol, out/protocol.name)
    write(out/'trace-seal.json', dict(status='OBSERVABLE_COUNTERFACTUAL_SEALED_BEFORE_EVALUATOR_PARSE',
          panels=evidence, source_seal_sha256=sha(study/'prediction-seal.json'),
          code_sha256=sha(Path(__file__)), protocol_sha256=sha(protocol)))
    summaries = {}; all_records = []
    for panel, (rows, cached, traces, counterfactuals) in packets.items():
        es = readrows(sources[panel]/'evaluator.jsonl'); vs = readrows(sources[panel]/'provenance.jsonl')
        assert [r['id'] for r in rows] == [e['id'] for e in es] == [v['id'] for v in vs]
        records = []
        for row, pred, q, e, v, tr, cf in zip(rows, cached['nominal'], cached['interval'], es, vs, traces, counterfactuals):
            _, supported = identity_pairs(row, pred, e, v, cf, hazard_only=False)
            cf_audit = [dict(slot=ret['slot'], proposal=ret['selected'], support=ret['support'],
                            identity_supported=any(pair['slot'] == ret['slot'] and pair['selected'] for pair in supported),
                            source=v['radar_slots'][ret['slot']]['kind'])
                        for ret in cf['returns'] if ret['selected'] is not None]
            records.append(dict(panel=panel, id=row['id'], episode_id=row['episode_id'], family=e['family'],
                time_s=row['time_s'], truth=truth(e), baseline=pred['baseline'], nominal=pred['candidate'],
                interval=q['candidate'], no_tof_gate=cf['candidate'],
                no_tof_association_audit=cf_audit, diagnostic=diagnose(row, pred, e, v, tr)))
        write(out/panel/'evaluator-only-audit.json', records)
        summaries[panel] = dict(overall=summarize(records), families={f: summarize([r for r in records if r['family'] == f])
                               for f in sorted({r['family'] for r in records})})
        all_records.extend(records)
    summary = dict(status='CONSUMED_DIAGNOSTIC_COMPLETE_NO_PROMOTION', frames=len(all_records),
                   overall=summarize(all_records), panels=summaries, seconds=time.perf_counter()-started,
                   backend='CPU: TASK_NOT_GPU_SUITABLE; scalar trace and cached proposal audit')
    write(out/'summary.json', summary)
    write(out/'completion.json', dict(status='PASS', output_hashes={str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob('*.json'))}, truth_authority='EVALUATOR_ONLY', resources_started=[]))
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
