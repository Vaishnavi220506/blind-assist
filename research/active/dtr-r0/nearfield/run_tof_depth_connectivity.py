"""Three frozen graph arms on the previously consumed 60-packet stress panel."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from run_tof_directional_stress import score, summarize
from tof_depth_connectivity import depth_readout, known_depth, near_view
from tof_directional_readout import readout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owners(result):
    return {i: n for n, region in enumerate(result['regions']) for i in region['zone_ids']}


def bridge_opportunity(base, truth, near_m):
    footprints = truth['near_footprints']
    truth_nodes = {i for nodes in footprints.values() for i in nodes}
    entries = {e['zone_id']: e for e in base['zone_evidence']}
    observed_near = truth_nodes & entries.keys()
    known_near = {i for i in observed_near if known_depth(entries[i]) is not None and known_depth(entries[i]) <= near_m}
    known_far = {i for i, e in entries.items() if i not in truth_nodes and
                 known_depth(e) is not None and known_depth(e) > near_m and
                 not any(t['status'] == 'SIM_MERGED' for t in e['targets'])}
    owner = owners(base)
    pairs = [[i, j] for i in sorted(known_near) for j in sorted(known_far) if owner[i] == owner[j]]
    mixed = any(set(r['zone_ids']) & observed_near and set(r['zone_ids']) - truth_nodes for r in base['regions'])
    return dict(pairs=pairs, eligible=bool(pairs), mixed_angular_frame=bool(mixed),
                observed_near=sorted(observed_near), truth_near=sorted(truth_nodes),
                known_near=sorted(known_near))


def bridge_success(raw, selected, truth, opportunity):
    if not opportunity['eligible']:
        return False
    raw_owner, selected_owner = owners(raw), owners(selected)
    # No deletion, missing endpoint or UNKNOWN output can masquerade as a cut.
    endpoints_survive = all(i in selected_owner and j in raw_owner for i, j in opportunity['pairs'])
    all_observed_near_survive = set(opportunity['observed_near']) <= selected_owner.keys()
    pairs_cut = all(i in raw_owner and j in raw_owner and raw_owner[i] != raw_owner[j]
                    for i, j in opportunity['pairs'])
    bilateral_not_merged = not score(raw, truth)['bilateral_merge']
    return endpoints_survive and all_observed_near_survive and pairs_cut and bilateral_not_merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    here = Path(__file__).parent
    protocol_path = here/'tof_depth_connectivity_protocol_20260914.json'
    protocol = json.loads(protocol_path.read_text())
    source = Path(protocol['source'])
    for name, digest in protocol['source_sha256'].items():
        assert sha(source/name) == digest, name
    packets = [json.loads(s) for s in (source/'packets.jsonl').read_text().splitlines()]
    assert len(packets) == 60 and len({r['id'] for r in packets}) == 60
    args.output.mkdir(parents=True, exist_ok=False)
    # Complete predictions before opening evaluator footprints or old outcomes.
    start = time.perf_counter()
    predictions = []
    for row in packets:
        arm_outputs = {}
        for arm, params in protocol['arms'].items():
            raw = readout(row, equal_weights=True) if params is None else depth_readout(row, **params)
            arm_outputs[arm] = dict(raw=raw, near=near_view(raw, protocol['near_view_m']))
        predictions.append(dict(id=row['id'], arms=arm_outputs))
    predict_seconds = time.perf_counter()-start
    prediction_path = args.output/'predictions.json'
    prediction_path.write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    truths = {r['id']: r for r in (json.loads(s) for s in (source/'evaluator.jsonl').read_text().splitlines())}
    old = {r['id']: r for r in json.loads((source/'predictions.json').read_text())}
    assert set(truths) == set(old) == {r['id'] for r in predictions}
    details = []
    for p in predictions:
        truth = truths[p['id']]['evaluator']
        baseline = p['arms']['equal_angular']['raw']
        assert baseline == old[p['id']]['arms']['equal']['output'], p['id']
        opportunity = bridge_opportunity(baseline, truth, protocol['near_view_m'])
        observed = {e['zone_id'] for e in baseline['zone_evidence']}
        valid_near = {e['zone_id'] for e in baseline['zone_evidence']
                      if known_depth(e) is not None and known_depth(e) <= protocol['near_view_m']}
        arms = {}
        for arm, output in p['arms'].items():
            raw, near = output['raw'], output['near']
            assert set(owners(raw)) == observed, 'Raw zone loss'
            assert raw['zone_evidence'] == baseline['zone_evidence'], 'Target tuple loss'
            arms[arm] = dict(raw=score(raw, truth), near=score(near, truth),
                bridge_cut=int(bridge_success(raw, near, truth, opportunity)),
                observed_near_kept=len(set(opportunity['observed_near']) & owners(near).keys()),
                valid_near_lost=len(valid_near - owners(near).keys()),
                known_near=sorted(valid_near),
                # Exact directions can be correct even while depth remains unresolved.
                unresolved_selected_zones=sum(known_depth(e) is None for e in near['zone_evidence'] if e['zone_id'] in owners(near)),
                direction_correct_with_unknown_depth=int(score(near, truth)['direction_correct'] and any(
                    known_depth(e) is None for e in near['zone_evidence'] if e['zone_id'] in owners(near))))
        assert p['arms']['absolute_depth']['raw']['graph']['accepted_edges'] == p['arms']['relative_depth']['raw']['graph']['accepted_edges']
        details.append(dict(id=p['id'], family=truths[p['id']]['family'], opportunity=opportunity, arms=arms))
    metrics = {}
    for view in ('raw', 'near'):
        metrics[view] = {a: summarize([d['arms'][a][view] for d in details]) for a in protocol['arms']}
    families = {f: {a: summarize([d['arms'][a]['near'] for d in details if d['family'] == f])
                for a in protocol['arms']} for f in sorted({d['family'] for d in details})}
    audit = {a: {k: sum(d['arms'][a][k] for d in details) for k in
             ('bridge_cut','observed_near_kept','valid_near_lost','unresolved_selected_zones','direction_correct_with_unknown_depth')}
             for a in protocol['arms']}
    for a in protocol['arms']:
        sequence = [d['arms'][a]['near']['actual'] for d in details if d['family'] == 'one_side_dropout']
        audit[a]['dropout_set_changes'] = sum(x != y for x, y in zip(sequence, sequence[1:]))
    b = metrics['near']['equal_angular']
    retained = []
    for a in ('absolute_depth', 'relative_depth'):
        m = metrics['near'][a]
        if (m['direction_correct'] > b['direction_correct'] and m['false_center'] < b['false_center'] and
            m['bilateral_merge'] <= b['bilateral_merge'] and audit[a]['bridge_cut'] > 0 and audit[a]['valid_near_lost'] == 0):
            retained.append(a)
    detail_path = args.output/'evaluation.json'
    detail_path.write_text(json.dumps(details, indent=2), encoding='utf-8')
    files = [protocol_path, Path(__file__), here/'tof_depth_connectivity.py',
             here/'tof_directional_readout.py', here/'run_tof_directional_stress.py',
             *[source/n for n in protocol['source_sha256']], prediction_path, detail_path]
    summary = dict(protocol=protocol['id'], authority='CONSUMED_HAND_CONSTRUCTED_PACKETS_NOT_SENSOR_VALIDATION',
        metrics=metrics, families=families, audit=audit,
        bridge_opportunities=sum(d['opportunity']['eligible'] for d in details),
        bridge_pairs=sum(len(d['opportunity']['pairs']) for d in details),
        mixed_angular_frames=sum(d['opportunity']['mixed_angular_frame'] for d in details),
        mixed_without_known_depth_opportunity=sum(d['opportunity']['mixed_angular_frame'] and not d['opportunity']['eligible'] for d in details),
        observed_anchor_zones=sum(len(d['opportunity']['observed_near']) for d in details),
        truth_anchor_zones=sum(len(d['opportunity']['truth_near']) for d in details),
        gates_identical_on_this_panel=True, retained_components=retained,
        decision='RETAIN_DEPTH_CONNECTIVITY_COMPONENT_ONLY' if retained else 'STOP_FIXED_DEPTH_CONNECTIVITY',
        backend='CPU: TASK_NOT_GPU_SUITABLE', three_arm_predict_seconds=predict_seconds,
        hashes={str(p): sha(p) for p in files})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
