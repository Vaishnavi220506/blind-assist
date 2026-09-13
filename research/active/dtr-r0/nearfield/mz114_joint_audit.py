"""Evaluator-only current-pair and temporal-lineage audit of sealed MZ114.

No prediction writes or source-policy changes. AABB correspondence is conditional
identity support; an assignment cost or unanimous solver choice is not truth.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from run_mz107_four_sensor import ROOT, readrows, sha, truth, write
from mz111_support_audit import owners, audit_spatial
from mz113_support_audit import audit_transport, supporting_evidence

COMPARISONS = {
    'joint_current_vs_incumbent': ('joint_current', 'combined_flow'),
    'joint_temporal_vs_incumbent': ('joint_temporal', 'combined_flow'),
    'joint_temporal_vs_joint_current': ('joint_temporal', 'joint_current'),
}


def current_pair(index, ret, rows, values, es, prov, object_cache):
    result = audit_spatial(index, ret, rows, values['nominal'], es, prov, object_cache)
    if ret['proposal'] is None:
        result['status'] = 'UNRESOLVED_CURRENT_NO_RGB_PAIR'
    return result


def audit_prior(index, prior, rows, values, es, prov, object_cache):
    trace = dict(prior, range_m=prior['predicted_range_m'])
    audit = audit_transport(index, trace, rows, es, prov, object_cache)
    failures = list(audit.get('failures', [])); source_index = prior['measurement_index']; slot = prior['measurement_slot']
    lineage = []
    if isinstance(source_index, int) and 0 <= source_index < index:
        source_returns = values['joint_temporal_spatial'][source_index]['spatial_evidence']
        sources = [ret for ret in source_returns if ret['slot'] == slot and ret['proposal'] is not None]
        if len(sources) != 1:
            failures.append('source_was_not_unique_resolved_measurement')
        else:
            audit['seed_current_pair'] = current_pair(source_index, sources[0], rows, values, es, prov, object_cache)
        if not math.isclose(prior['range_time_s'], rows[source_index]['time_s'], abs_tol=1e-8):
            failures.append('original_measurement_time_changed')
        if audit.get('source_measurement') and not math.isclose(prior['measured_range_m'], audit['source_measurement']['range_m'], abs_tol=1e-8):
            failures.append('original_measured_range_changed')
        for step in range(source_index+1, index+1):
            spatial = values['joint_temporal_spatial'][step]; priors = spatial['diagnostics']['prior_matches']
            matches = [p for p in priors if p['measurement_index'] == source_index and p['measurement_slot'] == slot]
            if len(matches) != 1:
                failures.append('missing_or_duplicate_visual_lineage_step_'+str(step)); continue
            matched = matches[0]
            if sum(p['proposal'] == matched['proposal'] for p in priors) != 1 or sum(p['track'] == matched['track'] for p in priors) != 1:
                failures.append('nonunique_current_flow_lineage_step_'+str(step))
            normalized = dict(matched, range_m=matched['predicted_range_m'])
            step_audit = audit_transport(step, normalized, rows, es, prov, object_cache)
            if step_audit.get('failures'):
                failures.append('invalid_transport_or_flow_step_'+str(step))
            if step < index and any(ret['proposal'] == matched['proposal'] for ret in spatial['spatial_evidence']):
                failures.append('old_measurement_survived_intervening_reseed_'+str(step))
            lineage.append(dict(index=step, id=rows[step]['id'], proposal=matched['proposal'], track=matched['track'],
                age_s=matched['age_s'], identity_status=step_audit['status'], visual=matched['visual']))
    audit['lineage'] = lineage; audit['failures'] = failures
    audit['lineage_contract_pass'] = not failures
    audit['lineage_identity_supported'] = (audit.get('seed_current_pair', {}).get('status') == 'CORRECT_SUPPORTED_PROXY'
        and all(step['identity_status'] == 'CORRECT_SUPPORTED_PROXY' for step in lineage))
    if failures:
        audit['status'] = 'MOTION_AGE_OR_LINEAGE_FAILURE'
    return audit


def spatial_arm(arm):
    return 'filtered_plane' if arm == 'combined_flow' else arm+'_spatial'


def joint_support(arm, index, rows, values, es, prov, object_cache):
    if arm == 'combined_flow':
        return supporting_evidence(arm, index, rows, values, es, prov, object_cache)
    spatial = values[spatial_arm(arm)][index]; n = values['nominal'][index]; result = []
    if spatial['tof_support']:
        result.append(dict(status='INDEPENDENT_TOF', component=arm))
    for ret in spatial['spatial_evidence']:
        if ret['support']:
            result.append(dict(current_pair(index, ret, rows, values, es, prov, object_cache), component=arm))
    if not n['candidate'] and values['flow'][index]['candidate']:
        result.extend(supporting_evidence('flow', index, rows, values, es, prov, object_cache))
    expected = bool(spatial['candidate'] or (values['flow'][index]['candidate'] and not n['candidate']))
    assert expected == bool(values[arm][index]['candidate'])
    return result


def summarize(records):
    changes = Counter(r['change'] for r in records)
    result = dict(changed_frames=len(records), changes=dict(changes), by_change={},
        new_resolved_pair_status=dict(Counter(pair['new_pair']['status'] for r in records for pair in r['changed_returns'] if pair['new_proposal'] is not None)),
        prior_status=dict(Counter(p['status'] for r in records for p in r['prior_audits'])),
        prior_lineage_failures=sum(not p['lineage_contract_pass'] for r in records for p in r['prior_audits']),
        prior_lineage_identity_unsupported=sum(not p['lineage_identity_supported'] for r in records for p in r['prior_audits']),
        no_attributed_alerting_evidence=sum(not r['alerting_evidence'] for r in records))
    for change in sorted(changes):
        subset = [r for r in records if r['change'] == change]
        result['by_change'][change] = dict(frames=len(subset),
            alerting_support_status_nonexclusive=dict(Counter(s for r in subset for s in {e['status'] for e in r['alerting_evidence']})),
            changed_resolved_pair_wrong_or_unsupported_frames=sum(any(p['new_proposal'] is not None and p['new_pair']['status'] != 'CORRECT_SUPPORTED_PROXY' for p in r['changed_returns']) for r in subset),
            support_changed_by_wrong_or_unsupported_pair_frames=sum(any(p['new_proposal'] is not None and p['old_support'] != p['new_support'] and p['new_pair']['status'] != 'CORRECT_SUPPORTED_PROXY' for p in r['changed_returns']) for r in subset))
    return result


def audit_panel(panel, rows, values, es, prov):
    cache = [owners(r, n, e) for r, n, e in zip(rows, values['nominal'], es)]
    output = {}
    for comparison, (candidate_arm, reference_arm) in COMPARISONS.items():
        records = []
        for index, row in enumerate(rows):
            candidate = bool(values[candidate_arm][index]['candidate']); reference = bool(values[reference_arm][index]['candidate'])
            if candidate == reference:
                continue
            gt = truth(es[index]); alerting = candidate_arm if candidate else reference_arm
            old = {r['slot']: r for r in values[spatial_arm(reference_arm)][index]['spatial_evidence']}
            new = values[spatial_arm(candidate_arm)][index]['spatial_evidence']; changed = []; prior_audits = {}
            for ret in new:
                previous = old.get(ret['slot']); old_proposal = previous.get('proposal') if previous else None
                old_support = bool(previous and previous['support'])
                if ret['proposal'] == old_proposal and bool(ret['support']) == old_support:
                    continue
                new_pair = current_pair(index, ret, rows, values, es, prov, cache)
                pair = dict(slot=ret['slot'], old_proposal=old_proposal, new_proposal=ret['proposal'], old_support=old_support,
                    new_support=bool(ret['support']), new_pair=new_pair,
                    old_pair=current_pair(index, previous, rows, values, es, prov, cache) if previous else None,
                    candidate_pair_costs=ret.get('pair_costs', []))
                changed.append(pair)
            # Every current prior can affect joint exclusion/selection, including
            # costs attached to unchosen competing pairs; audit each once.
            if candidate_arm == 'joint_temporal':
                for prior in values['joint_temporal_spatial'][index]['diagnostics']['prior_matches']:
                    p = audit_prior(index, prior, rows, values, es, prov, cache)
                    prior_audits[prior['proposal']] = p
                for pair in changed:
                    source = pair['new_pair'].get('source') or {}
                    pair['prior_pair_actor_relations'] = [dict(proposal=p['proposal'],
                        has_prior=p['prior'] is not None,
                        current_radar_same_actor_as_prior=bool(p['prior'] is not None and source.get('actor_id') is not None and
                            source['actor_id'] == (prior_audits[p['proposal']].get('source') or {}).get('actor_id')))
                        for p in pair['candidate_pair_costs']]
            change = ('GAINED_TP' if gt else 'ADDED_FP') if candidate else ('LOST_TP' if gt else 'REMOVED_FP')
            records.append(dict(panel=panel, frame_index=index, frame_id=row['id'], episode_id=row['episode_id'], family=es[index]['family'],
                truth=gt, candidate_arm=candidate_arm, reference_arm=reference_arm, candidate=candidate, reference=reference, change=change,
                alerting_arm=alerting, alerting_evidence=joint_support(alerting, index, rows, values, es, prov, cache),
                changed_returns=changed, prior_audits=list(prior_audits.values())))
        output[comparison] = records
    return output


def run(analysis, output):
    analysis = analysis.resolve(); output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    seal_path = analysis/'prediction-seal.json'; seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE'
    for name, digest in seal['code_sha256'].items():
        assert sha(analysis/name) == digest, name
    packets = {}; seen = set()
    for panel, item in seal['panels'].items():
        source = Path(item['capture']); path = analysis/panel/'predictions.json'
        assert sha(source/'receipt.json') == item['receipt_sha256'] and sha(source/'raw.jsonl') == item['raw_sha256']
        assert sha(path) == item['predictions_sha256']
        assert item['raw_sha256'] not in seen; seen.add(item['raw_sha256'])
        receipt = json.loads((source/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
        for name, digest in receipt['hashes'].items(): assert sha(source/name) == digest, name
        rows = readrows(source/'raw.jsonl'); values = json.loads(path.read_text())
        assert len(rows) == receipt['frames'] and len({r['id'] for r in rows}) == len(rows)
        assert all(len(v) == len(rows) for v in values.values())
        assert all('id' not in n or n['id'] == r['id'] for r, n in zip(rows, values['nominal']))
        packets[panel] = (source, rows, values)
    output.mkdir(parents=True); shutil.copyfile(Path(__file__), output/Path(__file__).name)
    panels = {}; merged = {}
    for panel, (source, rows, values) in packets.items():
        es = readrows(source/'evaluator.jsonl'); prov = readrows(source/'provenance.jsonl')
        assert [r['id'] for r in rows] == [e['id'] for e in es] == [p['id'] for p in prov]
        records = audit_panel(panel, rows, values, es, prov); target = output/panel; target.mkdir(); panels[panel] = {}
        for name, items in records.items():
            write(target/(name+'.json'), items); panels[panel][name] = summarize(items); merged.setdefault(name, []).extend(items)
    summary = dict(status='EVALUATOR_ONLY_JOINT_AUDIT_COMPLETE', unique_source_frames=sum(len(r) for _, r, _ in packets.values()),
        panels=panels, overall={name: summarize(records) for name, records in merged.items()},
        limits=['Projected-box identity support is conditional, not exact pixel visibility or physical Radar identity.',
            'A prior can legitimately disagree with a current Radar actor when penalizing a wrong candidate pair.',
            'Accepted current-prior diagnostics prove recorded unique lineage, not a global uniqueness theorem about all image motions.',
            'Changed-return status counts can overlap within a changed frame. Wrong-match FP removal is not correct ghost discrimination.',
            'Temporal priors change assignment costs; they do not substitute native truth or prior range for the current measured geometric readout.'])
    write(output/'summary.json', summary)
    write(output/'completion.json', dict(status='PASS', analysis_seal_sha256=sha(seal_path),
        audit_dependency_hashes={name: sha(Path(__file__).with_name(name)) for name in ('mz111_support_audit.py', 'mz113_support_audit.py')},
        output_hashes={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.analysis, args.output)
