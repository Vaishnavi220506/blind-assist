"""Posthoc observable matched readout control, not a retuned primary candidate.

Only identical current slot/proposal pairs may reuse the incumbent's already
filtered range. No new filter, threshold, assignment, fallback or image change.
Control predictions are sealed before evaluator parsing; sources are consumed.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from mz111_spatial_evidence import plane_inside
from run_mz107_four_sensor import ROOT, readrows, sha, truth, metrics, write


def control_frame(row, nominal, joint_spatial, incumbent_spatial, flow):
    """Observable-only cached-pair readout; never modifies either parent arm."""
    result = copy.deepcopy(joint_spatial)
    incumbent = {r['slot']: r for r in incumbent_spatial['spatial_evidence']}
    support = bool(result['tof_support']); reused = []
    for ret in result['spatial_evidence']:
        old = incumbent.get(ret['slot'])
        if ret['proposal'] is not None and old and old['proposal'] == ret['proposal'] and old.get('range_filtered', False):
            original_range = ret['range_m']; original_support = bool(ret['support'])
            ret['range_m'] = old['range_m']
            ret['support'] = bool(plane_inside(nominal['proposals'][ret['proposal']], old['range_m'], row, nominal['integrated_yaw_deg']))
            reused.append(dict(slot=ret['slot'], proposal=ret['proposal'], original_joint_range_m=original_range,
                reused_incumbent_range_m=old['range_m'], original_support=original_support, control_support=ret['support']))
        support |= bool(ret['support'])
    if result['diagnostics'].get('baseline_fallback', False):
        support |= bool(result['baseline'])
    result.update(candidate=bool(support), candidate_state='ALERT' if support else 'UNKNOWN')
    return dict(id=row['id'], candidate=bool(support or (flow['candidate'] and not nominal['candidate'])),
        spatial=result, reused_filtered_ranges=reused,
        original_flow_addition=bool(flow['candidate'] and not nominal['candidate']))


def run(analysis, output):
    analysis = analysis.resolve(); output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    source_seal_path = analysis/'prediction-seal.json'; source_seal = json.loads(source_seal_path.read_text())
    assert source_seal['status'] == 'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE'
    for name, digest in source_seal['code_sha256'].items(): assert sha(analysis/name) == digest, name
    assert sha(Path(__file__).with_name('mz111_spatial_evidence.py')) == source_seal['code_sha256']['mz111_spatial_evidence.py']
    packets = {}; seen = set()
    for panel, item in source_seal['panels'].items():
        source = Path(item['capture']); prediction_path = analysis/panel/'predictions.json'
        assert sha(source/'raw.jsonl') == item['raw_sha256'] and sha(source/'receipt.json') == item['receipt_sha256']
        assert sha(prediction_path) == item['predictions_sha256']
        assert item['raw_sha256'] not in seen; seen.add(item['raw_sha256'])
        receipt = json.loads((source/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
        for name, digest in receipt['hashes'].items(): assert sha(source/name) == digest, name
        rows = readrows(source/'raw.jsonl'); values = json.loads(prediction_path.read_text())
        assert len(rows) == receipt['frames'] and all(len(v) == len(rows) for v in values.values())
        assert all('id' not in n or n['id'] == r['id'] for r, n in zip(rows, values['nominal']))
        packets[panel] = (source, rows, values)
    output.mkdir(parents=True); shutil.copyfile(Path(__file__), output/Path(__file__).name)
    seals = {}; controls = {}; started = time.perf_counter()
    for panel, (_, rows, values) in packets.items():
        computed = {}
        for joint in ('joint_current', 'joint_temporal'):
            computed[joint+'_matched_readout'] = [control_frame(row, nominal, spatial, incumbent, flow)
                for row, nominal, spatial, incumbent, flow in zip(rows, values['nominal'], values[joint+'_spatial'], values['filtered_plane'], values['flow'])]
            assert all(not n['tof_support'] or c['candidate'] for n, c in zip(values['nominal'], computed[joint+'_matched_readout']))
        directory = output/panel; directory.mkdir(); write(directory/'predictions.json', computed)
        controls[panel] = computed; seals[panel] = dict(source=source_seal['panels'][panel], control_prediction_sha256=sha(directory/'predictions.json'))
    elapsed = time.perf_counter()-started
    write(output/'prediction-seal.json', dict(status='ALL_MATCHED_READOUT_CONTROLS_SEALED_BEFORE_EVALUATOR_PARSE',
        source_seal_sha256=sha(source_seal_path), panels=seals, control_sha256=sha(Path(__file__)),
        geometry_sha256=sha(Path(__file__).with_name('mz111_spatial_evidence.py')),
        scope='POSTHOC_CONSUMED_OBSERVABLE_READOUT_CONTROL'))
    overall = {}; panel_results = {}; all_comparisons = {}; all_truth = []; all_controls = {}; all_references = {}
    for panel, (source, rows, values) in packets.items():
        es = readrows(source/'evaluator.jsonl'); assert [r['id'] for r in rows] == [e['id'] for e in es]
        gt = np.asarray([truth(e) for e in es], bool); all_truth.extend(gt.tolist()); panel_results[panel] = {}
        for arm, predictions in controls[panel].items():
            p = np.asarray([c['candidate'] for c in predictions], bool); all_controls.setdefault(arm, []).extend(p.tolist())
            summary = dict(metrics=metrics(gt,p), reused_pair_ranges=sum(len(c['reused_filtered_ranges']) for c in predictions),
                changed_pair_supports=sum(r['original_support'] != r['control_support'] for c in predictions for r in c['reused_filtered_ranges']),comparisons={})
            primary = arm.removesuffix('_matched_readout')
            for reference in ('combined_flow',primary):
                q = np.asarray([c['candidate'] for c in values[reference]], bool); key=arm+'_vs_'+reference
                all_references.setdefault(key, []).extend(q.tolist())
                changes=[]
                for i,(row, candidate, original) in enumerate(zip(rows,p,q)):
                    if candidate == original:continue
                    change = ('GAINED_TP' if gt[i] else 'ADDED_FP') if candidate else ('LOST_TP' if gt[i] else 'REMOVED_FP')
                    changes.append(dict(panel=panel,id=row['id'],index=i,family=es[i]['family'],truth=bool(gt[i]),
                        candidate=bool(candidate),reference=bool(original),change=change,reused_filtered_ranges=predictions[i]['reused_filtered_ranges']))
                write(output/panel/(key+'-changes.json'), changes); all_comparisons.setdefault(key, []).extend(changes)
                summary['comparisons'][reference] = dict(gained_TP=int((gt&p&~q).sum()),lost_TP=int((gt&~p&q).sum()),added_FP=int((~gt&p&~q).sum()),removed_FP=int((~gt&~p&q).sum()))
            panel_results[panel][arm]=summary
    gt=np.asarray(all_truth,bool)
    for arm,predictions in all_controls.items():
        p=np.asarray(predictions,bool); primary=arm.removesuffix('_matched_readout'); comparisons={}
        for reference in ('combined_flow',primary):
            key=arm+'_vs_'+reference;q=np.asarray(all_references[key],bool)
            comparisons[reference]=dict(gained_TP=int((gt&p&~q).sum()),lost_TP=int((gt&~p&q).sum()),added_FP=int((~gt&p&~q).sum()),removed_FP=int((~gt&~p&q).sum()),
                exact_changes=all_comparisons[key])
        overall[arm]=dict(metrics=metrics(gt,p),comparisons=comparisons)
    result=dict(status='POSTHOC_MATCHED_READOUT_CONTROL_COMPLETE',scope='CONSUMED_DIAGNOSTIC_NOT_PRIMARY_RETUNING',frames=len(gt),panels=panel_results,overall=overall,
        observable_control_seconds=elapsed,backend='CPU_SMALL_SCALAR_CACHED_GEOMETRY',limits=[
            'Readout control chosen after observing a lost-TP range-smoothing confound; it is posthoc Development evidence.',
            'Only already filtered distances on identical slot/proposal pairs are reused; unmatched/new pairs retain original joint readout.',
            'No parameter choice, association retuning, new capture or candidate promotion.',
            'This control does not retroactively change the frozen primary results or their ghost/identity errors.'])
    write(output/'summary.json',result)
    write(output/'completion.json',dict(status='PASS',output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(overall,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.analysis,args.output)
