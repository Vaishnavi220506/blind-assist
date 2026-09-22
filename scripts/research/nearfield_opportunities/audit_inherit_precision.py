"""Read-only result audit; never invokes MILP or changes predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import active_view as av
import inherit_precision as candidate


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def audit(root, output):
    for seal in ('pre-inference-seal.json','prediction-seal.json','completion-seal.json'):
        av.verify_seal(root/seal)
    for path, expected in read(root/'input-hashes.json').items():
        assert av.file_hash(Path(path)) == expected
    summary = read(root/'summary.json')
    rows = read(root/'evaluation.json')
    queries = {q['query_id']:q['observations'] for q in read(root/'public-queries.json')}
    predictions = {r['query_id']:r['result'] for r in read(root/'predictions.json')}
    assert len(rows)==540 and len(queries)==len(predictions)==439
    model = candidate.model_for(.001)
    witnesses, receipts = 0, 0
    for qid,result in predictions.items():
        assert result['observations_used']==13 and result['solver_calls']==2
        assert result['query_frame']==candidate.original.QUERY_FRAME
        assert result['nuisance_bounds']==dict(range_m=[-.002,.002],pose_x_m=[-.001,.001],pose_z_m=[-.001,.001])
        for label,witness in result['witnesses'].items():
            if witness is not None:
                assert model.validate_witness(witness,queries[qid],label)
                witnesses+=1
        rmap = {r['requested_label']:r for r in result['solver_metadata']}
        receipts+=len(rmap)
        if result['decision']!='UNKNOWN':
            label=result['decision'].split('_')[0]
            opposite='OUT' if label=='IN' else 'IN'
            assert result['witnesses'][label] is not None and rmap[label]['solver_status']==0
            assert result['witnesses'][opposite] is None and rmap[opposite]['solver_status']==2
            assert rmap[opposite]['exclusion_supported']
    checked=0
    for cond,strata in summary['metrics'].items():
        for stratum, arms in strata.items():
            group=[r for r in rows if r['condition']==cond and (stratum=='all' or r['stratum']==stratum or
                stratum=='boundary' and r['stratum'].startswith('boundary_'))]
            for arm, metrics in arms.items():
                c=Counter((r['truth'],r[arm]) for r in group)
                tp=c[True,'IN_MODEL_CONDITIONAL']; fp=c[False,'IN_MODEL_CONDITIONAL']
                out=c[False,'OUT_MODEL_CONDITIONAL']; false=c[True,'OUT_MODEL_CONDITIONAL']
                positives=sum(r['truth'] for r in group); negatives=len(group)-positives
                expected=dict(n=len(group),positives=positives,negatives=negatives,tp=tp,fp=fp,fn=positives-tp,
                    false_out=false,correct_out=out,unknown=c[True,'UNKNOWN']+c[False,'UNKNOWN'],correct=tp+out,
                    recall=tp/positives if positives else None,precision=tp/(tp+fp) if tp+fp else None,
                    fpr=fp/negatives if negatives else None)
                assert metrics==expected,(cond,stratum,arm)
                checked+=1
            valid=lambda r,k: r[k]==('IN_MODEL_CONDITIONAL' if r['truth'] else 'OUT_MODEL_CONDITIONAL')
            expected=dict(gained_correct=[r['id'] for r in group if valid(r,'fine') and not valid(r,'coarse')],
                lost_correct=[r['id'] for r in group if valid(r,'coarse') and not valid(r,'fine')],
                fine_wrong=[r['id'] for r in group if r['fine']!='UNKNOWN' and not valid(r,'fine')])
            assert expected==summary['changes'][cond][stratum]
    containment=read(root/'containment-audit.json')
    assert len(containment)==540 and all(r['forward_valid'] and not r['true_class_excluded']
        and r['max_positive_residual']==0 for r in containment)
    assert all(r['fine']==predictions[r['query_id']]['decision'] for r in rows)
    stages=read(root/'stage-order.json')
    assert [r['name'] for r in stages]==['public_queries_and_code_sealed',
        'all_predictions_sealed_before_truth_and_old_decisions','evaluation_and_true_assignment_containment_complete']
    assert candidate.original.shared.base.RANGE_STEP==.1
    av.write_json(output,dict(status='PASS',rows=540,unique_queries=len(queries),
        validated_witnesses=witnesses,solver_receipts=receipts,metric_cells=checked,
        matching_gain_loss_sets=18,true_assignment_records=540,new_solver_calls=0,new_observations=0,
        scope='Saved output, forward witness validation, counts, identities, seals; not formal exclusion proof'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.root,args.output)
