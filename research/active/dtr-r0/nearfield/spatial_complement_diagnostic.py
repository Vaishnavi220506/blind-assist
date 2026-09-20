"""One saved-dev complementarity diagnostic. No model imports or test labels."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from tof_corridor_calibration import score_frame, decide

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
OUT = ROOT/'artifacts.local/work/ba-spatial-complement-diagnostic-20260921'
T = .4071309640537889
T0 = .007085703945147101
HIGH = 7.6612162590026855
NAMES = ('A', 'OR_zero', 'OR_high', 'OR_high_calibration', 'OR_group_score', 'OR_group_calibration')
FILES = ('protocol.json','observation-seal.json','fit-seal.json','development-seal.json',
         'operating-point.json','diagnostic.json','baseline.json','observations.npz',
         'dev-logits.npy','evaluator/metadata.json','evaluator/dev-labels.json',
         'source-admission.json','fit/head_last.pt','independent-audit.json')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def hold(raw, meta):
    result = np.asarray(raw, bool).copy()
    for i in range(1,len(raw)):
        if meta[i]['clip_id'] == meta[i-1]['clip_id']:
            assert abs(meta[i]['time_s']-meta[i-1]['time_s']-.2) < 1e-8
            result[i] |= raw[i-1]
    return result


def zero_fp_cutoff(scores):
    scores = np.asarray(scores, np.float64)
    assert np.isfinite(scores).all()
    return float(max(0., np.nextafter(scores.max(), np.inf))) if len(scores) else 0.


def grouped_screen(logits, a, truth, eligible, groups):
    logits = np.asarray(logits, np.float64)
    a, truth, eligible = (np.asarray(x, bool) for x in (a,truth,eligible))
    groups = np.asarray(groups)
    result = np.zeros(len(a), bool)
    cutoffs = np.zeros(len(a), np.float64)
    records = []
    for group in sorted(set(groups)):
        held_group = groups == group
        calibrating_negatives = ~held_group & ~a & ~truth & eligible
        threshold = zero_fp_cutoff(logits[calibrating_negatives])
        proposals = eligible & (logits >= threshold)
        assert not ((~held_group) & ~a & ~truth & proposals).any()
        result[held_group] = proposals[held_group]
        cutoffs[held_group] = threshold
        records.append(dict(held_group=str(group), calibration_groups=sorted(set(groups[~held_group].tolist())),
            cutoff=threshold, eligible_A_negative_calibration_negatives=int(calibrating_negatives.sum()),
            max_calibration_negative_logit=float(logits[calibrating_negatives].max()) if calibrating_negatives.any() else None,
            calibration_extra_raw_FP=0, held_indices=np.flatnonzero(held_group).tolist()))
    return result, cutoffs, records


def metric_module():
    spec = importlib.util.spec_from_file_location('complement_metrics',HERE/'full_event_metrics_20260920.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ARMS = tuple(n+'_'+s for n in NAMES for s in ('current','hold'))
    return module


def summarize(rows, report):
    result = {}
    for name in NAMES[1:]:
        arms, checks = {}, {}
        for stratum in ('Core','Boundary'):
            for suffix in ('current','hold'):
                base = report[stratum]['arms']['A_'+suffix]
                candidate = report[stratum]['arms'][name+'_'+suffix]
                key = stratum+'_'+suffix
                arms[key] = dict(A=[base['frames'][k] for k in ('TP','FP','FN')],
                    candidate=[candidate['frames'][k] for k in ('TP','FP','FN')],
                    extra_TP=candidate['frames']['TP']-base['frames']['TP'],
                    extra_FP=candidate['frames']['FP']-base['frames']['FP'],
                    A_segments=base['false_alert_segment_count'], candidate_segments=candidate['false_alert_segment_count'],
                    A_detected_events=base['detected_events'], candidate_detected_events=candidate['detected_events'])
                checks[key+'_no_extra_FP'] = arms[key]['extra_FP'] == 0
                checks[key+'_no_extra_segments'] = arms[key]['candidate_segments'] <= arms[key]['A_segments']
        checks['boundary_current_gain_10pp'] = arms['Boundary_current']['extra_TP'] >= 9
        checks['all_A_flags_retained'] = all(not r['flags']['A_'+s] or r['flags'][name+'_'+s]
                                           for r in rows for s in ('current','hold'))
        assert checks['all_A_flags_retained']
        raw_pass = checks['boundary_current_gain_10pp'] and all(checks[g+'_current_no_extra_FP'] for g in ('Core','Boundary'))
        complete_pass = raw_pass and all(checks[g+'_hold_no_extra_FP'] and checks[g+'_hold_no_extra_segments']
                                       and checks[g+'_current_no_extra_segments'] for g in ('Core','Boundary'))
        result[name] = dict(metrics=arms, checks=checks, current_complement=raw_pass, complete_complement=complete_pass)
    return result


def run(out):
    assert not out.exists(), 'One diagnostic output; no overwrite or revised screen'
    assert out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    assert read(SOURCE/'operating-point.json')['status'] == 'DEV_NO_ADMISSIBLE_OPERATING_POINT'
    for name in ('test-start.json','test-logits.npy','test-prediction-seal.json','test-metrics.json'):
        assert not (SOURCE/name).exists(), 'Retained test must remain unactivated'
    out.mkdir(parents=True)
    protocol = HERE/'SPATIAL_COMPLEMENT_PROTOCOL_20260921.md'
    (out/'protocol-before-run.md').write_bytes(protocol.read_bytes())
    source_hashes = {n:sha(SOURCE/n) for n in FILES}
    write(out/'protocol.json',dict(id='ba-spatial-complement-diagnostic-20260921',
        time_utc=datetime.now(timezone.utc).isoformat(), source=str(SOURCE),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_hashes=source_hashes, code_hashes={n:sha(HERE/n) for n in
            ('spatial_complement_diagnostic.py','tof_corridor_calibration.py','ba_camera_corridor.py','full_event_metrics_20260920.py')},
        protocol_sha256=sha(protocol), fixed_high_logit=HIGH, geometric_T=T, calibration_T0=T0,
        backend='TASK_NOT_GPU_SUITABLE', scope='CONSUMED_DEV_DIAGNOSTIC_NO_FRESH_CONFIRMATION',
        conditions=list(NAMES), splits=['dev'], new_neural_fits=0,new_image_inferences=0,new_captures=0))
    meta = [r for r in read(SOURCE/'evaluator/metadata.json') if r['split']=='dev']
    assert len(meta)==576 and len({r['base_group_id'] for r in meta})==8
    idx = np.array([r['index'] for r in meta],np.int64)
    logits = np.load(SOURCE/'dev-logits.npy').astype(np.float64)
    assert logits.shape==(576,) and np.isfinite(logits).all()
    base = read(SOURCE/'baseline.json')
    a = np.array([base[i]['current'] for i in idx],bool)
    unknown = np.array([base[i]['unknown'] for i in idx],bool)
    eligible = []
    with np.load(SOURCE/'observations.npz',allow_pickle=False) as obs:
        for i in idx:
            scored=score_frame(obs['boxes'],obs['ranges'][i])
            strong=decide(scored,T)
            assert strong['alert']==base[i]['current'] and strong['unknown']==base[i]['unknown']
            eligible.append(decide(scored,T0)['alert'])
    eligible=np.array(eligible,bool)
    fixed = {'A':a,'OR_zero':a|(logits>=0),'OR_high':a|(logits>=HIGH),
             'OR_high_calibration':a|(eligible&(logits>=HIGH))}
    public=[dict(id=r['id'],index=r['index'],logit=float(logits[i]),A_current=bool(a[i]),
        calibration_eligible=bool(eligible[i]),unknown=bool(unknown[i]),
        fixed_current_flags={n:bool(v[i]) for n,v in fixed.items()}) for i,r in enumerate(meta)]
    write(out/'public-fixed-predictions.json',public)
    write(out/'public-fixed-seal.json',dict(sha256=sha(out/'public-fixed-predictions.json'),
        label_access_for_fixed_flags=False))
    # Labels are used only for other-group threshold calibration and evaluation.
    label_rows=read(SOURCE/'evaluator/dev-labels.json')
    assert [r['index'] for r in label_rows]==idx.tolist()
    truth=np.array([r['truth'] for r in label_rows],bool)
    groups=[r['base_group_id'] for r in meta]
    folds={}
    for name,screen in (('OR_group_score',np.ones(576,bool)),('OR_group_calibration',eligible)):
        proposed,thresholds,records=grouped_screen(logits,a,truth,screen,groups)
        fixed[name]=a|proposed
        folds[name]=records
        for i,r in enumerate(public):
            r.setdefault('group_cutoffs',{})[name]=float(thresholds[i])
    write(out/'fold-calibration.json',folds)
    flags={}
    for name,raw in fixed.items():
        flags[name+'_current']=raw
        flags[name+'_hold']=hold(raw,meta)
        assert np.all(~a|raw) and np.all(~hold(a,meta)|flags[name+'_hold'])
    predictions=[dict(id=r['id'],index=r['index'],
        flags={n:bool(v[i]) for n,v in flags.items()}, current_unknown={n:bool(unknown[i]) for n in flags})
        for i,r in enumerate(meta)]
    write(out/'predictions.json',predictions)
    write(out/'prediction-seal.json',dict(hashes={n:sha(out/n) for n in
        ('predictions.json','fold-calibration.json','public-fixed-predictions.json')},
        held_group_truth_excluded_from_each_threshold=True))
    rows=[dict(**r,truth=bool(truth[i]),flags=predictions[i]['flags'],current_unknown=predictions[i]['current_unknown'])
          for i,r in enumerate(meta)]
    module=metric_module(); report=module.evaluate(rows)
    summary=summarize(rows,report)
    admission={r['id']:r for r in read(SOURCE/'source-admission.json')['frames'] if r['id'] in {r['id'] for r in meta}}
    opportunities={}
    for name in NAMES[1:]:
        opportunities[name]={}
        for suffix in ('current','hold'):
            added=[r for r in rows if r['flags'][name+'_'+suffix] and not r['flags']['A_'+suffix]]
            recovered=[r for r in added if r['truth']]
            wrong=[r for r in added if not r['truth']]
            opportunities[name][suffix]=dict(rescued_ids=[r['id'] for r in recovered],new_fp_ids=[r['id'] for r in wrong],
                recovered_with_returned_target_corridor_support=sum(admission[r['id']]['returned_target_corridor_samples']>0 for r in recovered),
                recovered_without_returned_target_corridor_support=sum(admission[r['id']]['returned_target_corridor_samples']==0 for r in recovered))
    grouped={g:module.evaluate([r for r in rows if r['base_group_id']==g]) for g in sorted(set(groups))}
    disagreement=[]
    for i,r in enumerate(meta):
        if not a[i] and logits[i]>=0:
            disagreement.append(dict(id=r['id'],base_group_id=r['base_group_id'],layout_relation=r['layout_relation'],
                layer=r['layer'],phase=r['phase'],truth=bool(truth[i]),logit=float(logits[i]),
                calibration_eligible=bool(eligible[i]),geometric_score=base[r['index']]['score']))
    write(out/'frame-results.json',rows);write(out/'metrics.json',report)
    write(out/'group-metrics.json',grouped);write(out/'incremental-ids.json',opportunities)
    write(out/'disagreements.json',disagreement)
    write(out/'result.json',dict(status='DIAGNOSTIC_COMPLETE_NO_PROMOTION',scope='CONSUMED_8_GROUP_DEV',
        frames=576,base_groups=8,arms=summary,test_inference_activated=False,
        selection='NO_ARM_SELECTED_FOR_DEPLOYMENT',source_preserved=all(sha(SOURCE/n)==h for n,h in source_hashes.items())))
    assert read(out/'result.json')['source_preserved']
    write(out/'evaluation-seal.json',{n:sha(out/n) for n in ('frame-results.json','metrics.json','group-metrics.json',
        'incremental-ids.json','disagreements.json','result.json')})
    print(json.dumps(read(out/'result.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUT)
    run(parser.parse_args().output)
