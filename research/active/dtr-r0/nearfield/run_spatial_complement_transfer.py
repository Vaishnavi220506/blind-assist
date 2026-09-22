"""One new-layout inference of a fully frozen supplementary policy. No fit."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import time

import numpy as np
from run_spatial_bce import read, write, sha, seal, check_seal, held, T, OLD, CHECKPOINT

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921'
SOURCE = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
HIGH = 7.6612162590026855
ARMS = ('A_current', 'C_current', 'A_hold', 'C_hold')
CODE = ('run_spatial_complement_transfer.py','spatial_complement_transfer_spec.py',
        'spatial_complement_transfer_data.py','spatial_complement_transfer_capture.py',
        'launch_spatial_complement_transfer.py','launch_spatial_bce.py','spatial_bce_spec.py','run_spatial_bce.py',
        'spatial_bce_model.py','core_transfer_spec.py','ue_capture_readiness.py',
        'run_core_transfer.py','full_event_metrics_20260920.py','tof_fov45_core.py',
        'tof_corridor_calibration.py','ba_camera_corridor.py','ba_camera_corridor_spec.py',
        'evaluate_ba_camera_corridor.py','tof_lateral_core.py','ba_camera_corridor_metrics.py')


def old_test_unactivated():
    for name in ('test-start.json','test-logits.npy','test-prediction-seal.json','test-metrics.json'):
        assert not (SOURCE/name).exists(), 'Original retained test must stay unactivated'


def verify(out):
    p = read(out/'protocol.json')
    assert sha(out/'spec.json') == p['spec_sha256']
    assert sha(out/'protocol-before-run.md') == p['protocol_text_sha256']
    for name, digest in p['code_hashes'].items():
        assert sha(HERE/name) == digest, name
    for name, digest in p['input_hashes'].items():
        assert sha(ROOT/name) == digest, name
    assert p['high_logit'] == HIGH and p['strong_threshold'] == T
    old_test_unactivated()
    return p


def freeze(out):
    from spatial_complement_transfer_spec import specification, check_spec
    assert not out.exists(), 'One new cohort, no overwrite'
    old_test_unactivated()
    original = read(SOURCE/'protocol.json')
    for name in set(CODE) & set(original['code_hashes']):
        assert sha(HERE/name) == original['code_hashes'][name], 'Original implementation drift: '+name
    spec = specification()
    old = [ROOT/'artifacts.local/work'/n/'spec.json' for n in OLD] + [SOURCE/'spec.json']
    admission = check_spec(spec, [read(p) for p in old])
    out.mkdir(parents=True)
    write(out/'spec.json', spec)
    write(out/'preflight.json', admission)
    (out/'protocol-before-run.md').write_bytes((HERE/'SPATIAL_COMPLEMENT_TRANSFER_PROTOCOL_20260921.md').read_bytes())
    deps = old + [CHECKPOINT, SOURCE/'fit/head_last.pt', SOURCE/'fit/train_receipt.json',
        SOURCE/'protocol.json', SOURCE/'operating-point.json',
        ROOT/'tools/research_backend.py', ROOT/'tools/run_obstacle_research.py',
        ROOT/'research/active/dtr-r0/unreal/street_process_lifecycle.py']
    write(out/'protocol.json', dict(id='ba-spatial-complement-transfer-20260921',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frames=1152, clips=48, frames_per_clip=24, groups=16, dt_s=.2, capture_timeout_s=1800,
        spec_sha256=sha(out/'spec.json'), protocol_text_sha256=sha(out/'protocol-before-run.md'),
        code_hashes={n:sha(HERE/n) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in deps},
        high_logit=HIGH, strong_threshold=T, arms=list(ARMS),
        scope='NEW_PROCEDURAL_GROUPS_SAME_SIMULATOR_DEVELOPMENT',
        stop='One capture and frozen inference; no tuning or automatic successor'))
    print('FROZEN', admission, flush=True)


def predict(out):
    import torch
    from spatial_bce_model import encode_rgb, build_inputs, load_head, predict as infer, _select
    verify(out); check_seal(out, 'observation-seal.json')
    write(out/'inference-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        model_sha256=sha(SOURCE/'fit/head_last.pt'), training=False, evaluator_label_access=False))
    ids = read(out/'identities.json')
    assert len(ids) == 1152
    for r in ids:
        assert sha(out/r['rgb_path']) == r['rgb_sha256']
    features = encode_rgb([out/r['rgb_path'] for r in ids], CHECKPOINT, out/'features')
    with np.load(out/'observations.npz',allow_pickle=False) as data:
        inputs = build_inputs(features, data['ranges'])
    head = load_head(SOURCE/'fit/head_last.pt')
    device, backend = _select(head, torch.from_numpy(inputs[:64].copy()), out/'head_backend.json')
    head.to(device).eval()
    started = time.perf_counter()
    logits = infer(head, inputs).astype(np.float64)
    elapsed = time.perf_counter()-started
    assert logits.shape == (1152,) and np.isfinite(logits).all()
    baseline = read(out/'baseline.json')
    a = np.asarray([r['current'] for r in baseline],bool)
    c = a | (logits >= HIGH)
    flags = dict(A_current=a,C_current=c,A_hold=held(a,ids),C_hold=held(c,ids))
    predictions = [dict(id=r['id'],index=r['index'],clip_id=r['clip_id'],time_s=r['time_s'],
        logit=float(logits[i]), flags={k:bool(v[i]) for k,v in flags.items()},
        current_unknown={k:bool(baseline[i]['unknown']) for k in ARMS}) for i,r in enumerate(ids)]
    np.save(out/'logits.npy',logits)
    write(out/'predictions.json',predictions)
    write(out/'inference-receipt.json',dict(backend=backend,head_elapsed_s=elapsed,
        head_sha256=sha(SOURCE/'fit/head_last.pt'),encoder_sha256=sha(CHECKPOINT),
        rows=len(ids),high_logit=HIGH,evaluator_labels_read=False))
    seal(out,'prediction-seal.json',['logits.npy','predictions.json','inference-receipt.json',
        'features/feature_receipt.json','features/encoder_backend.json','head_backend.json'])
    verify(out)
    print('PREDICTIONS_SEALED',len(ids),flush=True)


def metric_module():
    s = importlib.util.spec_from_file_location('transfer_metrics',HERE/'full_event_metrics_20260920.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.ARMS=ARMS
    return m


def evaluate(out):
    verify(out);check_seal(out,'observation-seal.json');check_seal(out,'prediction-seal.json')
    write(out/'evaluation-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(out/'prediction-seal.json')))
    predictions=read(out/'predictions.json')
    meta=read(out/'evaluator/metadata.json');labels=read(out/'evaluator/transfer-labels.json')
    native=read(out/'source-admission.json')['frames']
    assert len(meta)==len(labels)==len(predictions)==len(native)==1152
    rows=[]
    for p,m,y,n in zip(predictions,meta,labels,native):
        assert p['index']==m['index']==y['index'] and p['id']==m['id']==n['id']
        rows.append(dict(**m,truth=bool(y['truth']),logit=p['logit'],flags=p['flags'],
            current_unknown=p['current_unknown'],native_target_corridor_samples=n['returned_target_corridor_samples']))
    evaluator=metric_module(); report=evaluator.evaluate(rows)
    group_reports={g:evaluator.evaluate([r for r in rows if r['base_group_id']==g])
                   for g in sorted({r['base_group_id'] for r in rows})}
    assert len(group_reports)==16
    summary={};increments={}
    for group in ('Core','Boundary'):
        for suffix in ('current','hold'):
            a=report[group]['arms']['A_'+suffix];c=report[group]['arms']['C_'+suffix]
            key=group+'_'+suffix
            summary[key]=dict(A=[a['frames'][k] for k in ('TP','FP','FN')],
                C=[c['frames'][k] for k in ('TP','FP','FN')],
                extra_TP=c['frames']['TP']-a['frames']['TP'],extra_FP=c['frames']['FP']-a['frames']['FP'],
                recall_gain=c['frames']['recall']-a['frames']['recall'],
                A_segments=a['false_alert_segment_count'],C_segments=c['false_alert_segment_count'],
                A_events=a['detected_events'],C_events=c['detected_events'])
            subset=[r for r in rows if (r['layout_relation']=='BOUNDARY')==(group=='Boundary')]
            increments[key]={name:[r['id'] for r in subset if cond(r)] for name,cond in (
                ('added_TP',lambda r:r['truth'] and r['flags']['C_'+suffix] and not r['flags']['A_'+suffix]),
                ('added_FP',lambda r:not r['truth'] and r['flags']['C_'+suffix] and not r['flags']['A_'+suffix]),
                ('lost_A_flags',lambda r:r['flags']['A_'+suffix] and not r['flags']['C_'+suffix]))}
    retention=all(not r['flags']['A_'+s] or r['flags']['C_'+s] for r in rows for s in ('current','hold'))
    assert retention
    gaining_groups=sum(r['Boundary']['arms']['C_current']['frames']['TP']>
        r['Boundary']['arms']['A_current']['frames']['TP'] for r in group_reports.values())
    signal=summary['Boundary_current']['recall_gain']>=.10 and gaining_groups>=8 and retention
    raw=signal and all(summary[g+'_current']['extra_FP']==0 and summary[g+'_current']['C_segments']<=summary[g+'_current']['A_segments'] for g in ('Core','Boundary'))
    complete=raw and all(summary[g+'_hold']['extra_FP']==0 and summary[g+'_hold']['C_segments']<=summary[g+'_hold']['A_segments'] for g in ('Core','Boundary'))
    result=dict(status='COMPLETE',scope='NEW_SAME_SIMULATOR_DEVELOPMENT',groups=16,frames=1152,
        metrics=summary,boundary_gaining_groups=gaining_groups,all_A_flags_retained=retention,
        prospective_rescue_signal=signal,strict_current_upgrade=raw,strict_complete_upgrade=complete,
        disposition='COMPONENT_OR_CHALLENGER' if signal else 'NEGATIVE_CONTROL',
        inference_mode='FROZEN_SINGLE_HIGH_OR',original_test_activated=False,automatic_successor=False)
    write(out/'frame-results.json',rows);write(out/'metrics.json',report)
    write(out/'group-metrics.json',group_reports);write(out/'incremental-ids.json',increments)
    write(out/'result.json',result)
    seal(out,'evaluation-seal.json',['frame-results.json','metrics.json','group-metrics.json','incremental-ids.json','result.json'])
    verify(out)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('freeze','materialize','predict','evaluate'))
    args=parser.parse_args()
    if args.stage=='materialize':
        from spatial_complement_transfer_data import materialize
        materialize(OUT)
    else:
        globals()[args.stage](OUT)
