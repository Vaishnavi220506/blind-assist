"""Frozen readout transfer on disclosed, previously consumed MZ136 groups.

Select an exact logit threshold on original dev groups. Only if dev improves
nuisance >=20% at retained recall/timing, replay original test and shifted test.
No training, model selection, or claims of fresh independent confirmation.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from mz136_direct_readout import DirectReadoutNet
from run_mz136_corridor_pair import ROOT, load, pair_indices
from run_mz120_occupancy import batch
from run_mz107_four_sensor import truth, sha, write, readrows
from evaluate_mz136_corridor_pair import score, retention, pair_metrics, encode_rows
from mz136_incumbent import public_observations


def exact_thresholds(scores):
    values = np.unique(np.asarray(scores, dtype=np.float64))
    assert len(values) and np.isfinite(values).all()
    # Every distinct >= decision, including all-alert and no-alert. No coarse
    # sigmoid grid: large negative logits must retain their distinct decisions.
    return np.r_[values, np.nextafter(values[-1], np.inf)]


def run(root, out, fit_dir=None):
    root, out = root.resolve(), out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    fit = Path(fit_dir).resolve() if fit_dir is not None else root/'train-direct-readout-v1'
    model_hash = sha(fit/'model.pt')
    cap = root/'source/returned-v1/capture-v1'
    prep = root/'incumbent/fresh-v1'
    completed = json.loads((fit/'completion.json').read_text())
    assert completed['status'] == 'PASS'
    for name, digest in completed['hashes'].items():
        assert sha(fit/name) == digest, name
    original_freeze = json.loads((fit/'freeze.json').read_text())
    assert original_freeze['ordered'] is False
    for name, digest in original_freeze['source'].items():
        assert sha(cap/name) == digest
    for name in ('mz136_direct_readout.py', 'mz120_occupancy.py'):
        assert sha(Path(__file__).with_name(name)) == original_freeze['code'][name]
    backend = fit/'backend.json'
    assert json.loads(backend.read_text())['selected_device_type'] == 'cuda'
    seal = json.loads((prep/'prediction-seal.json').read_text())
    receipt = json.loads((prep/'completion.json').read_text())
    assert sha(prep/'prediction-seal.json') == receipt['prediction_seal_sha256']
    for arm, digest in seal['predictions_sha256'].items():
        assert sha(prep/arm/'predictions.json') == digest
    observations = json.loads((prep/'observation-seal.json').read_text())
    assert sha(prep/'observation-seal.json') == seal['observation_seal_sha256']
    for arm, digest in observations['hashes'].items():
        assert sha(prep/arm/'raw.jsonl') == digest
    snapshots = out/'source-snapshot'
    snapshots.mkdir()
    files = [Path(__file__), Path(__file__).with_name('mz136_direct_readout.py'),
             Path(__file__).with_name('evaluate_mz136_corridor_pair.py')]
    for f in files:
        shutil.copyfile(f, snapshots/f.name)
    write(out/'freeze.json', dict(authority='CONSUMED_DEVELOPMENT_TRANSFER_NOT_FRESH_CONFIRMATION',
        model_sha256=sha(fit/'model.pt'), source=original_freeze['source'],
        code={f.name:sha(f) for f in files}, backend_record_sha256=sha(backend),
        selection='original dev only; exact logit partitions; minimum FP at recall within2pp and per-event delay<=0.25s',
        continue_condition='dev FP <=80% MZ129 with retention; otherwise stop before test inference',
        training=False, threshold_zero_also_reported=True, test_threshold_tuning=False))
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    model = DirectReadoutNet().cuda().eval()
    model.load_state_dict(torch.load(fit/'model.pt', map_location='cuda', weights_only=True))
    rows, es, spec, data = load(cap, cap)
    pairs = pair_indices(rows, spec)
    assert readrows(prep/'nominal/raw.jsonl') == public_observations(rows)
    predictions = json.loads((prep/'nominal/predictions.json').read_text())['predictions']
    assert [p['id'] for p in predictions] == [r['id'] for r in rows]
    baseline = np.array([p['candidate'] for p in predictions], bool)

    def infer_partition(part, values, base_flags, prefix):
        selected = [p for p in pairs if p['split'] == part]
        idx = sorted(i for p in selected for i in (p['a'],p['b']))
        remap = {old:new for new,old in enumerate(idx)}
        local_pairs = [dict(p,a=remap[p['a']],b=remap[p['b']]) for p in selected]
        rr, ee = [rows[i] for i in idx], [es[i] for i in idx]
        # Only this admitted partition's truth is evaluated.
        gt = np.array([truth(e) for e in ee],bool)
        ss, bb, residual = [], [], []
        tick = time.perf_counter()
        with torch.no_grad():
            for start in range(0,len(idx),8):
                b, v = model.components(batch(values, np.array(idx[start:start+8])))
                r = model.readout((v-model.feature_mean)/model.feature_scale).squeeze(-1)
                ss.extend((b+r).cpu().tolist());bb.extend(b.cpu().tolist());residual.extend(r.cpu().tolist())
        elapsed = time.perf_counter()-tick
        scores = np.array(ss)
        np.save(out/(prefix+'-scores.npy'),scores)
        write(out/(prefix+'-cases.json'), [dict(id=r['id'],family=e['family'],target=bool(y),
            base_logit=float(b),residual_logit=float(d),score=float(s),mz129=bool(base_flags[i]))
            for r,e,y,b,d,s,i in zip(rr,ee,gt,bb,residual,scores,idx)])
        return rr,ee,gt,scores,base_flags[idx],local_pairs,elapsed

    rr,ee,gt,ss,base,pp,seconds = infer_partition('dev',data,baseline,'dev')
    ix = list(range(len(rr)))
    bm = score(rr,ee,gt,base,ix)
    zero = score(rr,ee,gt,ss>=0,ix,base)
    zero['paired'] = pair_metrics(gt,ss,ss>=0,pp)
    zero['retention'] = retention(zero,bm)
    curve = []
    for t in exact_thresholds(ss):
        value = score(rr,ee,gt,ss>=t,ix,base)
        value.update(logit_threshold=float(t),retention=retention(value,bm))
        curve.append(value)
    eligible = [v for v in curve if v['retention']['pass_retention']]
    assert eligible
    best = min(eligible,key=lambda v:(v['metrics']['FP'],-v['metrics']['TP'],-v['logit_threshold']))
    threshold = best['logit_threshold']
    best['paired'] = pair_metrics(gt,ss,ss>=threshold,pp)
    write(out/'dev-threshold-curve.json',curve)
    passes = bool(best['metrics']['FP'] <= .8*bm['metrics']['FP'] and bm['metrics']['FP']>0)
    write(out/'operating-point.json',dict(logit_threshold=threshold,dev_frame_ids=[r['id'] for r in rr],
        model_sha256=sha(fit/'model.pt'),dev_scores_sha256=sha(out/'dev-scores.npy'),
        proceed_to_consumed_test=passes,test_outcomes_used_for_selection=False))
    result = dict(authority='CONSUMED_DEVELOPMENT_ONLY',dev=dict(mz129=bm,zero=zero,selected=best),
                  dev_pass=passes,inference_seconds=dict(dev=seconds),test_inference=False)
    if passes:
        shifted_rows = readrows(prep/'shifted/raw.jsonl')
        for a,b in zip(public_observations(rows),shifted_rows):
            assert {k:v for k,v in a.items() if k!='tof_zones'}=={k:v for k,v in b.items() if k!='tof_zones'}
        shifted_predictions=json.loads((prep/'shifted/predictions.json').read_text())['predictions']
        assert [p['id'] for p in shifted_predictions]==[r['id'] for r in rows]
        shifted_base=np.array([p['candidate'] for p in shifted_predictions],bool)
        for prefix,values,base_flags in [('consumed_test',data,baseline),
            ('shifted_consumed_test',encode_rows(shifted_rows,data),shifted_base)]:
            rr,ee,gt,ss,base,pp,seconds=infer_partition('test',values,base_flags,prefix)
            ix=list(range(len(rr)));bm=score(rr,ee,gt,base,ix)
            candidate=score(rr,ee,gt,ss>=threshold,ix,base)
            candidate['retention']=retention(candidate,bm)
            candidate['paired']=pair_metrics(gt,ss,ss>=threshold,pp)
            result[prefix]=dict(mz129=bm,candidate=candidate)
            result['inference_seconds'][prefix]=seconds
        result['test_inference']=True
    result['decision']='TRAIN_FIT_COMPONENT_ONLY_DEV_ALERT_GATE_FAILED'
    if passes:
        joint = all(result[p]['candidate']['retention']['pass_retention'] and
                    result[p]['candidate']['metrics']['FP'] <= .8*result[p]['mz129']['metrics']['FP']
                    for p in ('consumed_test','shifted_consumed_test'))
        result['decision']='RETAIN_CONSUMED_DEVELOPMENT_TRANSFER_CHALLENGER' if joint else 'DEV_GAIN_DOES_NOT_TRANSFER_TO_CONSUMED_TEST'
    write(out/'summary.json',result)
    assert sha(fit/'model.pt') == model_hash
    write(out/'completion.json',dict(status='PASS',model_unchanged=True,
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(decision=result['decision'],dev={k:v['metrics'] for k,v in result['dev'].items()},
                         threshold=threshold,test_inference=result['test_inference']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit-dir',type=Path)
    args=parser.parse_args()
    run(args.root,args.output,args.fit_dir)
