"""Matched two-view symmetry pilot; only TRAIN fits, sealed consumed evaluation."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from mz159_reflection import views, averaged_score, METHOD
from mz145_causal_confirmation import fit_onset, predict
from run_mz143_corridor_evidence import (ROOT, CODE, CAP, INC, read, sha as path_sha, write,
    truth, selected_jsonl, public_observations, augmented_score, native_account,
    classifier, operating_threshold, local_dependencies)
from evaluate_mz136_corridor_pair import retention
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK = ROOT/'artifacts.local/work/mz159-reflection-invariance-20260916'
OLD = ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
ONSET = ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
M146 = ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
M158 = ROOT/'artifacts.local/work/mz158-crossview-agreement-20260916'
ARMS = ('repeat_control', 'reflection')


def sha(path):
    return path_sha(Path(path))


def features(rows, capture, out, name, start, cached):
    receipt = read(capture/'receipt.json'); xx = []; audits = []; rgb = {}
    yaw = 0.; episode = None; tick = time.perf_counter()
    for i, row in enumerate(rows):
        if row['episode_id'] != episode: yaw = 0.
        if row['imu_valid']: yaw += row['delta_yaw']
        episode = row['episode_id']; path = (capture/row['rgb_path']).resolve()
        assert path.is_relative_to(capture.resolve())
        assert sha(path) == receipt['hashes'][row['rgb_path']]
        rgb[str(path)] = sha(path)
        x, audit = views(row, cv2.imread(str(path)), yaw)
        np.testing.assert_array_equal(x[0], cached[i])
        xx.append(x); audits.append(dict(id=row['id'], **audit))
        assert time.perf_counter()-start < 1200
        if (i+1) % 48 == 0:
            print(json.dumps(dict(stage='features', panel=name, frames=i+1)), flush=True)
    x = np.stack(xx); np.savez_compressed(out/(name+'-features.npz'), views=x)
    write(out/(name+'-feature-audit.json'), audits)
    write(out/(name+'-feature-seal.json'), dict(ids=[r['id'] for r in rows],
        feature_sha256=sha(out/(name+'-features.npz')),
        audit_sha256=sha(out/(name+'-feature-audit.json')), rgb_sha256=rgb,
        original_cache_bitwise_equal=True, seconds=time.perf_counter()-tick,
        authority='PUBLIC_VIEWS_BEFORE_SELECTED_EVALUATOR_PARSE'))
    return x


def arm_score(model, x, arm):
    return averaged_score(model, x)[0] if arm == 'reflection' else model.predict_proba(x[:, 0])[:, 1]


def fit(x, y, arm):
    train = x if arm == 'reflection' else np.repeat(x[:, :1], 2, axis=1)
    model = classifier('fused_hgb'); model.set_params(min_samples_leaf=16)
    model.fit(train.reshape(-1, train.shape[-1]), np.repeat(y, 2),
              sample_weight=np.full(2*len(y), .5))
    return model


def later_panel(name):
    work = M146 if name == 'mz146' else M158
    capture = work/'source/returned-v1/capture-v1'
    receipt = read(capture/'receipt.json'); spec = read(capture/'spec.json')
    assert receipt['status'] == 'PASS' and sha(capture/'spec.json') == receipt['spec_sha256']
    for f in ('raw.jsonl', 'evaluator.jsonl'): assert sha(capture/f) == receipt['hashes'][f]
    rows = public_observations([json.loads(line) for line in (capture/'raw.jsonl').read_text().splitlines()])
    if name == 'mz146':
        folder = work/'evaluation-v1'; seal = read(folder/'prediction-seal.json')
        assert sha(folder/'features.npz') == seal['features_sha256']
        assert sha(folder/'predictions.json') == seal['predictions_sha256']
        saved = read(folder/'predictions.json')
        cached = np.load(folder/'features.npz')['features']
        folder = work/'incumbent-v1'; seal = read(folder/'prediction-seal.json')
    else:
        folder = work/'learned-v1'; seal = read(folder/'prediction-seal.json')
        assert sha(folder/'features.npz') == seal['context']['features_sha256']
        assert sha(folder/'predictions.json') == seal['predictions_sha256']
        saved = read(folder/'predictions.json')
        cached = np.load(folder/'features.npz')['static']
        folder = work/'baseline-v1'; seal = read(folder/'prediction-seal.json')
    assert sha(folder/'predictions.json') == seal['predictions_sha256']
    baseline = read(folder/'predictions.json')
    if isinstance(baseline, dict): baseline = baseline['predictions']
    assert [r['id'] for r in rows] == [p['id'] for p in baseline] == [p['id'] for p in saved]
    return capture, spec, rows, cached, np.array([p['candidate'] for p in baseline], bool)


def evaluate(name, capture, spec, split, rows, cached, baseline, out, start, models, cuts, oldmodel, oldcuts):
    bindings = {str(p):sha(p) for p in [capture/'receipt.json', capture/'spec.json',
        capture/'raw.jsonl', capture/'evaluator.jsonl', out/'model-seal.json']}
    write(out/(name+'-input-seal.json'), dict(inputs=bindings, authority='CONSUMED_DEVELOPMENT'))
    x = features(rows, capture, out, name, start, cached)
    scores = {a:arm_score(models[a], x, a) for a in ARMS}
    scores['mz145'] = oldmodel.predict_proba(x[:, 0])[:, 1]
    flags = {a:predict(rows, scores[a], **cuts[a]) for a in ARMS}
    flags['mz145'] = predict(rows, scores['mz145'], **oldcuts); flags['baseline'] = baseline
    write(out/(name+'-predictions.json'), [dict(id=r['id'],
        scores={a:float(s[i]) for a,s in scores.items()},
        flags={a:bool(p[i]) for a,p in flags.items()}) for i,r in enumerate(rows)])
    write(out/(name+'-prediction-seal.json'), dict(
        predictions_sha256=sha(out/(name+'-predictions.json')),
        input_seal_sha256=sha(out/(name+'-input-seal.json')),
        feature_seal_sha256=sha(out/(name+'-feature-seal.json')),
        model_seal_sha256=sha(out/'model-seal.json'),
        authority='SEALED_PUBLIC_PREDICTIONS_BEFORE_SELECTED_EVALUATOR_PARSE'))
    es = selected_jsonl(capture/'evaluator.jsonl', {r['id'] for r in rows})
    assert [e['id'] for e in es] == [r['id'] for r in rows]
    y = np.array([truth(e) for e in es], bool)
    reports = {a:augmented_score(rows, es, y, p.astype(float), .5, baseline, spec, split) for a,p in flags.items()}
    native = {a:native_account(rows, es, flags[a], baseline) for a in ARMS}
    for n in native.values(): n['radar_native_lineage'] = 'NOT_EVALUABLE'
    write(out/(name+'-native-contributors.json'), native)
    candidate = reports['reflection']; checks = {}; comparisons = {}
    for arm in ('baseline', 'mz145'):
        ref = reports[arm]; rr = retention(candidate, ref)
        lost = [r['id'] for i,r in enumerate(rows) if y[i] and flags[arm][i] and not flags['reflection'][i]]
        checks.update({arm+'_no_true_loss':not lost,
            arm+'_no_event_delay':all(v['relative_delay_s'] is not None and v['relative_delay_s']<=1e-9 for v in rr['per_event_delta']),
            arm+'_family_fp':all(candidate['families'][f]['FP']<=ref['families'][f]['FP'] for f in ref['families']),
            arm+'_false_segments':candidate['events']['false_alert_segments']<=ref['events']['false_alert_segments']})
        comparisons[arm] = dict(lost_true_frames=lost, retention=rr)
        if name != 'dev': checks[arm+'_precision'] = candidate['precision'] > ref['precision']
    if name == 'dev':
        checks['dev_counts'] = candidate['metrics']['TP']==24 and candidate['metrics']['FP']<=3
        checks['dev_segments'] = candidate['events']['false_alert_segments']<=2
    else:
        checks['baseline_fp_reduction'] = candidate['metrics']['FP']<=.7*reports['baseline']['metrics']['FP']
        checks['mz145_fp_reduction'] = candidate['metrics']['FP']<reports['mz145']['metrics']['FP']
    checks['native_retention'] = not native['reflection']['nonalert_with_native_corridor_contributors']
    result = dict(reports=reports, checks=checks, comparisons=comparisons,
        passed=all(checks.values()), authority='CONSUMED_DEVELOPMENT_NOT_FRESH',
        original_test_raw_or_evaluator_access=False, cached_test_predictions_decoded_but_unused=True)
    write(out/(name+'-summary.json'), result)
    assert all(sha(p)==h for p,h in bindings.items())
    print(json.dumps(dict(stage='evaluated', panel=name, passed=result['passed'],
        metrics={a:r['metrics'] for a,r in reports.items()}, checks=checks)), flush=True)
    return result


def run(out):
    out = out.resolve(); assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True); start = time.perf_counter()
    sources = local_dependencies(__file__); sources.update(local_dependencies(CODE/'mz159_reflection.py'))
    sources[str(CODE/'MZ159_PROTOCOL_20260916.md')] = sha(CODE/'MZ159_PROTOCOL_20260916.md')
    for p in sources:
        dest = out/'source-snapshot'/Path(p).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    spec = read(CAP/'spec.json'); receipt = read(CAP/'receipt.json'); oldfreeze = read(OLD/'freeze.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for f in ('raw.jsonl','evaluator.jsonl'): assert sha(CAP/f)==receipt['hashes'][f]
    ids = {s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    rows = {s:public_observations(selected_jsonl(CAP/'raw.jsonl', ids[s])) for s in ids}
    assert len(rows['train'])==192 and len(rows['dev'])==48
    cached = {}
    for s in ids:
        assert [r['id'] for r in rows[s]]==oldfreeze['selected_frame_ids'][s]
        assert sha(OLD/(s+'-features.npz'))==read(OLD/(s+'-feature-seal.json'))['feature_sha256']
        data = np.load(OLD/(s+'-features.npz')); cached[s] = np.c_[data['sensor'],data['geometry']]
    assert sha(INC/'nominal/predictions.json')==read(INC/'prediction-seal.json')['predictions_sha256']['nominal']
    base = {p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    assert sha(OLD/'fused_hgb.pkl')==read(OLD/'model-seal.json')['models']['fused_hgb']
    onset = read(ONSET/'onset-seal.json'); oldcuts = {k:onset[k] for k in ('low','high')}
    oldmodel = pickle.loads((OLD/'fused_hgb.pkl').read_bytes())
    inputs = [CAP/'spec.json',CAP/'receipt.json',CAP/'raw.jsonl',CAP/'evaluator.jsonl',
        OLD/'freeze.json',OLD/'train-features.npz',OLD/'dev-features.npz',OLD/'train-folds.json',
        OLD/'fused_hgb.pkl',OLD/'model-seal.json',ONSET/'onset-seal.json',INC/'nominal/predictions.json']
    write(out/'freeze.json', dict(method=METHOD, arms=ARMS, candidate='reflection',
        sources=sources, inputs={str(p):sha(p) for p in inputs},
        original_test_raw_or_evaluator_access=False, cached_test_predictions_decoded_but_unused=True, budget_seconds=1200,
        authority='FIXED_CONSUMED_DEVELOPMENT_RECIPE_BEFORE_FEATURES_AND_FIT'))
    first = rows['train'][0]
    select_backend('batch-tensor', cpu=BackendCandidate('opencv-sklearn-cpu','cpu',
        lambda:views(first,cv2.imread(str(CAP/first['rgb_path'])),first['delta_yaw'] if first['imu_valid'] else 0.),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__+' sklearn '+sklearn.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented OpenCV and sklearn HGB have no GPU backend'})
    x = features(rows['train'],CAP,out,'train',start,cached['train'])
    oldscores = oldmodel.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(-1,2)
    difference = np.abs(oldscores[:,0]-oldscores[:,1])
    write(out/'original-model-symmetry.json', dict(panel='TRAIN192_DIAGNOSTIC',
        mean_absolute_difference=float(difference.mean()), max_absolute_difference=float(difference.max()),
        q=np.quantile(difference,[0,.25,.5,.75,1]).tolist(), original_and_mirror_scores=oldscores.tolist()))
    es = selected_jsonl(CAP/'evaluator.jsonl',ids['train'])
    assert [e['id'] for e in es]==[r['id'] for r in rows['train']]
    y = np.array([truth(e) for e in es],bool); b=np.array([base[r['id']] for r in rows['train']],bool)
    ff = read(OLD/'train-folds.json'); assert [f['id'] for f in ff]==[r['id'] for r in rows['train']]
    folds = np.array([f['heldout_fold'] for f in ff]); assert all(np.sum(folds==f)==48 for f in range(4))
    write(out/'train-folds.json',dict(frames=ff,view_folds=np.repeat(folds,2).tolist(),
        view_weights=[.5]*(2*len(y)),flattening='FRAME_MAJOR_TWO_VIEWS'))
    models={};cuts={};oof={};train={};fit_seconds={}
    for arm in ARMS:
        tick=time.perf_counter(); scores=np.empty(len(y))
        for f in range(4):
            model=fit(x[folds!=f],y[folds!=f],arm)
            scores[folds==f]=arm_score(model,x[folds==f],arm)
            assert time.perf_counter()-start<1200
        low=operating_threshold(scores,y,b);high=fit_onset(rows['train'],scores,y,b,low)
        cuts[arm]=dict(low=low,high=high);oof[arm]=scores
        flags=predict(rows['train'],scores,low,high)
        train[arm]=augmented_score(rows['train'],es,y,flags.astype(float),.5,b,spec,'train')
        models[arm]=fit(x,y,arm);(out/(arm+'.pkl')).write_bytes(pickle.dumps(models[arm]))
        fit_seconds[arm]=time.perf_counter()-tick
        print(json.dumps(dict(stage='fit',arm=arm,cuts=cuts[arm],oof=train[arm]['metrics'],seconds=fit_seconds[arm])),flush=True)
    np.savez_compressed(out/'oof.npz',**oof);write(out/'train-summary.json',dict(reports=train,fit_seconds=fit_seconds))
    write(out/'model-seal.json',dict(models={a:sha(out/(a+'.pkl')) for a in ARMS},cutoffs=cuts,
        oof_sha256=sha(out/'oof.npz'),train_summary_sha256=sha(out/'train-summary.json'),
        freeze_sha256=sha(out/'freeze.json'),authority='SEALED_BEFORE_DEV_INFERENCE_AND_EVALUATOR_PARSE'))
    panels={}
    panels['dev']=evaluate('dev',CAP,spec,'dev',rows['dev'],cached['dev'],
        np.array([base[r['id']] for r in rows['dev']],bool),out,start,models,cuts,oldmodel,oldcuts)
    if panels['dev']['passed']:
        for name in ('mz146','mz158'):
            cap,ss,rr,cc,bb=later_panel(name)
            panels[name]=evaluate(name,cap,ss,'confirmation',rr,cc,bb,out,start,models,cuts,oldmodel,oldcuts)
    attribution=any(p['reports']['reflection']['metrics']['FP']<p['reports']['repeat_control']['metrics']['FP'] for p in panels.values())
    attribution=attribution and all(p['reports']['reflection']['metrics']['TP']>=p['reports']['repeat_control']['metrics']['TP'] for p in panels.values())
    passed=len(panels)==3 and all(p['passed'] for p in panels.values()) and attribution
    result=dict(panels=panels,attribution_passed=attribution,development_gain=passed,
        decision='REFLECTION_DEVELOPMENT_GAIN' if passed else 'REFLECTION_DEVELOPMENT_NOT_MET',
        seconds=time.perf_counter()-start,original_test_raw_or_evaluator_access=False, cached_test_predictions_decoded_but_unused=True,
        authority='CONSUMED_DEVELOPMENT_NOT_FRESH')
    write(out/'summary.json',result)
    assert all(sha(p)==h for p,h in sources.items())
    assert all(sha(p)==h for p,h in read(out/'freeze.json')['inputs'].items())
    assert time.perf_counter()-start<1200
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        model_seal_sha256=sha(out/'model-seal.json'),sources_inputs_unchanged=True,
        resources='Process-local CPU models; no remote allocations; process exits'))
    print(json.dumps(dict(decision=result['decision'],panels=list(panels),attribution_passed=attribution,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=4):run(parser.parse_args().output)
