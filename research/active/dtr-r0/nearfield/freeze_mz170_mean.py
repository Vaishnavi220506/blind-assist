"""Authenticate the unchanged mean recipe and selected-dev parity before capture.

Execution is separately authorized after the MZ170 protocol/runner are ready.
No fitting, cutoff selection, evaluator decoding, original test records or fresh
capture access occurs here. Only original dev48 RGB is replayed.
"""
from pathlib import Path
import difflib
import shutil
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from mz170_mean_ensemble import (ROOT,WORK,STATIC,ONSET,TEMPORAL,REFLECTION,PARENT,
    CHALLENGER,EXPERTS,MODEL_PATHS,METHOD,load_models,extract_features,decisions)
from run_mz143_corridor_evidence import (CAP,CODE,read,sha as _path_sha,write,
    selected_jsonl,public_observations,local_dependencies)


def sha(path):return _path_sha(Path(path))


def run():
    started=time.perf_counter();deadline=started+300.
    out=WORK/'recipe-freeze.json'
    if out.exists():raise FileExistsError('MZ170 recipe already frozen; no overwrite')
    for name in ('MZ170_PROTOCOL_20260916.md','run_mz170_mean_confirmation.py'):
        if not (CODE/name).is_file():raise FileNotFoundError('Complete root-owned protocol/runner first: '+name)
    WORK.mkdir(parents=True,exist_ok=True)
    sources=local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'run_mz170_mean_confirmation.py'))
    sources[str(CODE/'MZ170_PROTOCOL_20260916.md')]=sha(CODE/'MZ170_PROTOCOL_20260916.md')
    inputs={}
    def bind(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Authentication failed: '+str(path))
        if str(path) in inputs and inputs[str(path)]!=digest:raise ValueError('Input changed during freeze')
        inputs[str(path)]=digest
        return digest
    def document(path,expected=None):
        bind(path,expected);return read(Path(path))
    challenger=document(CHALLENGER)
    for path,digest in challenger['bindings'].items():bind(path,digest)
    for value in challenger['expert_models'].values():bind(value['path'],value['sha256'])
    static=document(STATIC/'model-seal.json');onset=document(ONSET/'onset-seal.json')
    temporal=document(TEMPORAL/'model-seal.json');reflection=document(REFLECTION/'model-seal.json')
    bind(STATIC/'model-seal.json',onset['model_seal_sha256'])
    if static['selected']!='fused_hgb' or onset['source_arm']!='fused_hgb':raise ValueError('Unexpected static expert')
    original_freezes={}
    for folder,seal in ((STATIC,static),(TEMPORAL,temporal),(REFLECTION,reflection)):
        freeze=document(folder/'freeze.json',seal.get('freeze_sha256'))
        original_freezes[str(folder)]=freeze
        for path,digest in freeze['sources'].items():
            if sha(path)==digest:
                bind(path,digest)
                continue
            # This historical orchestration file was mechanically repaired and
            # its data-access disclosure corrected after its saved source freeze.
            # Authenticate the original snapshot; never waive a live dependency.
            historical=Path(path)
            if folder!=REFLECTION or historical!=CODE/'run_mz159_reflection.py':
                raise ValueError('Unexpected historical source drift: '+path)
            if str(historical) in sources:
                raise ValueError('Historical source drift affects live execution: '+path)
            snapshot=folder/'source-snapshot'/historical.relative_to(ROOT.resolve())
            bind(snapshot,digest)
            current=bind(historical)
            correction=WORK/'historical-source-binding.json'
            write(correction,dict(original_path=str(historical),original_sha256=digest,
                authenticated_snapshot=str(snapshot),current_sha256=current,
                current_not_in_execution_dependencies=True,
                reason='HISTORICAL_ORCHESTRATION_PATH_NORMALIZATION_BASELINE_CONTAINER_AND_ACCESS_DISCLOSURE',
                diff=''.join(difflib.unified_diff(snapshot.read_text(encoding='utf-8').splitlines(True),
                    historical.read_text(encoding='utf-8').splitlines(True),fromfile=str(snapshot),tofile=str(historical))),
                first_failure_log_sha256=sha(WORK/'freeze-first-preflight-failure.log')))
            bind(correction)
        for path,digest in freeze['inputs'].items():bind(path,digest)
        oof_name='oof-scores.npz' if folder==STATIC else 'oof.npz'
        for key,name in (('oof_sha256',oof_name),('train_summary_sha256','train-summary.json')):
            if key in seal:bind(folder/name,seal[key])
    for path,digest in onset['sources'].items():bind(path,digest)
    parent_seal=document(PARENT/'run-v1/model-seal.json')
    parent_recipe=document(PARENT/'run-v1/recipe-freeze.json')
    for path,digest in parent_recipe['sources'].items():bind(path,digest)
    parent_complete=document(PARENT/'run-v1/completion.json')
    if parent_complete['status']!='PASS':raise ValueError('MZ169 did not complete')
    bind(PARENT/'run-v1/model-seal.json',parent_complete['model_seal_sha256'])
    bind(PARENT/'run-v1/summary.json',parent_complete['summary_sha256'])
    bind(PARENT/'run-v1/all-inputs.json',parent_complete['inputs_sha256'])
    if parent_seal['cutoffs']['mean']!=challenger['causal_cutoffs']:raise ValueError('Mean cutoff identity mismatch')
    receipt=document(CAP/'receipt.json')
    if receipt['status']!='PASS':raise ValueError('Original source receipt failed')
    spec=document(CAP/'spec.json',receipt['spec_sha256'])
    bind(CAP/'raw.jsonl',receipt['hashes']['raw.jsonl'])
    # Source-spec identities are bookkeeping. Only these 48 raw records and RGB
    # images are decoded; no evaluator JSONL is decoded.
    ids=original_freezes[str(STATIC)]['selected_frame_ids']['dev']
    if len(ids)!=48 or set(ids)!={f['id'] for f in spec['frames'] if f['split']=='dev'}:
        raise ValueError('Original dev48 identity mismatch')
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids)))
    if [r['id'] for r in rows]!=ids:raise ValueError('Original dev order mismatch')
    for folder in (REFLECTION,TEMPORAL):
        ps=document(folder/'dev-prediction-seal.json')
        bind(folder/'model-seal.json',ps['model_seal_sha256'])
        fs=document(folder/'dev-feature-seal.json',ps['feature_seal_sha256'])
        bind(folder/'dev-features.npz',fs['feature_sha256'])
        bind(folder/'dev-feature-audit.json',fs['audit_sha256'])
        bind(folder/'dev-predictions.json',ps['predictions_sha256'])
        if fs['ids']!=ids:raise ValueError('Frozen dev feature order mismatch')
        if 'input_seal_sha256' in ps:
            linked=document(folder/'dev-input-seal.json',ps['input_seal_sha256'])
            for path,digest in linked['inputs'].items():bind(path,digest)
    saved_seal=document(PARENT/'run-v1/dev-prediction-seal.json')
    bind(PARENT/'run-v1/model-seal.json',saved_seal['model_seal_sha256'])
    for path,digest in saved_seal['inputs'].items():bind(path,digest)
    saved=document(PARENT/'run-v1/dev-predictions.json',saved_seal['predictions_sha256'])
    if [p['id'] for p in saved]!=ids:raise ValueError('Parent dev predictions differ')
    for path,digest in sources.items():
        dest=WORK/'recipe-snapshot'/Path(path).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if sha(dest)!=digest:raise ValueError('Source snapshot mismatch')
    rgb={}
    for row in rows:
        path=(CAP/row['rgb_path']).resolve()
        if not path.is_relative_to(CAP.resolve()):raise ValueError('RGB path escapes admitted capture')
        rgb[str(path)]=bind(path,receipt['hashes'][row['rgb_path']])
    write(WORK/'parity-input-seal.json',dict(sources=sources,inputs=inputs,ids=ids,
        authority='SELECTED_DEV48_REPLAY_PRE_INPUT_SEAL_NO_EVALUATOR_OR_TEST_DECODE'))
    models,cuts=load_models()
    def loader(row):
        image=cv2.imread(str(CAP/row['rgb_path']))
        if image is None:raise ValueError('Unable to decode selected dev RGB')
        return image
    features,audits=extract_features(rows,loader,deadline)
    with np.load(REFLECTION/'dev-features.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(features['views'],data['views'])
    with np.load(TEMPORAL/'dev-features.npz',allow_pickle=False) as data:
        for arm in ('unregistered','compensated'):
            np.testing.assert_array_equal(features[arm],data[arm])
    scores,flags=decisions(rows,features,models,cuts)
    for index,arm in enumerate(EXPERTS):
        np.testing.assert_array_equal(scores[arm],[p['expert_scores'][index] for p in saved])
        parent_arm='mz145' if arm=='static' else arm
        np.testing.assert_array_equal(flags[arm],[p['flags'][parent_arm] for p in saved])
    np.testing.assert_array_equal(scores['mean'],[p['scores']['mean'] for p in saved])
    np.testing.assert_array_equal(flags['mean'],[p['flags']['mean'] for p in saved])
    np.savez_compressed(WORK/'dev-replay-features.npz',**features)
    write(WORK/'dev-replay-audit.json',audits)
    write(WORK/'dev-replay.json',[dict(id=r['id'],scores={k:float(v[i]) for k,v in scores.items()},
        flags={k:bool(v[i]) for k,v in flags.items()}) for i,r in enumerate(rows)])
    outputs={n:sha(WORK/n) for n in ('dev-replay-features.npz','dev-replay-audit.json','dev-replay.json')}
    parity=dict(frames=48,ids=ids,feature_shapes={k:list(v.shape) for k,v in features.items()},
        feature_values={k:int(v.size) for k,v in features.items()},bitwise_features={k:True for k in features},
        bitwise_scores={k:True for k in scores},bitwise_flags={k:True for k in flags},
        original_test_access=False,evaluator_decoded=False,new_fit=False,new_cutoffs=False,
        outputs=outputs,rgb_hashes=rgb,seconds=time.perf_counter()-started)
    write(WORK/'parity.json',parity)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Input changed after parity: '+path)
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('Source changed after parity: '+path)
    if time.perf_counter()>=deadline:raise TimeoutError('MZ170 parity exceeded fixed 300s deadline')
    write(out,dict(method=METHOD,sources=sources,inputs=inputs,cutoffs=cuts,
        models={arm:sha(path) for arm,path in MODEL_PATHS.items()},model_paths={arm:str(path) for arm,path in MODEL_PATHS.items()},
        mean_challenger_freeze_sha256=sha(CHALLENGER),parity_sha256=sha(WORK/'parity.json'),
        parity_input_seal_sha256=sha(WORK/'parity-input-seal.json'),parity=parity,
        seconds=time.perf_counter()-started,original_test_access=False,new_fit=False,new_threshold_selection=False,
        authority='UNCHANGED_MEAN_RECIPE_SEALED_BEFORE_FRESH_CAPTURE_AND_PREDICTIONS'))
    print(dict(recipe_freeze=str(out),sha256=sha(out),seconds=time.perf_counter()-started),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
