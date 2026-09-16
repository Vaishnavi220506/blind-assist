"""Unchanged frozen MZ169 equal-mean challenger for prospective confirmation.

Four correlated experts share the original observations and expert-training
cohort. No fitted stacking model, threshold fitting or sensor pruning is used.
Nonalert remains UNKNOWN, never a clear-space declaration.
"""
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np

from mz145_causal_confirmation import predict
from mz148_background_residual import CausalResidual
from mz159_reflection import averaged_score, views
from mz169_expert_stacking import mean_score


ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916'
STATIC=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
ONSET=ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
TEMPORAL=ROOT/'artifacts.local/work/mz148-background-residual-20260916/run-v1'
REFLECTION=ROOT/'artifacts.local/work/mz159-reflection-invariance-20260916/run-v1'
PARENT=ROOT/'artifacts.local/work/mz169-expert-stacking-20260916'
CHALLENGER=PARENT/'mean-challenger-freeze.json'
EXPERTS=('static','unregistered','compensated','reflection')
MODEL_PATHS=dict(static=STATIC/'fused_hgb.pkl',unregistered=TEMPORAL/'unregistered.pkl',
                 compensated=TEMPORAL/'compensated.pkl',reflection=REFLECTION/'reflection.pkl')
METHOD=dict(schema='MZ170_UNCHANGED_MZ169_EQUAL_LOGIT_MEAN_V1',expert_order=list(EXPERTS),
    weights=[.25,.25,.25,.25],clip_probability=[1e-6,1-1e-6],
    score='MZ169_MEAN_SCORE_OF_FOUR_EXPERT_PROBABILITIES',
    views='UNCHANGED_MZ159_ORIGINAL_AND_REFLECTED_PUBLIC_FEATURES',view_shape=[2,2485],
    temporal='UNCHANGED_MZ148_CAUSAL_RESIDUAL_BOTH_ARMS',temporal_features=3084,
    reflection_score='UNCHANGED_MZ159_AVERAGED_SCORE',
    readout='UNCHANGED_MZ145_STRONG_CURRENT_OR_CONSECUTIVE_WEAK',
    cutoffs='UNCHANGED_ORIGINAL_EXPERT_CUTOFFS_AND_MZ169_MEAN_CHALLENGER_CUTOFFS',
    new_fit=False,new_threshold_selection=False,new_hardware=False,
    independence='CORRELATED_EXPERTS_SHARE_SENSORS_AND_ORIGINAL_TRAIN192',
    no_alert='UNKNOWN_NOT_CLEAR',backend_reason='GPU_BACKEND_UNAVAILABLE')


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_models():
    """Authenticate and load the four frozen experts and five cutoff pairs."""
    challenger=_read(CHALLENGER)
    if (challenger['candidate']!='MZ169_FIXED_EQUAL_MEAN'
            or challenger['expert_order']!=list(EXPERTS)
            or challenger['weights']!=METHOD['weights']
            or challenger['clip_probability']!=METHOD['clip_probability']):
        raise ValueError('Unexpected inherited mean challenger recipe')
    for path,digest in challenger['bindings'].items():
        if _sha(path)!=digest:raise ValueError('Challenger binding changed: '+path)
    ss=_read(STATIC/'model-seal.json');ts=_read(TEMPORAL/'model-seal.json')
    rs=_read(REFLECTION/'model-seal.json');onset=_read(ONSET/'onset-seal.json')
    parent=_read(PARENT/'run-v1/model-seal.json')
    if ss['selected']!='fused_hgb' or onset['source_arm']!='fused_hgb':
        raise ValueError('Static expert selection differs')
    if _sha(STATIC/'model-seal.json')!=onset['model_seal_sha256']:
        raise ValueError('Static onset/model seal mismatch')
    expected=dict(static=ss['models']['fused_hgb'],unregistered=ts['models']['unregistered'],
                  compensated=ts['models']['compensated'],reflection=rs['models']['reflection'])
    models={}
    for arm,path in MODEL_PATHS.items():
        frozen=challenger['expert_models'][arm]
        if (Path(frozen['path']).resolve()!=path.resolve()
                or frozen['sha256']!=expected[arm] or _sha(path)!=expected[arm]):
            raise ValueError('Frozen expert identity mismatch: '+arm)
        models[arm]=pickle.loads(path.read_bytes())
    mean=dict(challenger['causal_cutoffs'])
    if mean!=parent['cutoffs']['mean']:
        raise ValueError('Mean cutoffs differ from sealed MZ169 calibration')
    cuts=dict(static={k:onset[k] for k in ('low','high')},
        unregistered=dict(ts['cutoffs']['unregistered']),compensated=dict(ts['cutoffs']['compensated']),
        reflection=dict(rs['cutoffs']['reflection']),mean=mean)
    if any(set(c)!= {'low','high'} or not np.isfinite([c['low'],c['high']]).all()
           or c['high']<c['low'] for c in cuts.values()):
        raise ValueError('Invalid inherited causal cutoffs')
    return models,cuts


def extract_features(rows,loader,deadline):
    """Use loader(row)->native BGR; deadline is absolute perf_counter time.

    Public yaw integrates only valid raw increments, reset at episode changes,
    identically to MZ159. MZ148 owns its unchanged temporal episode/gap reset.
    The caller authenticates observations/RGB and binds the returned audit.
    """
    if not math.isfinite(float(deadline)):raise ValueError('Finite absolute deadline required')
    if not len(rows):raise ValueError('Nonempty ordered public rows required')
    residual=CausalResidual();yaw=0.;episode=None
    features={key:[] for key in ('views','unregistered','compensated')};audits=[]
    def within():
        if time.perf_counter()>=deadline:raise TimeoutError('MZ170 feature extraction deadline exceeded')
    for index,row in enumerate(rows):
        within()
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];image=loader(row);within()
        view,view_audit=views(row,image,yaw);within()
        temporal=residual.update(row,image);within()
        if view.shape!=(2,2485) or not np.isfinite(view).all():raise ValueError('Unexpected frozen view features')
        features['views'].append(view)
        for arm in ('unregistered','compensated'):
            value=temporal[arm]
            if value.shape!=(3084,) or not np.isfinite(value).all():raise ValueError('Unexpected frozen temporal features')
            features[arm].append(value)
        audits.append(dict(id=row['id'],integrated_yaw_deg=float(yaw),views=view_audit,temporal=temporal['audit']))
        if (index+1)%48==0:
            print(json.dumps(dict(stage='mz170_features',frames=index+1,total=len(rows))),flush=True)
    return {key:np.stack(value) for key,value in features.items()},audits


def decisions(rows,features,models,cuts):
    """Return four expert probabilities, mean signed logit, and causal flags."""
    n=len(rows);view=np.asarray(features['views'])
    if view.shape!=(n,2,2485) or not np.isfinite(view).all():raise ValueError('Expected finite N x 2 x 2485 views')
    original=view[:,0]
    scores=dict(static=models['static'].predict_proba(original)[:,1])
    for arm in ('unregistered','compensated'):
        temporal=np.asarray(features[arm])
        if temporal.shape!=(n,3084) or not np.isfinite(temporal).all():raise ValueError('Expected finite N x 3084 temporal features')
        scores[arm]=models[arm].predict_proba(np.c_[original,temporal])[:,1]
    scores['reflection']=averaged_score(models['reflection'],view)[0]
    scores['mean']=mean_score(np.column_stack([scores[arm] for arm in EXPERTS]))
    flags={arm:predict(rows,value,**cuts[arm]) for arm,value in scores.items()}
    return scores,flags
