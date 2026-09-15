"""Fixed agreement of static geometry and raw causal-change classifiers.

The models share observations/training; agreement is not independent sensor
corroboration or a calibrated confidence. Non-alert means UNKNOWN.
"""
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
from mz143_corridor_features import extract
from mz148_background_residual import LAGS,summarize
from mz145_causal_confirmation import predict

ROOT=Path(__file__).resolve().parents[4]
STATIC=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
ONSET=ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
TEMPORAL=ROOT/'artifacts.local/work/mz148-background-residual-20260916/run-v1'
METHOD=dict(schema='MZ158_STATIC_AND_RAW_CHANGE_V1',models=['MZ143_fused_hgb','MZ148_unregistered'],
    static_features=2485,temporal_features=3084,lags=list(LAGS),rule='AND_OF_TWO_UNCHANGED_CAUSAL_ALERT_FLAGS',
    new_training=False,new_thresholds=False,new_sensors=False,no_alert='UNKNOWN_NOT_CLEAR',
    independence='CORRELATED_MODELS_SHARE_SENSORS_AND_ORIGINAL_TRAIN192',
    discovery='POSTHOC_CONSUMED_MZ146_COMPLEMENTARITY_REQUIRES_FRESH_CHECK',
    backend_reason='GPU_BACKEND_UNAVAILABLE')


class RawResidual:
    """Exactly MZ148's unregistered feature path, without unused ECC work."""
    def __init__(self):self.history=[];self.episode=None;self.time=None

    def update(self,row,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Native uint8 BGR required')
        if row['episode_id']!=self.episode or self.time is None or abs(row['time_s']-self.time-.25)>1e-6:
            self.history=[]
        self.episode=row['episode_id'];self.time=row['time_s']
        current=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        values=[]
        for lag in LAGS:
            if len(self.history)>=lag:
                reference=self.history[-lag]
                if reference.shape!=current.shape:raise ValueError('Resolution changed within episode')
                part=summarize(current,reference,np.ones_like(current,dtype=bool));head=[1.,1.,0.,1.]
            else:part=summarize(current,None,None);head=[0.,0.,0.,0.]
            values.extend(head);values.extend(part)
        self.history.append(current.copy());self.history=self.history[-max(LAGS):]
        result=np.asarray(values,np.float32)
        assert result.shape==(3084,) and np.isfinite(result).all()
        return result


def load_models():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    onset=read(ONSET/'onset-seal.json');seal=read(TEMPORAL/'model-seal.json')
    paths={'static':STATIC/'fused_hgb.pkl','raw_change':TEMPORAL/'unregistered.pkl'}
    models={key:pickle.loads(path.read_bytes()) for key,path in paths.items()}
    cuts={'static':dict(low=onset['low'],high=onset['high']),
          'raw_change':seal['cutoffs']['unregistered']}
    return models,cuts


def decisions(rows,static_features,raw_features,models,cuts):
    scores={'static':models['static'].predict_proba(static_features)[:,1],
            'raw_change':models['raw_change'].predict_proba(np.c_[static_features,raw_features])[:,1]}
    flags={key:predict(rows,value,**cuts[key]) for key,value in scores.items()}
    flags['agreement']=flags['static']&flags['raw_change']
    return scores,flags
