"""Strong current evidence or two consecutive weak observations; causal only."""
import numpy as np


def previous_scores(rows,scores):
    scores=np.asarray(scores,float)
    if len(rows)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid scores')
    previous=np.full(scores.shape,-np.inf)
    for i in range(1,len(rows)):
        if rows[i]['episode_id']==rows[i-1]['episode_id'] and abs(rows[i]['time_s']-rows[i-1]['time_s']-.25)<1e-6:
            previous[i]=scores[i-1]
    return previous


def predict(rows,scores,low,high):
    if not np.isfinite([low,high]).all() or high<low:raise ValueError('Invalid thresholds')
    scores=np.asarray(scores,float);previous=previous_scores(rows,scores)
    return (scores>=high)|((scores>=low)&(previous>=low))


def fit_onset(rows,scores,target,baseline,low):
    scores=np.asarray(scores,float);target=np.asarray(target,bool);baseline=np.asarray(baseline,bool)
    if scores.shape!=target.shape or target.shape!=baseline.shape:raise ValueError('Unmatched labels')
    previous=previous_scores(rows,scores)
    required=target&baseline
    if not required.any() or (scores[required]<low).any():raise ValueError('Frozen low threshold lacks required support')
    strong_required=required&(previous<low)
    high=float(scores[strong_required].min()) if strong_required.any() else 1.
    assert np.all(predict(rows,scores,low,high)[required])
    return high
