"""Causal two-state evidence filter with TRAIN-estimated transition counts.

This produces an empirical filtered score. Correlated, uncalibrated classifier
outputs do not become calibrated probabilities by passing through this filter.
Neither a low score nor state zero certifies clear space.
"""
import numpy as np


def adjacent(previous,current):
    return previous['episode_id']==current['episode_id'] and abs(current['time_s']-previous['time_s']-.25)<1e-6


def fit_parameters(rows,target):
    target=np.asarray(target,bool)
    if len(rows)!=len(target) or not len(rows):raise ValueError('Matched nonempty TRAIN rows required')
    transitions=np.zeros((2,2),int);starts=np.zeros(2,int)
    for i,y in enumerate(target):
        if i and adjacent(rows[i-1],rows[i]):transitions[int(target[i-1]),int(y)]+=1
        else:starts[int(y)]+=1
    matrix=(transitions+1)/(transitions.sum(1,keepdims=True)+2)
    return dict(transition=matrix.tolist(),initial_positive=float((starts[1]+1)/(starts.sum()+2)),
        training_positive_prior=float((target.sum()+1)/(len(target)+2)),
        transition_counts=transitions.tolist(),episode_start_counts=starts.tolist(),
        smoothing='LAPLACE_ONE_COUNT_FIXED',authority='TRAIN_LABEL_FREQUENCIES_ONLY')


def filter_scores(rows,scores,parameters):
    scores=np.asarray(scores,float)
    if len(rows)!=len(scores) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Finite matched classifier scores in [0,1] required')
    transition=np.asarray(parameters['transition'],float)
    if transition.shape!=(2,2) or (transition<=0).any() or not np.allclose(transition.sum(1),1):
        raise ValueError('Positive row-stochastic transition matrix required')
    prior=float(parameters['training_positive_prior']);initial=float(parameters['initial_positive'])
    if not 0<prior<1 or not 0<initial<1:raise ValueError('Interior priors required')
    result=np.empty(len(scores));state=initial
    for i,value in enumerate(scores):
        predicted=(1-state)*transition[0,1]+state*transition[1,1] if i and adjacent(rows[i-1],rows[i]) else initial
        q=float(np.clip(value,1e-6,1-1e-6))
        likelihood=q/(1-q)*(1-prior)/prior
        state=likelihood*predicted/(likelihood*predicted+1-predicted)
        result[i]=state
    return result
