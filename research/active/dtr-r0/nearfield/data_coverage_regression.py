"""Secondary consumed-cohort replay, only after primary evaluation is sealed."""
from datetime import datetime, timezone
import time
import numpy as np
import torch
from run_data_coverage import OUT, ROOT, TRANSFER, R_SOURCE, read, write, sha, seal, check, verify
from run_spatial_bce import check_seal, held
import spatial_bce_model as model
from run_corridor_relative import costs
from data_coverage_learning import metrics, ARMS


def run():
    verify();check('evaluation-seal.json');check('selection-seal.json')
    check_seal(TRANSFER,'prediction-seal.json');check_seal(R_SOURCE,'prediction-seal.json')
    receipt=read(TRANSFER/'features/feature_receipt.json')
    assert sha(TRANSFER/'features/rgb_features.npy')==receipt['cache_sha256']
    write(OUT/'regression-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        scope='CONSUMED_SECONDARY_REGRESSION',selection_allowed=False,
        primary_evaluation_seal_sha256=sha(OUT/'evaluation-seal.json')))
    ids=read(TRANSFER/'identities.json');baseline=read(TRANSFER/'baseline.json')
    old=read(TRANSFER/'predictions.json');old_r=read(R_SOURCE/'predictions.json')
    observations=read(TRANSFER/'observation-seal.json')
    for name in ('identities.json','baseline.json','observations.npz'):
        assert sha(TRANSFER/name)==observations['hashes'][name]
    with np.load(TRANSFER/'observations.npz',allow_pickle=False) as data:
        inputs=model.build_inputs(np.load(TRANSFER/'features/rgb_features.npy'),data['ranges'])
    head=model.load_head(OUT/'fit/head_last.pt');model._precision()
    device,backend=model._select(head,torch.from_numpy(inputs[:64]),OUT/'regression-backend.json')
    head.to(device);start=time.perf_counter();logits=model.predict(head,inputs).astype(float)
    assert len(logits)==len(ids)==len(old)==len(old_r)==1152
    selected=read(OUT/'selection.json')['arms']
    a=np.array([r['current'] for r in baseline],bool)
    b=np.array([r['logit'] for r in old],float)
    flags=dict(A_current=a,B_frozen_current=np.array([r['flags']['C_current'] for r in old]),
        R_frozen_current=np.array([r['flags']['R_current'] for r in old_r]),
        B_control_current=a|(b>=selected['B_control']['threshold']),
        N_current=a|(logits>=selected['N']['threshold']))
    for arm in tuple(flags):
        flags[arm.replace('_current','_hold')]=held(flags[arm],ids)
    predictions=[]
    for i,identity in enumerate(ids):
        assert identity['index']==baseline[i]['index']==old[i]['index']==old_r[i]['index']==i
        assert all(bool(flags['A_'+s][i])==old[i]['flags']['A_'+s]==old_r[i]['flags']['A_'+s] for s in ('current','hold'))
        assert bool(flags['B_frozen_hold'][i])==old[i]['flags']['C_hold']
        assert bool(flags['R_frozen_hold'][i])==old_r[i]['flags']['R_hold']
        predictions.append(dict(index=i,id=identity['id'],flags={k:bool(flags[k][i]) for k in ARMS},
            current_unknown={k:baseline[i]['unknown'] for k in ARMS},logit=float(logits[i])))
    np.save(OUT/'regression-logits.npy',logits)
    write(OUT/'regression-predictions.json',predictions)
    write(OUT/'regression-inference-receipt.json',dict(backend=backend,elapsed_s=time.perf_counter()-start,
        feature_cache_sha256=receipt['cache_sha256'],threshold=selected['N']['threshold'],retuning=False))
    seal('regression-prediction-seal.json',['regression-logits.npy','regression-predictions.json',
        'regression-inference-receipt.json','regression-backend.json'])
    metadata=read(TRANSFER/'evaluator/metadata.json');truth=read(TRANSFER/'evaluator/transfer-labels.json')
    rows=[]
    for p,m,y in zip(predictions,metadata,truth):
        assert p['index']==m['index']==y['index'] and p['id']==m['id']
        rows.append(dict(**m,truth=y['truth'],flags=p['flags'],current_unknown=p['current_unknown']))
    assert len(rows)==1152
    result=metrics(rows)
    write(OUT/'regression-metrics.json',result)
    write(OUT/'regression-costs.json',costs(logits,selected['N']['threshold'],[r['truth'] for r in rows],metadata,a))
    seal('regression-evaluation-seal.json',['regression-metrics.json','regression-costs.json'])
    print('CONSUMED_REGRESSION_COMPLETE',flush=True)


if __name__=='__main__':
    run()
