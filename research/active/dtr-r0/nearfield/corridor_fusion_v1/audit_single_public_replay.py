"""Authenticate the single public path on consumed input, without scoring fit data."""
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'artifacts.local/work'
HOME=WORK/'corridor-public-single-20260917'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    torch.set_num_threads(4)
    got=[json.loads(s) for s in (HOME/'consumed-public-replay.jsonl').read_text().splitlines()]
    original=read(WORK/'corridor-depth-confirmation-20260917/evaluation-v1/predictions.json')
    assert len(got)==len(original)==288
    for a,b in zip(got,original):
        assert a['id']==b['id'] and a['A_score']==b['A_score'] and a['A']==b['A_alert']
    prep=WORK/'corridor-public-positive-v2-20260917/preparation'
    data=np.load(prep/'public-tokens.npz');meta=read(prep/'metadata.json')
    ix=np.array([i for i,m in enumerate(meta) if m['cohort']=='new'])
    assert [meta[i]['id'] for i in ix]==[g['id'] for g in got]
    token=data['tokens'][ix,:128];v=data['valid'][ix,:128]
    units=np.array([4,2,3],np.float32);low=np.array([.2,-.3,.4],np.float32);high=np.array([3.6,.3,2.05],np.float32)
    center=token[:,:,5:8]*units;box=token[:,:,8:14].reshape(288,128,3,2)*units[:,None]
    x=np.concatenate([token,np.minimum(center-low,high-center),np.minimum(box[:,:,:,1]-low,high-box[:,:,:,0]),
        np.minimum(box[:,:,:,0]-low,high-box[:,:,:,1])],-1)
    state=torch.load(HOME/'bundle/positive.pt',weights_only=True,map_location='cpu')['state_dict']
    scores=[];slots=[]
    with torch.inference_mode():
        for i in range(288):
            valid=torch.tensor(v[i:i+1]);z=torch.tensor(x[i:i+1])
            z=(torch.where(valid[...,None],z,state['mean'])-state['mean'])/state['scale']
            for layer in [0,2,4]:
                z=torch.nn.functional.linear(z,state[f'net.{layer}.weight'],state[f'net.{layer}.bias'])
                if layer!=4:z=torch.relu(z)
            z=z.squeeze(-1).masked_fill(~valid,-30.).numpy()[0]
            scores.append(float(z.max()));slots.append(int(z.argmax()) if v[i].any() else None)
    np.testing.assert_array_equal(scores,[g['positive_score'] for g in got])
    assert slots==[g['max_return_slot'] for g in got]
    control=pickle.loads((HOME/'bundle/A-retrained.pkl').read_bytes())
    features=np.load(HOME/'a-control/features.npz')['base'][ix]
    np.testing.assert_array_equal(control.predict_proba(features)[:,1],[g['control_score'] for g in got])
    result=dict(status='PASS',frames=288,frozen_A_score_bitwise=True,
        independent_single_head_scores_and_slots_bitwise=True,retrained_A_cached_score_bitwise=True,
        evaluator_scoring=False,authority='CONSUMED_INPUT_IMPLEMENTATION_PARITY_NOT_GENERALIZATION')
    (HOME/'public-path-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result)


if __name__=='__main__':
    with threadpool_limits(4):main()
