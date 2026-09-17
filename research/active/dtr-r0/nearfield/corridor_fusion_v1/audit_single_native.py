"""Independent native geometry and saved public-forward confirmation audit."""
import argparse
import json
from pathlib import Path
import hashlib
import sys
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from single_positive_inference import SinglePositiveSystem
from public_return_tokens import encode_tokens
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mz171_return_labels import make_witness_labels

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-single-20260917'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(cap):
    torch.set_num_threads(4)
    out=HOME/'confirmation';receipt=read(cap/'receipt.json')
    for n,h in receipt['hashes'].items():assert sha(cap/n)==h,n
    rows,es=lines(cap/'raw.jsonl'),lines(cap/'evaluator.jsonl')
    pred,cases=read(out/'predictions.json'),read(out/'cases.json')
    system=SinglePositiveSystem(HOME/'bundle');yaw=0.;episode=None
    selected=[];zero_frames=0
    for row,e,p,c in zip(rows,es,pred,cases):
        assert row['id']==e['id']==p['id']==c['id']
        origin=np.array(e['body_origin_m'])
        margins=[]
        for obj in e['native_bounds']:
            low=np.array(obj['center_m'])-np.array(obj['extent_m'])-origin
            high=np.array(obj['center_m'])+np.array(obj['extent_m'])-origin
            if high[0]>=.2 and low[0]<=3.6 and high[2]>=.4 and low[2]<=2.05:
                margins.append(min(high[1]+.3,.3-low[1]))
        margin=max(margins) if margins else -np.inf
        truth=margin>=0
        state='positive' if margin>=.05 else 'negative' if margin<-.05 else 'boundary'
        assert truth==c['truth'] and state==c['stratum']
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id']
        actual=system.predict(row,cv2.imread(str(cap/row['rgb_path'])),yaw)
        for key in ['A_score','A','positive_score','positive','alert','max_return_slot','usable_tof_returns','control_score','control']:
            assert actual[key]==p[key],(row['id'],key)
        token=encode_tokens(row,yaw);lab=make_witness_labels(row,e)
        witness=bool(((lab['target'][:128]>0)&lab['known'][:128]).any())
        assert witness==c['sampled_witness']
        if not token['valid'][:128].any():
            zero_frames+=1;assert p['alert']==p['A']
        if p['alert'] and not p['A']:
            slot=p['max_return_slot'];assert token['valid'][slot]
            selected.append(dict(id=row['id'],truth=bool(truth),stratum=state,
                max_return_has_native_witness=bool(lab['known'][slot] and lab['target'][slot]>0)))
    assert len(rows)==len(es)==len(pred)==len(cases)==288
    result=dict(status='PASS',frames=288,all_public_scores_bitwise_replayed=True,
        native_truth_and_5cm_strata_independent=True,zero_return_frames=zero_frames,
        zero_fallback_retained=True,added_public_alerts=selected,
        prediction_seal_sha256=sha(out/'prediction-seal.json'),recipe_sha256=sha(HOME/'recipe-freeze.json'))
    (out/'native-public-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(4):main(args.capture)
