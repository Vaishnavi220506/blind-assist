"""Saved-output audit of conditional policy, fitting isolation and real call count."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from run_e1 import read,write,sha
from specialist_gate import policy


def audit(out):
    done=read(out/'completion.json');assert done['status']=='PASS'
    for file,key in [('summary.json','summary_sha256'),('prediction-seal.json','prediction_seal_sha256'),
                     ('predictions.csv','predictions_sha256'),('comparison.mp4','video_sha256')]:assert sha(out/file)==done[key]
    frozen=read(out/'freeze.json');selected=read(out/'selection-seal.json');seal=read(out/'prediction-seal.json')
    assert sha(out/'selection-seal.json')==seal['selection_sha256']
    assert sha(out/'report-predictions.npz')==seal['sha256']
    assert sha(out/'gate.pkl')==selected['gate_sha256']
    groups=[{r.rsplit('_',2)[0] for r in frozen[k]} for k in ('train_ids','dev_ids','report_ids')]
    assert not any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2)))
    p=np.load(out/'report-predictions.npz');a=p['A']>=frozen['A_threshold']
    final,invoked,veto=policy(a,p['gate'],selected['selected']['threshold'],p['C'],frozen['C_threshold'])
    assert np.array_equal(final,p['final']) and np.array_equal(invoked,p['invoked']) and np.array_equal(veto,p['veto'])
    assert not (final&~a).any()
    rows=list(csv.DictReader((out/'predictions.csv').open(encoding='utf-8')))
    assert [r['id'] for r in rows]==frozen['report_ids']
    y=np.array([int(r['truth']) for r in rows],bool)
    summary=read(out/'summary.json')
    for arm,rep in summary['reports'].items():
        flags=np.array([int(r[arm]) for r in rows],bool)
        for k,v in dict(TP=y&flags,FP=~y&flags,FN=y&~flags,TN=~y&~flags,UNKNOWN=~flags).items():assert rep['metrics'][k]==int(v.sum())
    timing=read(out/'latency-samples.json')
    assert len(timing)==288 and [r['id'] for r in timing]==frozen['report_ids']
    assert sum(r['invoked'] for r in timing)==int(invoked.sum())==done['actual_token_calls']==summary['actual_calls']
    rod=np.array([r['family']=='near_rod_farwall' for r in rows]);oracle=a&~(rod&(p['C']<frozen['C_threshold']))
    assert np.array_equal(oracle,np.array([int(r['rod_family_oracle']) for r in rows],bool))
    assert (int((y&oracle).sum()),int((~y&oracle).sum()),int((y&~oracle).sum()))==(137,17,7)
    cap=cv2.VideoCapture(str(out/'comparison.mp4'));n=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(480,640,3);n+=1
    finally:cap.release()
    assert n==288
    result=dict(status='PASS',frames=288,video_frames=n,actual_calls=int(invoked.sum()),only_veto=True,
                split_ids_disjoint=True,seals_counts_policy_latency_match=True,oracle='PRIVILEGED_DIAGNOSTIC_ONLY',no_new_fit_or_inference=True)
    write(out/'delivery-audit.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);audit(p.parse_args().output)
