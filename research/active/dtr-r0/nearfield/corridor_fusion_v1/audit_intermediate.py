"""Independent saved-score arithmetic and artifact integrity; no fitting/inference."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from run_e1 import read,sha,write


def audit(out):
    done=read(out/'completion.json');assert done['status']=='PASS'
    for name,key in [('summary.json','summary_sha256'),('predictions.csv','predictions_sha256'),('comparison.mp4','video_sha256'),('prediction-seal.json','prediction_seal_sha256')]:
        assert sha(out/name)==done[key]
    seal=read(out/'prediction-seal.json');selection=read(out/'selection-seal.json')
    assert sha(out/'report-scores.npy')==seal['scores_sha256']
    assert sha(out/'selection-seal.json')==seal['selection_sha256']
    assert sha(out/'C-model.pkl')==selection['head_sha256']
    assert sha(out/'pca.pkl')==selection['pca_sha256']
    feature=read(out/'feature-seal.json');frozen=read(out/'freeze.json')
    assert feature['pca_fit_ids']==frozen['ids']['train']
    groups=[set(frozen['groups'][k]) for k in ('train','dev','report')]
    assert not any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2)))
    for k,h in feature['features'].items():
        assert sha(out/(k+'-features.npz'))==h
        data=np.load(out/(k+'-features.npz'))
        assert data['intermediate'].shape==(len(frozen['ids'][k]),816)
        assert list(data['ids'])==frozen['ids'][k]
    rows=list(csv.DictReader((out/'predictions.csv').open(encoding='utf-8')))
    assert [r['id'] for r in rows]==frozen['ids']['report']
    y=np.array([int(r['truth']) for r in rows],bool);score=np.load(out/'report-scores.npy')
    assert np.array_equal(score,np.array([float(r['score']) for r in rows]))
    summary=read(out/'summary.json')
    for arm,key in [('C_f1','dev_best'),('C_recall95','dev_recall95')]:
        flags=score>=selection['selection'][key]['threshold']
        assert np.array_equal(flags,np.array([int(r[arm]) for r in rows],bool))
        rep=summary['reports'][arm]
        for k,v in dict(TP=y&flags,FP=~y&flags,FN=y&~flags,TN=~y&~flags,UNKNOWN=~flags).items():assert rep['metrics'][k]==int(v.sum())
        for family,m in rep['families'].items():
            ix=np.array([r['family']==family for r in rows])
            assert m['TP']==int((y&flags&ix).sum()) and m['FP']==int((~y&flags&ix).sum())
    cap=cv2.VideoCapture(str(out/'comparison.mp4'));n=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(480,640,3);n+=1
    finally:cap.release()
    assert n==24
    result=dict(status='PASS',frames=len(rows),video_frames=n,groups_disjoint=True,train_only_pca=True,
                scores_thresholds_counts_match=True,feature_dimensions=816,no_training_or_inference=True)
    write(out/'delivery-audit.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);audit(p.parse_args().output)
