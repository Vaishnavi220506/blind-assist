"""Read-only arithmetic, delivery and sealed-output verification, no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def audit(path):
    def read(name):return json.loads((path/name).read_text())
    def sha(name):return hashlib.sha256((path/name).read_bytes()).hexdigest()
    summary=read('summary.json');rows=list(csv.DictReader((path/'predictions.csv').open()))
    assert len(rows)==288 and len({r['id'] for r in rows})==288
    done=read('completion.json');assert done['status']=='PASS'
    for filename,key in [('summary.json','summary_sha256'),('predictions.csv','predictions_sha256'),('comparison.mp4','video_sha256')]:
        assert sha(filename)==done[key]
    seal=read('prediction-seal.json')
    assert sha('report-scores.npz')==seal['scores_sha256']
    assert sha('selection-seal.json')==seal['selection_sha256']
    assert summary['reports']['MZ129']['pr_auc'] is None
    y=np.array([int(r['truth']) for r in rows],bool)
    for name,report in summary['reports'].items():
        flags=np.array([int(r[name+'_alert']) for r in rows],bool)
        for key,v in dict(TP=y&flags,FP=~y&flags,FN=y&~flags,TN=~y&~flags,UNKNOWN=~flags).items():
            assert report['metrics'][key]==int(v.sum()),(name,key)
        tp=int((y&flags).sum());fp=int((~y&flags).sum());fn=int((y&~flags).sum())
        assert abs(report['f1']-2*tp/(2*tp+fp+fn))<1e-12
    for arm in ('A','B'):
        prob=np.load(path/'report-scores.npz')[arm]
        for suffix,key in [('f1','dev_best'),('recall95','dev_recall95')]:
            flags=prob>=summary['selection'][arm][key]['threshold']
            assert np.array_equal(flags,np.array([int(r[arm+'_'+suffix+'_alert']) for r in rows],bool))
    static=np.array([float(r['MZ145_score']) for r in rows])
    assert np.array_equal(static,np.load(path/'report-scores.npz')['A'])
    cap=cv2.VideoCapture(str(path/'comparison.mp4'));frames=0
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(480,1280,3)
            frames+=1
    finally:cap.release()
    assert frames==24
    recovery=read('delivery-recovery.json')
    for filename,digest in recovery['sealed'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    result=dict(status='PASS',frames=288,arms=7,video_decoded_frames=frames,
                A_equals_old_static_scores=True,sealed_predictions_unchanged=True,
                binary_reference_ranking='NOT_EVALUABLE',no_training_or_inference=True)
    (path/'delivery-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);audit(p.parse_args().output)
