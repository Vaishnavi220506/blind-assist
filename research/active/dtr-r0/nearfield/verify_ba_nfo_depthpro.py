"""Independent saved-array recount for the frozen reference, no model calls."""
import json
from collections import defaultdict

import numpy as np

from ba_nfo_depthpro import BASE, OUT, ROOT, SOURCE, sha, write


def main():
    rows=json.loads((OUT/'manifest.json').read_text())
    report=json.loads((OUT/'results.json').read_text())
    frames=json.loads((OUT/'frame-results.json').read_text())
    seal=json.loads((OUT/'prediction-seal.json').read_text())
    with np.load(BASE/'predictions.npz') as a:
        baseline=np.unpackbits(a['masks'][:,0],axis=1,bitorder='little').reshape(500,192,256).astype(bool)
    totals=defaultdict(lambda: np.zeros(4,np.int64)); unknown=0
    for index,(row,frame) in enumerate(zip(rows,frames)):
        assert row['id']==frame['id']
        assert sha(SOURCE/row['prepared'])==row['sha256']
        with np.load(SOURCE/row['prepared']) as a:
            depth=a['depth']; known=np.isfinite(depth)&(depth>0); truth=known&(depth<2)
            unknown+=int((~known).sum())
            domains={name:np.zeros_like(known) for name in ['mixed','small_foreground','far_small','pure_far','public_far','inside','far_small_separated']}
            for box,value in zip(a['boxes'],a['values']):
                y0,x0,y1,x1=box; sl=np.s_[y0:y1,x0:x1]
                n=int(known[sl].sum()); p=int(truth[sl].sum())
                domains['inside'][sl]=True
                far=np.isfinite(value) and value>=2
                if far:
                    domains['public_far'][sl]=known[sl]
                    if n and p==0:domains['pure_far'][sl]=known[sl]
                if n and 0<p<n:
                    domains['mixed'][sl]=known[sl]
                    if p/n<=.2:
                        domains['small_foreground'][sl]=known[sl]
                        if far:
                            domains['far_small'][sl]=known[sl]
                            near_depth=depth[sl][truth[sl]]
                            far_depth=depth[sl][known[sl]&~truth[sl]]
                            q90=np.quantile(near_depth,.9); q10=np.quantile(far_depth,.1)
                            if value>=2.2 and q90<=1.8 and q10>=2.2 and q10-q90>=.5:
                                domains['far_small_separated'][sl]=known[sl]
            inside=domains.pop('inside'); domains['full']=known; domains['outside']=known&~inside
            domains['far_small_other']=domains['far_small']&~domains['far_small_separated']
            domains['far_small_near_le1p8']=domains['far_small']&truth&(depth<=1.8)
            domains['far_small_near_gt1p8']=domains['far_small']&truth&(depth>1.8)
        predictions={'nfo':baseline[index]}
        for arm in ['low','native']:
            path=OUT/'predictions'/arm/f'{row["id"]}.npz'
            assert sha(path)==seal['outputs'][f'predictions/{arm}/{row["id"]}']['sha256']
            with np.load(path) as a:
                predicted=a['depth']; assert np.isfinite(predicted).all() and (predicted>0).all()
                if arm=='native':np.testing.assert_array_equal(predicted,a['native_depth'][::4,::4])
            predictions[arm]=predicted<2
        for arm,predicted in predictions.items():
            code=truth.astype(np.uint8)*2+predicted.astype(np.uint8)
            for domain,mask in domains.items():
                # 00TN,01FP,10FN,11TP: independently reconstruct joint bins.
                c=np.bincount(code[mask],minlength=4)[[3,1,2,0]]
                actual=np.array([frame['metrics'][arm][domain][k] for k in ['tp','fp','fn','tn']])
                np.testing.assert_array_equal(c,actual)
                totals[arm,domain]+=c
    assert len(rows)==len(frames)==500 and unknown==350140
    for (arm,domain),counts in totals.items():
        np.testing.assert_array_equal(counts,[report['metrics'][arm][domain][k] for k in ['tp','fp','fn','tn']])
    write(OUT/'verification.json',dict(status='PASS',frames=500,arms=3,domains=11,
        unknown=unknown,all_frame_and_aggregate_counts_exact=True,native_sampling_exact=True,
        original_baseline_preserved=True,model_calls=0,
        verifier_sha256=sha(__file__),results_sha256=sha(OUT/'results.json'),
        backend='CPU TASK_NOT_GPU_SUITABLE: saved-array integer histogram recount'))
    print('INDEPENDENT_RECOUNT_PASS 500 frames,3 arms,11 domains')


if __name__=='__main__':
    main()
