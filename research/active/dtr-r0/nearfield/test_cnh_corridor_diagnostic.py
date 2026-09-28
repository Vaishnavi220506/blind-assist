import unittest
import contextlib
import io
import json
import time
import numpy as np
import cnh_corridor_diagnostic as D


class MotionTests(unittest.TestCase):
    def test_old_hard_arm_unchanged(self):
        rng=np.random.default_rng(5)
        h=rng.uniform(0,20,(8,8,8,16));a=np.ones((8,8,8));b=np.ones((8,8,16))
        poses=np.repeat(np.eye(4)[None],8,0);poses[:,2,3]=np.linspace(-.8,0,8)
        q=np.eye(4)
        expected,coverage=D.old.point_score(h,a,b,poses,q)
        actual,cov,counts=D.point_score(h,a,b,poses,q,True)
        np.testing.assert_allclose(actual,expected,rtol=1e-12)
        np.testing.assert_allclose(cov,coverage)
        self.assertEqual(counts.shape,(2,9))

    def test_partial_reclaims_field_of_view(self):
        poses=np.repeat(np.eye(4)[None],8,0);poses[:4,0,3]=4
        h=np.ones((8,8,8,16));a=np.ones((8,8,8));b=np.zeros((8,8,16))
        hard,hcov,_=D.point_score(h,a,b,poses,np.eye(4),True)
        soft,scov,counts=D.point_score(h,a,b,poses,np.eye(4))
        self.assertTrue((scov>hcov).all())
        self.assertTrue((soft>hard).all())
        self.assertTrue((counts[:,1:8].sum(1)>0).all())

    def test_split_disjoint_and_size(self):
        s={k:set(v) for k,v in D.SPLITS.items()}
        self.assertEqual([len(s[k]) for k in ['train','calib','evaluation']],[96,24,48])
        self.assertFalse(s['train']&s['calib'] or s['calib']&s['evaluation'] or s['train']&s['evaluation'])
        self.assertFalse(set(range(18))&set.union(*s.values()))

    def test_evaluator_retains_negative_only_clusters(self):
        original=D.OUT
        target=original.parent/'cnh-corridor-evaluator-tests'/str(time.time_ns())
        folder=target/'predictions';folder.mkdir(parents=True)
        rng=np.random.default_rng(18)
        try:
            D.OUT=target
            for u in D.SPLITS['calib']+D.SPLITS['evaluation']:
                family=np.array(['boundary']*6+['mixed_surface']*6+['sidewall']*6+['general']*4)
                margins=np.array([-.12,-.045,-.015,.015,.045,.12]*3+[-.22,-.18,.20,.30])
                groups=np.full(22,u%2);labels=np.zeros((22,2),bool)
                labels[:,u%2]=margins<0
                score=labels.astype(float)*2-1+rng.uniform(-.1,.1,(22,2))
                np.savez(folder/f'unit{u}.npz',labels=labels,scores=np.repeat(score[:,None,:],6,1),family=family,margin=margins,group=groups,
                         coverage=np.ones((22,2)),hardcov=np.ones((22,2)),seen_counts=np.zeros((22,2,9),int))
            with contextlib.redirect_stdout(io.StringIO()):D.evaluate()
            result=json.loads((target/'results.json').read_text())
            self.assertTrue(all(c['paired_units']==48 for c in result['comparisons']))
            self.assertTrue(all(c['ber_delta']==0 for c in result['comparisons']))
            self.assertFalse(any(c.get('advance',False) for c in result['comparisons']))
            self.assertEqual(len((target/'sample_ledger.jsonl').read_text().splitlines()),2112)
            self.assertTrue(all(r['positive']+r['negative']==r['tp']+r['fp']+r['fn']+r['tn'] for r in result['rows']))
            self.assertTrue(all(r['ap'] is None for r in result['rows'] if not r['positive'] or not r['negative']))
        finally:D.OUT=original


if __name__=='__main__':unittest.main()
