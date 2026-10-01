import unittest
import numpy as np
import cnh_corridor_stopping as M
class Tests(unittest.TestCase):
    def box(self,lo,hi):return dict(lo=lo,hi=hi,rho=.5)
    def test_ground_not_collision(self):
        d,ids=M.first_collision([self.box([-8,1.65,-8],[8,1.8,9])]);self.assertTrue(np.isinf(d));self.assertEqual(ids,[])
    def test_continuous_thin_obstacle(self):
        d,ids=M.first_collision([self.box([-.1,.5,1.123],[.1,.6,1.124])]);self.assertAlmostEqual(d,1.123-.15-M.START)
    def test_lateral_clear_and_boundary(self):
        self.assertTrue(np.isinf(M.first_collision([self.box([.3,.5,1],[.4,.6,1.1])])[0]))
        self.assertTrue(np.isfinite(M.first_collision([self.box([.299,.5,1],[.4,.6,1.1])])[0]))
    def test_query_volume_rotation(self):
        b=[self.box([.1,.5,1],[.2,.6,1.1])];p=np.eye(4)[None];y=M.query_labels(b,p);self.assertEqual(y[0,3],1)
        p[0,:3,:3]=M.S.ry(90);self.assertEqual(M.query_labels(b,p)[0,3],0)
    def test_zero_yaw_same_labels(self):
        s=M.scenes(6102)[0];np.testing.assert_array_equal(M.query_labels(s['boxes'],s['head']),M.query_labels(s['boxes'],s['travel']))
    def test_threshold_matches_brute(self):
        scores=np.random.default_rng(0).normal(size=(20,34));vals=np.unique(scores);expected=float(np.nextafter(vals[-1],np.inf))
        for v in vals[::-1]:
            if M.episodes(scores,v)[0]/(scores.size*.2/60)<=2:expected=float(v)
            else:break
        self.assertEqual(M.calibrate(scores),expected)
    def test_always_never_reaction(self):
        dist=np.array([5.,np.inf]);s=np.ones((2,34));a=M.task_metrics(s,.5,dist,.6)[0];n=M.task_metrics(s,2,dist,.6)[0]
        self.assertEqual(a['collisions'],0);self.assertEqual(a['unnecessary'],1);self.assertEqual(n['collisions'],1);self.assertEqual(n['unnecessary'],0)
    def test_start_margin_sufficient(self):
        for s in M.scenes(6100):
            d,_=M.first_collision(s['boxes'])
            if np.isfinite(d):self.assertGreater(d,M.SPEED*(1.4+1.5)+M.SPEED**2/3)
if __name__=='__main__':unittest.main()
