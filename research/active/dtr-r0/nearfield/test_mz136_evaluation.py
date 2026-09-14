"""Decision checks: held-out labels cannot select thresholds; misses stay visible."""
import copy
import unittest
import numpy as np
from evaluate_mz136_corridor_pair import choose,score,retention,flags_at_threshold,pair_metrics


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(id=str(i),episode_id='dev' if i<4 else 'test',time_s=(i%4)*.25) for i in range(8)]
        es=[dict(family='fixture') for _ in rows]
        y=np.array([False,True,True,False]*2)
        return rows,es,y

    def test_no_test_outcome_threshold_influence(self):
        rows,es,y=self.fixture();base=y.copy();s=np.array([-2.,2.,1.,-1.,4.,-4.,0.,5.])
        first,_=choose(rows,es,y,s,list(range(4)),base)
        other=y.copy();other[4:]=~other[4:];s[4:]*=-100;base[4:]=~base[4:]
        second,_=choose(rows,es,other,s,list(range(4)),base)
        self.assertEqual(first,second)

    def test_missed_event_fails_even_if_remaining_delay_zero(self):
        rows,es,y=self.fixture();base=score(rows,es,y,y,list(range(8)))
        p=y.copy();p[5:7]=False
        candidate=score(rows,es,y,p,list(range(8)))
        self.assertEqual(candidate['events']['positive_segments'],2)
        self.assertEqual(candidate['events']['missed_positive_segments'],1)
        self.assertEqual(candidate['events']['max_detected_delay_s'],0.)
        self.assertFalse(retention(candidate,base)['pass_retention'])

    def test_each_event_timing_not_only_maximum(self):
        rows,es,y=self.fixture();base=score(rows,es,y,y,list(range(8)))
        candidate=copy.deepcopy(base)
        candidate['event_details'][0]['first_alert_s']+=.5
        self.assertFalse(retention(candidate,base)['incumbent_events_and_timing_retained'])

    def test_endpoint_threshold_preserves_pair_order(self):
        s=np.array([1000.,999.]);y=np.array([True,False]);pairs=[dict(a=0,b=1)]
        for t in (0.,1.):
            flags=flags_at_threshold(s,t)
            self.assertTrue((flags==(t==0)).all())
            m=pair_metrics(y,s,flags,pairs)
            self.assertEqual(m['order_correct'],1)
            self.assertEqual(m['both_correct'],0)


if __name__=='__main__':unittest.main()
