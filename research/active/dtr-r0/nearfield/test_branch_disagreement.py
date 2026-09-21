import unittest
import numpy as np
from branch_disagreement import current_rule,partition,held


class DisagreementTest(unittest.TestCase):
    def test_all_boolean_patterns_and_A_retention(self):
        b=[False,False,True,True];n=[False,True,False,True]
        for mask in range(8):
            self.assertEqual(current_rule([False]*4,b,n,mask),
                [False,bool(mask&4),bool(mask&2),bool(mask&1)])
            self.assertEqual(current_rule([True]*4,b,n,mask),[True]*4)

    def test_hold_cross_time_agreement_is_not_current_consensus(self):
        rows=[dict(clip_id='one',time_s=i*.2) for i in range(3)]
        b=[True,False,False];n=[False,True,False]
        intersection=held(current_rule([False]*3,b,n,1),rows)
        direct=held(b,rows)&held(n,rows)
        self.assertFalse(intersection.any());self.assertTrue(direct[1])

    def test_hold_does_not_cross_clip_boundary(self):
        rows=[dict(clip_id='one',time_s=0.),dict(clip_id='two',time_s=0.)]
        np.testing.assert_array_equal(held([True,False],rows),[True,False])

    def test_partition_excludes_A_and_preserves_neither_as_silence(self):
        def row(i,a,b,n,y):
            return dict(id=str(i),truth=y,flags=dict(A_current=a,B_control_current=b,N_current=n))
        got=partition([row(0,True,True,True,True),row(1,False,True,True,True),
            row(2,False,True,False,False),row(3,False,False,True,True),
            row(4,False,False,False,True)],'current')
        self.assertEqual(got['common']['positive_ids'],['1'])
        self.assertEqual(got['B_only']['negative_ids'],['2'])
        self.assertEqual(got['N_only']['positive_ids'],['3'])
        self.assertEqual(got['neither']['positive_ids'],['4'])


if __name__=='__main__':
    unittest.main()
