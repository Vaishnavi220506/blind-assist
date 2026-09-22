import unittest
from return_lineage_core import classify


class LineageTest(unittest.TestCase):
    def test_one_target_sample_is_not_discarded(self):
        d=dict(target=3);s=dict(target=1,target_eligible=1)
        w=dict(target_count=0,count=30,min_m=7.,corridor_count=0)
        self.assertEqual(classify(d,s,3,dict(observed=True),w,2.),'B_RETURN_WINNER_FLIP_CAPABLE')

    def test_transport_needs_projected_and_visible_absence(self):
        d=dict(target=0);s=dict(target=0,target_eligible=0)
        w=dict(target_count=0,count=30,min_m=7.,corridor_count=0)
        self.assertEqual(classify(d,s,0,dict(observed=True),w,None),'A_ABSENT')
        self.assertTrue(classify(d,s,1,dict(observed=True),w,None).startswith('C_'))

    def test_dense_target_sampled_out_is_not_transport(self):
        self.assertEqual(classify(dict(target=3),dict(target=0,target_eligible=0),1,dict(observed=True),
            dict(target_count=0,count=20,min_m=6.,corridor_count=0),None),'C_NATIVE_TARGET_MISSED_BY_SAMPLING')

    def test_mixed_and_missing_not_background_victory(self):
        self.assertEqual(classify({}, {}, 1,dict(observed=False,reason='SIMULATED_DROPOUT'),None,None),'C_SIMULATED_DROPOUT')
        self.assertEqual(classify({}, {}, 1,dict(observed=True),dict(target_count=1,count=2),2),'C_MIXED_WINNER')


if __name__=='__main__':unittest.main()
