import unittest
import numpy as np
from spatial_complement_diagnostic import zero_fp_cutoff,grouped_screen,hold


class ComplementTests(unittest.TestCase):
    def test_nextafter_does_not_round_back_to_float32_negative(self):
        values=np.array([1.,7.6612162590026855],np.float32).astype(np.float64)
        cut=zero_fp_cutoff(values)
        self.assertGreater(cut,values.max())
        self.assertFalse((values>=cut).any())
        self.assertEqual(zero_fp_cutoff([]),0.)
        self.assertEqual(zero_fp_cutoff([-2.,-1.]),0.)

    def test_group_own_negative_never_sets_its_cutoff(self):
        proposed,cuts,records=grouped_screen([1.,100.,2.,3.],[False]*4,[False]*4,
            [True]*4,['a','a','b','b'])
        self.assertLess(cuts[0],4.)
        self.assertGreater(cuts[2],100.)
        self.assertTrue(proposed[1])  # Honest unseen-group false positive.
        self.assertEqual(records[0]['calibration_groups'],['b'])
        self.assertEqual(records[1]['calibration_groups'],['a'])

    def test_only_eligible_A_negative_negatives_calibrate(self):
        _,cuts,_=grouped_screen([1.,100.,200.,300.,2.],[False,True,False,False,False],
            [False,False,True,False,False],[True,True,True,False,True],['a','b','b','b','b'])
        self.assertGreater(cuts[0],2.)
        self.assertLess(cuts[0],3.)

    def test_union_and_hold_preserve_baseline_without_recursive_extension(self):
        meta=[dict(clip_id='a',time_s=.2*i) for i in range(4)]+[dict(clip_id='b',time_s=0.)]
        a=np.array([False,True,False,False,False]);b=np.array([True,False,True,False,False]);c=a|b
        self.assertTrue(np.all(~hold(a,meta)|hold(c,meta)))
        self.assertFalse(hold(c,meta)[4])
        np.testing.assert_array_equal(hold([True,False,False,False,False],meta),[True,True,False,False,False])


if __name__=='__main__':unittest.main()
