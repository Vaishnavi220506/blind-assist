import unittest
import numpy as np
from appearance_feature_location import transplant,allocate


class MediationTests(unittest.TestCase):
    def test_disjoint_partition_preserves_all_sensor_inputs_and_original_arrays(self):
        ref=np.zeros((2,220,8,8));alt=np.ones_like(ref);alt[:,216:]=17
        a=transplant(ref,alt,'center');b=transplant(ref,alt,'outer')
        np.testing.assert_array_equal(a[:,:216]+b[:,:216],alt[:,:216])
        np.testing.assert_array_equal(a[:,216:],ref[:,216:])
        np.testing.assert_array_equal(b[:,216:],ref[:,216:])
        self.assertEqual(ref.sum(),0);self.assertEqual(alt[0,219,0,0],17)

    def test_interacting_function_allocation_is_complete_and_symmetric(self):
        # f(x,y)=3x+5y+7xy, each binary; interaction shared equally.
        a=allocate(0,15,3,5)
        self.assertEqual(a,dict(center=6.5,outer=8.5,interaction=7))
        b=allocate(0,15,5,3)
        self.assertEqual(a['center'],b['outer'])

    def test_identical_features_are_exact_identity(self):
        x=np.arange(220*64,dtype=np.float32).reshape(1,220,8,8)
        for region in ('center','outer'):np.testing.assert_array_equal(transplant(x,x,region),x)


if __name__=='__main__':unittest.main()
