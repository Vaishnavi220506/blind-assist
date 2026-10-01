import unittest
import numpy as np
from cnh_proposal_observation import perturb, fit_response, generator_hash


class ResponseTests(unittest.TestCase):
    def test_resume_identity_catches_generator_change(self):
        self.assertNotEqual(generator_hash('def perturb():\n return 1'),generator_hash('def perturb():\n return 2'))

    def test_resume_identity_allows_evaluator_only_change(self):
        self.assertEqual(generator_hash('def perturb():\n return 1\ndef evaluate():\n return 1'),
                         generator_hash('def perturb():\n return 1\ndef evaluate():\n return 2'))

    def test_nominal_preserves_observation(self):
        h=np.arange(64*16,dtype=np.float32).reshape(1,8,8,16)
        a=np.full((1,8,8),4.,np.float32)
        hh,aa=perturb(h,a,('nominal',1,0,1),0,0)
        np.testing.assert_array_equal(hh,h)
        np.testing.assert_array_equal(aa,a)

    def test_affine_fit_recovers_known_invertible_response(self):
        h=np.random.default_rng(7).normal(20,4,(128,8,8,16))
        target=2*h+24
        f=fit_response(h,target)
        self.assertAlmostEqual(f['gain'],2)
        np.testing.assert_allclose((target-f['offset'])/f['gain'],h,atol=1e-12)

    def test_noise_has_declared_variance_and_no_offset(self):
        h=np.zeros((20000,1,1,16))
        a=np.full((20000,1,1),4.)
        hh,aa=perturb(h,a,('ambient2',1,0,2),0,0)
        self.assertLess(abs(float(hh.mean())),.1)
        self.assertLess(abs(float(hh.var())-64),1)
        np.testing.assert_array_equal(aa,8)

    def test_noise_is_paired_reproducible_and_unit_distinct(self):
        h=np.zeros((12,8,8,16)); a=np.ones((12,8,8))
        c=('ambient4',1,0,4)
        x=perturb(h,a,c,0,0)[0]
        np.testing.assert_array_equal(x,perturb(h,a,c,0,0)[0])
        self.assertFalse(np.array_equal(x,perturb(h,a,c,1,0)[0]))


if __name__=='__main__':
    unittest.main()
