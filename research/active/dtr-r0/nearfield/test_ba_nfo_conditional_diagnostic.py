"""Focused guards for exact threshold and spatial diagnostic semantics."""
import unittest
import numpy as np
from ba_nfo_conditional_diagnostic import curve, ap, distance


class ConditionalDiagnosticTests(unittest.TestCase):
    def test_ties_are_atomic_and_monotone_transform_preserves_curve(self):
        score = np.array([.9,.8,.8,.1])
        truth = np.array([True,False,True,False])
        c = curve(score,truth)
        np.testing.assert_array_equal(c['tp'],[0,1,2,2])
        np.testing.assert_array_equal(c['fp'],[0,0,1,2])
        self.assertAlmostEqual(ap(c),5/6)
        shifted = curve(score*.5+.2,truth)
        for key in ['tp','fp','recall','precision','iou']:
            np.testing.assert_array_equal(c[key],shifted[key])

    def test_distance_missing_rescue_and_euclidean_geometry(self):
        mask = np.zeros((9,9),bool)
        self.assertTrue(np.isinf(distance(mask)).all())
        mask[4,4] = True
        d = distance(mask)
        self.assertAlmostEqual(float(d[7,8]),5.)
        self.assertEqual(d[4,4],0)

    def test_iou_identity(self):
        c = curve(np.array([.8,.4,.3,.2]),np.array([False,True,False,True]))
        p,r = c['precision'][1:],c['recall'][1:]
        expected = np.divide(p*r,p+r-p*r,out=np.zeros(len(p)),where=p+r-p*r>0)
        np.testing.assert_allclose(c['iou'][1:],expected)


if __name__ == '__main__':
    unittest.main()
