"""Control-mask ordering, saturation and deterministic registration checks."""
import unittest
import numpy as np
from cnh_segment_controls_masks import sequence_masks, coverage_stats, CONDITIONS
from cnh_segment_learned_masks import rng_for, shift_mask, coverage


class ControlMaskTests(unittest.TestCase):
    def setUp(self):
        self.objects = [dict(id=1, category='SMALL'), dict(id=100, category='BACKGROUND')]
        self.pool = [np.array([[0, 0]])]

    def test_background_includes_all_hits_but_not_missing_rays(self):
        oid = np.full((2, 8, 8, 256), -1, int)
        oid[:, 2, 3, :128] = 100
        oid[:, 4, 5, :64] = 1
        result = sequence_masks(oid, self.objects, 96, 0, 'v2', self.pool)
        self.assertEqual(set(result), set(CONDITIONS))
        self.assertEqual(result['IDEAL'][0].sum(), .25)
        self.assertEqual(result['BG'][0].sum(), .75)
        self.assertEqual(result['BG'][0, 2, 3], .5)
        self.assertFalse((result['FA5'] < result['IDEAL']).any())
        self.assertFalse((result['FA10'] < result['FA5']).any())
        self.assertFalse((result['BG_FA10_SHIFT05'] < result['BG_FA5_SHIFT05']).any())
        for values in result.values():
            self.assertEqual(values.dtype, np.float16)
            self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_full_background_saturates_before_whole_union_shift(self):
        oid = np.full((3, 8, 8, 256), 100, int)
        result = sequence_masks(oid, self.objects, 96, 2, 'v2', self.pool)
        self.assertTrue((result['BG'] == 1).all())
        self.assertFalse(result['IDEAL'].any())
        angle = rng_for('v2', 96, 2, -1, 'registration').uniform(0, 2*np.pi)
        dx, dy = np.rint(8*np.array([np.cos(angle), np.sin(angle)])).astype(int)
        expected = coverage(shift_mask(np.ones((128, 128), bool), dx, dy))
        for condition in ('BG_FA5_SHIFT05', 'BG_FA10_SHIFT05'):
            for frame in result[condition]:
                np.testing.assert_array_equal(frame, expected)
        stats = coverage_stats(result)
        self.assertEqual(stats['BG']['fraction_frames_fully_saturated'], 1.)
        self.assertLess(stats['BG_FA5_SHIFT05']['mean_coverage'], 1.)

    def test_determinism_and_irrelevant_metadata(self):
        oid = np.full((2, 8, 8, 256), 100, int); oid[:, 3, 3, :20] = 1
        a = sequence_masks(oid, self.objects, 97, 1, 'v2', self.pool)
        metadata = [dict(obj, distance=-999, positive=True) for obj in self.objects]
        b = sequence_masks(oid, metadata, 97, 1, 'v2', self.pool)
        for key in CONDITIONS:
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == '__main__':
    unittest.main()
