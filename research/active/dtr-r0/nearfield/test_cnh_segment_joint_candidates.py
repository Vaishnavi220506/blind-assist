"""Focused angular-mask tests; no experiment outcomes or GPU required."""
import unittest
import numpy as np
import cnh_segment_joint_candidates as c
from cnh_candidate_quality_readout import SEED, fake_footprints, to_pixels


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.oid = np.full((2, 8, 8, 256), 100, dtype=np.int16)
        self.oid[:, 3, 4, 100:110] = 1
        self.objects = [dict(id=1, category='LOW'), dict(id=2, category='HIGH'),
                        dict(id=100, category='BACKGROUND')]
        self.pool = [np.array([[0, 0], [0, 1]])]

    def test_subray_morphology_and_fov_edges(self):
        mask = np.zeros((128, 128), bool)
        mask[32, 32] = True
        self.assertEqual(c.morph(mask, True).sum(), 9)
        self.assertFalse(c.morph(mask, False).any())
        mask[:] = False
        mask[0, 0] = True
        self.assertEqual(c.morph(mask, True).sum(), 4)
        self.assertFalse(c.morph(mask, True)[-1].any())
        self.assertEqual(c.morph(np.ones_like(mask), False).sum(), 126**2)

    def test_coverage_uses_solid_angle_not_uniform_area(self):
        mask = np.zeros((128, 128), bool)
        mask[0:16, 0:16] = True
        coverage = c.mask_coverage(mask)
        self.assertEqual(coverage.dtype, np.float32)
        self.assertEqual(coverage.sum(), 1.)
        mask[:] = False
        mask[0, 0] = True
        self.assertNotAlmostEqual(float(c.mask_coverage(mask)[0, 0]), 1/256, places=6)

    def test_rng_matches_previous_and_fake_pool(self):
        out = c.candidate_frames(self.oid, self.objects, 96, 0, self.pool)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 96, 0, 0, 1]))
        angle, drop = rng.uniform(0, 2*np.pi), rng.random()
        xy, _ = c.angular_geometry()
        expected = to_pixels(xy[self.oid[0] == 1], np.array([np.cos(angle), np.sin(angle)]), .5)
        np.testing.assert_array_equal(out['SHIFT05'][0][0]['mask'], expected)
        self.assertEqual(len(out['DROP20'][0]), int(drop >= .2))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 96, 0, 0, 99999]))
        fake, _ = fake_footprints(self.pool, rng, 10)
        self.assertEqual(out['FA1'][0][-1]['id'], -1)
        np.testing.assert_array_equal(out['FA1'][0][-1]['coverage'], fake[0])
        self.assertEqual(out['FA1'][0][-1]['mask'].sum(), 512)

    def test_only_id_and_category_enter_no_query_or_truth(self):
        a = c.candidate_frames(self.oid, self.objects, 96, 0, self.pool)
        objects = [dict(o, distance_m=-999, positive=True, query='arbitrary') for o in self.objects]
        b = c.candidate_frames(self.oid, objects, 96, 0, self.pool)
        for condition in c.CONDITIONS:
            for af, bf in zip(a[condition], b[condition]):
                self.assertEqual([x['id'] for x in af], [x['id'] for x in bf])
                for ac, bc in zip(af, bf):
                    self.assertEqual(set(ac), {'id', 'coverage', 'mask'})
                    np.testing.assert_array_equal(ac['mask'], bc['mask'])
        self.assertEqual([x['id'] for x in a['IDEAL'][0]], [1])
        direct = (self.oid[0] == 1).any(-1)
        np.testing.assert_array_equal(a['IDEAL'][0][0]['coverage'] > 0, direct)

    def test_shape_and_scope_guard(self):
        for unit in (95, 143, 192):
            with self.assertRaises(ValueError):
                c.candidate_frames(self.oid, self.objects, unit, 0, self.pool)
        with self.assertRaises(ValueError):
            c.candidate_frames(self.oid[..., :16], self.objects, 96, 0, self.pool)


if __name__ == '__main__':
    unittest.main()
