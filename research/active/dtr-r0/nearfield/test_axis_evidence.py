"""Observable-evidence invariants, not fitted policy performance."""
import unittest
import numpy as np
from audit_axis_evidence import describe, interval_state
from tof_fov45_core import boxes45
from tof_lateral_core import lateral_relation


class AxisEvidenceTest(unittest.TestCase):
    def test_missing_is_not_outside(self):
        d = describe(boxes45(), np.full(64, np.nan))
        self.assertEqual(d['depth_state'], 'MISSING')
        self.assertTrue(d['unknown'])
        self.assertFalse(any(d['witnesses'].values()))

    def test_far_return_has_undefined_conditional_overlap(self):
        d = describe(boxes45(), np.full(64, 6.))
        self.assertEqual(d['depth_state'], 'OUTSIDE_ONLY')
        self.assertTrue(d['unknown'])
        self.assertTrue(all(z['angular_given_depth'] is None for z in d['zones']))

    def test_exact_depth_contact_and_full_horizontal_interval(self):
        self.assertEqual(interval_state([3., 3.5]), 'PARTIAL')
        self.assertEqual(interval_state([.3, 3.]), 'CONTAINED')
        self.assertEqual(lateral_relation([.05, .2], [2., 3.]), 'CROSSING')

    def test_input_validity_and_unclamped_interval(self):
        values = np.full(64, np.nan)
        values[:4] = [0., .09, 8., 7.9]
        d = describe(boxes45(), values)
        self.assertEqual(d['valid_zones'], 1)
        self.assertGreater(d['zones'][3]['interval_m'][1], 8.)


if __name__ == '__main__':
    unittest.main()
