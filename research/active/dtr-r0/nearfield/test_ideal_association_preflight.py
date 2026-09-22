"""Focused missing-return/coverage accounting checks, not scientific validation."""
import unittest
import numpy as np

from audit_ideal_association_preflight import coverage
from ba_camera_corridor import sample_indices
from tof_fov45_core import boxes45


class CoverageTests(unittest.TestCase):
    def fixture(self):
        boxes = boxes45()
        values = np.full(64, np.nan, dtype=np.float32)
        traces = [dict(zone_id=i, observed=False, distance_m=None, pixel_indices=[]) for i in range(64)]
        return boxes, values, traces

    def test_missing_categories_partition_without_depth_spread(self):
        boxes, values, traces = self.fixture()
        sy, sx = sample_indices()
        y, x = map(int, boxes[0][:2])
        y1, x1 = map(int, boxes[1][:2])
        mask = np.zeros((360,640), bool)
        for py, px in ((y,x), (y,x+1), (y1,x1), (0,0)):
            mask[sy[py],sx[px]] = True
        values[0] = 2.
        traces[0].update(observed=True, distance_m=2., pixel_indices=[y*256+x])
        r = coverage(mask, boxes, values, traces)
        self.assertEqual(r['returned_target_pixels'], 1)
        self.assertEqual(r['sampled_target_pixels'], 4)
        self.assertEqual(r['missing_reasons'], dict(outside_footprint=1,
            no_observed_zone=1, observed_zone_noncontributor=1))
        self.assertFalse(r['sampled_target_fully_covered'])

    def test_sampled_coverage_does_not_imply_native_coverage(self):
        boxes, values, traces = self.fixture()
        sy,sx = sample_indices()
        y,x = map(int, boxes[0][:2])
        mask = np.zeros((360,640), bool)
        mask[sy[y],sx[x]] = True
        self.assertNotIn(int(sx[x]+1), sx)
        mask[sy[y],sx[x]+1] = True
        values[0] = 2.
        traces[0].update(observed=True, distance_m=2., pixel_indices=[y*256+x])
        r = coverage(mask,boxes,values,traces)
        self.assertTrue(r['sampled_target_fully_covered'])
        self.assertFalse(r['native_target_fully_covered'])

    def test_unobserved_zone_cannot_supply_contributors(self):
        boxes,values,traces = self.fixture()
        y,x = map(int,boxes[0][:2])
        traces[0]['pixel_indices'] = [y*256+x]
        with self.assertRaises(AssertionError):
            coverage(np.zeros((360,640),bool), boxes, values, traces)


if __name__ == '__main__':
    unittest.main()
