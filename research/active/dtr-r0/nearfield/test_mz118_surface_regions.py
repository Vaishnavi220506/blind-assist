import unittest

import numpy as np

import mz108_competitive_association as legacy
import mz118_surface_regions as surface


class SurfaceRegionTests(unittest.TestCase):
    def test_broad_bottom_clipped_shape_restored(self):
        image = np.full((120, 200, 3), 20, dtype=np.uint8)
        image[4:, 125:180] = 180
        self.assertEqual(legacy.proposals(image), [])
        regions = surface.proposals(image)
        self.assertTrue(any(b[0] <= 126 and b[2] >= 179 and b[1] <= 5 and b[3] == 120 for b in regions))
        self.assertEqual(regions, surface.proposals(image))

    def test_legacy_regions_retained_in_original_order(self):
        image = np.full((120, 200, 3), 20, dtype=np.uint8)
        image[25:70, 30:50] = 180
        image[10:80, 90:130] = 130
        previous = legacy.proposals(image)
        self.assertTrue(previous)
        self.assertEqual(surface.proposals(image)[:len(previous)], previous)

    def test_nested_alternatives_preserved_near_duplicates_coalesced(self):
        previous = [[30, 20, 34, 90]]
        additions = [[0, 0, 100, 100], [1, 0, 100, 100], [10, 0, 90, 100], [30, 20, 34, 90]]
        self.assertEqual(surface._append_unique(previous, additions),
                         previous+[[0, 0, 100, 100], [10, 0, 90, 100]])
        self.assertEqual(previous, [[30, 20, 34, 90]])

    def test_missing_and_uniform_blank_fallback(self):
        self.assertEqual(surface.proposals(None), [])
        for value in (0, 127, 255):
            self.assertEqual(surface.proposals(np.full((120, 200, 3), value, dtype=np.uint8)), [])


if __name__ == '__main__': unittest.main()
