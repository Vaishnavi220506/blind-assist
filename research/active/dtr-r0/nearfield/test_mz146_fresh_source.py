import copy
import unittest

import mz136_paired_source as inherited
from mz146_fresh_source import check_source, source


class FreshSourceTests(unittest.TestCase):
    def test_seed_restoration_confirmation_and_pair_geometry(self):
        original = inherited.SEED
        spec = source()
        self.assertEqual(inherited.SEED, original)
        audit = check_source(spec)
        self.assertEqual(audit['frames'], 288)
        self.assertEqual(audit['positive_frames'], 144)
        self.assertEqual(audit['split_frames'], {'confirmation': 288})
        self.assertEqual(len({f['scene_group'] for f in spec['frames']}), 24)
        self.assertEqual(spec, source())

    def test_accidental_old_partition_or_geometry_change_rejected(self):
        spec = source(); bad = copy.deepcopy(spec)
        bad['frames'][0]['split'] = 'test'
        with self.assertRaises(AssertionError):
            check_source(bad)
        bad = copy.deepcopy(spec)
        bad['frames'][0]['objects'][0]['center_m'][0] += 1.
        with self.assertRaises(AssertionError):
            check_source(bad)


if __name__ == '__main__':
    unittest.main()
