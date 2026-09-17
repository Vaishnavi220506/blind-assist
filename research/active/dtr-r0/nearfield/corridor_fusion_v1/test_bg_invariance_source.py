import copy
import unittest
from bg_invariance_source import source, check_source, pair_indices


class FactorialTests(unittest.TestCase):
    def test_counts_and_group_split(self):
        spec = source()
        self.assertEqual(check_source(spec)['texture_pairs'], 144)
        groups = {}
        for f in spec['frames']:
            groups.setdefault(f['physical_group'], set()).add(f['split'])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))

    def test_context_geometry_change_rejected(self):
        spec = source()
        _, pairs = pair_indices(spec)
        spec['frames'][pairs[0][1]]['objects'][1]['center_m'][0] += .1
        with self.assertRaises(AssertionError):
            check_source(spec)

    def test_target_appearance_change_rejected(self):
        spec = source()
        _, pairs = pair_indices(spec)
        spec['frames'][pairs[0][1]]['objects'][0]['texture_seed'] += 1
        with self.assertRaises(AssertionError):
            check_source(spec)

    def test_sensor_seed_change_rejected(self):
        spec = source()
        _, pairs = pair_indices(spec)
        spec['frames'][pairs[0][1]]['tof_sensor_seed'] += 1
        with self.assertRaises(AssertionError):
            check_source(spec)


if __name__ == '__main__':
    unittest.main()
