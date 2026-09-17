"""Focused observable-association and native-evidence contracts."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import representation_features as f
from run_representation import WORK, lines


class AssociationTests(unittest.TestCase):
    def test_iou_and_lexicographic_tie(self):
        obs = {'box': [0, 0, 10, 10]}
        a = {'box': [0, 0, 5, 10]}
        b = {'box': [5, 0, 10, 10]}
        broad = {'box': [-100, -100, 100, 100]}
        self.assertEqual(f.select_descriptors(obs, [b, broad, a], 'single'), [a])
        self.assertEqual(f.select_descriptors(obs, [b, broad, a], 'multi'), [b, broad, a])

    def test_no_intersection_and_one_candidate(self):
        obs = {'box': [0, 0, 10, 10]}
        outside = {'box': [20, 20, 30, 30]}
        inside = {'box': [1, 1, 9, 9]}
        self.assertEqual(f.select_descriptors(obs, [outside], 'single'), [])
        self.assertEqual(f.select_descriptors(obs, [inside], 'single'), [inside])

    @classmethod
    def setUpClass(cls):
        cls.cap = WORK/'mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
        cls.row = next(r for r in lines(cls.cap/'raw.jsonl') if r['tof_packet_received'])
        cls.image = cv2.imread(str(cls.cap/cls.row['rgb_path']))

    def test_empty_rgb_proposals_preserve_sensor(self):
        original = f.extract(self.row, self.image, 0.)
        with patch.object(f, 'foreground', return_value=([], np.zeros(self.image.shape[:2], np.uint8))), \
             patch.object(f, 'proposals', return_value=([], None)):
            out = f.extract(self.row, self.image, 0., 'single')
        np.testing.assert_array_equal(out['sensor'], original['sensor'])
        np.testing.assert_array_equal(out['geometry'][10:], 0.)
        self.assertEqual(out['audit']['hypothesis_count'], 0)

    def test_merged_and_missing_preserve_explicit_state(self):
        row = copy.deepcopy(self.row)
        for zone in row['tof_zones']:
            for t in zone['targets']:
                t['status'] = 'SIM_MERGED'
        out = f.extract(row, self.image, 0., 'single')
        self.assertEqual(out['audit']['hypothesis_count'], 0)
        self.assertEqual(out['audit']['encoded_slots'], 128)
        row['tof_packet_received'] = False
        missing = f.extract(row, self.image, 0., 'single')
        names = missing['sensor_names']
        usable = [i for i, name in enumerate(names) if name.endswith('.usable')]
        np.testing.assert_array_equal(missing['sensor'][usable], 0.)
        self.assertEqual(missing['sensor'][names.index('global.tof_packet')], 0.)

    def test_raw_mask_and_single_native_parity(self):
        multi = f.extract(self.row, self.image, 0., 'multi')
        single = f.extract(self.row, self.image, 0., 'single')
        np.testing.assert_array_equal(multi['sensor'], single['sensor'])
        names = multi['sensor_names'] + multi['geometry_names']
        kept = [names[i] for i in f.raw_indices(names)]
        self.assertEqual(len(kept), 2224)
        self.assertNotIn('rgb.narrow_seeds', kept)
        self.assertNotIn('rgb.distinct_boxes', kept)
        self.assertFalse(any(n.startswith(('all.', 'nearest.', 'forward_height.', 'hypotheses.')) for n in kept))


if __name__ == '__main__':
    unittest.main()
