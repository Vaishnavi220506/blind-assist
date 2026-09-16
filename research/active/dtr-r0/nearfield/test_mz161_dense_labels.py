"""Synthetic-only dense-label checks; no real capture/evaluator file access."""
import copy
import json
import unittest

import numpy as np

from mz161_dense_labels import make_labels


def scene():
    row = dict(id='synthetic', rgb_intrinsics=dict(width=7, height=5, fx=4., fy=4., cx=3., cy=2.),
               camera_in_body_m=[0., 0., 1.], camera_pitch_deg=0.)
    evaluation = dict(id='synthetic', body_origin_m=[0., 0., 0.],
        camera=dict(x=0., y=0., z=1., yaw=0., pitch=0., roll=0.),
        native_bounds=[dict(name='shape0', center_m=[2., 0., 1.], extent_m=[.1, .4, .4]),
                       dict(name='env0', center_m=[4., 0., 1.], extent_m=[.1, 10., 10.])])
    return row, evaluation


class DenseLabelTests(unittest.TestCase):
    def test_exact_first_hit_and_balanced_class_mass(self):
        row, evaluation = scene()
        labels = make_labels(row, evaluation, 0.)
        expected = np.zeros((5, 7), dtype=bool); expected[2, 3] = True
        np.testing.assert_array_equal(labels['target'], expected)
        np.testing.assert_array_equal(labels['target_visible'], expected)
        np.testing.assert_array_equal(labels['target_corridor'], expected)
        self.assertTrue(labels['known'].all())
        self.assertEqual(labels['weights'].dtype, np.float32)
        self.assertAlmostEqual(float(labels['weights'][expected].sum()), .5)
        self.assertAlmostEqual(float(labels['weights'][~expected].sum()), .5)
        self.assertEqual(labels['audit']['target_columns'], [dict(column=3, visible_pixels=1, corridor_pixels=1)])
        self.assertEqual(labels['audit']['pose_authority'], 'SOURCE_COMMANDED_REFERENCE')
        json.dumps(labels['audit'], allow_nan=False)

    def test_camera_forward_depth_and_exact_corridor_points(self):
        row, evaluation = scene()
        evaluation['native_bounds'] = [dict(name='shape0', center_m=[2., 0., 1.], extent_m=[.1, 10., 10.])]
        labels = make_labels(row, evaluation, 0.)
        expected = np.zeros((5, 7), dtype=bool); expected[:4, 3] = True
        np.testing.assert_array_equal(labels['target'], expected)
        self.assertTrue(labels['target_visible'].all())
        # A 3.7 m forward surface is beyond the corridor even off the optical axis.
        evaluation['native_bounds'][0]['center_m'][0] = 3.8
        self.assertFalse(make_labels(row, evaluation, 0.)['target'].any())

    def test_unknown_and_single_class_mass(self):
        row, evaluation = scene()
        evaluation['native_bounds'] = evaluation['native_bounds'][:1]
        labels = make_labels(row, evaluation, 0.)
        self.assertEqual(int(labels['known'].sum()), 1)
        self.assertEqual(float(labels['weights'].sum()), 1.)
        self.assertTrue((labels['weights'][~labels['known']] == 0).all())
        evaluation['native_bounds'][0]['center_m'][0] = 4.
        labels = make_labels(row, evaluation, 0.)
        self.assertFalse(labels['target'].any())
        self.assertAlmostEqual(float(labels['weights'].sum()), 1.)
        evaluation['native_bounds'] = []
        labels = make_labels(row, evaluation, 0.)
        self.assertFalse(labels['known'].any())
        self.assertEqual(float(labels['weights'].sum()), 0.)
        self.assertEqual(labels['audit']['unknown_pixels'], 35)

    def test_all_actors_and_occluded_volume_truth(self):
        row, evaluation = scene()
        evaluation['native_bounds'].append(dict(name='other_actor', center_m=[1., 0., 1.], extent_m=[.05, .04, .04]))
        labels = make_labels(row, evaluation, 0.)
        self.assertTrue(labels['target'][2, 3])
        self.assertFalse(labels['target_visible'][2, 3])
        self.assertEqual(labels['audit']['actors'][-1]['visible_corridor_pixels'], 1)
        # This occluder is entirely before the corridor and hides a risky volume.
        evaluation['native_bounds'][-1] = dict(name='near_occluder', center_m=[.125, 0., 1.], extent_m=[.025, 2., 2.])
        labels = make_labels(row, evaluation, 0.)
        self.assertTrue(labels['known'].all())
        self.assertFalse(labels['target'].any())
        self.assertFalse(labels['target_visible'].any())
        self.assertTrue(labels['audit']['native_aabb_corridor_truth'])
        self.assertTrue(labels['audit']['native_volume_truth_vs_visible_mismatch'])

    def test_reference_pose_body_origin_and_input_immutability(self):
        row, evaluation = scene()
        before = copy.deepcopy((row, evaluation))
        expected = make_labels(row, evaluation, 25.)
        self.assertEqual((row, evaluation), before)
        self.assertEqual(expected['audit']['reference_minus_public_yaw_deg'], -25.)
        shift = [10., 20., 3.]
        evaluation['body_origin_m'] = shift
        for key, value in zip(('x', 'y', 'z'), shift): evaluation['camera'][key] += value
        for actor in evaluation['native_bounds']:
            actor['center_m'] = [a+b for a, b in zip(actor['center_m'], shift)]
        actual = make_labels(row, evaluation, 25.)
        for key in ('target', 'known', 'target_visible', 'target_corridor', 'weights'):
            np.testing.assert_array_equal(actual[key], expected[key])
        evaluation['camera'].pop('yaw')
        with self.assertRaisesRegex(ValueError, 'source-commanded'):
            make_labels(row, evaluation, 0.)


if __name__ == '__main__':
    unittest.main()
