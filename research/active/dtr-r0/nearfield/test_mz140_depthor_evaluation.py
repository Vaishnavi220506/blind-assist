"""Synthetic evaluator checks; no dataset/evaluator files or model are loaded."""
import copy
import json
import unittest

import numpy as np

from evaluate_mz140_depthor import evaluate_frame, rasterize_native_bounds


def scene():
    row = dict(id='opaque', rgb_intrinsics=dict(width=7, height=5, fx=4., fy=4., cx=3., cy=2.),
               camera_in_body_m=[0., 0., 1.], camera_pitch_deg=0.)
    evaluation = dict(id='opaque', family='synthetic', body_origin_m=[0., 0., 0.],
        camera=dict(x=0., y=0., z=1., yaw=0., pitch=0., roll=0.),
        native_bounds=[dict(name='shape0', center_m=[2., 0., 1.], extent_m=[.1, .4, .4]),
                       dict(name='background', center_m=[4., 0., 1.], extent_m=[.1, 10., 10.])],
        zonal_tof_native=[])
    return row, evaluation, dict(integrated_yaw_deg=0.)


class DepthorGeometryEvaluationTests(unittest.TestCase):
    def test_axis_depth_is_not_slant_and_native_pose_overrides_public_drift(self):
        row, e, cached = scene()
        e['native_bounds'][0]['extent_m'] = [.1, 10., 10.]
        cached['integrated_yaw_deg'] = 20.
        raster = rasterize_native_bounds(row, e, cached)
        np.testing.assert_allclose(raster['depth'], 1.9)
        self.assertEqual(raster['pose_source'], 'NATIVE_EVALUATOR_CAMERA')
        result = evaluate_frame(row, e, raster['depth'], cached)
        self.assertEqual(result['pose_diagnostic']['native_minus_public_yaw_deg'], -20.)
        self.assertEqual(result['target']['retained_fraction'], 1.)
        self.assertEqual(result['target']['transverse_columns']['any_agreement_fraction'], 1.)

    def test_occluder_removes_target_pixels_and_ring_errors_reduce_local_precision(self):
        row, e, cached = scene()
        original = rasterize_native_bounds(row, e, cached)
        e['native_bounds'].append(dict(name='foreground', center_m=[1., 0., 1.], extent_m=[.05, .04, .04]))
        raster = rasterize_native_bounds(row, e, cached)
        self.assertFalse(raster['target_visible'][2, 3])
        self.assertEqual(raster['owner'][2, 3], 2)
        self.assertLess(raster['target_visible'].sum(), original['target_visible'].sum())
        # Restore scene: one background pixel impersonates the target's depth.
        e['native_bounds'].pop()
        depth = original['depth'].copy()
        pixel = tuple(np.argwhere(~original['target_visible'])[0])
        depth[pixel] = 1.9
        result = evaluate_frame(row, e, depth, cached)
        self.assertEqual(result['target']['retained_fraction'], 1.)
        self.assertEqual(result['local_target_surface']['spurious_pixels'], 1)
        self.assertLess(result['local_target_surface']['precision'], 1.)
        json.dumps(result, allow_nan=False)

    def test_unknown_background_is_not_scored_as_spurious(self):
        row, e, cached = scene()
        e['native_bounds'] = e['native_bounds'][:1]
        result = evaluate_frame(row, e, np.full((5, 7), 1.9), cached)
        self.assertGreater(result['silhouette_neighborhood']['unknown_reference_pixels'], 0)
        self.assertEqual(result['local_target_surface']['spurious_pixels'], 0)
        self.assertGreater(result['unknown_reference_pixels'], 0)

    def test_returned_hit_projection_separates_numerical_and_012m_proximity(self):
        row, e, cached = scene()
        e['zonal_tof_native'] = [dict(zone_id=0,
            returned_lineage=[dict(target_index=0, hit_indices=[0, 1])],
            private_rays=[dict(subray=0, hit_point_m=[1.9, 0., 1.], actor_id='shape0'),
                          dict(subray=1, hit_point_m=[1.9, 3., 1.], actor_id='shape0'),
                          dict(subray=2, hit_point_m=[1.9, 0., 1.], actor_id='unreturned')])]
        result = evaluate_frame(row, e, np.full((5, 7), 1.95), cached)
        audit = result['native_returned_contributors']['summary']
        self.assertEqual(audit['returned_contributor_records'], 2)
        self.assertEqual(audit['all']['inside_rgb'], 1)
        self.assertEqual(audit['all']['outside_rgb'], 1)
        self.assertEqual(audit['corridor']['numerical_agreement'], 0)
        self.assertEqual(audit['corridor']['diagnostic_012m_agreement'], 1)
        self.assertEqual(audit['corridor']['diagnostic_012m_fraction'], 1.)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
