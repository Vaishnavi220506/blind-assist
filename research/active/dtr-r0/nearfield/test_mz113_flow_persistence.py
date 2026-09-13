"""Synthetic translated-image checks, without captured source outcomes."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import mz113_flow_persistence as model
from test_mz111_temporal_geometry import frame


def texture():
    image = np.zeros((360, 640), np.uint8)
    rng = np.random.default_rng(113)
    image[135:225, 365:395] = rng.integers(25, 255, (90, 30), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), .6)


def translated(image, dx):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (image.shape[1], image.shape[0]))


def observations():
    old = frame(0., radar=True, distance=2., velocity=None, box=[[365.,135.,395.,225.]])
    old[0]['radar_angle'] = [10.]
    new = frame(.25, box=[[330.,135.,360.,225.]])
    return [old, new]


def run(frames, images, use_flow=True):
    rows = [dict(r, id=str(k)) for k, (r, _) in enumerate(frames)]
    return model.predict(rows, [p for _, p in frames], lambda row: images[int(row['id'])], use_flow=use_flow)


class FlowPersistence(unittest.TestCase):
    def test_pixels_recover_crossing_where_angular_overlap_fails(self):
        old = texture(); new = translated(old, -35.)
        pair = observations()
        out = run(pair, [old, new])
        self.assertFalse(out[0]['candidate'])
        self.assertTrue(out[1]['candidate'])
        trace = out[1]['diagnostics']['propagated'][0]
        self.assertEqual(trace['velocity_state'], 'UNKNOWN')
        self.assertEqual(trace['range_assumption'], 'HOLD_RANGE_VELOCITY_UNKNOWN')
        self.assertEqual(trace['range_m'], 2.)
        self.assertGreaterEqual(trace['visual']['valid_points'], 3)
        self.assertLessEqual(trace['visual']['max_fb_error_px'], 1.5)
        self.assertFalse(run(pair, [old, new], use_flow=False)[1]['candidate'])

    def test_texture_failure_and_missing_rgb_break_history(self):
        old = texture(); new = translated(old, -35.)
        self.assertFalse(run(observations(), [np.zeros_like(old), np.zeros_like(old)])[1]['candidate'])
        pair = observations(); middle = frame(.1, box=pair[0][1]['proposals'])
        self.assertFalse(run([pair[0], middle, pair[1]], [old, None, new])[-1]['candidate'])

    def test_forward_backward_inconsistency_rejects_flow(self):
        points = np.array([[[370.,150.]], [[375.,170.]], [[380.,190.]]], np.float32)
        with patch.object(cv2, 'goodFeaturesToTrack', return_value=points), patch.object(cv2, 'calcOpticalFlowPyrLK',
            side_effect=[(points+1, np.ones((3,1), np.uint8), None), (points+3, np.ones((3,1), np.uint8), None)]):
            result = model.flow_box(texture(), texture(), [365.,135.,395.,225.])
        self.assertEqual(result['state'], 'FLOW_UNAVAILABLE')
        self.assertEqual(result['valid_points'], 0)

    def test_ambiguous_current_radar_and_split_regions_reset(self):
        old = texture(); new = translated(old, -35.)
        pair = observations(); pair[1][0].update(radar_range_m=[2.,3.], radar_angle=[0.,0.], radar_valid=[True,True], radar_velocity=[None,None])
        self.assertFalse(run(pair, [old, new])[1]['candidate'])
        pair = observations(); pair[1][1]['proposals'].append([332.,137.,362.,227.])
        self.assertFalse(run(pair, [old, new])[1]['candidate'])

    def test_velocity_and_range_age_are_preserved(self):
        old = texture(); new = translated(old, -35.)
        pair = observations(); pair[0][0].update(radar_range_m=[2.5], radar_velocity=[-1.])
        later = frame(.5, box=pair[1][1]['proposals']); stale = frame(.75, box=pair[1][1]['proposals'])
        out = run(pair+[later, stale], [old, new, new, new])
        self.assertAlmostEqual(out[1]['diagnostics']['propagated'][0]['range_m'], 2.25)
        self.assertEqual(out[2]['diagnostics']['propagated'][0]['measurement_index'], 0)
        self.assertFalse(out[3]['candidate'])

    def test_bad_time_episode_and_imu_reset(self):
        old = texture(); new = translated(old, -35.)
        for field, value in [('time_s', None), ('time_s', 0.), ('episode_id', 'new'), ('imu_valid', False)]:
            pair = observations(); pair[1][0][field] = value
            self.assertFalse(run(pair, [old, new])[1]['candidate'])

    def test_causal_prefix_and_nominal_support_are_immutable(self):
        old = texture(); new = translated(old, -35.)
        pair = observations(); before = copy.deepcopy(pair)
        past = run(pair, [old, new])
        future = frame(.5, box=pair[1][1]['proposals'])
        self.assertEqual(past, run(pair+[future], [old, new, new])[:2])
        self.assertEqual(pair, before)
        pair[0][1]['candidate'] = True; pair[1][1]['tof_support'] = True
        self.assertTrue(all(p['candidate'] for p in run(pair, [None, None])))

    def test_angular_control_has_same_missing_velocity_seed_policy(self):
        frames = [frame(0., radar=True, distance=2., velocity=None), frame(.25)]
        out = run(frames, [texture(), texture()], use_flow=False)
        self.assertTrue(out[1]['candidate'])
        self.assertEqual(out[1]['diagnostics']['propagated'][0]['velocity_state'], 'UNKNOWN')

    def test_large_isolated_jump_abstains_when_lk_cannot_track(self):
        old = texture()
        result = model.flow_box(old, translated(old, -70.), [365.,135.,395.,225.])
        self.assertEqual(result['state'], 'FLOW_UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
