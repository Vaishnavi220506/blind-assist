"""Synthetic causal range transport checks; no captured source outcomes."""
import copy
import math
import unittest

import mz111_temporal_geometry as model


def frame(t, radar=False, distance=4., velocity=-2., box=None, yaw=0., episode='episode'):
    row = dict(episode_id=episode, time_s=t, imu_valid=True, camera_pitch_deg=0., camera_in_body_m=[0.,0.,1.7],
        rgb_intrinsics=dict(cx=320., cy=180., fx=320., fy=320.), radar_packet_received=True,
        radar_range_m=[distance] if radar else [], radar_angle=[0.] if radar else [],
        radar_valid=[True] if radar else [], radar_velocity=[velocity] if radar else [])
    nominal = dict(baseline=False, candidate=False, tof_support=False, candidate_state='UNKNOWN',
        proposals=[[300.,150.,340.,210.]] if box is None else box, integrated_yaw_deg=yaw, associations=[])
    return row, nominal


def run(frames):
    return model.predict([r for r, _ in frames], [p for _, p in frames])


class TemporalGeometry(unittest.TestCase):
    def test_velocity_moves_range_into_corridor_and_no_current_spatial_addition(self):
        frames = [frame(0., radar=True), frame(.25)]
        moving = run(frames)
        self.assertFalse(moving[0]['candidate'])
        self.assertTrue(moving[1]['candidate'])
        self.assertAlmostEqual(moving[1]['diagnostics']['propagated'][0]['range_m'], 3.5)
        stationary = run([frame(0., radar=True, velocity=0.), frame(.25)])
        self.assertFalse(stationary[1]['candidate'])

    def test_age_is_measured_from_range_and_never_refreshed_by_rgb(self):
        out = run([frame(0., radar=True), frame(.25), frame(.5), frame(.75)])
        self.assertTrue(out[1]['candidate']); self.assertTrue(out[2]['candidate'])
        self.assertFalse(out[3]['candidate'])
        self.assertEqual(out[2]['diagnostics']['propagated'][0]['measurement_index'], 0)

    def test_missing_current_rgb_breaks_track(self):
        out = run([frame(0., radar=True), frame(.1, box=[]), frame(.2)])
        self.assertFalse(out[2]['candidate'])

    def test_imu_compensates_camera_turn(self):
        # Same world-bearing interval after a 10-degree camera yaw.
        original = [300.,150.,340.,210.]
        shifted = [320+320*math.tan(math.atan((u-320)/320)-math.radians(10)) for u in (original[0],original[2])]
        box = [[shifted[0],150.,shifted[1],210.]]
        out = run([frame(0., radar=True), frame(.25, box=box, yaw=10.)])
        self.assertTrue(out[1]['candidate'])
        uncompensated = run([frame(0., radar=True), frame(.25, box=box, yaw=0.)])
        self.assertFalse(uncompensated[1]['candidate'])

    def test_competing_current_returns_reset_instead_of_rescue(self):
        contested = frame(.25, radar=True)
        contested[0].update(radar_range_m=[2.,3.], radar_angle=[0.,0.], radar_valid=[True,True], radar_velocity=[0.,0.])
        out = run([frame(0., radar=True), contested, frame(.4)])
        self.assertFalse(out[1]['candidate']); self.assertFalse(out[2]['candidate'])
        self.assertEqual(out[1]['diagnostics']['ambiguous'], 1)

    def test_split_visual_correspondence_resets_track(self):
        split = [[300.,150.,340.,210.], [302.,152.,342.,212.]]
        out = run([frame(0., radar=True), frame(.2, box=split), frame(.4)])
        self.assertFalse(out[1]['candidate']); self.assertFalse(out[2]['candidate'])

    def test_episode_time_and_imu_reset(self):
        for second in (frame(.25, episode='other'), frame(0.), frame(-.1)):
            self.assertFalse(run([frame(0., radar=True), second])[-1]['candidate'])
        gap = frame(.1); gap[0]['imu_valid'] = False
        out = run([frame(0., radar=True), gap, frame(.2, radar=True), frame(.3)])
        self.assertFalse(out[-1]['candidate'])

    def test_invalid_velocity_cannot_seed_and_nominal_support_is_immutable(self):
        for velocity in (None, float('nan')):
            self.assertFalse(run([frame(0., radar=True, velocity=velocity), frame(.25)])[-1]['candidate'])
        frames = [frame(0.), frame(.25)]
        frames[0][1]['candidate'] = True
        frames[1][1]['tof_support'] = True
        before = copy.deepcopy(frames)
        self.assertTrue(all(p['candidate'] for p in run(frames)))
        self.assertEqual(frames, before)

    def test_future_input_does_not_change_past_output(self):
        prefix = [frame(0., radar=True), frame(.25)]
        self.assertEqual(run(prefix), run(prefix+[frame(.5, radar=True, distance=1.)])[:2])


if __name__ == '__main__':
    unittest.main()
