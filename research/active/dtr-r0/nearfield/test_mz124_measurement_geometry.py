"""Geometric exclusion cannot follow from a center sample or missing support."""
import copy
import math
import random
import unittest

from mz124_measurement_geometry import CORRIDOR, intersects, refine_support, predict_frame


def row():
    return dict(time_s=0., camera_pitch_deg=-3., camera_in_body_m=[0., 0., 1.7], imu_valid=True,
        rgb_intrinsics=dict(cx=320., cy=180., fx=460., fy=460.), tof_packet_received=True,
        radar_packet_received=True, radar_range_m=[], radar_angle=[], radar_valid=[], tof_zones=[])


def cached():
    return dict(integrated_yaw_deg=0., proposals=[], candidate=True, common_radar=False, guard_events=[])


class GeometryTests(unittest.TestCase):
    def test_random_point_witness_is_never_excluded(self):
        rng = random.Random(124)
        r = row(); witnesses = 0
        for _ in range(100):
            u, v = rng.uniform(260, 380), rng.uniform(110, 250)
            box = [u-15, v-15, u+15, v+15]; distance = rng.uniform(.5, 3.4)
            a = (u-320)/460; b = (180-v)/460; norm = math.sqrt(1+a*a+b*b)
            p = math.radians(-3.)
            xyz = [distance*(math.cos(p)-b*math.sin(p))/norm,
                   distance*a/norm, 1.7+distance*(math.sin(p)+b*math.cos(p))/norm]
            if intersects([(x, x) for x in xyz]):
                witnesses += 1
                self.assertTrue(refine_support(box, (distance-.01, distance+.01), r, 0.)['possible'])
        self.assertGreater(witnesses, 20)

    def test_far_range_exclusion(self):
        self.assertFalse(refine_support([310, 170, 330, 190], (8., 9.), row(), 0.)['possible'])

    def test_no_returns_and_missing_packets_cannot_retract(self):
        for key in ('tof_packet_received', 'radar_packet_received', 'imu_valid'):
            r = row(); r[key] = False
            p = predict_frame(r, cached())
            self.assertTrue(p['flags']['accountable_correction'])
            self.assertFalse(p['correction_eligible'])

    def test_independent_radar_never_retracted_by_no_rgb(self):
        r = row(); r.update(radar_range_m=[2.45], radar_angle=[0.], radar_valid=[True])
        c = cached(); c['common_radar'] = True
        p = predict_frame(r, c)
        self.assertTrue(p['flags']['sensor_measurement'])
        self.assertTrue(p['flags']['rgb_measurement'])
        self.assertTrue(p['flags']['accountable_correction'])

    def test_truth_metadata_does_not_change_prediction(self):
        r = row(); c = cached(); first = predict_frame(r, c)
        r.update(truth=False, native_bounds=[dict(center_m=[900, 900, 900])], family='fake')
        c['evaluator'] = {'truth': True}
        self.assertEqual(first, predict_frame(r, c))

    def test_raw_radar_possible_prevents_strict_correction_even_if_incumbent_rejected(self):
        from unittest.mock import patch
        r = row(); r.update(radar_range_m=[2.5], radar_angle=[10.], radar_valid=[True])
        r['tof_zones'] = [dict(zone_id=0, theta_bounds_deg=[0., 5.], phi_bounds_deg=[0., 5.],
            targets=[dict(distance_m=2.5, range_noise_sigma_m=.04, status='SIM_VALID')])]
        with patch('mz124_measurement_geometry.refine_support', return_value=dict(possible=False, nodes=1, reason='fixture')):
            p = predict_frame(r, cached())
        self.assertTrue(p['inherited_radar_exclusion'])
        self.assertFalse(p['correction_exclusion'])
        self.assertTrue(p['flags']['accountable_correction'])


if __name__ == '__main__':
    unittest.main()
