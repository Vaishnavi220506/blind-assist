import copy
import json
import unittest

import numpy as np

from mz115_zonal_tof import zone_geometry
from mz143_corridor_features import extract


def inputs():
    zones = [dict(zone_geometry(i)[0], targets=[]) for i in range(64)]
    row = dict(camera_in_body_m=[0., 0., 1.7], camera_pitch_deg=0.,
        tof_packet_received=True, imu_valid=True, radar_packet_received=True,
        radar_range_m=[None]*4, radar_angle=[None]*4, radar_velocity=[None]*4,
        radar_valid=[False]*4, tof_zones=zones,
        rgb_intrinsics=dict(width=640, height=360, fx=457., fy=457., cx=319.5, cy=179.5))
    image = np.full((360, 640, 3), 40, np.uint8)
    image[40:340, 345:353] = 210
    return row, image


def target(distance=2., status='SIM_VALID'):
    return dict(distance_m=distance, range_noise_sigma_m=.04,
                signal_strength_proxy=.1, status=status)


def feature(result, name):
    return result['sensor'][result['sensor_names'].index(name)]


class FeatureTests(unittest.TestCase):
    def test_identity_truth_metadata_invariance_and_no_mutation(self):
        row, image = inputs(); row['tof_zones'][28]['targets'] = [target()]
        original = copy.deepcopy(row); pixels = image.copy()
        result = extract(row, image, 0.)
        contaminated = copy.deepcopy(row)
        contaminated.update(id='known_positive', family='rod', truth=True,
                            native_bounds=[dict(secret=3)], body_origin_m=[999]*3,
                            camera=dict(yaw=90), pair_id='different', time_s=999)
        contaminated['tof_zones'][28]['targets'][0]['private_hit_point'] = [99]*3
        again = extract(contaminated, image, 0.)
        np.testing.assert_array_equal(result['sensor'], again['sensor'])
        np.testing.assert_array_equal(result['geometry'], again['geometry'])
        self.assertEqual(result['audit'], again['audit'])
        self.assertEqual(row, original); np.testing.assert_array_equal(image, pixels)
        json.dumps(result['audit'], allow_nan=False)

    def test_second_slot_preserved_and_merged_not_exact_depth(self):
        row, image = inputs(); row['tof_zones'][28]['targets'] = [target(3.7)]
        first = extract(row, image, 0.)
        row['tof_zones'][28]['targets'].append(target(2.))
        second = extract(row, image, 0.)
        self.assertEqual(feature(first, 'zone28.slot1.present'), 0)
        self.assertEqual(feature(second, 'zone28.slot1.range_m'), 2.)
        self.assertEqual(second['audit']['public_slots'], 2)
        row['tof_zones'][28]['targets'][1]['status'] = 'SIM_MERGED'
        merged = extract(row, image, 0.)
        self.assertEqual(feature(merged, 'zone28.slot1.merged'), 1.)
        self.assertLess(feature(merged, 'zone28.slot1.support_x_lo'), .1)
        self.assertGreater(feature(merged, 'zone28.slot1.support_x_hi'), 3.9)
        self.assertEqual(merged['geometry'][merged['geometry_names'].index('slots.usable_valid')], 1.)

    def test_imu_rotates_center_and_translates_full_support(self):
        row, image = inputs(); row['tof_zones'][28]['targets'] = [target()]
        zero = extract(row, image, 0.); rotated = extract(row, image, 90.)
        x = feature(zero, 'zone28.slot0.center_x'); y = feature(zero, 'zone28.slot0.center_y')
        self.assertAlmostEqual(feature(rotated, 'zone28.slot0.center_x'), -y, places=5)
        self.assertAlmostEqual(feature(rotated, 'zone28.slot0.center_y'), x, places=5)
        row['camera_in_body_m'][0] = .2
        translated = extract(row, image, 0.)
        self.assertAlmostEqual(feature(translated, 'zone28.slot0.support_x_lo')-
                               feature(zero, 'zone28.slot0.support_x_lo'), .2, places=5)

    def test_missing_and_zero_image_finite_same_dimensions(self):
        row, image = inputs(); base = extract(row, image, 0.)
        row['tof_packet_received'] = False; row['imu_valid'] = False
        row['tof_zones'][0]['targets'] = [target()]
        empty = extract(row, np.zeros_like(image), 0.)
        self.assertEqual(base['sensor_names'], empty['sensor_names'])
        self.assertEqual(base['geometry_names'], empty['geometry_names'])
        self.assertLess(len(empty['sensor'])+len(empty['geometry']), 2500)
        self.assertTrue(np.isfinite(empty['sensor']).all() and np.isfinite(empty['geometry']).all())
        self.assertEqual(feature(empty, 'zone00.slot0.present'), 1.)
        self.assertEqual(feature(empty, 'zone00.slot0.usable'), 0.)
        self.assertEqual(empty['audit']['hypothesis_count'], 0)

    def test_zone_input_order_does_not_change_encoding(self):
        row, image = inputs(); row['tof_zones'][28]['targets'] = [target()]
        a = extract(row, image, 0.)
        row['tof_zones'].reverse()
        b = extract(row, image, 0.)
        np.testing.assert_array_equal(a['sensor'], b['sensor'])
        np.testing.assert_array_equal(a['geometry'], b['geometry'])

    def test_thin_rgb_changes_geometry_without_clearing_partial_fov_slot(self):
        row, image = inputs()
        row['tof_zones'][0]['targets'] = [target()]
        row['tof_zones'][28]['targets'] = [target()]
        bright = extract(row, image, 0.)
        blank = extract(row, np.full_like(image, 40), 0.)
        np.testing.assert_array_equal(bright['sensor'], blank['sensor'])
        self.assertFalse(np.array_equal(bright['geometry'], blank['geometry']))
        self.assertEqual(feature(blank, 'zone00.slot0.usable'), 1.)
        outside = next(s for s in blank['audit']['slots'] if s['zone'] == 0)
        self.assertTrue(outside['outside_or_partial_rgb'])
        self.assertEqual(bright['audit']['public_slots'], blank['audit']['public_slots'])


if __name__ == '__main__':
    unittest.main()
