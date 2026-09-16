"""Synthetic geometry only; no dataset files, trained models or capture access."""
import copy
import json
import math
import unittest

import numpy as np

from mz161_dense_labels import make_labels
from mz167_range_readout import ARMS, reference_readouts, uniform_bin_mass


def scene():
    row = dict(id='synthetic', rgb_intrinsics=dict(width=7, height=5, fx=4., fy=4., cx=3., cy=2.),
               camera_in_body_m=[0., 0., 1.], camera_pitch_deg=0.)
    evaluation = dict(id='synthetic', body_origin_m=[0., 0., 0.],
        camera=dict(x=0., y=0., z=1., yaw=0., pitch=0., roll=0.),
        native_bounds=[dict(name='shape0', center_m=[2., 0., 1.], extent_m=[.1, 10., 10.])])
    return row, evaluation


class RangeReadoutTests(unittest.TestCase):
    def test_half_bin_positive_negative_boundary_and_narrow_interval(self):
        # Both true-inside and true-outside points in the same category can have
        # exactly half uniform mass. Fixed >= .5 deliberately includes both.
        s = np.array([1.015625, 1.046875, 1.0625, 1.0625, 1.015625, 1.015625])
        lo = np.array([1., 1., 1., 1.0625, 1., 1.])
        hi = np.array([1.03125, 1.03125, 1.0625, 1.125, 1.015625, 1.03125-2**-20])
        mass = uniform_bin_mass(s, lo, hi, np.ones(6,bool), np.ones(6,bool))
        np.testing.assert_array_equal(mass[:5], [.5, .5, 0., 1., .25])
        self.assertLess(mass[-1], .5)
        self.assertTrue((mass[:2] >= .5).all())
        self.assertTrue(lo[0] <= s[0] <= hi[0])
        self.assertFalse(lo[1] <= s[1] <= hi[1])
        self.assertEqual(mass.dtype, np.float32)

    def test_empty_invalid_unknown_overflow_and_validation(self):
        s=np.array([1., np.nan, 8., 12., -9.])
        lo=np.array([1., np.nan, 0., 0., -10.])
        hi=np.array([1., np.nan, 7.99, 7.99, -11.])
        valid=np.array([True,False,True,True,False]);known=np.array([True,False,True,True,False])
        np.testing.assert_array_equal(uniform_bin_mass(s,lo,hi,valid,known),np.zeros(5,np.float32))
        self.assertEqual(uniform_bin_mass(np.array([]),np.array([]),np.array([]),np.array([],bool),np.array([],bool)).shape,(0,))
        for left,right in [(-.01,1.),(2.,1.),(0.,8.),(0.,np.inf)]:
            with self.assertRaises(ValueError):
                uniform_bin_mass(np.array([1.]),np.array([left]),np.array([right]),np.array([True]),np.array([True]))
        with self.assertRaisesRegex(ValueError,'positive'):
            uniform_bin_mass(np.array([-1.]),np.array([0.]),np.array([1.]),np.array([True]),np.array([True]))
        with self.assertRaisesRegex(ValueError,'identical'):
            uniform_bin_mass(np.ones(2),np.zeros(1),np.ones(2),np.ones(2,bool),np.ones(2,bool))
        # The overflow premise is checked even where the reference is unknown.
        with self.assertRaisesRegex(ValueError,'strictly'):
            uniform_bin_mass(np.array([np.nan]),np.array([0.]),np.array([8.]),np.array([True]),np.array([False]))

    def test_axis_to_slant_original_rays_and_native_parity(self):
        row,e=scene();before=copy.deepcopy((row,e))
        result=reference_readouts(row,e,0.)
        expected=np.empty((5,7),np.float64)
        for y in range(5):
            for x in range(7):
                expected[y,x]=1.9*math.sqrt(1+((x-3)/4)**2+((2-y)/4)**2)
        np.testing.assert_allclose(result['reference_slant'],expected,rtol=1e-15,atol=0)
        self.assertEqual(result['reference_slant'].dtype,np.float64)
        self.assertTrue(result['known'].all())
        np.testing.assert_array_equal(result['masks']['native'],make_labels(row,e,0.)['target'])
        np.testing.assert_array_equal(result['label']['target'],result['masks']['native'])
        np.testing.assert_array_equal(result['masks']['public_exact'],result['masks']['native'])
        self.assertEqual((row,e),before)
        for arm in ARMS:
            self.assertEqual(result['masks'][arm].dtype,np.bool_)
            self.assertEqual(result['scores'][arm].dtype,np.float32)
        self.assertEqual(result['audit']['pose_authority'],'SOURCE_COMMANDED_REFERENCE')
        json.dumps(result['audit'],allow_nan=False)

    def test_midpoint_and_mass_can_add_false_positive_at_fixed_range_edge(self):
        row,e=scene()
        e['native_bounds'][0]['center_m'][0]=3.71  # First-hit axis/slant at the central pixel is 3.61 m.
        result=reference_readouts(row,e,0.)
        self.assertFalse(result['masks']['native'][2,3])
        self.assertFalse(result['masks']['public_exact'][2,3])
        self.assertTrue(result['masks']['bin_midpoint'][2,3])
        self.assertTrue(result['masks']['bin_mass'][2,3])
        # Fixed [3.5625,3.625) category overlaps [0.2,3.6] by about 60%.
        self.assertAlmostEqual(float(result['scores']['bin_mass'][2,3]),.6,places=5)

    def test_nonzero_camera_offset_and_public_yaw_are_respected(self):
        row,e=scene()
        row['camera_in_body_m']=[0.,.2,1.]
        e['camera']['y']=.2;e['camera']['yaw']=10.
        matched=reference_readouts(row,e,10.)
        np.testing.assert_array_equal(matched['masks']['public_exact'],matched['masks']['native'])
        mismatched=reference_readouts(row,e,-10.)
        # Reference first hit uses the source-commanded pose, independently of
        # which public yaw supplies the corridor interval.
        np.testing.assert_array_equal(matched['reference_slant'],mismatched['reference_slant'])
        np.testing.assert_array_equal(matched['masks']['native'],mismatched['masks']['native'])
        self.assertGreater(mismatched['audit']['exact_public_changed_pixels'],0)
        self.assertEqual(mismatched['audit']['reference_minus_public_yaw_deg'],20.)
        # Translating camera, body and all AABBs together changes no geometry.
        shift=np.array([11.,-7.,3.]);moved=copy.deepcopy(e)
        moved['body_origin_m']=shift.tolist()
        for k,v in zip(('x','y','z'),shift):moved['camera'][k]+=float(v)
        for actor in moved['native_bounds']:actor['center_m']=(np.array(actor['center_m'])+shift).tolist()
        translated=reference_readouts(row,moved,10.)
        for arm in ARMS:np.testing.assert_array_equal(translated['masks'][arm],matched['masks'][arm])
        np.testing.assert_allclose(translated['reference_slant'],matched['reference_slant'],rtol=0,atol=1e-6)

    def test_unknown_occlusion_and_overflow_reference(self):
        row,e=scene();e['native_bounds']=[]
        result=reference_readouts(row,e,0.)
        self.assertFalse(result['known'].any())
        self.assertTrue(np.isnan(result['reference_slant']).all())
        self.assertIsNone(result['audit']['reference_slant_min_m'])
        for arm in ARMS:
            self.assertFalse(result['masks'][arm].any())
            self.assertFalse(result['scores'][arm].any())
        row,e=scene();e['native_bounds'].append(dict(name='occluder',center_m=[.125,0.,1.],extent_m=[.025,2.,2.]))
        occluded=reference_readouts(row,e,0.)
        self.assertTrue(occluded['known'].all())
        self.assertTrue(occluded['audit']['native_volume_truth'])
        self.assertTrue(occluded['audit']['native_volume_truth_vs_visible_mismatch'])
        self.assertFalse(occluded['masks']['native'].any())
        row,e=scene();e['native_bounds'][0]['center_m'][0]=9.
        distant=reference_readouts(row,e,0.)
        self.assertEqual(distant['audit']['reference_overflow_pixels'],35)
        self.assertFalse(distant['masks']['bin_midpoint'].any())
        self.assertFalse(distant['scores']['bin_mass'].any())


if __name__=='__main__':
    unittest.main()
