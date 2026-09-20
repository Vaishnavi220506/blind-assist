"""Synthetic geometry checks only; no capture, payload, model or training access."""
import copy
import unittest

import numpy as np

from reuse_corridor_geometry import (CUBE_ASSET, QUERY_LO, QUERY_HI, camera_obb,
    classify_cube, obb_intersects_aabb, ue_rotation)


def camera(**changes):
    return dict(dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.), **changes)


def cube(origin=(1., 0., 0.), scale=(.2, .2, .2), **rotation):
    return dict(mesh_asset=CUBE_ASSET, actor_origin_m=list(origin), scale=list(scale),
                rotation_deg=dict(dict(pitch=0., yaw=0., roll=0.), **rotation),
                mesh_local_bounds_cm=dict(min=[-50.]*3, max=[50.]*3))


class ReuseCorridorGeometryTests(unittest.TestCase):
    def test_ue_rotator_matches_local_capture_basis(self):
        from mz101_stereo_capture import basis
        for pose in (camera(), camera(yaw=90.), camera(pitch=90.), camera(roll=90.),
                     camera(yaw=63., pitch=-17., roll=31.)):
            r = ue_rotation({k: pose[k] for k in ('pitch', 'yaw', 'roll')})
            np.testing.assert_allclose(r, np.asarray(basis(pose)).T, atol=1e-15)
            np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(ue_rotation(dict(pitch=0.,yaw=0.,roll=90.))[:, 1], [0,0,-1], atol=1e-15)
        np.testing.assert_allclose(ue_rotation(dict(pitch=90.,yaw=0.,roll=0.))[:, 0], [0,0,1], atol=1e-15)

    def test_extent_not_center_and_closed_contact(self):
        self.assertTrue(classify_cube(camera(), cube((1.,.4,-.2),(.2,.4,.2)))['intersects'])
        touching = classify_cube(camera(), cube((1.,.4,-.2),(.2,.2,.2)))
        self.assertTrue(touching['intersects'])
        self.assertFalse(touching['near_lateral_outside'])
        outside = classify_cube(camera(), cube((1.,.4001,-.2),(.2,.2,.2)))
        self.assertFalse(outside['intersects'])
        self.assertTrue(outside['near_lateral_outside'])
        self.assertAlmostEqual(outside['lateral_separation_m'], .0001)
        self.assertFalse(classify_cube(camera(), cube((4.,.6,0.)))['near_lateral_outside'])
        self.assertFalse(classify_cube(camera(), cube((1.,.6,2.)))['near_lateral_outside'])

    def test_oblique_rod_aabb_false_positive(self):
        # Image-plane rod x+y=1.85; query has x+y<=1.2. Bounding envelopes overlap.
        result = classify_cube(camera(), cube((1., .65, -1.2), (.1, 2., .08), roll=-45.))
        self.assertTrue(np.all(np.array(result['aabb_hi']) >= QUERY_LO)
                        and np.all(np.array(result['aabb_lo']) <= QUERY_HI))
        self.assertFalse(result['intersects'])

    def test_cross_axis_is_required_beyond_six_face_axes(self):
        axes = ue_rotation(dict(pitch=18.553723123700067, yaw=-65.5227121270495, roll=42.46487866895407))
        center = np.array([.513795208256618, -.722437430910377, .8164417022940736])
        half = np.array([.3801573229792044, .7332400371933465, .4739785927649358])
        delta = center-(QUERY_LO+QUERY_HI)/2
        query_half = (QUERY_HI-QUERY_LO)/2
        for axis in [*np.eye(3), *axes.T]:
            self.assertLessEqual(abs(delta@axis), query_half@np.abs(axis)+half@np.abs(axes.T@axis))
        self.assertFalse(obb_intersects_aabb(center,axes,half))

    def test_yaw_translation_invariance(self):
        obj = cube((1.2,.31,-.23),(.15,.65,.09),pitch=17.,yaw=33.,roll=-21.)
        reference = camera_obb(camera(), obj)
        rotated = copy.deepcopy(obj)
        yaw = 127.
        r = ue_rotation(dict(pitch=0.,yaw=yaw,roll=0.))
        offset = np.array([101.,-48.,12.])
        rotated['actor_origin_m'] = (r@np.array(obj['actor_origin_m'])+offset).tolist()
        rotated['rotation_deg']['yaw'] += yaw
        pose = camera(x=offset[0],y=offset[1],z=offset[2],yaw=yaw)
        actual = camera_obb(pose, rotated)
        for k in reference:
            np.testing.assert_allclose(reference[k], actual[k], atol=1e-13)
        self.assertEqual(classify_cube(camera(),obj)['intersects'], classify_cube(pose,rotated)['intersects'])

    def test_full_camera_attitude_and_component_union(self):
        pose = camera(x=4.,y=-3.,z=2.,pitch=23.,yaw=-48.,roll=37.)
        angles = {k:pose[k] for k in ('pitch','yaw','roll')}
        r = ue_rotation(angles)
        location = r@np.array([1.,.2,-.1])+np.array([pose[k] for k in ('x','y','z')])
        obj = cube(location, (.3,.4,.5), **angles)
        np.testing.assert_allclose(camera_obb(pose,obj)['center'], [.2,.1,1.], atol=1e-14)
        self.assertTrue(classify_cube(pose,obj)['intersects'])
        # Two side posts enclose the query in aggregate AABB but their union is clear.
        components = [classify_cube(camera(),cube((1.,side,0.),(.2,.1,.2))) for side in (-.6,.6)]
        self.assertFalse(any(c['intersects'] for c in components))
        self.assertTrue(all(c['near_lateral_outside'] for c in components))

    def test_reject_unverified_non_cube_and_bad_transforms(self):
        for key,value in (('mesh_asset','/Engine/BasicShapes/Cylinder.Cylinder'),('scale',[1.,0.,1.]),
                          ('actor_origin_m',[float('nan'),0.,0.])):
            obj=cube();obj[key]=value
            with self.assertRaises(ValueError):classify_cube(camera(),obj)
        obj=cube();obj['mesh_local_bounds_cm']['max'][0]=51.
        with self.assertRaises(ValueError):classify_cube(camera(),obj)
        with self.assertRaises(ValueError):obb_intersects_aabb([0,0,1],np.ones((3,3)),[1,1,1])


if __name__ == '__main__':
    unittest.main()
