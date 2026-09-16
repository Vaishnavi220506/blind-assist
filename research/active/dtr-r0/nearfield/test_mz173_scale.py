import copy
import json
import math
import unittest
import numpy as np
from mz173_scale import calibrate


K = dict(width=640, height=360, fx=500., fy=450., cx=319.5, cy=179.5)
YY, XX = np.mgrid[:360, :640]
NORM = np.sqrt(1+((XX-K['cx'])/K['fx'])**2+((K['cy']-YY)/K['fy'])**2)


def target(distance=4., status='SIM_VALID'):
    return dict(distance_m=distance, status=status, range_noise_sigma_m=.02, signal_strength_proxy=.5)


def zone(zid=0, box=(430.25, 70.25, 620.75, 150.75), targets=None):
    l,t,r,b=box
    return dict(zone_id=zid, theta_bounds_deg=[math.degrees(math.atan((x-K['cx'])/K['fx'])) for x in (l,r)],
        phi_bounds_deg=[math.degrees(math.atan((K['cy']-y)/K['fy'])) for y in (b,t)],
        targets=[target()] if targets is None else targets)


def row(name='frame0', zones=None, received=True):
    return dict(id=name, rgb_intrinsics=copy.deepcopy(K), tof_packet_received=received,
                tof_zones=[zone()] if zones is None else zones)


class ScaleTests(unittest.TestCase):
    def test_off_axis_full_pixel_slant_scale_and_residual(self):
        z=3/NORM;r=row(zones=[zone(targets=[target(7.2)])])
        output,a=calibrate([r],[z])
        self.assertAlmostEqual(a['scale'],2.4,places=12)
        self.assertAlmostEqual(a['anchors'][0]['predicted_mean_slant'],3.,places=12)
        self.assertGreater(abs(float(z[71:151,431:621].mean())-3.),.05)
        np.testing.assert_allclose(output,7.2/NORM,rtol=1e-7)
        self.assertEqual(a['anchors'][0]['pixel_count'],80*190)
        self.assertAlmostEqual(a['anchors'][0]['abs_residual_m'],0.,places=12)
        self.assertEqual(a['ratio_mad'],0.);self.assertEqual(output.dtype,np.float32)

    def test_excluded_dual_merged_missing_partial_and_bad_pixels_no_mutation(self):
        zones=[zone(),zone(1,targets=[target(),target(3.)]),
               zone(2,targets=[target(),target(3.,'SIM_MERGED')]),zone(3,targets=[]),
               zone(4,box=(-10.,20.,10.,50.)),zone(5,box=(10.25,20.25,30.75,40.75))]
        rows=[row('first',zones),row('missing',received=False)]
        maps=[2/NORM,2/NORM];maps[0][21,11]=np.nan
        before_rows=copy.deepcopy(rows);before_maps=[m.copy() for m in maps]
        output,a=calibrate(rows,maps)
        self.assertEqual(a['anchor_count'],1);self.assertAlmostEqual(a['scale'],2.)
        self.assertEqual(a['exclusion_counts'],dict(MULTIPLE_USABLE_VALID_TARGETS=1,MERGED_TARGET_PRESENT=1,
            NO_USABLE_VALID_TARGET=1,INCOMPLETE_RGB_ZONE=1,INVALID_RELATIVE_ZONE_PIXELS=1,MISSING_TOF_PACKET=1))
        self.assertEqual(rows,before_rows)
        for old,new in zip(before_maps,maps):np.testing.assert_array_equal(old,new)
        self.assertTrue(np.isfinite(output).all());json.dumps(a,allow_nan=False)

    def test_median_unique_anchor_weight_and_last_duplicate_map(self):
        rows=[row('a'),row('b'),row('c')];maps=[2/NORM,(4/3)/NORM,.04/NORM]
        _,a=calibrate(rows,maps)
        self.assertAlmostEqual(a['scale'],3.);self.assertAlmostEqual(a['ratio_mad'],1.)
        # Repeated current has two view-position outputs: use only the last.
        duplicate=row('repeat');out,b=calibrate([duplicate,duplicate],[1/NORM,2/NORM])
        self.assertEqual(b['anchor_count'],1);self.assertEqual(b['duplicates_removed'],1)
        self.assertEqual(b['used_frame_ids'],['repeat']);self.assertAlmostEqual(b['scale'],2.)
        np.testing.assert_allclose(out,4/NORM,rtol=1e-7)

    def test_only_supplied_history_and_invalid_current_pixels_stay_unknown(self):
        rows=[row('past'),row('current',received=False),row('future')]
        maps=[2/NORM,3/NORM,.01/NORM]
        maps[1][0,:4]=[np.nan,np.inf,0.,-1.]
        out,a=calibrate(rows[:2],maps[:2])
        self.assertEqual(a['used_frame_ids'],['past','current'])
        self.assertEqual({x['frame_id'] for x in a['anchors']},{'past'})
        self.assertAlmostEqual(a['scale'],2.);self.assertTrue(np.isnan(out[0,:4]).all())
        self.assertEqual(a['current_invalid_pixels'],4)
        self.assertAlmostEqual(float(out[100,100]),float(maps[1][100,100]*2),places=6)
        # Native/evaluator-looking metadata cannot influence this public adapter.
        altered=copy.deepcopy(rows[:2]);altered[0].update(truth=False,native_bounds=[{'private':'ignored'}])
        out2,a2=calibrate(altered,maps[:2]);np.testing.assert_array_equal(out,out2);self.assertEqual(a,a2)

    def test_no_anchor_unknown_and_malformed_identity_rejected(self):
        out,a=calibrate([row(received=False)],[2/NORM])
        self.assertTrue(np.isnan(out).all());self.assertFalse(a['valid']);self.assertIsNone(a['scale'])
        self.assertIsNone(a['ratio_mad']);self.assertEqual(a['status'],'UNKNOWN_NO_ANCHORS')
        json.dumps(a,allow_nan=False)
        with self.assertRaises(ValueError):calibrate([],[])
        with self.assertRaises(ValueError):calibrate([row()],[np.ones((2,2))])
        with self.assertRaises(ValueError):calibrate([row(zones=[zone(),zone()])],[2/NORM])


if __name__=='__main__':unittest.main()
