"""Synthetic-only calibrated geometry, tracking and causal-state tests."""
import copy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mz168_visual_yaw import CausalVisualYaw, _decompose_yaw, cv_to_body, estimate_pair


INTR=dict(width=640,height=360,fx=500.,fy=500.,cx=320.,cy=180.)
K=np.array([[500.,0.,320.],[0.,500.,180.],[0.,0.,1.]])


def analytic_pair(pitch_ref=2.,pitch_cur=3.,ref_yaw=0.,cur_yaw=.6,translation=(0.,.01,0.)):
    r=cv_to_body(pitch_cur,cur_yaw).T@cv_to_body(pitch_ref,ref_yaw)
    h=K@(r+np.outer(translation,[0.,0.,1.]))@np.linalg.inv(K)
    xx,yy=np.meshgrid(np.linspace(100.,540.,10),np.linspace(60.,300.,7))
    p=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float32)
    q=cv2.perspectiveTransform(p[:,None],h)[:,0]
    return h,p,q,r


def row(episode='a',time=0.,delta=0.):
    return dict(episode_id=episode,time_s=time,delta_yaw=delta,imu_valid=True,
                rgb_intrinsics=dict(INTR),camera_pitch_deg=0.,camera_in_body_m=[0.,0.,1.5])


class VisualYawTests(unittest.TestCase):
    def test_calibrated_rotation_translation_sign_and_projection(self):
        h,p,q,r=analytic_pair()
        result=_decompose_yaw(h,K,p,q,2.,3.,0.,.65)
        self.assertTrue(result['accepted'],result['audit'])
        self.assertAlmostEqual(result['yaw_deg'],.6,places=7)
        self.assertEqual(result['audit']['positive_depth_filter'],'NORMALIZED_VISIBLE_REFERENCE_POINTS')
        self.assertLessEqual(result['audit']['surviving_branch_yaw_spread_deg'],.1)
        chosen=result['audit']['candidates'][result['audit']['best_solution_index']]['rotation']
        np.testing.assert_allclose(chosen,r,rtol=0,atol=1e-9)
        self.assertGreater(len(result['audit']['candidates']),1)
        json.dumps(result,allow_nan=False)
        # If a build cannot discard opposite normals, +/-t,n duplicates still
        # count as one rotation each, not as additional ambiguous yaw branches.
        with patch('mz168_visual_yaw.cv2.filterHomographyDecompByVisibleRefpoints',return_value=np.arange(4,dtype=np.int32)[:,None]):
            unfiltered=_decompose_yaw(h,K,p,q,2.,3.,0.,.65)
        self.assertTrue(unfiltered['accepted'],unfiltered['audit'])
        self.assertEqual(len(unfiltered['audit']['candidates']),4)
        self.assertEqual(unfiltered['audit']['distinct_rotation_count'],2)

    def test_pure_rotation_nonzero_reference_and_imu_rejection(self):
        h,p,q,_=analytic_pair(pitch_ref=4.,pitch_cur=1.,ref_yaw=19.,cur_yaw=19.7,translation=(0.,0.,0.))
        result=_decompose_yaw(h,K,p,q,4.,1.,19.,19.8)
        self.assertTrue(result['accepted'],result['audit'])
        self.assertAlmostEqual(result['yaw_deg'],19.7,places=8)
        self.assertEqual(result['audit']['distinct_rotation_count'],1)
        self.assertIn('PURE_ROTATION',result['audit']['positive_depth_filter'])
        rejected=_decompose_yaw(h,K,p,q,4.,1.,19.,23.)
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['yaw_deg'],23.)
        self.assertEqual(rejected['audit']['reason'],'ROTATION_DISAGREES_WITH_IMU')

    def test_ambiguous_oblique_translation_is_not_resolved_by_imu(self):
        h,p,q,_=analytic_pair(pitch_ref=0.,pitch_cur=0.,cur_yaw=.6,translation=(.02,0.,.05))
        result=_decompose_yaw(h,K,p,q,0.,0.,0.,.6)
        self.assertFalse(result['accepted'],result['audit'])
        self.assertEqual(result['audit']['reason'],'AMBIGUOUS_YAW_BRANCHES')
        self.assertGreater(result['audit']['surviving_branch_yaw_spread_deg'],.1)
        self.assertLess(result['audit']['best_difference_deg'],1e-6)
        self.assertIsNotNone(result['audit']['best_second_margin_deg'])
        self.assertEqual(result['yaw_deg'],.6)

    def test_near_pure_nonrotation_and_unavailable_filter_fail_closed(self):
        # OpenCV 4.10's near-pure shortcut returns normalized H for this tiny
        # translation. It must not masquerade as an exact SO(3) estimate.
        h,p,q,_=analytic_pair(pitch_ref=0.,pitch_cur=0.,cur_yaw=.2,translation=(.0001,0.,0.))
        result=_decompose_yaw(h,K,p,q,0.,0.,0.,.2)
        self.assertFalse(result['accepted'],result['audit'])
        self.assertEqual(result['audit']['reason'],'INVALID_ROTATION_SOLUTION')
        self.assertGreater(result['audit']['candidates'][0]['orthogonality_frobenius_error'],1e-6)
        h,p,q,_=analytic_pair()
        with patch('mz168_visual_yaw.cv2.filterHomographyDecompByVisibleRefpoints',side_effect=cv2.error('synthetic unavailable')):
            failed=_decompose_yaw(h,K,p,q,2.,3.,0.,.65)
        self.assertFalse(failed['accepted'])
        self.assertEqual(failed['audit']['reason'],'POSITIVE_DEPTH_FILTER_FAILED')
        self.assertEqual(failed['yaw_deg'],.65)

    def test_native_tracking_and_no_feature_fallback(self):
        blank=np.zeros((360,640,3),np.uint8)
        result=estimate_pair(blank,blank,INTR,0.,0.,0.,.2)
        self.assertEqual(result['audit']['reason'],'INSUFFICIENT_CORNERS')
        self.assertEqual(result['yaw_deg'],.2)
        rng=np.random.default_rng(168016)
        image=cv2.GaussianBlur(rng.integers(0,256,blank.shape,dtype=np.uint8),(5,5),.8)
        h,_,_,_=analytic_pair(pitch_ref=0.,pitch_cur=0.,cur_yaw=.3,translation=(0.,.01,0.))
        moved=cv2.warpPerspective(image,h,(640,360))
        before=image.copy();result=estimate_pair(image,moved,INTR,0.,0.,0.,.35)
        self.assertTrue(result['accepted'],result['audit'])
        self.assertAlmostEqual(result['yaw_deg'],.3,delta=.05)
        self.assertGreaterEqual(result['audit']['inliers'],40)
        np.testing.assert_array_equal(image,before)
        again=estimate_pair(image,moved,INTR,0.,0.,0.,.35)
        self.assertEqual(result,again)

    def test_anchor_gap_reset_expiration_and_no_visual_feedback(self):
        image=np.zeros((360,640,3),np.uint8)
        def proposal(reference,current,intr,p0,p1,y0,imu):
            return dict(yaw_deg=imu+.2,imu_yaw_deg=imu,accepted=True,audit=dict(reason='ACCEPTED'))
        with patch('mz168_visual_yaw.estimate_pair',side_effect=proposal) as called:
            state=CausalVisualYaw()
            self.assertEqual(state.update(row(time=.25),image)['audit']['reason'],'NONZERO_TIME_EPISODE_START')
            self.assertFalse(state.update(row(time=.5,delta=.1),image)['accepted'])
            self.assertEqual(called.call_count,0)
            self.assertEqual(state.update(row('b',0.),image)['audit']['reason'],'ANCHOR_ESTABLISHED')
            a=state.update(row('b',.25,.1),image)
            b=state.update(row('b',.5,.1),image)
            self.assertAlmostEqual(a['yaw_deg'],.3)
            self.assertAlmostEqual(b['imu_yaw_deg'],.2)
            self.assertAlmostEqual(b['yaw_deg'],.4)
            self.assertEqual(state.update(row('b',1.,.1),image)['audit']['reason'],'TIME_GAP_INVALIDATED_ANCHOR')
            self.assertFalse(state.update(row('b',1.25,.1),image)['accepted'])
            self.assertEqual(called.call_count,2)
            state.update(row('c',0.),image)
            for i in range(1,7):self.assertTrue(state.update(row('c',i*.25,.1),image)['accepted'])
            expired=state.update(row('c',1.75,.1),image)
            self.assertEqual(expired['audit']['reason'],'REFERENCE_EXPIRED')
            state.update(row('d',0.),image)
            invalid=row('d',.25,float('nan'));invalid['imu_valid']=False
            failure=state.update(invalid,image)
            self.assertEqual(failure['audit']['reason'],'INVALID_IMU_INVALIDATED_ANCHOR')
            self.assertEqual(failure['imu_yaw_deg'],0.)
            resumed=state.update(row('d',.5,.1),image)
            self.assertEqual(resumed['audit']['reason'],'INVALID_IMU_INVALIDATED_ANCHOR')
            self.assertAlmostEqual(resumed['imu_yaw_deg'],.1)
            json.dumps(failure,allow_nan=False)

    def test_metadata_immutability_and_prefix_causality(self):
        image=np.zeros((360,640,3),np.uint8);images=[image.copy() for _ in range(4)]
        rows=[row(time=i*.25,delta=.03 if i else 0.) for i in range(4)]
        decorated=copy.deepcopy(rows)
        for r in decorated:r.update(id='ignored',family='private_fake',truth=True,native_bounds=[{'yaw':50}])
        before=copy.deepcopy(decorated)
        def execute(rr):
            state=CausalVisualYaw()
            return [state.update(r,im) for r,im in zip(rr,images)]
        reference=execute(rows)
        self.assertEqual(reference,execute(decorated))
        self.assertEqual(decorated,before)
        for n in range(1,5):self.assertEqual(execute(rows[:n]),reference[:n])
        for im in images:np.testing.assert_array_equal(im,image)


if __name__=='__main__':
    unittest.main()
