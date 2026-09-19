"""Synthetic geometry/observability checks, never cohort-selected thresholds."""
import unittest

import cv2
import numpy as np

import ba_contour_parallax as m


class ContourGeometryTests(unittest.TestCase):
    def setUp(self):
        self.pose=dict(x=0.,y=0.,z=1.7,pitch=0.,yaw=0.,roll=0.)
        f=64/np.tan(np.deg2rad(50))
        self.k=dict(width=128,height=128,fx=f,fy=f,cx=63.5,cy=63.5)

    def test_projection_and_pitch(self):
        source={**self.pose,'y':-.2}
        uv,z=m.project([[63.5,63.5]],[.5,.25],source,self.pose,self.k)
        np.testing.assert_allclose(uv[:,0,0],63.5+self.k['fx']*.2*np.array([.5,.25]))
        np.testing.assert_allclose(z[:,0],[2.,4.])
        rotation,_=m.pose_matrix({**self.pose,'pitch':-10})
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
        self.assertLess(rotation[2,2],0)

    def test_rotation_has_no_depth_dependent_displacement(self):
        uv,_=m.project([[40,60],[80,60]],[.1,.5,1.5],{**self.pose,'yaw':5},self.pose,self.k)
        np.testing.assert_allclose(uv[0],uv[1],atol=1e-12)
        np.testing.assert_allclose(uv[1],uv[2],atol=1e-12)

    def test_known_plane_and_missing_motion(self):
        rng=np.random.default_rng(51)
        reference=rng.uniform(.1,.9,(128,128)).astype(np.float32)
        offsets=[-.2,-.1,0.]
        images=[cv2.warpAffine(reference,np.array([[1,0,-offset*self.k['fx']/2],[0,1,0]],np.float32),
                   (128,128),borderMode=cv2.BORDER_REFLECT) for offset in offsets]
        poses=[{**self.pose,'y':offset} for offset in offsets]
        ends=np.array([[28.,64.],[100.,64.]])
        result=m.match_line(images,poses,self.k,ends,72.)
        self.assertTrue(result['accepted'],result)
        self.assertLess(abs(result['depth_m']-2),.1)
        self.assertLessEqual(result['interval_m'][0],2.)
        self.assertGreaterEqual(result['interval_m'][1],2.)
        zero=m.match_line([reference]*3,[self.pose]*3,self.k,ends,72.)
        self.assertFalse(zero['accepted'])
        self.assertFalse(zero['checks']['metric_baseline'])

    def test_parallel_infinite_edge_stays_ambiguous(self):
        image=np.full((128,128),.2,np.float32);image[64:]=.8
        poses=[{**self.pose,'y':offset} for offset in (-.2,-.1,0.)]
        result=m.match_line([image]*3,poses,self.k,np.array([[28.,64.],[100.,64.]]),72.)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['checks']['unique'])

    def test_textureless_input_never_accepts_depth(self):
        image=np.full((128,128),.5,np.float32)
        poses=[{**self.pose,'x':offset} for offset in (-.2,-.1,0.)]
        result=m.match_line([image]*3,poses,self.k,np.array([[28.,64.],[100.,64.]]),72.)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['checks']['correlation'])


if __name__=='__main__':
    cv2.setNumThreads(1);unittest.main()
