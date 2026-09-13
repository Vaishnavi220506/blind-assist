import unittest
import numpy as np
import mz119_parallax as m

INTR=dict(fx=457.,fy=457.,cx=320.,cy=180.)


def pixel(p):return [320+457*p[0]/p[2],180+457*p[1]/p[2]]


def anchors(t,R=np.eye(3)):
    out=[]
    for iy,y in enumerate((-.8,-.3,.3,.8)):
        for ix,x in enumerate((-.8,-.3,.3,.8)):
            p=np.array([x,y,3.5]);out.append(dict(zone_id=iy*8+ix,reference_pixel=pixel(p),pixel=pixel(R@p+t),range_m=float(np.linalg.norm(p))))
    return out


class ParallaxTests(unittest.TestCase):
    def test_static_metric_translation_and_foreground_depth_contained(self):
        t=np.array([-.12,0.,0.]);a=anchors(t);selected,estimate=m.robust_pose(a,np.eye(3),INTR)
        self.assertTrue(np.allclose(estimate,t));poly,bounds,zero=m.pose_poly(selected,np.eye(3),np.zeros((3,3)),INTR)
        self.assertFalse(zero);self.assertTrue(all(lo<=v<=hi for v,(lo,hi) in zip(t,bounds)))
        point=np.array([.08,.05,2.6]);r=m.depth_bounds(poly,pixel(point),pixel(point+t),np.eye(3),np.zeros((3,3)),INTR)
        self.assertLess(r[1]-r[0],2);self.assertTrue(r[0]<=np.linalg.norm(point+t)<=r[1])

    def test_no_motion_and_uncertain_pure_rotation_include_zero(self):
        for angle in (0.,.2):
            R=m.basis(-3,angle).T@m.basis(-3,0);a=anchors(np.zeros(3),R)
            poly,bounds,zero=m.pose_poly(a,np.eye(3),np.abs(R-np.eye(3))+1e-7,INTR)
            self.assertTrue(zero)

    def test_coherent_anchor_motion_is_explicitly_not_identifiable(self):
        # Same images/ranges as translated camera: static-scene interpretation is conditional.
        a=anchors(np.array([.12,0,0]));_,estimate=m.robust_pose(a,np.eye(3),INTR)
        self.assertAlmostEqual(estimate[0],.12)
        self.assertIsNone(m.robust_pose(a[:3],np.eye(3),INTR))

    def test_normalization_error_includes_sampled_pixel_perturbations(self):
        p=[100,250];q=m.ray(p,INTR);err=m.ray_error(p,INTR)
        for u in np.linspace(-.5,.5,7):
            for v in np.linspace(-.5,.5,7):self.assertTrue(np.all(np.abs(m.ray([p[0]+u,p[1]+v],INTR)-q)<=err))


if __name__=='__main__':unittest.main()
