import itertools
import unittest
import numpy as np
import mz118_interval_planes as m
from mz115_spatial_allocation import zone_box
from mz115_zonal_tof import zone_geometry

INTR=dict(cx=320.,cy=180.,fx=457.,fy=457.)


class PlaneTests(unittest.TestCase):
    def test_true_plane_in_every_feasible_anchor_and_roi_bound(self):
        beta=np.array([.4,.03,-.02]);anchors=[]
        for z in range(64):
            g,_=zone_geometry(z);b=zone_box(g,INTR)
            u=b[0]+.3*(b[2]-b[0]);v=b[1]+.7*(b[3]-b[1])
            ray=np.array([1.,(u-320)/457,(180-v)/457]);ray/=np.linalg.norm(ray)
            anchors.append(dict(zone_id=z,zone_box=b,range_m=1/(ray@beta)+(.03 if z%2 else -.03)))
        polys=m.octants(anchors,INTR);self.assertTrue(polys)
        self.assertTrue(any(all((np.array(p['A'])@beta)<=np.array(p['b'])+1e-10) and all(x*s>=0 for x,s in zip(beta,p['signs'])) for p in polys))
        for z in (0,19,44,63):
            b=anchors[z]['zone_box'];bounds=m.range_bounds(polys,b,INTR);self.assertIsNotNone(bounds)
            for u,v in itertools.product(np.linspace(b[0],b[2],5),np.linspace(b[1],b[3],5)):
                q=np.array([1.,(u-320)/457,(180-v)/457]);q/=np.linalg.norm(q);r=1/(q@beta)
                self.assertTrue(bounds[0]<=r<=bounds[1])

    def test_zero_coefficients_allowed_on_octant_boundary(self):
        anchors=[]
        for z in range(64):
            g,_=zone_geometry(z);b=zone_box(g,INTR)
            anchors.append(dict(zone_id=z,zone_box=b,range_m=2.5/m.prior.center_ray(b,INTR)[0]))
        polys=m.octants(anchors,INTR);self.assertTrue(polys)
        beta=np.array([.4,0,0])
        self.assertTrue(any(np.all(np.array(p['A'])@beta<=np.array(p['b'])) for p in polys))

    def test_unbounded_and_nonpositive_inversion_abstain(self):
        unbounded=[dict(signs=[1,1,1],A=[[0.,0,0]],b=[1.],bounds=[(0,None)]*3)]
        self.assertIsNone(m.range_bounds(unbounded,[300,170,340,190],INTR))
        bounded=[dict(signs=[1,1,1],A=np.eye(3).tolist(),b=[1.,1.,1.],bounds=[(0,None)]*3)]
        self.assertIsNone(m.range_bounds(bounded,[300,170,340,190],INTR))
        self.assertIsNone(m.octants([],INTR))

    def test_valid_and_missing_visual_preserved(self):
        item=dict(zone_id=0,status='SIM_VALID',zone_box=[0,0,30,30],range_m=2.,target_slot=0)
        pred=dict(proposals=[],spatial_evidence=[item],integrated_yaw_deg=0.)
        self.assertEqual(m.refine(dict(rgb_intrinsics=INTR),pred)['spatial_evidence'],[item])


if __name__=='__main__':unittest.main()
