import itertools
import unittest
import numpy as np
from mz115_zonal_tof import zone_geometry
from mz115_spatial_allocation import zone_box
import mz117_surface_intervals as m

INTR=dict(cx=320.,cy=180.,fx=457.,fy=457.)


class SurfaceTests(unittest.TestCase):
    def test_unit_intervals_include_interior_normalization_extrema(self):
        b=[260,130,380,230];intervals=m.unit_ray_intervals(b,INTR)
        for u,v in itertools.product(np.linspace(b[0],b[2],13),np.linspace(b[1],b[3],13)):
            q=np.array([1.,(u-320)/457,(180-v)/457]);q/=np.linalg.norm(q)
            self.assertTrue(all(lo-1e-12<=x<=hi+1e-12 for x,(lo,hi) in zip(q,intervals)))

    def test_true_plane_enclosed_with_within_zone_anchor_offsets(self):
        beta=np.array([.4,.03,-.02]);anchors=[]
        for z in range(64):
            geometry,_=zone_geometry(z);b=zone_box(geometry,INTR)
            u=b[0]+.3*(b[2]-b[0]);v=b[1]+.7*(b[3]-b[1]);q=np.array([1.,(u-320)/457,(180-v)/457]);q/=np.linalg.norm(q)
            r=1/float(q@beta)+(.03 if z%2 else -.03)
            anchors.append(dict(zone_id=z,target_slot=0,zone_box=b,range_m=r))
        model=m.fit_plane(anchors,INTR)
        self.assertIsNotNone(model);self.assertIsNotNone(model['coefficient_error'])
        self.assertTrue(all(abs(a-b)<=e for a,b,e in zip(beta,model['beta'],model['coefficient_error'])))
        for z in (0,19,44,63):
            b=anchors[z]['zone_box'];bounds=m.range_interval(model,b,INTR);self.assertIsNotNone(bounds)
            for u,v in itertools.product(np.linspace(b[0],b[2],5),np.linspace(b[1],b[3],5)):
                q=np.array([1.,(u-320)/457,(180-v)/457]);q/=np.linalg.norm(q);r=1/float(q@beta)
                self.assertTrue(bounds[0]<=r<=bounds[1])

    def test_rank_and_denominator_reject_without_fabricated_range(self):
        self.assertIsNone(m.fit_plane([],INTR))
        self.assertIsNone(m.range_interval(dict(beta=[0,0,0],coefficient_error=[1,1,1]),[300,170,340,190],INTR))
        self.assertIsNone(m.range_interval(dict(beta=[.4,0,0],coefficient_error=None),[300,170,340,190],INTR))

    def test_missing_anchors_and_valid_return_are_preserved(self):
        item=dict(zone_id=0,target_slot=0,status='SIM_VALID',zone_box=[10,10,20,20],range_m=2.,range_bounds=[1.88,2.12],localized_xyz=[[1,3],[0,.2],[1,2]])
        p=dict(proposals=[],spatial_evidence=[item],integrated_yaw_deg=0.)
        out=m.refine(dict(rgb_intrinsics=INTR),p)
        self.assertEqual(out['spatial_evidence'],[item]);self.assertEqual(out['surface_models'],[])

    def test_proxy_padding_precedes_sensor_clipping(self):
        model=dict(beta=[1/4.05,0.,0.],coefficient_error=[0.,0.,0.])
        box=[319.99,179.99,320.01,180.01]
        self.assertIsNone(m.range_interval(model,box,INTR))
        bounds=m.range_interval(model,box,INTR,m.RANGE_ERROR_M)
        self.assertAlmostEqual(bounds[0],3.92,places=7)
        self.assertEqual(bounds[1],4.)


if __name__=='__main__':unittest.main()
