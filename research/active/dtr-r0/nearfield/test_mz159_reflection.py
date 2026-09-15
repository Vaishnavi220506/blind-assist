import copy
import unittest
import numpy as np
from mz143_corridor_features import _public,extract
from mz136_boundary_geometry import camera_to_body,ray
from mz145_causal_confirmation import predict
from mz159_reflection import reflected,views,averaged_score
from test_mz143_corridor_features import inputs,target,feature


class ReflectionTests(unittest.TestCase):
    def test_involution_and_observed_slots(self):
        row,image=inputs();row['rgb_intrinsics']['cx']=311.2
        row['camera_in_body_m']=[.1,.04,1.6];row['camera_pitch_deg']=8.
        row.update(radar_range_m=[1.,1.,2.,None],radar_angle=[-15.,10.,30.,None],
                   radar_velocity=[-.2,.1,.3,None],radar_valid=[True,True,True,False])
        row['tof_zones'][0]['targets']=[target(2.),target(3.,'SIM_MERGED')]
        before=copy.deepcopy(row);pixels=image.copy()
        a,im,y=reflected(row,image,13.)
        self.assertEqual(row,before)
        b,im2,y2=reflected(a,im,y)
        self.assertEqual(b,_public(row));self.assertEqual(y2,13.)
        self.assertEqual(a['radar_angle'][:3],[-10.,15.,-30.])
        self.assertEqual(a['tof_zones'][7]['targets'],row['tof_zones'][0]['targets'])
        self.assertEqual(row,before);np.testing.assert_array_equal(image,pixels)
        np.testing.assert_array_equal(im2,image)

    def test_reflected_rays_and_supports_with_extrinsics(self):
        row,image=inputs();row['camera_in_body_m']=[.1,.07,1.6];row['camera_pitch_deg']=11.
        row['rgb_intrinsics']['cx']=313.25;row['tof_zones'][28]['targets']=[target()]
        rr,ii,yy=reflected(row,image,17.);reflection=np.diag([1.,-1.,1.])
        for u,v in [(20.,30.),(350.,179.),(611.,350.)]:
            np.testing.assert_allclose(ray(rr,639-u,v,yy),reflection@ray(row,u,v,17.),atol=1e-12)
        a=extract(row,image,17.);b=extract(rr,ii,yy);z=27
        for axis in ('x','z'):
            for endpoint in ('lo','hi'):
                self.assertAlmostEqual(feature(a,f'zone28.slot0.support_{axis}_{endpoint}'),
                                       feature(b,f'zone{z}.slot0.support_{axis}_{endpoint}'),places=5)
        self.assertAlmostEqual(feature(a,'zone28.slot0.support_y_lo'),-feature(b,'zone27.slot0.support_y_hi'),places=5)
        self.assertAlmostEqual(feature(a,'zone28.slot0.support_y_hi'),-feature(b,'zone27.slot0.support_y_lo'),places=5)

    def test_score_symmetry_and_causal_flags(self):
        row,image=inputs();row['tof_zones'][28]['targets']=[target()]
        x,_=views(row,image,7.);rr,ii,yy=reflected(row,image,7.);xx,_=views(rr,ii,yy)
        np.testing.assert_array_equal(x,xx[::-1])
        class Asymmetric:
            def predict_proba(self,x):
                p=1/(1+np.exp(-x[:,3]))
                return np.c_[1-p,p]
        a,pa=averaged_score(Asymmetric(),np.stack([x,x,x]))
        b,pb=averaged_score(Asymmetric(),np.stack([xx,xx,xx]))
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(pa,pb[:,::-1])
        rows=[dict(episode_id='a',time_s=i*.25) for i in range(3)]
        np.testing.assert_array_equal(predict(rows,a,.4,.6),predict(rows,b,.4,.6))


if __name__=='__main__':unittest.main()
