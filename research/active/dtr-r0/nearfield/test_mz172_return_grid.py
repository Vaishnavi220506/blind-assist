import copy
import math
import unittest
import numpy as np
from mz172_return_grid import make_zone_grid


def fixture():
    zones=[]
    for zid in range(64):
        iy,ix=divmod(zid,8);lo=-22.5+ix*5.625;hi=22.5-iy*5.625
        zones.append(dict(zone_id=zid,theta_bounds_deg=[lo,lo+5.625],phi_bounds_deg=[hi-5.625,hi],targets=[]))
    return dict(rgb_intrinsics=dict(width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.),tof_zones=zones)


class ZoneGridTest(unittest.TestCase):
    def test_order_coordinates_and_full_box(self):
        row=fixture();row['tof_zones'].reverse();out=make_zone_grid(row)
        self.assertEqual(out['grid'].shape,(64,64,2));self.assertEqual(out['grid'].dtype,np.float32)
        self.assertEqual(out['visible'].shape,(64,64));self.assertEqual(out['visible'].dtype,np.bool_)
        z=next(z for z in row['tof_zones'] if z['zone_id']==0);k=row['rgb_intrinsics']
        l=k['cx']+k['fx']*math.tan(math.radians(z['theta_bounds_deg'][0]))
        r=k['cx']+k['fx']*math.tan(math.radians(z['theta_bounds_deg'][1]))
        t=k['cy']-k['fy']*math.tan(math.radians(z['phi_bounds_deg'][1]))
        b=k['cy']-k['fy']*math.tan(math.radians(z['phi_bounds_deg'][0]))
        expected=np.array([[2*(l+(i+.5)*(r-l)/8)/639-1,2*(t+(j+.5)*(b-t)/8)/359-1]
            for j in range(8) for i in range(8)],np.float32)
        np.testing.assert_allclose(out['grid'][0],expected,rtol=0,atol=1e-7)
        pixels=(out['grid'][0].astype(float)+1)*[639/2,359/2]
        self.assertTrue(np.all((pixels[:,0]>l)&(pixels[:,0]<r)&(pixels[:,1]>t)&(pixels[:,1]<b)))
        row['tof_zones'].reverse();np.testing.assert_array_equal(out['grid'],make_zone_grid(row)['grid'])

    def test_partial_and_wholly_outside_never_shrink(self):
        row=fixture();row['tof_zones'][0].update(theta_bounds_deg=[0.,2.],phi_bounds_deg=[20.,24.])
        row['tof_zones'][1].update(theta_bounds_deg=[60.,65.],phi_bounds_deg=[0.,2.])
        out=make_zone_grid(row)
        self.assertGreater(out['visible'][0].sum(),0);self.assertLess(out['visible'][0].sum(),64)
        self.assertFalse(out['visible'][1].any());self.assertTrue(np.all(out['grid'][1,:,0]>1))
        self.assertTrue(np.any(out['grid'][0,:,1]<-1))
        self.assertEqual(out['audit']['sensor_slots_dropped'],0)

    def test_packet_private_metadata_and_input_immutability(self):
        row=fixture();saved=copy.deepcopy(row);a=make_zone_grid(row);self.assertEqual(row,saved)
        row.update(id='private-id',family='private-family',truth=False,tof_packet_received=False,radar_packet_received=False,imu_valid=False)
        for z in row['tof_zones']:z['targets']=[dict(distance_m=None,status='SIM_MERGED')]
        b=make_zone_grid(row)
        np.testing.assert_array_equal(a['grid'],b['grid']);np.testing.assert_array_equal(a['visible'],b['visible'])
        self.assertEqual(a['audit'],b['audit'])

    def test_reject_malformed_geometry(self):
        for mutate in (lambda r:r['tof_zones'].pop(),lambda r:r['tof_zones'][0].update(zone_id=1),
            lambda r:r['tof_zones'][0].update(theta_bounds_deg=[1.,1.]),
            lambda r:r['tof_zones'][0].update(phi_bounds_deg=[float('nan'),1.]),
            lambda r:r['rgb_intrinsics'].update(fx=0.),lambda r:r['rgb_intrinsics'].update(width=0)):
            row=fixture();mutate(row)
            with self.assertRaises(ValueError):make_zone_grid(row)


if __name__=='__main__':unittest.main()
