"""Focused pre-score checks for native geometry and ambiguity fallbacks."""
import copy
import unittest

from mz131_native_angular import native_cohorts
from mz115_zonal_tof import zone_geometry
from mz124_measurement_geometry import zone_box


def packet(entries):
    zones=[]
    for i,targets in entries.items():
        z,_=zone_geometry(i)
        z['targets']=[dict(status=s,distance_m=r,range_noise_sigma_m=.04) for s,r in targets]
        zones.append(z)
    return dict(tof_packet_received=True,tof_zones=zones)


class NativeAngularTest(unittest.TestCase):
    def test_native_centroid_span_and_unspanned_axis(self):
        p=packet({27:[('SIM_VALID',1.)],28:[('SIM_VALID',1.)]})
        selected,groups,_=native_cohorts(p)
        self.assertEqual(set(selected),{27,28})
        self.assertEqual(groups[0]['angular']['theta_bounds_deg'],[-2.8125,2.8125])
        self.assertEqual(groups[0]['angular']['phi_bounds_deg'],[0.,5.625])
        self.assertEqual(groups[0]['centroid_deg']['theta_bounds_deg'],0.)

    def test_single_merged_and_multitarget_fallback(self):
        p=packet({27:[('SIM_VALID',1.)],28:[('SIM_VALID',1.),('SIM_VALID',3.)],
            35:[('SIM_MERGED',1.)]})
        chosen,_,reasons=native_cohorts(p)
        self.assertFalse(chosen)
        self.assertEqual(reasons[27],'SINGLE_ZONE')
        self.assertEqual(reasons[28],'MULTI_TARGET_OR_EMPTY')
        self.assertEqual(reasons[35],'NON_VALID_OR_MERGED')

    def test_depth_chain_is_not_single_surface(self):
        p=packet({24:[('SIM_VALID',1.)],25:[('SIM_VALID',1.2)],26:[('SIM_VALID',1.4)]})
        chosen,groups,_=native_cohorts(p)
        self.assertFalse(chosen)
        self.assertEqual(groups[0]['reason'],'GLOBAL_DEPTH_CONFLICT')

    def test_no_row_wrap_and_order_or_rgb_dependence(self):
        p=packet({7:[('SIM_VALID',1.)],8:[('SIM_VALID',1.)]})
        self.assertFalse(native_cohorts(p)[0])
        p=packet({27:[('SIM_VALID',1.)],28:[('SIM_VALID',1.1)]})
        expected=native_cohorts(p)
        p['tof_zones'].reverse();p['proposals']=[[-1000,-1000,1000,1000]]
        p['rgb_intrinsics']={'width':1,'height':1}
        self.assertEqual(native_cohorts(p),expected)

    def test_missing_packet_and_off_image_native_bounds(self):
        p=packet({60:[('SIM_VALID',1.)],61:[('SIM_VALID',1.)]})
        chosen,_,_=native_cohorts(p)
        intr=dict(cx=320,cy=180,fx=450,fy=450,width=640,height=360)
        box=zone_box(chosen[60],intr)
        self.assertGreater(box[3],intr['height'])
        q=copy.deepcopy(p);q['tof_packet_received']=False
        self.assertFalse(native_cohorts(q)[0])


if __name__=='__main__':unittest.main()
