import copy
import unittest

import numpy as np

from mz153_temporal_depth import make_input, full_rgb_depth, prefix_indices


def observation():
    return dict(id='test', tof_packet_received=True, rgb_intrinsics=dict(
        width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.),
        tof_zones=[dict(zone_id=r*8+c,theta_bounds_deg=[-22.5+c*5.625,-22.5+(c+1)*5.625],
                        phi_bounds_deg=[22.5-(r+1)*5.625,22.5-r*5.625],targets=[])
                   for r in range(8) for c in range(8)])


class AdapterTests(unittest.TestCase):
    def test_orientation_strongest_and_missing(self):
        row=observation()
        row['tof_zones'][10]['targets']=[dict(status='SIM_VALID',distance_m=2.,signal_strength_proxy=1.),
                                        dict(status='SIM_VALID',distance_m=3.,signal_strength_proxy=5.)]
        row['tof_zones'][11]['targets']=[dict(status='SIM_MERGED',distance_m=1.,signal_strength_proxy=10.)]
        rgb,d,audit=make_input(row,np.full((360,640,3),255,np.uint8))
        self.assertEqual(audit['used'][0]['cell'],[1,2])
        self.assertEqual(audit['used'][0]['slot'],1)
        self.assertEqual(np.count_nonzero(d),1)
        self.assertLess(d[0,1,2],.75)
        self.assertEqual(audit['missing_cells'],63)
        self.assertTrue(np.all(rgb[:,0]==0))
        self.assertTrue(np.all(rgb[:,64]==1))

    def test_roundtrip_constant_preserves_meters_and_unknown_edges(self):
        depth,shared=full_rgb_depth(observation(),np.full((128,128),.5,np.float32))
        self.assertAlmostEqual(depth[180,320],2.)
        self.assertFalse(shared[180,0])
        self.assertTrue(np.isnan(depth[180,0]))
        self.assertTrue(np.allclose(depth[np.isfinite(depth)],2.))

    def test_gap_reset_and_no_future(self):
        rows=[dict(episode_id='a',time_s=t) for t in [0,.25,.5,1.]]
        rows.append(dict(episode_id='b',time_s=0.))
        self.assertEqual(prefix_indices(rows,0),[0,0])
        self.assertEqual(prefix_indices(rows,2),[0,1,2])
        self.assertEqual(prefix_indices(rows,3),[3,3])
        self.assertEqual(prefix_indices(rows,4),[4,4])
        changed=copy.deepcopy(rows);changed[3]['time_s']=-99
        self.assertEqual(prefix_indices(rows,2),prefix_indices(changed,2))

    def test_unknown_interpolation_contributor_stays_unknown(self):
        for invalid in (0.,-.5,float('nan')):
            raw=np.full((128,128),.5,np.float32);raw[64,64]=invalid
            depth,shared=full_rgb_depth(observation(),raw)
            self.assertTrue(shared[180,320])
            self.assertTrue(np.isnan(depth[180,320]))


if __name__=='__main__':
    unittest.main()
