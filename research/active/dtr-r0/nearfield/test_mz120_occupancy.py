"""Focused integrity checks for the new spatial experiment boundary."""
import copy
import unittest
import numpy as np
from mz120_occupancy import encode, CORE, HEAD, CELLS
from run_mz120_occupancy import labels, evaluate, split


class OccupancyBoundaryTests(unittest.TestCase):
    def frame(self, objects):
        return dict(body_origin_m=[0,0,0], native_bounds=objects, family='near_rod_farwall_mixture')

    def obj(self, center, extent): return dict(center_m=center, extent_m=extent)

    def test_far_wall_and_above_head_do_not_become_current_alert(self):
        wall = labels(self.frame([self.obj([3.8,0,1.5],[.04,2,1.5])]))
        above = labels(self.frame([self.obj([2.8,0,2.4],[.1,.1,.1])]))
        self.assertFalse(wall[CORE].any()); self.assertFalse(above[CORE].any())
        head = labels(self.frame([self.obj([2.8,0,1.9],[.1,.1,.1])]))
        self.assertTrue(head[HEAD].any())

    def test_two_events_in_one_episode_are_not_hidden_by_one_alert(self):
        rows = [dict(episode_id='one', time_s=i*.25) for i in range(3)]
        target = np.zeros((3,len(CELLS)),bool); target[[0,2],np.flatnonzero(CORE)[0]]=True
        pred = target.astype(float); pred[2]=0
        result = evaluate(rows,[self.frame([])]*3,target,pred,.5,np.ones(3,bool))
        self.assertEqual(result['event_hits'], {'one/0':True,'one/1':False})
        self.assertEqual(result['events']['missed_positive_segments'],1)

    def test_wrong_height_alarm_does_not_mask_head_cell_miss(self):
        target = np.zeros((1,len(CELLS)),bool); target[0,np.flatnonzero(HEAD)[0]]=True
        pred = np.zeros_like(target,float); pred[0,np.flatnonzero(CORE & ~HEAD)[0]]=1
        r = evaluate([dict(episode_id='x',time_s=0)],[self.frame([])],target,pred,.5,np.ones(1,bool))
        self.assertEqual(r['metrics']['TP'],1); self.assertEqual(r['spatial']['HEAD']['FN'],1)

    def test_complete_episode_split(self):
        rows=[dict(id=str(i),episode_id='a' if i<2 else 'b') for i in range(4)]
        tr,dv=split([dict(name='test',rows=rows,es=[dict(family='pair')]*4)])
        self.assertEqual(tr.tolist(),[0,1]);self.assertEqual(dv.tolist(),[2,3])

    def test_weak_rod_in_mixed_family_is_not_hidden_by_bright_body(self):
        rod = dict(self.obj([2.4,0,1.1],[.03,.02,1.1]), evaluation_role='near_weak_rod')
        body = dict(self.obj([3.4,0,1.1],[.03,.15,1.1]), evaluation_role='far_bright_body')
        e = dict(self.frame([rod,body]),family='multitarget_competing_depth')
        target=labels(e)[None]; pred=labels(self.frame([body]))[None].astype(float)
        r=evaluate([dict(episode_id='mixed',time_s=0)],[e],target,pred,.5,np.ones(1,bool))
        self.assertEqual(r['metrics']['TP'],1)
        self.assertEqual(r['spatial']['rod_true_cells']['TP'],0)
        self.assertGreater(r['spatial']['rod_true_cells']['FN'],0)

    def test_packet_dropout_masks_stale_values_and_metadata_is_unused(self):
        zones=[dict(theta_bounds_deg=[-2,2],phi_bounds_deg=[-2,2], targets=[dict(
            status='SIM_MERGED',distance_m=3.,range_noise_sigma_m=.04,signal_strength_proxy=.1)])]
        row=dict(rgb_intrinsics=dict(width=640,height=360,cx=320,cy=180,fx=457,fy=457),
                 camera_pitch_deg=-3,camera_in_body_m=[0,0,1.7],tof_zones=zones,tof_packet_received=False,
                 radar_range_m=[3.],radar_angle=[0.],radar_velocity=[None],radar_valid=[True],
                 radar_packet_received=False,imu_valid=True)
        a=encode(row,0); changed=copy.deepcopy(row)
        changed.update(native_bounds='poison',id='label',family='positive',time_s=1000)
        changed['tof_zones'][0]['targets'][0]['distance_m']=1000
        changed['radar_range_m'][0]=1000
        b=encode(changed,0)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['tof'][0,3],0);self.assertEqual(a['radar'][0,4],0)


if __name__ == '__main__': unittest.main()
