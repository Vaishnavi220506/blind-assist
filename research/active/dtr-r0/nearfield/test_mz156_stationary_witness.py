import copy
import unittest
from mz156_stationary_witness import predict


def row(i=0,episode='a',**overrides):
    value=dict(id=f'{episode}{i}',episode_id=episode,time_s=i*.25,
        camera_in_body_m=[0.,0.,1.7],camera_pitch_deg=-3.,
        rgb_intrinsics=dict(fx=450.,fy=450.,cx=320.,cy=180.),
        imu_valid=True,delta_yaw=0.,delta_pitch=0.,tof_packet_received=True,
        tof_zones=[dict(zone_id=28,theta_bounds_deg=[0.,5.625],phi_bounds_deg=[0.,5.625],
            targets=[dict(status='SIM_VALID',distance_m=1.2,range_noise_sigma_m=.04)])])
    value.update(overrides);return value


class StationaryWitnessTests(unittest.TestCase):
    def test_missing_packet_preserves_only_past_positive_and_baseline(self):
        rows=[row(),row(1,tof_packet_received=False),row(2,tof_packet_received=False)]
        got=predict(rows,[False,False,True],stationary_world=True)
        self.assertEqual([p['current_witness'] for p in got],[True,False,False])
        self.assertEqual([p['prefix_candidate'] for p in got],[True,True,True])
        self.assertEqual(got[-1]['prefix_supports'][0]['source_id'],'a0')
        self.assertEqual(got[-1]['prefix_supports'][0]['evidence_age_s'],.5)
        self.assertEqual(predict(rows,[False,False,True]),predict(rows,[False,False,True],stationary_world=False))
        self.assertEqual([p['prefix_candidate'] for p in predict(rows,[False,False,True])],[False,False,True])

    def test_invalid_pose_and_gaps_disable_until_new_aligned_episode(self):
        rows=[row(),row(1,imu_valid=False),row(2),row(0,episode='b'),row(2,episode='b'),row(3,episode='b')]
        got=predict(rows,[False]*6,stationary_world=True)
        self.assertEqual([p['prefix_witness'] for p in got],[True,False,False,True,False,False])
        self.assertFalse(predict([row(1)],[False],stationary_world=True)[0]['active'])
        changed=row(1);changed['rgb_intrinsics']['fx']=500.
        got=predict([row(),changed,row(2)],[False]*3,stationary_world=True)
        self.assertEqual([p['active'] for p in got],[True,False,False])

    def test_full_support_range_and_slot_status(self):
        r=row();second=copy.deepcopy(r['tof_zones'][0]['targets'][0])
        r['tof_zones'][0]['targets'][0]['status']='SIM_MERGED';r['tof_zones'][0]['targets'].append(second)
        witness=predict([r],[False],stationary_world=True)[0]['current_supports'][0]
        self.assertEqual(witness['target_slot'],1)
        self.assertAlmostEqual(witness['range_bounds_m'][0],.97)
        self.assertAlmostEqual(witness['range_bounds_m'][1],1.43)
        r['tof_zones'][0]['theta_bounds_deg']=[0.,22.5]
        self.assertFalse(predict([r],[False],stationary_world=True)[0]['current_witness'])

    def test_window_causality_and_no_metadata_dependence(self):
        rows=[row(i) for i in range(7)]
        full=predict(rows,[False]*7,stationary_world=True)
        self.assertFalse(full[6]['active']);self.assertFalse(full[6]['prefix_witness'])
        self.assertEqual(full[:3],predict(rows[:3],[False]*3,stationary_world=True))
        changed=copy.deepcopy(rows)
        for r in changed:r.update(truth=False,family='hidden',native_bounds=[1,2],observation_arm='unknown')
        self.assertEqual(full,predict(changed,[False]*7,stationary_world=True))


if __name__=='__main__':unittest.main()
