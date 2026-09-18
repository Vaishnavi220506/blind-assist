import copy
import unittest
from mz179_temporal_ceiling import ceiling, features, prepare, radar_features, slope


class CeilingTests(unittest.TestCase):
    def frame(self,t,radar):
        return dict(t=t,radar=radar,tof_valid=0,tof_packet=False,tof_invalid=0,
                    yaw_delta=0,pitch_delta=0,angular_velocity=None,raw_score=.5,astar_score=.5)

    def test_unknown_velocity_not_zero(self):
        f=radar_features([self.frame(0,[dict(range=2,bearing=0,velocity=None)])],.25)
        self.assertIsNone(f['doppler_median'])
        self.assertIsNone(f['doppler_sign_consistency'])

    def test_cells_do_not_bridge_missing_frame(self):
        p=dict(range=2.1,bearing=0,velocity=0)
        h=[self.frame(0,[p]),self.frame(.25,[]),self.frame(.5,[p])]
        f=radar_features(h,.75)
        self.assertEqual(f['consecutive_support'],1)
        self.assertEqual(f['occupancy_density'],2/3)
        self.assertEqual(f['time_since_last_support'],.25)

    def test_two_returns_one_frame_one_cell_vote(self):
        f=radar_features([self.frame(0,[dict(range=2.1,bearing=0,velocity=0)]*2)],.25)
        self.assertEqual(f['occupancy_density'],1)
        self.assertEqual(f['consecutive_support'],1)

    def test_missing_feature_cannot_separate(self):
        result=ceiling([dict(truth=True,features={'x':None}),dict(truth=False,features={'x':1})],['x'])
        self.assertEqual(result['single_cuts_tested'],0)
        self.assertEqual(result['missing_features'],['x'])

    def test_slope_requires_two_times(self):
        self.assertIsNone(slope([(0,1)]))
        self.assertEqual(slope([(0,1),(1,2)]),1)

    def test_public_prepare_reset_and_future_independence(self):
        row=dict(id='a0',episode_id='a',time_s=0,imu_valid=True,delta_yaw=0,delta_pitch=0,
                 radar_packet_received=True,radar_range_m=[2],radar_angle=[0],radar_velocity=[None],
                 radar_valid=[True],tof_zones=[],tof_packet_received=False)
        rows=[];ps=[]
        for ep,t,dy in [('a',0,0),('a',.25,10),('a',.5,100),('b',0,0)]:
            r=copy.deepcopy(row);r.update(id=f'{ep}{t}',episode_id=ep,time_s=t,delta_yaw=dy);rows.append(r)
            ps.append(dict(id=r['id'],scores={'raw':.7,'multi':.4}))
        a=prepare(rows,ps);rows[2]['radar_range_m']=[99];b=prepare(rows,ps)
        self.assertEqual(a[:2],b[:2])
        self.assertEqual(a[3]['radar'][0]['bearing'],0)
        history=[r for r in a if r['episode']=='a' and 0<=r['t']<.5]
        self.assertEqual([r['t'] for r in history],[0,.25])
        self.assertEqual(features(history,.5)['radar_nearest_range'],2)


if __name__=='__main__':unittest.main()
