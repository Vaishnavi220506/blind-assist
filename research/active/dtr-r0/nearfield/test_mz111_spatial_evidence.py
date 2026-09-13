"""Spatial representation and no-feedback controls on synthetic observations."""
import copy
import unittest
import mz108_competitive_association as visual
import mz111_spatial_evidence as spatial
from test_mz107_rgb_association import fixture
from run_mz111_spatial_temporal import event_metrics


class SpatialEvidence(unittest.TestCase):
    def fixture(self):
        row,image=fixture();row.update(time_s=0.,episode_id='example',imu_valid=True,radar_velocity=[-.5])
        return row,visual.predict_frame(row,image,0.,use_regions=False)

    def test_plane_center_corridor_and_outside(self):
        row,p=self.fixture()
        self.assertTrue(spatial.plane_inside([300,140,340,220],2.,row,0.))
        self.assertFalse(spatial.plane_inside([450,140,490,220],2.,row,0.))

    def test_filter_uses_velocity_but_never_future_or_truth(self):
        row,p=self.fixture();nextrow=dict(row,time_s=.25,radar_range_m=[2.4])
        source=copy.deepcopy([row,nextrow]);pred=copy.deepcopy([p,p])
        full=spatial.predict(source,pred,'plane',True)
        first=spatial.predict(source[:1],pred[:1],'plane',True)
        self.assertEqual(full[:1],first)
        self.assertTrue(full[1]['spatial_evidence'][0]['range_filtered'])
        self.assertEqual(source,[row,nextrow]);self.assertEqual(pred,[p,p])
        jumped=dict(nextrow,radar_range_m=[3.5])
        self.assertFalse(spatial.predict([row,jumped],[p,p],'plane',True)[1]['spatial_evidence'][0]['range_filtered'])

    def test_ambiguity_preserves_independent_return(self):
        row,p=self.fixture();row.update(radar_range_m=[2.5,2.5],radar_angle=[0.,0.],radar_valid=[True,True],radar_velocity=[-.5,-.5])
        result=spatial.predict([row],[p],'plane')[0]
        self.assertTrue(result['candidate'])
        self.assertTrue(all(r['state']=='AMBIGUOUS' and r['height_state']=='HEIGHT_UNKNOWN' for r in result['spatial_evidence']))

    def test_event_denominators_do_not_hide_missed_segments(self):
        rows=[dict(episode_id='one',time_s=i*.25) for i in range(6)]
        score=event_metrics(rows,[True,True,False,True,False,False],[False,True,True,False,True,True])
        self.assertEqual(score['positive_segments'],2);self.assertEqual(score['missed_positive_segments'],1)
        self.assertEqual(score['first_alert_delay_s'],[.25]);self.assertEqual(score['false_alert_segments'],2)
        self.assertEqual(score['false_alert_bin_duration_s'],.75)


if __name__=='__main__':unittest.main()
