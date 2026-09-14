"""Narrow mathematical and independent-coverage checks for the fixed model."""
import unittest

from mz134_joint_attribution import feasible_intervals, falsifier, predict_frame


class JointFeasibilityTest(unittest.TestCase):
    def test_exact_marginal_bounds_and_no_observation_column(self):
        values=feasible_intervals([[1.,1.,0.],[2.,0.,0.]],[2.,3.])
        self.assertEqual([v['upper'] for v in values],[1.5,2.,None])
        self.assertTrue(all(v['lower']==0 for v in values))
        self.assertEqual(values[0]['unresolved_residual'],[1.25,1.5])

    def test_contradictory_observed_zero_excludes_positive_coefficient(self):
        # An actual observed zero can constrain the algebra. Missing zones in
        # the real experiment are not inserted as zero-signal observations.
        value=feasible_intervals([[1.],[1.]],[0.,2.])[0]
        self.assertEqual(value['upper'],0.)

    def test_hidden_corridor_source_is_not_forced_visible(self):
        self.assertEqual(falsifier()['status'],'PASS_HIDDEN_CORRIDOR_EXPLANATION_UNRESOLVED')

    def test_empty_visible_masks_preserve_return(self):
        zones=[dict(zone_id=j,theta_bounds_deg=[j-4.,j-3.],phi_bounds_deg=[-2.,2.],targets=[])
               for j in range(8)]
        zones[3]['targets']=[dict(distance_m=2.,range_noise_sigma_m=.01,
            status='SIM_VALID',signal_strength_proxy=1.)]
        row=dict(id='unobserved-mask-case',tof_packet_received=True,tof_zones=zones,
            rgb_intrinsics=dict(width=640,height=360,cx=320.,cy=180.,fx=320.,fy=320.),
            time_s=0.,camera_pitch_deg=0.,camera_in_body_m=[0.,0.,1.5])
        prediction,details=predict_frame(row,0.,[])
        self.assertTrue(prediction['candidate'])
        self.assertEqual(len(details['returns']),1)
        self.assertEqual(details['returns'][0]['state'],'UNRESOLVED_NATIVE_SUPPORT_RETAINED')
        self.assertEqual(details['groups'][0]['mandatory_visible_sources'],0)


if __name__=='__main__':
    unittest.main()
