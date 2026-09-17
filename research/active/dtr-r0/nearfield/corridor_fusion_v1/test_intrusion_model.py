"""Focused geometry, public-input, and residual learning contract checks."""
import copy
import unittest
import numpy as np
import torch
from intrusion_model import QUERY,native_margins,local_inputs,IntrusionNet,objective
from mz115_zonal_tof import zone_geometry


def observation(packet=True):
    zones=[]
    for i in range(64):
        geometry,_=zone_geometry(i)
        zones.append(dict(geometry,targets=[dict(status='SIM_MERGED',distance_m=2.,
            range_noise_sigma_m=.04,signal_strength_proxy=.1)]))
    return dict(rgb_intrinsics=dict(width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.),
        camera_pitch_deg=-3.,camera_in_body_m=[0.,0.,1.7],tof_zones=zones,tof_packet_received=packet,
        radar_range_m=[2.],radar_angle=[0.],radar_velocity=[0.],radar_valid=[True],
        radar_packet_received=packet,imu_valid=True)


def obj(center,extent):return dict(center_m=center,extent_m=extent)
def ev(objects,origin=(0,0,0)):return dict(native_bounds=objects,body_origin_m=origin)


class NativeGeometryContract(unittest.TestCase):
    def test_eight_queries_cover_body_and_head_with_correct_forward_bins(self):
        self.assertEqual(QUERY.shape,(8,6))
        np.testing.assert_allclose(QUERY[:4,0],[.2,1.4,2.6,3.2])
        np.testing.assert_allclose(QUERY[:4,3],[1.4,2.6,3.2,3.6])
        np.testing.assert_allclose(QUERY[:,1],-.3);np.testing.assert_allclose(QUERY[:,4],.3)
        np.testing.assert_allclose(QUERY[:4,2],.4);np.testing.assert_allclose(QUERY[:4,5],1.5)
        np.testing.assert_allclose(QUERY[4:,2],1.5);np.testing.assert_allclose(QUERY[4:,5],2.05)

    def test_all_objects_union_and_band_signs(self):
        body=obj([2.,0.,.9],[.1,.1,.1]);head=obj([2.9,0.,1.8],[.1,.1,.1])
        far=obj([4.2,0.,1.],[.1,2.,1.])
        b=native_margins(ev([body]));h=native_margins(ev([head]))
        self.assertGreater(b[1],0);self.assertTrue((b[4:]<0).all())
        self.assertGreater(h[6],0);self.assertTrue((h[:4]<0).all())
        self.assertTrue((native_margins(ev([far]))<0).all())
        combined=native_margins(ev([far,body,head]))
        np.testing.assert_allclose(combined,np.maximum.reduce([native_margins(ev([far])),b,h]))
        np.testing.assert_array_equal(combined,native_margins(ev([head,far,body])))

    def test_signed_separation_and_body_origin_translation(self):
        # Binary exactly representable vertical contact at BODY top=1.5.
        contact=obj([2.,0.,1.625],[.125,.125,.125])
        self.assertEqual(native_margins(ev([contact]))[1],0)
        separate=obj([2.,.6,.9],[.1,.1,.1])
        self.assertAlmostEqual(float(native_margins(ev([separate]))[1]),-.2,places=6)
        delta=np.array([10.,-4.,2.]);shift=copy.deepcopy(separate)
        shift['center_m']=(np.array(shift['center_m'])+delta).tolist()
        np.testing.assert_allclose(native_margins(ev([separate])),native_margins(ev([shift],delta)))


class ObservableInputsContract(unittest.TestCase):
    def test_shapes_missing_slots_null_token_and_rgb_validity(self):
        row=observation(False);image=np.full((360,640,3),128,np.uint8)
        x=local_inputs(row,image,0.)
        self.assertEqual(x['patch'].shape,(8,3,21,21));self.assertEqual(x['tof'].shape,(129,8))
        self.assertEqual(x['local'].shape,(8,129));self.assertEqual(x['local'].dtype,np.bool_)
        self.assertTrue(x['local'][:,-1].all());self.assertFalse(x['tof'][:,:5].any())
        self.assertFalse(x['tof'][:,-1].any());self.assertFalse(x['tof'][-1].any())
        validity=x['patch'][:,2]
        self.assertTrue((validity>=0).all() and (validity<=1).all())
        self.assertTrue((validity==0).any());self.assertTrue((validity==1).any())
        self.assertTrue(all(np.isfinite(v).all() for v in x.values()))

    def test_evaluator_identity_and_stale_packet_values_are_not_features(self):
        row=observation(False);image=np.zeros((360,640,3),np.uint8);a=local_inputs(row,image,0.)
        changed=copy.deepcopy(row)
        changed.update(id='leaked',family='positive',truth=1,native_bounds='poison',wall_distance_m=4.5,
            target_group='hidden',counterfactual_group='hidden',C_score=1.,evaluator='poison')
        for z in changed['tof_zones']:z['targets'][0]['distance_m']=999.
        changed['radar_range_m']=[999.]
        b=local_inputs(changed,image,0.)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_valid_merged_return_and_missing_second_slot_are_distinguished(self):
        x=local_inputs(observation(),np.zeros((360,640,3),np.uint8),0.)
        np.testing.assert_array_equal(x['tof'][:-1:2,3:5],np.ones((64,2)))
        self.assertFalse(x['tof'][1:-1:2,:5].any())
        self.assertTrue((x['tof'][:-1,7]==1).all())


class ModelContract(unittest.TestCase):
    def test_initial_residual_equals_a_and_both_losses_backward_finite(self):
        torch.set_num_threads(4);torch.manual_seed(123)
        x=local_inputs(observation(False),np.zeros((360,640,3),np.uint8),0.)
        batch={k:torch.from_numpy(np.stack([v]*4)) for k,v in x.items()}
        batch.update(base=torch.randn(4,2485),A_logit=torch.tensor([-3.,.1,2.,-.5]),
            truth=torch.tensor([0.,1.,1.,0.]),field=torch.zeros(4,8),margin=torch.full((4,8),-.2))
        model=IntrusionNet();pred=model(batch)
        torch.testing.assert_close(pred['logit'],batch['A_logit'],rtol=0,atol=0)
        self.assertEqual(pred['field'].shape,(4,8));self.assertEqual(pred['margin'].shape,(4,8))
        for auxiliary in (False,True):
            model.zero_grad();pred=model(batch)
            loss=objective(pred,batch,auxiliary=auxiliary,pairs=torch.tensor([[0,1],[3,2]]),
                invariance=torch.tensor([[0,3],[1,2]]))
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(float(model.frame[-1].weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
