"""Mechanical scene/renderer invariants; no experimental metrics or tuning."""
import unittest
from dataclasses import replace

import numpy as np

import cnh_proposal_attribution_scenes as S
from cnh_track_a_geometry import box_mesh, raycast
from cnh_route_sensor import synthesize_response


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes=S.make_scenes(0)

    def test_frozen_design_labels_and_poses(self):
        self.assertEqual(len(self.scenes),22)
        self.assertEqual([sum(s['family']==f for s in self.scenes) for f in ['boundary','mixed_surface','sidewall','general']],[6,6,6,4])
        for s in self.scenes:
            self.assertEqual(s['poses'].shape,(16,4,4))
            self.assertEqual(s['labels'].shape,(16,2))
            np.testing.assert_allclose(s['travel'][-1],np.eye(4),atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(np.diff(s['travel'][:,:3,3],axis=0),axis=1),.16,atol=1e-14)
            self.assertEqual(s['labels'][-1,s['group']],int(s['margin']<0))
            self.assertEqual(s['labels'][-1,1-s['group']],0)
        np.testing.assert_array_equal(S.labels_for(self.scenes[0]['boxes'],self.scenes[0]['travel']),self.scenes[0]['labels'])

    def test_zero_yaw_and_world_rotation_label_equivariance(self):
        s=self.scenes[0]
        # Zero relative yaw gives byte-identical follow/travel query truth.
        np.testing.assert_array_equal(S.labels_for(s['boxes'],s['travel']),s['labels'])
        turn=S.ry(90)
        boxes=[]
        for b in s['boxes']:
            corners=box_mesh(b['lo'],b['hi']).reshape(-1,3)@turn.T
            boxes.append(dict(lo=corners.min(0),hi=corners.max(0),rho=b['rho']))
        poses=s['travel'].copy()
        poses[:,:3,:3]=turn@poses[:,:3,:3]
        poses[:,:3,3]=poses[:,:3,3]@turn.T
        np.testing.assert_array_equal(S.labels_for(boxes,poses),s['labels'])

    def test_head_motion_and_actual_turn_are_distinct(self):
        sway=S.make_scenes(1)[0]
        turn=S.make_scenes(2)[0]
        np.testing.assert_allclose(sway['travel'][:,:3,:3],np.broadcast_to(np.eye(3),(16,3,3)),atol=1e-14)
        self.assertGreater(np.linalg.norm(sway['head'][0,:3,:3]-sway['head'][-1,:3,:3]),.1)
        self.assertGreater(np.linalg.norm(turn['travel'][0,:3,:3]-turn['travel'][-1,:3,:3]),.1)
        np.testing.assert_array_equal(turn['travel'],turn['head'])
        np.testing.assert_allclose(np.linalg.norm(np.diff(turn['travel'][:,:3,3],axis=0),axis=1),.16,atol=1e-14)

    def test_analytic_raycast_matches_frozen_triangles(self):
        s=self.scenes[0];p=s['poses'][-1]
        d,_=S.ray_grid();d=d.reshape(-1,3)[::199]@p[:3,:3].T
        boxes=s['boxes']
        meshes=[box_mesh(b['lo'],b['hi']) for b in boxes]
        reference=raycast(p[:3,3],d,np.concatenate(meshes),
            np.concatenate([np.full(len(m),i) for i,m in enumerate(meshes)]),
            np.concatenate([np.full(len(m),b['rho']) for m,b in zip(meshes,boxes)]))
        actual=S.raycast_boxes(p[:3,3],d,boxes)
        for key in ['distance','rho','cos']:
            np.testing.assert_allclose(actual[key],reference[key],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(actual['object_id'],reference['object_id'])

    def test_expected_energy_and_paired_noise_budget(self):
        scene=dict(self.scenes[0],poses=self.scenes[0]['poses'][-1:])
        quiet=S.render(scene,431,noise_scale=0)
        p=scene['poses'][0];d,w=S.ray_grid()
        hit=S.raycast_boxes(p[:3,3],d@p[:3,:3].T,scene['boxes'])
        params,_=S.nominal_parameters()
        ref=synthesize_response(hit['distance'],hit['rho'],hit['cos'],w,params=replace(params,noise_scale=0),seed=1)
        coarse=ref['histogram'].reshape(8,8,16,8).sum(-1)
        np.testing.assert_allclose(quiet['hist'][0],coarse,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(quiet['ambient'][0],ref['ambient'],rtol=0,atol=0)
        noisy=S.render(scene,987,static=True)
        self.assertEqual(noisy['hist'].shape,(8,8,8,16))
        np.testing.assert_array_equal(noisy['hist'],noisy['finehist'].reshape(8,8,2,8,2,16).sum(axis=(2,4)))
        np.testing.assert_array_equal(noisy['ambient'],noisy['fineambient'].reshape(8,8,2,8,2).sum(axis=(2,4)))
        self.assertFalse(np.array_equal(noisy['hist'][0],noisy['hist'][1]))
        # Removing/changing evaluator-only fields cannot affect observation.
        observation_only={k:v for k,v in scene.items() if k in ('boxes','poses')}
        repeat=S.render(observation_only,431,noise_scale=0)
        np.testing.assert_array_equal(quiet['hist'],repeat['hist'])


if __name__=='__main__':unittest.main()
