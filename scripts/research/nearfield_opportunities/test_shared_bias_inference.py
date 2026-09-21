"""Analytic fixtures for shared-bias closure; no consumed pilot query scoring."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import active_view as sensor
import bent_path_inference as paths
import shared_bias_inference as model


def witness(x=.12,width=.16,z=1.3,b=0.,px=0.,pz=0.):
    return dict(scene=dict(wall_z=4.2,boxes=[dict(x=x,z=z,width=width,thickness=.04)]),
                bias=dict(range_m=b,pose_x_m=px,pose_z_m=pz))


def observations(w,poses):
    scene=sensor.Scene(tuple(sensor.Box(**b) for b in w['scene']['boxes']))
    b=w['bias'];answer=[]
    for nominal in poses:
        camera=(round(nominal[0]+b['pose_x_m'],12),round(nominal[1]+b['pose_z_m'],12))
        measured=[min((4.2-camera[1])/math.cos(a),sensor.hit_distance(scene.boxes[0],camera,a))+b['range_m']
                  for a in sensor.ANGLES]
        answer.append(dict(camera=list(nominal),bins=[math.floor(t/.1+.5) for t in measured]))
    return answer


def residual(c,w):
    b=w['scene']['boxes'][0];bias=w['bias']
    values=[b['x']-b['width']/2,b['x']+b['width']/2,b['z']-.02,0.,
            bias['range_m'],bias['pose_x_m'],bias['pose_z_m']]+[0.]*(len(c.lower)-7)
    branches={}
    for co,limit in c.rows:
        bs=[(i,a) for i,a in co.items() if i>=7 and a>0]
        if bs:
            assert len(bs)==1
            i,a=bs[0]
            branches[i]=sum(v*values[j] for j,v in co.items() if j<7)+a-limit
    for co,limit in c.rows:
        if co and limit==-1. and all(i>=7 and a==-1. for i,a in co.items()):
            values[min(co,key=lambda i:branches[i])]=1.
    return max([0.,*(sum(v*values[j] for j,v in co.items())-lim for co,lim in c.rows),
                *(max(lo-v,v-hi) for v,lo,hi in zip(values,c.lower,c.upper,strict=True))])


class SharedBiasTests(unittest.TestCase):
    def test_public_inputs_and_shared_nuisance_validation(self):
        obs=observations(witness(),[(0.,0.)])
        with self.assertRaises(ValueError): model.infer(obs,'fitted')
        with self.assertRaises(ValueError): model.infer([dict(obs[0],actual_bias=.002)])
        w=witness(px=.001)
        self.assertIn('OUTSIDE_DECLARED_BIAS_DOMAIN',model.validation_detail(w,obs,'IN','range_only')['failures'])
        self.assertTrue(model.validation_detail(witness(),obs,'IN')['valid'])

    def test_known_geometry_and_extreme_biases_satisfy_outer_rows(self):
        fixtures=[witness(),witness(x=-.71,width=.11,z=2.2),
                  witness(x=.38+sensor.EPS,width=.16,z=1.3),
                  witness(x=-.3-.08-sensor.EPS,width=.16,z=3.02)]
        biases=[(.002,.001,.001),(-.002,-.001,-.001),(0.,0.,0.),
                (.00123,-.00073250000049,.00048750000051)]
        for original in fixtures:
            for b,px,pz in biases:
                w=dict(scene=original['scene'],bias=dict(range_m=b,pose_x_m=px,pose_z_m=pz))
                obs=observations(w,paths.PATHS['x_then_z'])
                label='IN' if sensor.intersects_query(model.geometry._scene(w['scene'])) else 'OUT'
                self.assertTrue(model.validate_witness(w,obs,label))
                c=model._constraints(obs,label,'range_pose')
                self.assertLessEqual(residual(c,w),1e-10)

    def test_range_only_subset_and_seven_continuous_variables(self):
        w=witness(x=-.6,width=.2,b=-.002)
        obs=observations(w,paths.PATHS['z_then_x'])
        for mode in model.MODES:
            c=model._constraints(obs,'OUT',mode)
            self.assertLessEqual(residual(c,w),1e-10)
            args=c.scipy()
            self.assertEqual(list(args['integrality'][:7]),[0]*7)
            self.assertTrue(all(v==1 for v in args['integrality'][7:]))
            self.assertEqual(args['options'],model.base.SOLVER_OPTIONS)
            self.assertTrue(model.validate_witness(w,obs,'OUT',mode))
        zero=witness()
        obs=observations(zero,paths.PATHS['straight_x'])
        self.assertTrue(model.validate_witness(zero,obs,'IN','range_only'))

    def test_wall_bin_shift_is_explained_by_shared_bias_not_target(self):
        w=witness(x=.7,width=.1,z=1.8,b=-.002)
        obs=observations(w,[(.06,.01)])
        self.assertEqual(obs[0]['bins'][0],44)
        c=model._constraints(obs,'OUT','range_only')
        self.assertEqual(c.rays[0]['branch'],'WALL')
        self.assertLessEqual(residual(c,w),1e-10)
        bad=[dict(camera=[.06,.01],bins=[40]*8)]
        impossible=model._constraints(bad,'OUT','range_pose')
        self.assertTrue(all(r['branch']=='IMPOSSIBLE' for r in impossible.rays))
        self.assertIn(({},-1.),impossible.rows)

    def test_bin_closure_does_not_admit_failed_forward_candidate(self):
        a=sensor.ANGLES[5];front=((14+.5)*.1)*math.cos(a)
        w=witness(x=(front+.02)*math.tan(a),width=.1,z=front+.02)
        obs=observations(w,[(0.,0.)])
        obs[0]['bins'][5]=14
        c=model._constraints(obs,'IN','range_only')
        self.assertLessEqual(residual(c,w),1e-10)
        self.assertFalse(model.validate_witness(w,obs,'IN','range_only'))

    def test_two_solves_status_gating_and_bias_preserving_projection(self):
        vector=[.3+sensor.EPS,math.nextafter(.8,math.inf),1.18,0.,.001,0.,0.]
        decoded=model._candidate(vector,'range_pose')
        obs=observations(decoded,[(0.,0.),(.06,.03)])
        for status in (0,1):
            calls=[]
            def fake(**kwargs):
                calls.append(kwargs)
                if len(calls)==2:return SimpleNamespace(status=2,message='fixture infeasible',x=None)
                values=np.zeros(len(kwargs['c']));values[:7]=vector
                return SimpleNamespace(status=status,message='fixture',x=values)
            with patch.object(model,'milp',fake): result=model.infer(obs)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c['options']==model.base.SOLVER_OPTIONS for c in calls))
            self.assertEqual(result['decision'],'IN_MODEL_CONDITIONAL' if status==0 else 'UNKNOWN')
            if status==0:
                self.assertTrue(result['attempts'][0]['projected_valid'])
                self.assertEqual(result['witnesses']['IN']['bias'],decoded['bias'])
                self.assertEqual(result['attempts'][0]['raw_solver_vector'][:7],vector)
            else:self.assertFalse(result['attempts'][0]['projection_eligible'])

    def test_small_unconsumed_fixture_real_solver_and_exact_witness_replay(self):
        w=witness(x=-.71,width=.11,z=2.2,b=-.002,px=.001,pz=.001)
        obs=observations(w,[(0.,0.),(.06,.01)])
        result=model.infer(obs,'range_pose')
        self.assertEqual(result['solver_calls'],2)
        self.assertEqual(result['solver_metadata'][0]['budget'],model.base.SOLVER_OPTIONS)
        for label,candidate in result['witnesses'].items():
            if candidate is not None:self.assertTrue(model.validate_witness(candidate,obs,label))
        self.assertFalse(result['exclusion_metadata']['OUT']['reported_infeasible'])
        self.assertTrue(any(candidate is not None for candidate in result['witnesses'].values()))


if __name__=='__main__':unittest.main()
