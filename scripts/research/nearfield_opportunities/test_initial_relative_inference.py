"""Hand-built query-frame fixtures; no consumed-cohort optimizer execution."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initial_relative_inference as model
import shared_bias_inference as old
from test_shared_bias_inference import observations, residual, witness


class InitialRelativeTests(unittest.TestCase):
    def test_exact_complete_public_path_only(self):
        obs=observations(witness(),model.PATH)
        self.assertEqual(len(model._inputs(obs)),13)
        for bad in (obs[:-1],obs[::-1],[dict(obs[0],truth=True),*obs[1:]],
                    observations(witness(),old.paths.PATHS['z_then_x'])):
            with self.assertRaises(ValueError):model.infer(bad)

    def test_joint_translation_preserves_relative_extent_label_and_bins(self):
        original=witness(x=.3795,width=.16,z=1.3)
        alternate=witness(x=.3805,width=.16,z=1.3,px=.001)
        self.assertEqual(model.relative_label(original),'IN')
        self.assertEqual(model.relative_label(alternate),'IN')
        self.assertFalse(old.sensor.intersects_query(old.geometry._scene(alternate['scene'])))
        obs=observations(original,model.PATH)
        self.assertEqual(obs,observations(alternate,model.PATH))
        self.assertTrue(model.validate_witness(alternate,obs,'IN'))
        self.assertFalse(old.validate_witness(alternate,obs,'IN'))
        # Initial query stays fixed across the path, not at the last camera.
        self.assertEqual(model.reference_camera(alternate),(.001,0.))

    def test_query_prefix_changes_only_and_outer_true_fixtures(self):
        fixtures=[witness(x=.3805,width=.16,z=1.3,px=.001),
                  witness(x=-.3805,width=.16,z=1.3,px=-.001),
                  witness(x=.12,width=.16,z=3.0205,pz=.001),
                  witness(x=.3815,width=.16,z=1.3,px=.001),
                  witness(x=.12,width=.16,z=3.0215,pz=.001)]
        for w in fixtures:
            obs=observations(w,model.PATH);label=model.relative_label(w)
            c=model._constraints(obs,label);previous=old._constraints(obs,label,'range_pose')
            prefix=7 if label=='IN' else 8
            self.assertEqual(c.rows[prefix:],previous.rows[prefix:])
            self.assertEqual(c.lower,previous.lower);self.assertEqual(c.upper,previous.upper)
            self.assertEqual(c.rays,previous.rays)
            self.assertLessEqual(residual(c,w),1e-10)
            self.assertTrue(model.validate_witness(w,obs,label))
            self.assertEqual(c.scipy()['options'],old.base.SOLVER_OPTIONS)

    def test_round12_query_envelope_and_shifted_projection(self):
        px=.00012345678951;pz=-.00012345678951
        w=witness(x=round(px,12)+.3+.08,width=.16,z=1.3,px=px,pz=pz)
        obs=observations(w,model.PATH);c=model._constraints(obs,'IN')
        self.assertEqual(c.rows[4],({0:1.,5:-1.},.3+old.sensor.EPS+old.ROUND_ENVELOPE))
        self.assertEqual(model.reference_camera(w),(round(px,12),round(pz,12)))
        self.assertLessEqual(residual(c,w),1e-10)
        original=witness(x=.8,width=.4,z=3.3,px=.001,pz=-.001,b=.001)
        proposed=model.propose_projected_in(original)
        self.assertAlmostEqual(proposed['scene']['boxes'][0]['x'],.501)
        self.assertAlmostEqual(proposed['scene']['boxes'][0]['z'],3.019)
        self.assertEqual(proposed['bias'],original['bias'])
        self.assertEqual(proposed['scene']['boxes'][0]['width'],.4)
        self.assertEqual(model.relative_label(proposed),'IN')
        self.assertFalse(model.validate_witness(proposed,observations(original,model.PATH),'IN'))

    def test_world_bounds_and_forward_checks_are_not_removed(self):
        w=witness(x=.901,width=.1,px=.001)
        obs=observations(w,model.PATH)
        detail=model.validation_detail(w,obs,model.relative_label(w))
        self.assertIn('OUTSIDE_DECLARED_GEOMETRY_DOMAIN',detail['failures'])
        self.assertFalse(detail['valid'])
        inside=witness();obs=observations(inside,model.PATH)
        obs[2]['bins'][3]=1
        self.assertIn('OBSERVATION_BIN_MISMATCH',model.validation_detail(inside,obs,'IN')['failures'])

    def test_two_original_budget_solves_and_status_failure_unknown(self):
        v=[.3005,.4605,1.28,0.,0.,.001,0.]
        w=old._candidate(v,'range_pose');obs=observations(w,model.PATH)
        original_solver=old.milp;original_constructor=old._constraints
        for status in (0,1):
            calls=[]
            def fake(**kwargs):
                calls.append(kwargs)
                if len(calls)==2:return SimpleNamespace(status=2,message='fixture infeasible',x=None)
                values=np.zeros(len(kwargs['c']));values[:7]=v
                return SimpleNamespace(status=status,message='fixture',x=values)
            with patch.object(model,'milp',fake):r=model.infer(obs)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(k['options']==old.base.SOLVER_OPTIONS for k in calls))
            self.assertEqual(r['decision'],'IN_MODEL_CONDITIONAL' if status==0 else 'UNKNOWN')
            self.assertEqual(r['witnesses']['IN'],w)
            self.assertEqual(r['query_frame'],model.QUERY_FRAME)
        self.assertIs(old.milp,original_solver);self.assertIs(old._constraints,original_constructor)
        with patch.object(model,'milp',side_effect=RuntimeError('fixture exception')):
            r=model.infer(obs)
        self.assertEqual(r['decision'],'UNKNOWN')
        self.assertTrue(all(p['outcome']=='SOLVER_EXCEPTION' and not p['exclusion_supported'] for p in r['solver_metadata']))


if __name__=='__main__':unittest.main()
