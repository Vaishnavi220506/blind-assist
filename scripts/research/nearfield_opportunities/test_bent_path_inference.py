"""Bounded interface and readout fixtures, not the path pilot cohort."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import active_view as sensor
import bent_path_inference as bent
import path_constraint_inference as model


def observations(candidate, poses):
    scene = model._scene(candidate)
    return [dict(camera=list(p), bins=list(sensor.observe(scene,p))) for p in poses]


def fake_solver(vector, status=0):
    calls=[]
    def solve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            return SimpleNamespace(status=2, message='fixture infeasible', x=None)
        x=np.zeros(len(kwargs['c']))
        x[:4]=vector
        return SimpleNamespace(status=status,message='fixture',x=x)
    return solve,calls


class BentPathTests(unittest.TestCase):
    def test_paths_and_public_whitelist(self):
        self.assertEqual(len(bent.ALLOWED_POSES),30)
        for path in bent.PATHS.values():
            self.assertEqual(len(path),13)
            self.assertAlmostEqual(sum(math.dist(a,b) for a,b in zip(path,path[1:])),.12)
        row=dict(camera=[.06,.03],bins=[42]*8)
        self.assertEqual(bent._inputs([row])[0]['camera'],(.06,.03))
        for bad in ([dict(row,truth=True)],[dict(row,camera=[.01,.01])],
                    [dict(row,camera=[-.01,0])],[dict(row,bins=[True]*8)], [row]*14):
            with self.assertRaises(ValueError): bent._inputs(bad)

    def test_original_fixture_semantics_and_private_binding(self):
        vector=[0.,.2,1.18,0.]
        obs=observations(model.base._candidate(vector),bent.PATHS['straight_x'])
        first,_=fake_solver(vector)
        with patch.object(model,'milp',first): original=model.infer(obs)
        second,calls=fake_solver(vector)
        old_solver,old_inputs=model.milp,model._inputs
        with patch.object(bent,'milp',second): result=bent.infer(obs)
        self.assertIs(model.milp,old_solver)
        self.assertIs(model._inputs,old_inputs)
        for key in ('decision','witnesses'): self.assertEqual(result[key],original[key])
        self.assertEqual(result['solver_calls'],2)
        self.assertTrue(all(c['options']==model.base.SOLVER_OPTIONS for c in calls))

    def test_new_offaxis_trace_validates_original_geometry(self):
        vector=[0.,.2,1.18,0.]
        obs=observations(model.base._candidate(vector),bent.PATHS['x_then_z'])
        with self.assertRaises(ValueError): model._inputs(obs)
        solve,_=fake_solver(vector)
        with patch.object(bent,'milp',solve): result=bent.infer(obs)
        self.assertEqual(result['decision'],'IN_MODEL_CONDITIONAL')
        self.assertEqual(result['projection_added_labels'],[])
        self.assertTrue(model.validate_witness(result['witnesses']['IN'],obs,'IN'))

    def test_projection_requires_exact_label_and_all_bins(self):
        vector=[.3+sensor.EPS,math.nextafter(.8,math.inf),1.18,0.]
        obs=observations(model.base._candidate(vector),[(0.,0.),(.06,.03)])
        solve,_=fake_solver(vector)
        with patch.object(bent,'milp',solve): result=bent.infer(obs)
        self.assertEqual(result['projection_added_labels'],['IN'])
        self.assertEqual(result['decision'],'IN_MODEL_CONDITIONAL')
        bad=[dict(row,bins=[1]*8) for row in obs]
        solve,_=fake_solver(vector)
        with patch.object(bent,'milp',solve): result=bent.infer(bad)
        self.assertEqual(result['decision'],'UNKNOWN')
        self.assertFalse(result['attempts'][0]['projected_valid'])

    def test_limits_and_exclusion_conflicts_fail_closed(self):
        vector=[0.,.2,1.18,0.]
        obs=observations(model.base._candidate(vector),[(.06,.03)])
        solve,_=fake_solver(vector,status=1)
        with patch.object(bent,'milp',solve): result=bent.infer(obs)
        self.assertEqual(result['decision'],'UNKNOWN')
        witness=model.base._candidate(vector)
        receipts={k:dict(solver_status=2,exclusion_supported=True) for k in ('IN','OUT')}
        decision,reason,conflicts=bent._readout({'IN':witness,'OUT':None},receipts)
        self.assertEqual(decision,'UNKNOWN')
        self.assertEqual(conflicts,['IN'])


if __name__=='__main__': unittest.main()
