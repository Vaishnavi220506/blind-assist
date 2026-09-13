"""Check counterfactual isolation and evaluator identity boundaries."""
import copy
import unittest
from unittest.mock import patch
import mz108_competitive_association as model
import mz110_association_diagnostic as audit
from test_mz107_rgb_association import fixture


class Diagnostic(unittest.TestCase):
    def make(self):
        row, image = fixture(); row.update(time_s=0., episode_id='episode')
        pred = model.predict_frame(row, image, 0., use_regions=False)
        return row, pred

    def test_missing_tof_opens_only_counterfactual(self):
        row, pred = self.make(); row['tof_packet_received'] = False
        frozen = copy.deepcopy(pred)
        self.assertIsNone(audit.observable_trace(row, pred)['returns'][0]['selected'])
        cf = audit.observable_trace(row, pred, False)
        self.assertEqual(cf['returns'][0]['selected'], 0)
        self.assertEqual(pred, frozen)
        row['radar_angle'] = [-50.]
        self.assertIsNone(audit.observable_trace(row, pred, False)['returns'][0]['selected'])

    def test_gate_removal_keeps_reciprocal_contention(self):
        row, pred = self.make(); row['tof_packet_received'] = False
        row.update(radar_range_m=[2.5,2.5], radar_angle=[0.,0.], radar_valid=[True,True])
        cf = audit.observable_trace(row, pred, False)
        self.assertTrue(all(r['selected'] is None for r in cf['returns']))
        self.assertTrue(cf['candidate'])  # Independent Radar fallback.

    def test_no_rgb_and_independent_tof_survive(self):
        row, pred = self.make(); pred.update(proposals=[], tof_support=True)
        self.assertTrue(audit.observable_trace(row, pred, False)['candidate'])

    def test_ghost_and_ambiguous_box_do_not_become_identity_truth(self):
        row, pred = self.make(); trace = audit.observable_trace(row, pred, False)
        obj = dict(name='target', center_m=[2.,0.,1.], extent_m=[.1,.1,.1])
        evaluation = dict(native_bounds=[obj], body_origin_m=[0.,0.,0.])
        ghost = dict(radar_slots=[dict(kind='persistent_ghost', actor_id=None)])
        real = dict(radar_slots=[dict(kind='real_actor', actor_id='episode/target')])
        with patch.object(audit, 'projected_objects', return_value=[pred['proposals'][0]]):
            self.assertEqual(audit.identity_pairs(row,pred,evaluation,ghost,trace), (0, []))
            count, pairs = audit.identity_pairs(row,pred,evaluation,real,trace)
            self.assertEqual(count, 1); self.assertEqual(len(pairs), 1)
            evaluation['native_bounds'].append(dict(obj, name='occluder'))
            count, pairs = audit.identity_pairs(row,pred,evaluation,real,trace)
            self.assertEqual(count, 1); self.assertEqual(pairs, [])


if __name__ == '__main__':
    unittest.main()
