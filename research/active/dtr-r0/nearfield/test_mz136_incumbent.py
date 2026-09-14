"""Contract checks; full sealed 288-frame replay is the module CLI regression."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from mz136_incumbent import (SOURCE, OBSERVABLE_FIELDS, public_observations,
    shifted_geometry, regenerate_shifted_tof, predict_incumbent)
from mz133_angular_resolution import geometry


class IncumbentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(s) for s in (SOURCE/'capture-v1/raw.jsonl').read_text().splitlines()]
        cls.spec = json.loads((SOURCE/'capture-v1/spec.json').read_text())
        cls.evaluations = [json.loads(s) for s in (SOURCE/'capture-v1/evaluator.jsonl').read_text().splitlines()]

    def one_frame(self):
        spec = copy.deepcopy(self.spec); spec['frames'] = spec['frames'][:1]
        return copy.deepcopy(self.rows[:1]), spec, copy.deepcopy(self.evaluations[:1])

    def test_public_contract_discards_private_fields_without_mutation(self):
        rows = copy.deepcopy(self.rows[:1])
        rows[0].update(truth=True, family='DO_NOT_USE', objects=[{'private':1}], sensor_seed=1)
        rows[0]['tof_zones'][0]['private_rays'] = ['DO_NOT_USE']
        rows[0]['rgb_intrinsics']['true_camera'] = 'DO_NOT_USE'
        before = copy.deepcopy(rows)
        public = public_observations(rows)
        self.assertEqual(public, self.rows[:1])
        self.assertEqual(rows, before)
        self.assertLessEqual(set(public[0]), set(OBSERVABLE_FIELDS))

    def test_shift_retains_zone_bounds_and_signal_budget(self):
        zones, shifted = shifted_geometry(); old_zones, old = geometry(8,3)
        self.assertEqual(zones, old_zones)
        self.assertEqual(shifted.shape, (64,9,3))
        np.testing.assert_allclose(np.degrees(np.arctan(shifted[:,:,1]))-
                                   np.degrees(np.arctan(old[:,:,1])), .1875, atol=1e-12)
        np.testing.assert_allclose(np.degrees(np.arctan(shifted[:,:,2]))-
                                   np.degrees(np.arctan(old[:,:,2])), -.1875, atol=1e-12)
        for zone, ray in zip(zones,shifted):
            theta = np.degrees(np.arctan(ray[:,1])); phi = np.degrees(np.arctan(ray[:,2]))
            self.assertTrue(np.all(theta > zone['theta_bounds_deg'][0]))
            self.assertTrue(np.all(theta < zone['theta_bounds_deg'][1]))
            self.assertTrue(np.all(phi > zone['phi_bounds_deg'][0]))
            self.assertTrue(np.all(phi < zone['phi_bounds_deg'][1]))

    def test_native_admission_and_non_tof_invariance(self):
        rows, spec, evaluations = self.one_frame()
        before = copy.deepcopy((rows,spec,evaluations))
        generated, report = regenerate_shifted_tof(rows,spec,evaluations)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['sample_rays'],576)
        self.assertEqual({k:v for k,v in generated[0].items() if k != 'tof_zones'},
                         {k:v for k,v in rows[0].items() if k != 'tof_zones'})
        self.assertEqual((rows,spec,evaluations),before)
        self.assertEqual(regenerate_shifted_tof(rows,spec,evaluations),(generated,report))

    def test_failed_or_truncated_native_admission_does_not_export(self):
        rows, spec, evaluations = self.one_frame()
        evaluations[0]['zonal_tof_native'][0]['private_rays'][0]['range_m'] = .1
        generated, report = regenerate_shifted_tof(rows,spec,evaluations)
        self.assertEqual(generated,[]); self.assertEqual(report['status'],'FAIL')
        evaluations[0]['zonal_tof_native'][0]['private_rays'].pop()
        with self.assertRaises(ValueError): regenerate_shifted_tof(rows,spec,evaluations)
        with self.assertRaises(ValueError): regenerate_shifted_tof(rows,spec,[])

    def test_inference_projects_inputs_before_any_original_pipeline(self):
        rows = copy.deepcopy(self.rows[:1]); rows[0]['truth'] = True
        def check(projected, image_loader):
            self.assertEqual(projected, self.rows[:1])
            raise RuntimeError('checked contract before original inference')
        with patch('mz136_incumbent.cv2.__version__','5.0.0'), patch('mz136_incumbent.mz116.predict',side_effect=check):
            with self.assertRaisesRegex(RuntimeError,'checked contract'):
                predict_incumbent(rows,lambda _:None)


if __name__ == '__main__': unittest.main()
