"""Focused capture/projection fixtures; no258-query diagnostic replay."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import active_view as sensor
import candidate_diagnostics as diag
import continuous_boundary_witness as base
import path_constraint_inference as model


def scene(x=.1, width=.2, z=1.2):
    return dict(wall_z=4.2, boxes=[dict(x=x, z=z, width=width, thickness=.04)])


def observations(value):
    s = sensor.Scene(tuple(sensor.Box(**b) for b in value["boxes"]))
    return [dict(camera=[0., 0.], bins=list(sensor.observe(s, (0., 0.))))]


class CandidateDiagnosticsTests(unittest.TestCase):
    def test_raw_face_bound_can_leave_query_after_coordinate_decode_and_rounding(self):
        vector = [.3 + sensor.EPS, math.nextafter(.8, math.inf), 1.18, 0.]
        self.assertLessEqual(vector[0], .3 + sensor.EPS)
        raw = diag.unrounded_candidate(vector)
        decoded = base._candidate(vector)
        obs = observations(decoded)
        detail = diag.validation_detail(decoded, obs, "IN")
        self.assertEqual(detail["failures"], ["QUERY_LABEL_MISMATCH"])
        self.assertTrue(detail["domain_valid"])
        self.assertTrue(detail["all_bins_match"])
        self.assertGreater(detail["decoded_faces"]["L"], .3 + sensor.EPS)
        self.assertGreater(diag.validation_detail(raw, obs, "IN")["decoded_faces"]["L"], vector[0])
        proposed = diag.propose_projected_in(decoded)
        self.assertTrue(diag.validation_detail(proposed, obs, "IN")["valid"])
        self.assertEqual(proposed["boxes"][0]["width"], decoded["boxes"][0]["width"])
        self.assertLess(abs(proposed["boxes"][0]["x"] - decoded["boxes"][0]["x"]), 2e-12)

    def test_projection_is_coordinate_clipping_and_never_guarantees_bins(self):
        original = scene(.8, .4, 3.3)
        proposed = diag.propose_projected_in(original)
        self.assertEqual(proposed, scene(.5, .4, 3.02))
        self.assertEqual(diag.propose_projected_in(proposed), proposed)
        detail = diag.validation_detail(proposed, observations(original), "IN")
        self.assertTrue(detail["label_matches"])
        self.assertFalse(detail["valid"])
        self.assertIn("OBSERVATION_BIN_MISMATCH", detail["failures"])
        inside = scene()
        self.assertEqual(diag.propose_projected_in(inside), inside)

    def test_failure_categories_match_frozen_validator(self):
        obs = observations(scene())
        bad = scene(width=.8)
        detail = diag.validation_detail(bad, obs, "IN")
        self.assertIn("OUTSIDE_DECLARED_DOMAIN", detail["failures"])
        detail = diag.validation_detail(scene(), obs, "OUT")
        self.assertEqual(detail["failures"], ["QUERY_LABEL_MISMATCH"])
        self.assertEqual(diag.validation_detail({"truth": True}, obs, "IN")["failures"],
                         ["STRUCTURE_OR_NONFINITE_COORDINATES"])
        with self.assertRaises(ValueError):
            diag.collect([{**obs[0], "case_id": "forbidden"}])

    def test_collector_calls_original_infer_once_two_solves_and_preserves_baseline(self):
        raw = [.3 + sensor.EPS, math.nextafter(.8, math.inf), 1.18, 0.]
        obs = observations(base._candidate(raw))
        calls = []
        def fake(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return SimpleNamespace(status=2, message="fixture infeasible", x=None)
            x = np.zeros(len(kwargs["c"]))
            x[:4] = raw
            return SimpleNamespace(status=0, message="fixture candidate", x=x)
        with patch.object(model, "milp", fake):
            result = diag.collect(obs)
            self.assertIs(model.milp, fake)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c["options"] == base.SOLVER_OPTIONS for c in calls))
        self.assertEqual(result["baseline"]["decision"], "UNKNOWN")
        self.assertIsNone(result["baseline"]["witnesses"]["IN"])
        attempt = result["attempts"][0]
        self.assertEqual(attempt["raw_solver_vector"][:4], raw)
        self.assertEqual(attempt["validation"]["failures"], ["QUERY_LABEL_MISMATCH"])
        self.assertTrue(attempt["projected_valid"])
        self.assertTrue(attempt["projection_eligible"])
        self.assertFalse(attempt["baseline_candidate_changed"])
        self.assertFalse(result["attempts"][1]["projection_eligible"])
        self.assertTrue(result["projected_proposal_is_not_a_decision"])

    def test_solver_capture_is_restored_after_failure_and_valid_candidates_are_not_moved(self):
        obs = observations(scene())
        def failure(**kwargs):
            raise RuntimeError("fixture failure")
        with patch.object(model, "milp", failure):
            result = diag.collect(obs)
            self.assertIs(model.milp, failure)
        self.assertEqual(result["solver_calls"], 2)
        self.assertTrue(all(a["solver_exception"]["type"] == "RuntimeError" for a in result["attempts"]))
        def valid(**kwargs):
            x = np.zeros(len(kwargs["c"]))
            x[:4] = [0., .2, 1.18, 0.]
            return SimpleNamespace(status=0, message="fixture", x=x)
        with patch.object(model, "milp", valid):
            result = diag.collect(obs)
        self.assertIsNotNone(result["baseline"]["witnesses"]["IN"])
        self.assertFalse(result["attempts"][0]["projection_eligible"])
        self.assertIsNone(result["attempts"][0]["projected_candidate"])


if __name__ == "__main__":
    unittest.main()
