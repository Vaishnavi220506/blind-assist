"""Focused solver mechanics fixtures, never the fresh180-scene cohort."""
import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import active_view as av
import continuous_boundary_witness as witness


def scene(x, width, z):
    return {"wall_z": 4.2, "boxes": [{"x": x, "z": z, "width": width, "thickness": .04}]}


def public(value, cameras=((0., 0.),)):
    return [{"camera": list(camera), "bins": list(witness._bins(value, camera))}
            for camera in cameras]


class ContinuousWitnessTests(unittest.TestCase):
    def test_public_interface_rejects_identity_truth_raw_and_unbudgeted_pose(self):
        self.assertEqual(list(inspect.signature(witness.infer).parameters), ["observations"])
        row = public(scene(.1, .2, 1.2))[0]
        for name in ("id", "truth", "scene", "raw_radial_m"):
            with self.assertRaises(ValueError):
                witness.infer([{**row, name: None}])
        with self.assertRaises(ValueError):
            witness.infer([{**row, "camera": [.001, 0]}])
        with self.assertRaises(ValueError):
            witness.infer([{**row, "bins": [True] * 8}])

    def test_independent_forward_matches_frozen_geometry_on_mechanical_fixtures(self):
        for value in (scene(.1, .2, 1.2), scene(-.37, .17, 2.13), scene(.72, .03, .63)):
            b = value["boxes"][0]
            original = av.Scene((av.Box(b["x"], b["z"], b["width"], b["thickness"]),))
            for camera in witness.POSES:
                self.assertEqual(witness._bins(value, camera), av.observe(original, camera))
        self.assertTrue(witness._truth(scene(.4, .2, 1.2)))
        self.assertFalse(witness._truth(scene(.401, .2, 1.2)))
        self.assertTrue(witness._truth(scene(0., .2, 3.02)))
        self.assertFalse(witness._truth(scene(0., .2, 3.021)))

    def test_continuous_solver_constructs_and_validates_two_wall_signature_witnesses(self):
        # An analytic wall signature is a mechanical ambiguity fixture, not a
        # geometry from the fresh evaluation source.
        bins = [math.floor(4.2 / math.cos(a) / .1 + .5) for a in witness.ANGLES]
        observations = [{"camera": [0., 0.], "bins": bins}]
        result = witness.infer(observations)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["valid_witness_count"], 2, result["solver_metadata"])
        self.assertEqual(result["solver_calls"], 2)
        for label, value in result["witnesses"].items():
            self.assertTrue(witness.validate_witness(value, observations, label))
            self.assertGreater(result["solver_metadata"][0]["binary_variables"], 0)
        self.assertFalse(result["unique_label_certified"])

    def test_two_view_target_witness_replays_every_observation(self):
        observations = public(scene(.1, .2, 1.2), ((0., 0.), (.12, 0.)))
        result = witness.infer(observations)
        self.assertIsNotNone(result["witnesses"]["IN"], result["solver_metadata"])
        self.assertTrue(witness.validate_witness(result["witnesses"]["IN"], observations, "IN"))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["search_failure_implies_nonexistence"])

    def test_solver_failure_cannot_become_uniqueness_or_out(self):
        failure = SimpleNamespace(status=2, message="infeasible", x=None)
        with patch.object(witness, "milp", return_value=failure) as solver:
            result = witness.infer(public(scene(.1, .2, 1.2)))
        self.assertEqual(solver.call_count, 2)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["valid_witness_count"], 0)
        self.assertEqual(result["reason"], "SOLVER_INCOMPLETE_OPPOSING_WITNESS_NOT_ESTABLISHED")
        self.assertFalse(result["unique_label_certified"])
        self.assertTrue(all(not r["infeasible_status_is_nonexistence_certificate"]
                            for r in result["solver_metadata"]))

    def test_invalid_numerical_candidate_rejected_without_retry(self):
        # Pretend an OUT request returned an obviously central intersecting box.
        fake = SimpleNamespace(status=0, message="mock", x=[-.1, .1, 1.18, .0001])
        with patch.object(witness, "milp", return_value=fake) as solver:
            result = witness.infer(public(scene(0., .2, 1.2)))
        self.assertEqual(solver.call_count, 2)
        self.assertIsNone(result["witnesses"]["OUT"])
        self.assertEqual(result["solver_metadata"][1]["outcome"], "CANDIDATE_REJECTED_BY_FORWARD_VALIDATION")
        self.assertEqual(result["status"], "UNKNOWN")

    def test_opposing_reason_is_only_a_pair_specific_forecast(self):
        # Disclosed consumed transfer witness pair: same initial/-X bins, +X differs.
        separable = {"IN": scene(.52, .48, 2.5), "OUT": scene(.49, .28, 2.5)}
        observations = public(separable["IN"], ((0., 0.), (-.12, 0.)))
        for label, value in separable.items():
            self.assertTrue(witness.validate_witness(value, observations, label))
        analysis = witness._opposing_analysis(separable)
        self.assertIn(0, analysis["separating_actions"])
        # Disclosed consumed raw-ray alias pair, not a fresh-cohort case.
        alias = {"IN": scene(-.385, .18, 2.35), "OUT": scene(-.395, .18, 2.35)}
        analysis = witness._opposing_analysis(alias)
        self.assertTrue(analysis["all_four_actions_indistinguishable"])
        self.assertEqual(analysis["scope"], "THIS_VALIDATED_WITNESS_PAIR_ONLY_NOT_UNIVERSAL_IMPOSSIBILITY")
        self.assertEqual(analysis["actual_scene_observations_generated"], 0)


if __name__ == "__main__":
    unittest.main()
