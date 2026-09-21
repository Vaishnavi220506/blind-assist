"""Mechanical outer-relaxation and conditional-authority fixtures only."""
import inspect
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import active_view as av
import continuous_boundary_witness as old
import path_constraint_inference as path


def scene(x=.1, width=.2, z=1.2):
    return dict(wall_z=4.2, boxes=[dict(x=x, z=z, width=width, thickness=.04)])


def public(value, cameras=((0., 0.),)):
    model = av.Scene(tuple(av.Box(**b) for b in value["boxes"]))
    return [dict(camera=list(p), bins=list(av.observe(model, p))) for p in cameras]


def continuous(value):
    b = value["boxes"][0]
    return [b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["z"] - .02, 0.]


def relaxation_contains(c, value):
    """Choose feasible Boolean alternatives with fixed geometry; never run MILP."""
    x = continuous(value) + [0.] * (len(c.lower) - 4)
    for coefficients, bound in c.rows:
        indicators = [(j, a) for j, a in coefficients.items() if j >= 4 and a > 0]
        if indicators:
            assert len(indicators) == 1
            j, coefficient = indicators[0]
            residual = sum(a * x[k] for k, a in coefficients.items() if k < 4)
            if residual + coefficient <= bound + 1e-13:
                x[j] = 1.
    return (all(lo - 1e-13 <= v <= hi + 1e-13 for v, lo, hi in zip(x, c.lower, c.upper, strict=True))
            and all(sum(a * x[j] for j, a in coefficients.items()) <= bound + 1e-13
                    for coefficients, bound in c.rows))


class PathConstraintTests(unittest.TestCase):
    def test_public_api_and_frozen_pose_union(self):
        self.assertEqual(list(inspect.signature(path.infer).parameters), ["observations"])
        self.assertEqual(len(path.ALLOWED_POSES), 53)
        thirteen = public(scene(), [(round(i * .01, 10), 0.) for i in range(13)])
        self.assertEqual(len(path._inputs(thirteen)), 13)
        tree = {(0., 0.), (.06, 0.), (-.06, 0.), (0., .06), (0., -.06),
                (.12, 0.), (-.12, 0.), (0., .12), (0., -.12),
                (.06, .06), (.06, -.06), (-.06, .06), (-.06, -.06)}
        self.assertTrue(tree <= path.ALLOWED_POSES)
        for invalid in ([{**thirteen[0], "truth": True}], thirteen + [thirteen[0]],
                        [{**thirteen[0], "camera": [.005, 0.]}],
                        [{**thirteen[0], "bins": [True] * 8}]):
            with self.assertRaises(ValueError):
                path.infer(invalid)

    def test_query_epsilon_strip_is_in_outer_in_class(self):
        value = scene(x=.4 + .5 * av.EPS, width=.2, z=1.2)
        model = path._scene(value)
        self.assertTrue(av.intersects_query(model))
        point = continuous(value)
        self.assertGreater(point[0], .3)  # Exact ideal constraint omits this strip.
        observations = path._inputs(public(value))
        before = old._constraints(observations, "IN")
        after, changes = path._outer_constraints(observations, "IN")
        self.assertGreater(point[0], before.rows[4][1])
        self.assertLessEqual(point[0], after.rows[4][1])
        self.assertEqual(changes["query_IN_bounds_expanded"], 3)
        self.assertTrue(relaxation_contains(after, value))
        far = scene(x=0., z=3.02 + .5 * av.EPS)
        self.assertTrue(av.intersects_query(path._scene(far)))
        after, _ = path._outer_constraints(path._inputs(public(far)), "IN")
        self.assertLessEqual(continuous(far)[2], after.rows[6][1])
        self.assertTrue(relaxation_contains(after, far))

    def test_ray_corner_epsilon_strip_uses_direction_scaled_envelope(self):
        ray, z = 5, 1.4
        angle = av.ANGLES[ray]
        ux, uz = math.sin(angle), math.cos(angle)
        # Positive-X entry just beyond the depth exit by .5*EPS is accepted by
        # the frozen scalar slab law, though exact-real intervals do not touch.
        left = (z + .02) * ux / uz + ux * .5 * av.EPS
        value = scene(x=left + .02, width=.04, z=z)
        observations = path._inputs(public(value))
        wall = math.floor((4.2 / uz) / .1 + .5)
        self.assertNotEqual(observations[0]["bins"][ray], wall)
        before = old._constraints(observations, "IN")
        after, _ = path._outer_constraints(observations, "IN")
        point = continuous(value)
        fixed = []
        for i, ((coef, bound), (_, widened)) in enumerate(zip(before.rows, after.rows, strict=True)):
            if i < 7 or widened == bound:
                continue
            residual = sum(a * point[j] for j, a in coef.items())
            if residual > bound:
                fixed.append(i)
                self.assertLessEqual(residual, widened)
        self.assertTrue(fixed, "Fixture must expose the omitted overlap strip")
        self.assertTrue(relaxation_contains(after, value))

    def test_half_open_upper_bin_is_relaxed_but_not_a_valid_witness(self):
        ray, lower_bin = 3, 10
        upper = (lower_bin + .5) * .1
        value = scene(x=0., width=.2, z=upper * math.cos(av.ANGLES[ray]) + .02)
        observations = public(value)
        self.assertEqual(observations[0]["bins"][ray], lower_bin + 1)
        observations[0]["bins"][ray] = lower_bin
        c, _ = path._outer_constraints(path._inputs(observations), "IN")
        self.assertTrue(relaxation_contains(c, value))
        self.assertFalse(path.validate_witness(value, observations, "IN"))

    def test_actual_outer_fixtures_and_inherited_settings_are_unchanged(self):
        settings = dict(old.SOLVER_OPTIONS)
        for value in (scene(), scene(x=.44, width=.16, z=1.33), scene(x=-.4, width=.2, z=1.2)):
            label = "IN" if av.intersects_query(path._scene(value)) else "OUT"
            poses = [(round(i * .01, 10), 0.) for i in range(13)]
            observations = path._inputs(public(value, poses))
            c, _ = path._outer_constraints(observations, label)
            self.assertTrue(relaxation_contains(c, value))
            self.assertTrue(path.validate_witness(value, observations, label))
            self.assertEqual(c.lower[3], 0.)
            self.assertEqual(c.upper[3], old.SLACK_MAX)
            self.assertEqual(c.scipy()["options"], settings)
        self.assertEqual(old.SOLVER_OPTIONS, settings)

    @staticmethod
    def fake(status, x=None, message="fixture"):
        return SimpleNamespace(status=status, x=x, message=message)

    def test_validated_side_plus_opposite_reported_infeasible_only(self):
        value = scene()
        valid = self.fake(0, continuous(value))
        with patch.object(path, "milp", side_effect=[valid, self.fake(2)]) as calls:
            result = path.infer(public(value))
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(result["decision"], "IN_MODEL_CONDITIONAL")
        self.assertEqual(result["authority"], "NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY")
        self.assertTrue(result["exclusion_metadata"]["OUT"]["reported_infeasible"])
        self.assertFalse(result["sensor_clearance_certified"])
        self.assertFalse(result["universal_uniqueness_proven"])
        outside = scene(x=.5, width=.1, z=1.2)
        with patch.object(path, "milp", side_effect=[self.fake(2), self.fake(0, continuous(outside))]):
            result = path.infer(public(outside))
        self.assertEqual(result["decision"], "OUT_MODEL_CONDITIONAL")
        self.assertFalse(result["exclusion_metadata"]["IN"]["is_formal_certificate"])

    def test_limits_no_candidate_invalid_candidate_and_status_conflicts_remain_unknown(self):
        value = scene()
        for failed in (self.fake(1, message="Time limit reached"), self.fake(0), self.fake(3), self.fake(4),
                       self.fake(0, continuous(value)), self.fake(2, continuous(value))):
            with patch.object(path, "milp", side_effect=[self.fake(0, continuous(value)), failed]):
                result = path.infer(public(value))
            self.assertEqual(result["decision"], "UNKNOWN")
            self.assertFalse(result["exclusion_metadata"]["OUT"]["reported_infeasible"])
        with patch.object(path, "milp", side_effect=[self.fake(1, continuous(value), "Time limit"), self.fake(2)]):
            result = path.infer(public(value))
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertIsNotNone(result["witnesses"]["IN"])
        self.assertEqual(result["solver_metadata"][0]["limit_kind"], "TIME_LIMIT")
        with patch.object(path, "milp", side_effect=[self.fake(2), self.fake(2)]):
            result = path.infer(public(value))
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertIn("MODEL_INCONSISTENCY", result["reason"])

    def test_small_genuine_solver_fixture_replays_all_thirteen_bins(self):
        observations = public(scene(), [(round(i * .01, 10), 0.) for i in range(13)])
        result = path.infer(observations)
        self.assertEqual(result["solver_calls"], 2)
        self.assertIsNotNone(result["witnesses"]["IN"], result["solver_metadata"])
        self.assertTrue(path.validate_witness(result["witnesses"]["IN"], observations, "IN"))
        self.assertNotEqual(result["decision"], "OUT_MODEL_CONDITIONAL")
        self.assertEqual(result["authority"], path.AUTHORITY)


if __name__ == "__main__":
    unittest.main()
