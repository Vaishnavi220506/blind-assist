"""Harness invariants; no transfer-cohort observations or scoring."""
import unittest
from unittest.mock import patch

import active_view as av
import opportunity_transfer as tr


class TransferTests(unittest.TestCase):
    def test_shifts_every_solid_and_keeps_parent_immutable(self):
        source = [dict(case_id="a", group="appendage", solids=[[[0., 0., 1.], [.2, .1, 1.1]],
                                                                [[.2, 0., 1.], [.4, .1, 1.1]]])]
        shape, active = tr.shifted_sources(source, [])
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0]["solids"][1][0][0], .19)
        self.assertEqual(shape[1]["solids"][0][0][0], .01)
        self.assertEqual(source[0]["solids"][0][0][0], 0.)
        self.assertEqual(active, [])

    def test_empty_excluded_and_consumed_geometry_rejected(self):
        empty = dict(id="empty", stratum="in", scene=dict(boxes=[], wall_z=4.2))
        self.assertEqual(tr.shifted_sources([], [empty]), ([], []))
        first = dict(case_id="a", group="box", solids=[[[0., 0., 1.], [.2, .1, 1.1]]])
        second = dict(case_id="b", group="box", solids=[[[.01, 0., 1.], [.21, .1, 1.1]]])
        with self.assertRaises(ValueError):
            tr.shifted_sources([first, second], [])

    def test_choices_need_no_actual_future_or_truth(self):
        bank = av.ForecastBank(((1,), (1,)), (((1,), (2,)),)*4, (True, False))
        with patch.object(av, "observe", side_effect=AssertionError("future")), \
             patch.object(av, "intersects_query", side_effect=AssertionError("truth")), \
             patch.object(tr.pos, "source_scene", side_effect=AssertionError("geometry")):
            choices = tr.choose_all([dict(id="opaque", ranges=[1])], bank)
        self.assertTrue(choices[0]["initial_ambiguous"])
        self.assertEqual(choices[0]["actions"], dict(fixed=0, old_adaptive=0, positive_priority=0))

    def test_truth_aware_retention_does_not_hide_false_commitments(self):
        rows = [dict(id="new_fp", truth=False, policies={"fixed": dict(decision="UNKNOWN"),
                  "positive_priority": dict(decision="INTERSECTS")}),
                dict(id="lost_tp", truth=True, policies={"fixed": dict(decision="INTERSECTS"),
                  "positive_priority": dict(decision="NONINTERSECTING_HYPOTHESES")})]
        d = tr.active_contrast(rows, "fixed")
        self.assertEqual(d["FP"]["gained"], ["new_fp"])
        self.assertEqual(d["TP"]["gained"], [])
        self.assertEqual(d["TP"]["lost"], ["lost_tp"])
        self.assertEqual(d["false_OUT"]["gained"], ["lost_tp"])


if __name__ == "__main__":
    unittest.main()
