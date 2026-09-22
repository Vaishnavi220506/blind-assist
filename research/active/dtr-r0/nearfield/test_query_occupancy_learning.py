"""Independent synthetic checks of label, cutoff and evidence semantics.

These fixtures contain no captured observations or train/dev/held file access.
No experiment model is fitted or optimized.
"""
import math
import unittest

import numpy as np
import torch

from ba_camera_corridor_metrics import evaluate_rows
from query_occupancy_data import geometric_labels, visible_labels
from query_occupancy_evaluate import ARMS, metric_rows
from query_occupancy_learning import frame_truth, operating_point, scores, supervised_loss
from query_occupancy_model import distance_cdf


def fixture_labels(truth):
    classes = np.full((len(truth), 6), 6, np.int64)
    classes[np.flatnonzero(truth), 1] = 2
    return dict(classes=classes, valid=np.ones(classes.shape, bool))


def fixture_identities(count):
    return [dict(id=f"synthetic-{i}", clip_id="synthetic-clip", frame_in_clip=i,
                 time_s=i * .2, base_group_id="synthetic-layout", type_id="cuboid",
                 layer="BODY", layout_relation="INSIDE",
                 baseline=dict(alert=False, unknown=True)) for i in range(count)]


def fixture_rows(labels, probabilities, threshold):
    identities = fixture_identities(len(probabilities))
    outputs = {name: dict(probability=np.asarray(probabilities, np.float32))
               for name in ("classifier", "occupancy")}
    selection = {name: dict(selection=dict(threshold=threshold)) for name in outputs}
    return metric_rows(identities, labels, outputs, selection)


class QueryOccupancyLearningTests(unittest.TestCase):
    def test_no_alert_cutoff_replays_exactly_after_float32_storage(self):
        labels = fixture_labels([True, False, False])
        probability = np.full((3, 6), .5, np.float32)
        truth, valid = frame_truth(labels)
        selected, curve = operating_point(probability[:, 1], truth, valid, fixture_identities(3))
        self.assertEqual((selected["TP"], selected["FP"]), (0, 0))
        # Only two attainable points exist; equal scores cannot be split by a cutoff.
        self.assertEqual({(row["TP"], row["FP"]) for row in curve}, {(0, 0), (1, 2)})
        rows = fixture_rows(labels, probability, selected["threshold"])
        self.assertFalse(any(row["predictions"]["classifier"]["alert"] for row in rows))
        self.assertFalse(any(row["predictions"]["occupancy"]["alert"] for row in rows))

    def test_atomic_ties_never_invent_a_low_false_positive_operating_point(self):
        # A tied negative shares the positive's score. Splitting it would invent
        # 100% recall at 0% FPR; the only admissible point has zero recall here.
        score = np.array([.8, .8, .1], np.float32)
        selected, curve = operating_point(score, [True, False, False], [True] * 3, fixture_identities(3))
        self.assertEqual(selected["recall"], 0.)
        self.assertNotIn((1, 0), {(point["TP"], point["FP"]) for point in curve})

    def test_selection_requires_known_positive_and_negative_denominators(self):
        for truth, valid in (([True, True], [True, True]),
                             ([False, False], [True, True]),
                             ([True, False], [False, False])):
            with self.assertRaisesRegex(ValueError, "NOT_EVALUABLE"):
                operating_point([.8, .2], truth, valid, fixture_identities(2))

    def test_unknown_truth_and_unknown_sensor_axis_have_distinct_denominators(self):
        labels = fixture_labels([True, False, True, False])
        labels["valid"][1, 4] = False
        probability = np.repeat(np.array([.9, .9, .1, .1], np.float32)[:, None], 6, axis=1)
        rows = fixture_rows(labels, probability, .5)
        result = evaluate_rows(rows, arms=ARMS)
        arm = result["arms"]["occupancy"]
        count = arm["frames"]["all_known"]
        self.assertEqual([row["truth"] for row in rows], [True, None, True, False])
        self.assertEqual((result["known_truth_frames"], result["unknown_truth_frames"]), (3, 1))
        self.assertEqual((count["TP"], count["FP"], count["FN"], count["TN"]), (1, 0, 1, 0))
        self.assertEqual(count["abstained_negative"], 1)
        self.assertEqual(count["false_alert_rate_known_negative"], 0.)
        self.assertEqual(arm["alerts_on_unknown_truth"], 1)
        # The UNKNOWN truth sample splits observed positive runs, preserving the gap.
        self.assertEqual((arm["detected_events"], arm["event_count"]), (1, 2))
        self.assertTrue(arm["events"][1]["preceding_truth_unknown"])

    def test_invisible_geometric_positive_is_supervised_without_invented_mask(self):
        labels = dict(classes=torch.tensor([[2]]), valid=torch.tensor([[True]]),
                      mask=torch.zeros(1, 1, 2, 2), coverage=torch.zeros(1, 2, 2))
        output = dict(distance_logits=torch.zeros(1, 1, 7), mask_logits=torch.full((1, 1, 2, 2), 9.))
        self.assertAlmostEqual(float(supervised_loss(output, labels, "occupancy")), math.log(7), places=6)
        # Binary supervision retains the same invisible geometric positive.
        self.assertAlmostEqual(float(supervised_loss(dict(query_logits=torch.zeros(1, 1)), labels, "classifier")),
                               math.log(2), places=6)
        labels["valid"].fill_(False)
        self.assertEqual(float(supervised_loss(output, labels, "occupancy")), 0.)

    def test_partial_native_coverage_is_not_a_negative_pixel(self):
        # One quarter of the coarse cell is observed and fully occupied. The
        # unknown three quarters must not make its conditional target 0.25.
        labels = dict(classes=torch.tensor([[1]]), valid=torch.tensor([[True]]),
                      mask=torch.tensor([[[[.25]]]]), coverage=torch.tensor([[[.25]]]))
        logits = torch.zeros(1, 1, 7)
        near_one = supervised_loss(dict(distance_logits=logits, mask_logits=torch.tensor([[[[8.]]]])), labels, "occupancy")
        near_zero = supervised_loss(dict(distance_logits=logits, mask_logits=torch.tensor([[[[-8.]]]])), labels, "occupancy")
        self.assertLess(float(near_one), float(near_zero))

    def test_geometric_first_hit_uses_all_extents_and_closed_upper_bin_edges(self):
        # The first object intersects only at its edge, despite an outside center.
        bounds = [(np.array([.3, .5, 1.25]), np.array([.7, .8, 1.7])),
                  (np.array([-.1, .5, 2.]), np.array([.1, .8, 2.5]))]
        query = np.array([[-.3, .3, .42, .9]])
        classes, distances = geometric_labels(bounds, query)
        self.assertAlmostEqual(float(distances[0]), 1.25)
        # CDF uses <= upper edge; exactly 1.25m belongs to the second bin.
        self.assertEqual(int(classes[0]), 1)
        none, distance = geometric_labels([], query)
        self.assertEqual(int(none[0]), 6)
        self.assertTrue(np.isnan(distance[0]))

    def test_unexplained_thin_native_surface_is_retained_in_visibility_audit(self):
        native = np.full((360, 640), np.nan, np.float32)
        native[250, 320] = 1.75
        visible = visible_labels(native, [], np.array([[-.3, .3, .42, .9]]))
        self.assertEqual(visible["count"], [1])
        self.assertEqual(visible["unexplained"], [1])
        self.assertAlmostEqual(float(visible["mask"].sum()), 1 / 64)
        self.assertAlmostEqual(float(visible["coverage"].sum()), 1 / 64)

    def test_occupancy_score_is_same_categorical_mass_as_distance_cdf(self):
        logits = torch.tensor([[[1., 2., -3., .5, 0., -2., 4.],
                                [0., 0., 0., 0., 0., 0., 20.]]])
        alert = scores(dict(distance_logits=logits), "occupancy")
        torch.testing.assert_close(alert, distance_cdf(logits)[..., -1], atol=0, rtol=1e-6)
        self.assertGreater(float(alert[0, 1]), 0.)


if __name__ == "__main__":
    unittest.main()
