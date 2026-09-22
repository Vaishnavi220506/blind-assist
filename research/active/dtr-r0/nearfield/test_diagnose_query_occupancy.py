"""Independent synthetic statistics checks; no source arrays or checkpoints."""
import unittest

import numpy as np

from diagnose_query_occupancy import pair_summary, query_pairs, rank_metrics, split_metrics


def classification_fixture(rows):
    """Rows are (group, frame, relation, positive, score, frame_valid)."""
    probability = np.full((len(rows), 6), .05, np.float32)
    classes = np.full((len(rows), 6), 6, np.int64)
    valid = np.ones((len(rows), 6), bool)
    identities = []
    for i, (group, frame, relation, positive, score, known) in enumerate(rows):
        probability[i, 1] = score
        if positive:
            classes[i, 1] = 2
        if not known:
            valid[i, 4] = False
        identities.append(dict(base_group_id=group, frame_in_clip=frame,
                               layout_relation=relation, baseline=dict(unknown=True)))
    return dict(probability=probability), dict(classes=classes, valid=valid), identities


class QueryOccupancyDiagnosticTests(unittest.TestCase):
    def test_mixed_native_area_and_atomic_ties_have_hand_calculated_ranks(self):
        # The tied high-score block has foreground=1, background=0.5. The
        # lower block has background=1, so AUC=(1+0.5*0.5)/1.5 and AP=1/1.5.
        score = np.array([.9, .9, .2])
        foreground = np.array([.75, .25, 0.])
        background = np.array([.25, .25, 1.])
        result = rank_metrics(score, foreground, background)
        self.assertAlmostEqual(result["auc"], 5 / 6)
        self.assertAlmostEqual(result["ap"], 2 / 3)
        self.assertAlmostEqual(result["prevalence"], .4)
        order = np.array([1, 0, 2])
        self.assertEqual(result, rank_metrics(score[order], foreground[order], background[order]))
        tied = rank_metrics(np.zeros(3), foreground, background)
        self.assertEqual(tied["auc"], .5)
        self.assertEqual(tied["ap"], tied["prevalence"])

    def test_zero_coverage_high_score_does_not_create_nan_or_background(self):
        result = rank_metrics([.99, .8, .1], [0, 1, 0], [0, 0, 1])
        self.assertEqual(result, dict(auc=1., ap=1., prevalence=.5))

    def test_ranking_empty_denominators_are_explicit(self):
        for score, pos, neg, prevalence in (([], [], [], None),
                                           ([.5], [0], [0], None),
                                           ([.5], [1], [0], 1.),
                                           ([.5], [0], [1], 0.)):
            self.assertEqual(rank_metrics(score, pos, neg),
                             dict(auc=None, ap=None, prevalence=prevalence))
        result = pair_summary([])
        self.assertEqual((result["n"], result["wins"], result["ties"], result["losses"]), (0, 0, 0, 0))
        self.assertIsNone(result["win_rate"])

    def test_same_image_pairs_orient_positive_over_negative_and_exclude_invalid(self):
        score = np.array([[.3, .8, .4, .2, .7, .9]])
        classes = np.array([[6, 2, 6, 6, 3, 6]])
        valid = np.array([[True, True, False, True, True, True]])
        rows = query_pairs(score, classes, valid)
        got = {(r["positive_query"], r["negative_query"]): r for r in rows}
        self.assertEqual(set(got), {(1, 0), (1, 3), (1, 5), (4, 0), (4, 3), (4, 5)})
        self.assertEqual(sum(r["kind"] == "lateral" for r in rows), 3)
        self.assertEqual(sum(r["kind"] == "height" for r in rows), 3)
        self.assertAlmostEqual(got[1, 0]["margin"], .5)
        self.assertAlmostEqual(got[4, 5]["margin"], -.2)
        self.assertEqual(got[4, 0]["kind"], "height")
        summary = pair_summary([1., 0., -1.])
        self.assertEqual((summary["wins"], summary["ties"], summary["losses"], summary["win_rate"]),
                         (1, 1, 1, .5))

    def test_corridor_pairs_use_same_group_and_frame_not_row_position(self):
        pred, labels, identities = classification_fixture([
            ("a", 0, "INSIDE", True, .8, True),
            ("b", 0, "OUTSIDE", False, .6, True),
            ("a", 0, "BOUNDARY", True, .7, True),
            ("a", 0, "OUTSIDE", False, .2, True),
            ("b", 0, "INSIDE", True, .65, True),
            ("a", 1, "INSIDE", True, .99, True),
            ("a", 1, "OUTSIDE", False, .1, False),
        ])
        report, details = split_metrics(pred, labels, identities, "classifier", .5)
        rows = {(r["row"], r["outside_row"]): r for r in details["relation_pairs"]}
        self.assertEqual(set(rows), {(0, 3), (2, 3), (4, 1)})
        self.assertAlmostEqual(rows[0, 3]["margin"], .6, places=6)
        self.assertAlmostEqual(rows[2, 3]["margin"], .5, places=6)
        self.assertAlmostEqual(rows[4, 1]["margin"], .05, places=6)
        self.assertEqual(report["paired_corridor_order"]["INSIDE"]["n"], 2)
        self.assertEqual(report["paired_corridor_order"]["BOUNDARY"]["n"], 1)
        self.assertEqual(report["known_frames"], 6)
        self.assertEqual(report["unknown_sensor_frames"], 7)

    def test_fixed_float64_cutoff_preserves_no_alert_sentinel(self):
        pred, labels, identities = classification_fixture([
            ("a", 0, "INSIDE", True, .5, True),
            ("a", 0, "OUTSIDE", False, .5, True),
        ])
        threshold = float(np.nextafter(np.float64(.5), np.inf))
        report, _ = split_metrics(pred, labels, identities, "classifier", threshold)
        self.assertEqual({k: report["fixed_cutoff"][k] for k in ("TP", "FP", "FN")}, dict(TP=0, FP=0, FN=1))
        at_tie, _ = split_metrics(pred, labels, identities, "classifier", .5)
        self.assertEqual((at_tie["fixed_cutoff"]["TP"], at_tie["fixed_cutoff"]["FP"]), (1, 1))

    def test_fixed_cutoff_empty_positive_denominator_is_not_zero_recall(self):
        pred, labels, identities = classification_fixture([("a", 0, "OUTSIDE", False, .2, True)])
        report, _ = split_metrics(pred, labels, identities, "classifier", .5)
        self.assertIsNone(report["fixed_cutoff"]["recall"])
        self.assertEqual(report["fixed_cutoff"]["fpr"], 0.)

    def test_subthreshold_mixed_foreground_can_rank_and_respond_to_query(self):
        pred, labels, identities = classification_fixture([
            ("a", 0, "INSIDE", True, .2, True),
            ("a", 0, "OUTSIDE", False, .1, True),
        ])
        pred["distribution"] = np.full((2, 6, 7), 1 / 7, np.float32)
        pred["mask"] = np.broadcast_to(np.array([.1, .2, .99], np.float32), (2, 6, 1, 3)).copy()
        pred["mask"][0, 1, 0] = [.2, .1, .99]
        labels["coverage"] = np.tile(np.array([[[1., 1., 0.]]], np.float32), (2, 1, 1))
        labels["mask"] = np.zeros((2, 6, 1, 3), np.float32)
        labels["mask"][0, 1, 0, 0] = .25
        report, details = split_metrics(pred, labels, identities, "occupancy", .5)
        self.assertEqual(len(details["spatial_queries"]), 1)
        row = details["spatial_queries"][0]
        self.assertAlmostEqual(row["auc"], 11 / 14)
        self.assertAlmostEqual(row["ap"], .25)
        self.assertAlmostEqual(row["prevalence"], .125)
        self.assertEqual(row["fixed_iou"], 0.)
        self.assertAlmostEqual(row["soft_iou"], .1, places=6)
        self.assertEqual(row["peak_occupied_fraction"], .25)
        self.assertFalse(row["label_has_majority_foreground_cell"])
        self.assertAlmostEqual(row["wrong_query_mean_ap"], .125)
        self.assertAlmostEqual(report["correct_minus_wrong_query_ap"]["mean"], .125)
        self.assertEqual(report["centre_visible_positive_ranking"]["ap"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
