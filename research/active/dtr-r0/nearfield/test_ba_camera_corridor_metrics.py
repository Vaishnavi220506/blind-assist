import unittest

from ba_camera_corridor_metrics import ARMS, evaluate_rows


def make_rows(truths, alerts, unknown=(), ambiguous=(), boundary=(), clip_id="clip"):
    return [{"clip_id": clip_id, "frame_in_clip": i, "time_s": i * 0.2,
             "truth": truth, "boundary": i in boundary,
             "predictions": {arm: {"alert": alerts[i], "unknown": i in unknown,
                                    "ambiguous": i in ambiguous} for arm in ARMS}}
            for i, truth in enumerate(truths)]


class CameraCorridorMetricsTest(unittest.TestCase):
    def test_unknown_truth_and_abstention_never_become_clear(self):
        rows = make_rows([True, False, None, True, False, False],
                         [False, False, True, True, True, False],
                         unknown=(0, 1, 2, 3, 4), ambiguous=(3, 4))
        report = evaluate_rows(rows)
        arm = report["arms"]["raw_tof"]
        frame = arm["frames"]["all_known"]
        self.assertEqual([frame[k] for k in ("TP", "FP", "FN", "TN")], [1, 1, 1, 1])
        self.assertEqual(frame["abstained_positive"], 1)
        self.assertEqual(frame["abstained_negative"], 1)
        self.assertEqual(frame["prediction_unknown_positive"], 2)
        self.assertEqual(frame["prediction_unknown_negative"], 2)
        self.assertEqual(frame["ambiguous_alerts"], 2)
        self.assertEqual(report["unknown_truth_frames"], 1)
        self.assertEqual(arm["prediction_unknown_frames"], 5)
        self.assertEqual(arm["alerts_on_unknown_truth"], 1)
        self.assertAlmostEqual(frame["false_alert_rate_known_negative"], 1 / 3)
        self.assertEqual(arm["event_count"], 2)
        self.assertEqual(arm["event_recall"], 0.5)

    def test_multi_event_unknown_split_miss_and_preexisting_alert(self):
        rows = make_rows([True, True, None, True, False, True, True, False, True],
                         [False, True, True, False, True, True, True, False, False],
                         unknown=(0, 3, 8))
        arm = evaluate_rows(rows)["arms"]["nfo"]
        events = arm["events"]
        self.assertEqual([(e["start_frame"], e["end_frame"]) for e in events],
                         [(0, 1), (3, 3), (5, 6), (8, 8)])
        self.assertTrue(events[0]["left_censored"])
        self.assertIsNone(events[0]["preexisting_alert_at_entry"])
        self.assertAlmostEqual(events[0]["first_alert_relative_to_entry_s"], 0.2)
        self.assertTrue(events[1]["preceding_truth_unknown"])
        self.assertIsNone(events[1]["first_alert_time_s"])
        self.assertIsNone(events[1]["first_alert_relative_to_entry_s"])
        self.assertTrue(events[2]["preexisting_alert_at_entry"])
        self.assertEqual(events[2]["first_alert_relative_to_entry_s"], 0.0)
        self.assertFalse(events[3]["detected"])
        self.assertEqual(arm["event_recall"], 0.5)

    def test_false_segments_split_by_unknown_and_clip_sampled_duration(self):
        rows = make_rows([False, False, None, False, True, False], [True] * 6)
        rows += make_rows([False, False], [True, False], clip_id="second")
        arm = evaluate_rows(reversed(rows))["arms"]["depthpro_global"]
        segments = arm["false_alert_segments"]
        self.assertEqual(len(segments), 4)
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in segments],
                         [(0, 1), (3, 3), (5, 5), (0, 0)])
        self.assertAlmostEqual(segments[0]["sampled_duration_s"], 0.4)
        self.assertAlmostEqual(segments[0]["end_time_s"] - segments[0]["start_time_s"], 0.2)
        self.assertAlmostEqual(arm["false_alert_sampled_duration_s"], 1.0)
        self.assertTrue(segments[-1]["left_censored"])

    def test_boundary_contact_keeps_supplied_positive_truth(self):
        rows = make_rows([True, False, True, None], [False, True, True, True],
                         boundary=(0, 1, 3))
        report = evaluate_rows(rows)
        frames = report["arms"]["raw_tof"]["frames"]
        self.assertEqual(report["boundary_frames"], 3)
        self.assertEqual(frames["boundary"]["frames"], 2)
        self.assertEqual(frames["boundary"]["FN"], 1)
        self.assertEqual(frames["boundary"]["FP"], 1)
        self.assertEqual(frames["interior"]["TP"], 1)
        self.assertEqual(frames["interior"]["FN"], 0)

    def test_bad_timestamps_or_missing_frames_rejected(self):
        rows = make_rows([False, True], [False, True])
        rows[1]["time_s"] = 0.3
        with self.assertRaisesRegex(ValueError, "timestamp"):
            evaluate_rows(rows)
        with self.assertRaisesRegex(ValueError, "indices"):
            evaluate_rows(rows[1:])
        rows = make_rows([False], [False])
        rows[0]["truth"] = 0
        with self.assertRaisesRegex(ValueError, "truth"):
            evaluate_rows(rows)

    def test_interior_events_remain_whole_intervals_boundary_hits_not_interior_hits(self):
        rows = make_rows([True, True, True, False, False, False, False, True],
                         [False, True, False, True, True, True, False, True],
                         boundary=(1, 4, 7))
        arm = evaluate_rows(rows)["arms"]["raw_tof"]
        self.assertEqual(arm["event_count"], 2)
        self.assertEqual(arm["detected_events"], 2)
        self.assertEqual(arm["interior_event_count"], 1)
        self.assertEqual(arm["interior_detected_events"], 0)
        self.assertEqual(arm["false_alert_segment_count"], 1)
        self.assertEqual(arm["interior_false_alert_segment_count"], 2)
        self.assertAlmostEqual(arm["false_alert_sampled_duration_s"], 0.6)
        self.assertAlmostEqual(arm["interior_false_alert_sampled_duration_s"], 0.4)

    def test_no_evaluable_events_has_null_rate(self):
        arm = evaluate_rows(make_rows([None], [False], unknown=(0,)))["arms"]["raw_tof"]
        self.assertIsNone(arm["event_recall"])
        self.assertIsNone(arm["frames"]["all_known"]["recall"])
        self.assertEqual(arm["frames"]["all_known"]["TN"], 0)


if __name__ == "__main__":
    unittest.main()
