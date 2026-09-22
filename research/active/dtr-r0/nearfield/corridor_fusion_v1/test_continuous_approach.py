"""Synthetic-only regression tests; no capture files or model outcomes read."""

import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

from evaluate_continuous_approach import (
    METHODS, aggregate, evaluate, geometry, metric_for_method, onset,
    target_tof_ranges, verify_sealed_inputs, visible_proxy,
)


def public_slot(status="SIM_VALID", distance=2.0):
    return dict(status=status, distance_m=distance, range_noise_sigma_m=0.02, signal_strength_proxy=1.0)


def tof_fixture():
    raw = dict(tof_packet_received=True, tof_zones=[dict(zone_id=0, targets=[public_slot()])])
    ev = dict(zonal_tof_native=[dict(zone_id=0, packet_received=True,
        private_rays=[dict(subray=0, range_m=1.9, actor_id="ep/target"),
                      dict(subray=1, range_m=2.0, actor_id="ep/background")],
        returned_lineage=[dict(target_index=0, hit_indices=[1])])])
    return raw, ev


def metric_frames(risks, alerts, contacts=None, supports=None, dt=0.1):
    contacts = contacts or [False] * len(risks)
    supports = supports or [False] * len(risks)
    return [dict(id=f"f{i}", time_s=round(i * dt, 6), risk_truth=risk,
                 corridor_truth=risk, contact_truth=contacts[i], target_returned=supports[i],
                 forward_clearance_m=3.0 - i * dt, phase="FORWARD",
                 predictions={method: dict(alert=bool(alerts[i]), score=0.8 if alerts[i] else 0.1) for method in METHODS})
            for i, risk in enumerate(risks)]


def synthetic_full_source():
    # Source generation contains no trained predictions or captured evidence.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from continuous_approach_source import source
    spec = source()
    raw, ev, preds = [], [], []
    state = {}
    for frame in spec["frames"]:
        target = frame["objects"][0]
        raw.append(dict(id=frame["id"], episode_id=frame["episode"], time_s=frame["time_s"],
                        tof_packet_received=True, tof_zones=[]))
        ev.append(dict(id=frame["id"], episode_id=frame["episode"], time_s=frame["time_s"],
                       family=frame["family"], camera=frame["camera"], body_origin_m=frame["body_origin_m"],
                       native_bounds=[dict(name="target", center_m=target["center_m"], extent_m=[s / 2 for s in target["size_m"]])],
                       zonal_tof_native=[]))
        # Explicit synthetic constant-off output exercises missed-event coverage.
        state[frame["episode"]] = False
        preds.append(dict(id=frame["id"], episode_id=frame["episode"], time_s=frame["time_s"],
                          tof_packet_received=True, astar_score=0.1, astar_threshold=0.55, astar_alert=False,
                          raw_hgb_score=0.1, raw_hgb_threshold=0.5, raw_hgb_alert=False,
                          astar_hysteresis_on_threshold=0.55, astar_hysteresis_off_threshold=0.53,
                          astar_hysteresis_alert=False))
    return spec, raw, ev, preds


class ReturnedTargetSupportTest(unittest.TestCase):
    def test_private_hit_not_selected_is_not_public_support(self):
        raw, ev = tof_fixture()
        result = target_tof_ranges(raw, ev)
        self.assertEqual(result["private_target_hit_count"], 1)
        self.assertEqual(result["valid_returned_target_slot_count"], 0)
        self.assertFalse(result["target_returned"])

    def test_merged_target_lineage_counts_one_public_slot(self):
        raw, ev = tof_fixture()
        raw["tof_zones"][0]["targets"][0]["status"] = "SIM_MERGED"
        ev["zonal_tof_native"][0]["returned_lineage"][0]["hit_indices"] = [0, 1]
        result = target_tof_ranges(raw, ev)
        self.assertTrue(result["target_returned"])
        self.assertEqual(result["valid_returned_target_slot_count"], 1)
        self.assertEqual(result["target_public_return_min_m"], 2.0)
        self.assertEqual(result["private_target_hit_min_m"], 1.9)

    def test_packet_status_and_nonfinite_range_gate_public_support(self):
        for change in ("packet", "status", "nan", "sigma"):
            with self.subTest(change=change):
                raw, ev = tof_fixture()
                ev["zonal_tof_native"][0]["returned_lineage"][0]["hit_indices"] = [0]
                if change == "packet":
                    raw["tof_packet_received"] = False
                    ev["zonal_tof_native"][0]["packet_received"] = False
                elif change == "status":
                    raw["tof_zones"][0]["targets"][0]["status"] = "SIM_INVALID"
                elif change == "nan":
                    raw["tof_zones"][0]["targets"][0]["distance_m"] = float("nan")
                else:
                    raw["tof_zones"][0]["targets"][0]["range_noise_sigma_m"] = -1
                result = target_tof_ranges(raw, ev)
                self.assertFalse(result["target_returned"])
                self.assertEqual(result["private_target_hit_count"], 1)

    def test_unresolved_usable_lineage_is_unknown(self):
        raw, ev = tof_fixture()
        ev["zonal_tof_native"][0]["returned_lineage"] = []
        result = target_tof_ranges(raw, ev)
        self.assertIsNone(result["target_returned"])
        self.assertEqual(result["attribution_unknown_slot_count"], 1)
        ev["zonal_tof_native"][0]["returned_lineage"] = [dict(target_index=0, hit_indices=[999])]
        self.assertIsNone(target_tof_ranges(raw, ev)["target_returned"])

    def test_duplicate_lineage_and_out_of_range_slot_rejected(self):
        raw, ev = tof_fixture()
        zone = ev["zonal_tof_native"][0]
        zone["returned_lineage"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            target_tof_ranges(raw, ev)
        zone["returned_lineage"] = [dict(target_index=1, hit_indices=[0])]
        with self.assertRaisesRegex(ValueError, "public slot"):
            target_tof_ranges(raw, ev)


class TimelineMetricsTest(unittest.TestCase):
    def test_contact_inside_corridor_near_blind_range_stays_risk(self):
        bound = dict(center_m=[0.05, 0.0, 1.0], extent_m=[0.02, 0.02, 0.1])
        result = geometry(bound, [0.0, 0.0, 0.0])
        self.assertFalse(result["corridor"])
        self.assertTrue(result["contact"])
        self.assertTrue(result["risk"])

    def test_body_relative_height_and_unknown_geometry(self):
        bound = dict(center_m=[1.0, 0.0, 3.0], extent_m=[0.1, 0.1, 0.1])
        self.assertTrue(geometry(bound, [0.0, 0.0, 2.0])["risk"])
        self.assertIsNone(geometry(bound, None))

    def test_first_correct_alert_ignores_earlier_false_alert(self):
        frames = metric_frames([False, True, True, True], [True, False, True, True], [False, False, False, True])
        result = metric_for_method(frames, "astar", "approach")
        self.assertEqual(result["first_alert_s"], 0.0)
        self.assertEqual(result["first_correct_alert_s"], 0.2)
        self.assertAlmostEqual(result["first_correct_alert_lead_before_contact_s"], 0.1)
        self.assertEqual(result["confusion"]["TP"], 2)
        self.assertEqual(result["confusion"]["FP"], 1)
        self.assertEqual(result["confusion"]["FN"], 1)

    def test_alert_at_contact_is_not_advance_warning(self):
        result = metric_for_method(metric_frames([True, True], [False, True], [False, True]), "astar", "approach")
        self.assertFalse(result["correct_alert_before_contact"])
        self.assertIsNone(result["first_correct_alert_lead_before_contact_s"])
        self.assertEqual(result["first_correct_alert_to_contact_signed_s"], 0.0)

    def test_native_supported_fn_and_unknown_not_true_negative(self):
        frames = metric_frames([True, True, None, False], [False, True, False, False], supports=[True, True, None, False])
        result = metric_for_method(frames, "astar", "approach")
        self.assertEqual(result["native_supported_FN_ids"], ["f0"])
        self.assertEqual(result["native_supported_recall"], 0.5)
        self.assertEqual(result["confusion"]["TN"], 1)
        self.assertEqual(result["confusion"]["unknown_truth_frames"], 1)
        self.assertEqual(result["confusion"]["precision"], 1.0)
        self.assertEqual(result["confusion"]["recall"], 0.5)

    def test_release_starts_first_negative_after_last_risk_interval(self):
        frames = metric_frames([True, False, True, True, False, False], [True, False, True, True, True, False])
        result = metric_for_method(frames, "astar", "stop_back")
        self.assertEqual(result["release_first_nonrisk_id"], "f4")
        self.assertEqual(result["release_frame_id"], "f5")
        self.assertAlmostEqual(result["release_delay_s"], 0.1)
        frames[-2]["predictions"]["astar"]["alert"] = False
        self.assertEqual(metric_for_method(frames, "astar", "stop_back")["release_delay_s"], 0.0)

    def test_release_censoring_and_absent_mean_are_null(self):
        frames = metric_frames([True, True], [True, True])
        method = metric_for_method(frames, "astar", "stop_back")
        self.assertEqual(method["release_status"], "UNKNOWN_RIGHT_CENSORED_RISK")
        sequence = dict(sequence_id="censored", process="stop_back", frames=frames, contact_first_s=None,
                        methods={"astar": method})
        summary = aggregate([sequence], "astar")
        self.assertIsNone(summary["stop_back_release_delay_mean_observed_s"])
        self.assertEqual(summary["stop_back_release_denominator"], 1)

    def test_release_cannot_bridge_unknown_risk_or_reuse_old_alert(self):
        frames = metric_frames([True, False, True, False], [True, False, False, False])
        self.assertEqual(metric_for_method(frames, "astar", "stop_back")["release_status"], "NO_PRIOR_ALERT_IN_LAST_RISK_INTERVAL")
        frames = metric_frames([True, None, False], [True, False, False])
        self.assertEqual(metric_for_method(frames, "astar", "stop_back")["release_status"], "UNKNOWN_EXIT_OR_LATER_RISK")

    def test_head_turn_is_stationary_occupancy_not_collision_success(self):
        result = metric_for_method(metric_frames([True, True], [True, True], [False, True]), "astar", "head_turn")
        self.assertEqual(result["complete_event_kind"], "STATIONARY_CORRIDOR_OCCUPANCY")
        self.assertEqual(result["contact_warning_status"], "NOT_APPLICABLE_STATIONARY_OCCUPANCY")
        self.assertIsNone(result["correct_alert_before_contact"])

    def test_sidepass_segments_and_projection_left_censor(self):
        frames = metric_frames([False] * 5, [True, True, False, True, False])
        result = metric_for_method(frames, "astar", "side_pass")
        self.assertEqual(result["side_pass_false_alert_segments"], 2)
        self.assertAlmostEqual(result["side_pass_false_alert_duration_s"], 0.3)
        first = onset([dict(id="f0", time_s=0.0, visible=True)], "visible", "NONE")
        self.assertEqual(first["status"], "LEFT_CENSORED_AT_FIRST_SAMPLE")
        camera = dict(x=0.0, y=0.0, z=1.7, yaw=0.0, pitch=0.0, roll=0.0)
        rig = dict(width=640, height=360, hfov_deg=70.0)
        bound = dict(center_m=[2.0, 0.0, 1.7], extent_m=[0.1, 0.1, 0.1])
        self.assertTrue(visible_proxy(bound, camera, rig))
        bound["center_m"][0] = -2.0
        self.assertFalse(visible_proxy(bound, camera, rig))


class IntegrityAndCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_full_source()

    def test_all_24_synthetic_sequences_included_even_if_no_alerts(self):
        spec, raw, ev, preds = self.fixture
        sequences, summary = evaluate(spec, list(reversed(raw)), list(reversed(ev)), list(reversed(preds)))
        self.assertEqual((len(sequences), summary["frames"]), (24, 1920))
        self.assertEqual(len(summary["methods"]["astar"]["no_alert_sequence_ids"]), 24)
        self.assertEqual(summary["methods"]["astar"]["stationary_occupancy_sequences"], 6)
        self.assertEqual(len(summary["methods"]["astar"]["missed_sequence_ids"]), 18)
        self.assertIsNone(summary["methods"]["astar"]["stop_back_release_delay_mean_observed_s"])
        self.assertTrue(all(row["target_first_visible_authority"].startswith("PROJECTION_PROXY") for row in sequences))
        for row in sequences:
            if row["process"] == "stop_back":
                self.assertEqual({phase["phase"] for phase in row["motion_phases"]}, {"FORWARD", "STATIONARY", "BACK"})

    def test_duplicate_prediction_id_is_rejected(self):
        spec, raw, ev, preds = copy.deepcopy(self.fixture)
        preds[-1]["id"] = preds[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate(spec, raw, ev, preds)

    def test_time_mismatch_and_score_alert_disagreement_rejected(self):
        for error in ("time", "score_alert", "hysteresis"):
            with self.subTest(error=error):
                spec, raw, ev, preds = copy.deepcopy(self.fixture)
                if error == "time":
                    preds[1]["time_s"] = 100.0
                elif error == "score_alert":
                    preds[1]["astar_alert"] = True
                else:
                    preds[1]["astar_hysteresis_alert"] = True
                with self.assertRaises(ValueError):
                    evaluate(spec, raw, ev, preds)


class SealedInputTest(unittest.TestCase):
    def fixture(self):
        capture = (Path.cwd() / "synthetic-capture").resolve()
        prediction_root = (Path.cwd() / "synthetic-predictions").resolve()
        prediction_file = prediction_root / "predictions.jsonl"
        docs = {
            prediction_root / "completion.json": dict(status="PASS", evaluator_read=False,
                outputs={"predictions.jsonl": "p", "summary.json": "s", "input-seal.json": "i"}),
            prediction_root / "summary.json": dict(status="PASS", evaluator_read=False, source_labels_used=False, frames=1),
            prediction_root / "input-seal.json": dict(evaluator_read=False, inputs={str(capture / "raw.jsonl"): "r"}),
            capture / "receipt.json": dict(status="PASS", frames=1, spec_sha256="spec", hashes={
                "raw.jsonl": "r", "evaluator.jsonl": "e", "provenance.jsonl": "v", "manifest.json": "m"}),
            capture / "spec.json": dict(frame_count=1),
        }
        hashes = {prediction_file: "p", prediction_root / "summary.json": "s", prediction_root / "input-seal.json": "i",
                  prediction_root / "completion.json": "pc", capture / "receipt.json": "receipt",
                  capture / "spec.json": "spec", capture / "raw.jsonl": "r", capture / "evaluator.jsonl": "e",
                  capture / "provenance.jsonl": "v", capture / "manifest.json": "m"}
        return capture, prediction_file, docs, hashes

    def run_verifier(self, capture, predictions, docs, hashes, events):
        def read(path, *args, **kwargs):
            events.append(("read", path))
            return json.dumps(docs[path])
        def digest(path):
            events.append(("hash", path))
            return hashes[path]
        with patch.object(Path, "read_text", read), patch.object(Path, "is_file", return_value=True), patch("evaluate_continuous_approach.sha", side_effect=digest):
            return verify_sealed_inputs(capture, predictions)

    def test_prediction_hashes_verified_before_evaluator_is_touched(self):
        capture, predictions, docs, hashes = self.fixture()
        events = []
        result = self.run_verifier(capture, predictions, docs, hashes, events)
        self.assertEqual(result["status"], "PASS")
        self.assertLess(events.index(("hash", predictions)), events.index(("hash", capture / "evaluator.jsonl")))

    def test_bad_prediction_completion_blocks_evaluator_access(self):
        capture, predictions, docs, hashes = self.fixture()
        docs[predictions.parent / "completion.json"]["status"] = "RUNNING"
        events = []
        with self.assertRaisesRegex(ValueError, "prediction completion"):
            self.run_verifier(capture, predictions, docs, hashes, events)
        self.assertFalse(any(path.parent == capture for _, path in events))

    def test_bad_capture_hash_and_different_raw_input_are_rejected(self):
        for error in ("capture_hash", "raw_input"):
            with self.subTest(error=error):
                capture, predictions, docs, hashes = self.fixture()
                if error == "capture_hash":
                    hashes[capture / "evaluator.jsonl"] = "changed"
                else:
                    docs[predictions.parent / "input-seal.json"]["inputs"][str(capture / "raw.jsonl")] = "different"
                with self.assertRaises(ValueError):
                    self.run_verifier(capture, predictions, docs, hashes, [])


if __name__ == "__main__":
    unittest.main()
