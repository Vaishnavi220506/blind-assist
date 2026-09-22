"""Synthetic auditor tests only: no captured/held files, no training, no UE."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import audit_query_occupancy as audit


def identities(count):
    return [dict(id=f"synthetic-{i}", clip_id="clip", frame_in_clip=i, time_s=i * .2,
                 base_group_id="group", type_id="cuboid", layer="BODY", layout_relation="INSIDE",
                 baseline=dict(alert=False, unknown=True)) for i in range(count)]


def labels(truth):
    classes = np.full((len(truth), 6), 6, np.int64)
    valid = np.ones_like(classes, bool)
    for i, value in enumerate(truth):
        if value is None:
            valid[i, 4] = False
        elif value:
            classes[i, 1] = 2
    return dict(classes=classes, valid=valid)


def rows(truth, alerts):
    p = np.repeat(np.asarray(alerts, np.float32)[:, None], 6, 1)
    predictions = {arm: dict(probability=p) for arm in audit.LEARNED}
    selection = {arm: dict(selection=dict(threshold=.5)) for arm in audit.LEARNED}
    return audit.rebuild_rows(identities(len(truth)), labels(truth), predictions, selection)


def synthetic_seals():
    root = Path("synthetic-root")
    fit, pred = audit.stage(root, "fit"), audit.stage(root, "predictions")
    hashes, documents = {}, {}
    def reserve(path):
        value = hashlib.sha256(str(path).encode()).hexdigest()
        hashes[path] = value
        return value
    names = [f"query_occupancy_{part}.py" for part in ("data", "learning", "model", "evaluate")]
    names += ["run_query_occupancy.py", "ba_camera_corridor_metrics.py", "../../../../tools/research_backend.py"]
    source = dict(code={name: reserve(audit.HERE / name) for name in names}, held_labels_opened=False,
        protocol_sha256=reserve(root / "plan/learning-protocol.md"),
        materialization_sha256=reserve(audit.stage(root, "prepared") / "materialization.json"),
        used_inputs={name: reserve(audit.stage(root, "prepared") / name) for name in (
            "observations/rgb.npy", "observations/tof.npy", "observations/identities.json", "labels/train.npz", "labels/dev.npz")})
    selection = {arm: dict(selection=dict(threshold=.5), checkpoint_sha256=reserve(fit / f"{arm}.pt")) for arm in audit.LEARNED}
    seal = dict(status="PASS", held_labels_opened=False, frames=1,
        selection_sha256=reserve(fit / "selection.json"), source_seal_sha256=reserve(fit / "learning-source-seal.json"),
        models={arm: dict(predictions_sha256=reserve(pred / f"{arm}.npz"),
                         checkpoint_sha256=selection[arm]["checkpoint_sha256"]) for arm in audit.LEARNED})
    documents.update({fit / "selection.json": selection, fit / "learning-source-seal.json": source,
                      pred / "prediction-seal.json": seal})
    return root, hashes, documents


class QueryOccupancyAuditTests(unittest.TestCase):
    def test_missing_prediction_seal_never_opens_held_arrays(self):
        with patch.object(audit, "read_json", side_effect=FileNotFoundError("seal")) as reader, \
             patch.object(audit.np, "load") as loader:
            with self.assertRaises(FileNotFoundError):
                audit.audit(Path("synthetic-root"))
            loader.assert_not_called()
            self.assertEqual(len(reader.call_args_list), 1)
            self.assertEqual(reader.call_args.args[0].name, "prediction-seal.json")

    def test_full_hash_gate_accepts_frozen_relative_dependency(self):
        root, hashes, documents = synthetic_seals()
        with patch.object(audit, "read_json", side_effect=lambda p: documents[p]), \
             patch.object(audit, "digest", side_effect=lambda p: hashes[p]):
            seal, _, source = audit.verify_prediction_gate(root)
        self.assertEqual(seal["status"], "PASS")
        self.assertIn("../../../../tools/research_backend.py", source["code"])

    def test_corrupt_prediction_hash_blocks_every_held_read(self):
        root, hashes, documents = synthetic_seals()
        hashes[audit.stage(root, "predictions") / "occupancy.npz"] = "0" * 64
        with patch.object(audit, "read_json", side_effect=lambda p: documents[p]) as reader, \
             patch.object(audit, "digest", side_effect=lambda p: hashes[p]), patch.object(audit.np, "load") as loader:
            with self.assertRaisesRegex(audit.AuditFailure, "SHA256 mismatch"):
                audit.audit(root)
            loader.assert_not_called()
            self.assertTrue(all("evaluation" not in str(call.args[0]) for call in reader.call_args_list))

    def test_world_prism_uses_background_and_extent_contact(self):
        camera = dict(x=0., y=0., z=2., pitch=0., yaw=0., roll=0.)
        objects = [dict(name="target", render_bounds_center_m=[2.5, 0., 1.4], render_bounds_extent_m=[.2, .1, .1]),
                   dict(name="background", render_bounds_center_m=[1.5, .5, 1.4], render_bounds_extent_m=[.25, .2, .1])]
        # Background's center is outside the corridor but its closed extent hits.
        cls, distance = audit.world_box_labels(objects, camera, np.asarray([[-.3, .3, .42, .9]]))
        self.assertEqual(cls.tolist(), [1])
        self.assertEqual(distance.tolist(), [1.25])
        empty, no_distance = audit.world_box_labels([], camera)
        self.assertTrue((empty == 6).all())
        self.assertTrue(np.isnan(no_distance).all())

    def test_near_clip_upper_edges_and_depth_disjoint_boxes(self):
        camera = dict(x=0., y=0., z=2., pitch=0., yaw=0., roll=0.)
        query = np.asarray([[-.3, .3, .42, .9]])
        for front, expected in ((.2, 0), (.75, 0), (1.25, 1), (3., 5), (3.01, 6)):
            obj = dict(render_bounds_center_m=[front + .1, 0., 1.4], render_bounds_extent_m=[.1, .1, .1])
            cls, _ = audit.world_box_labels([obj], camera, query)
            self.assertEqual(int(cls[0]), expected)

    def test_unknown_truth_splits_events_and_false_segments(self):
        sample = rows([False, True, True, None, True, False, False, None, False],
                      [True, False, True, True, False, True, True, True, True])
        result = audit.recompute_metrics(sample)
        arm = result["arms"]["occupancy"]
        count = arm["frames"]["all_known"]
        self.assertEqual((count["TP"], count["FP"], count["FN"], count["TN"]), (1, 4, 2, 0))
        self.assertEqual((result["known_truth_frames"], result["unknown_truth_frames"]), (7, 2))
        self.assertEqual((arm["detected_events"], arm["event_count"]), (1, 2))
        self.assertEqual(arm["false_alert_segment_count"], 3)
        self.assertEqual(arm["alerts_on_unknown_truth"], 2)
        self.assertTrue(arm["events"][1]["preceding_truth_unknown"])
        self.assertAlmostEqual(arm["events"][0]["first_alert_relative_to_entry_s"], .2)
        self.assertAlmostEqual(arm["false_alert_sampled_duration_s"], .8)

    def test_sensor_unknown_silent_negative_is_abstention_not_tn(self):
        result = audit.recompute_metrics(rows([False, True], [False, False]))
        count = result["arms"]["classifier"]["frames"]["all_known"]
        self.assertEqual((count["TN"], count["FN"], count["abstained_negative"], count["abstained_positive"]), (0, 1, 1, 1))
        self.assertEqual(count["false_alert_rate_known_negative"], 0.)

    def test_float64_no_alert_cutoff_and_nonrecursive_hold(self):
        ids = identities(3)
        ids[0]["baseline"]["alert"] = True
        p = np.full((3, 6), .5, np.float32)
        pred = {arm: dict(probability=p) for arm in audit.LEARNED}
        select = {arm: dict(selection=dict(threshold=float(np.nextafter(.5, np.inf)))) for arm in audit.LEARNED}
        sample = audit.rebuild_rows(ids, labels([True, True, True]), pred, select)
        self.assertEqual([r["predictions"]["A_hold"]["alert"] for r in sample], [True, True, False])
        self.assertFalse(any(r["predictions"]["occupancy"]["alert"] for r in sample))

    def test_distribution_integrity_and_index_corruption(self):
        distribution = np.full((1, 6, 7), 1. / 7, np.float32)
        probability = distribution[..., :6].sum(-1)
        predictions = {arm: dict(indices=np.array([4]), probability=probability.copy()) for arm in audit.LEARNED}
        predictions["occupancy"].update(distribution=distribution, mask=np.zeros((1, 6, 45, 80), np.float32))
        self.assertTrue(audit.prediction_checks(predictions, np.array([4]))["monotone_cdf"])
        corrupt = copy.deepcopy(predictions)
        corrupt["occupancy"]["probability"][0, 0] = .1
        with self.assertRaisesRegex(audit.AuditFailure, "CDF and occupancy"):
            audit.prediction_checks(corrupt, np.array([4]))
        corrupt = copy.deepcopy(predictions)
        corrupt["classifier"]["indices"][0] = 3
        with self.assertRaisesRegex(audit.AuditFailure, "index/order"):
            audit.prediction_checks(corrupt, np.array([4]))

    def test_comparison_rejects_missing_event_and_changed_unknown(self):
        result = audit.recompute_metrics(rows([True, False], [True, False]))
        corrupt = copy.deepcopy(result)
        corrupt["arms"]["classifier"]["events"] = []
        with self.assertRaises(audit.AuditFailure):
            audit.same_tree(corrupt, result)
        corrupt = copy.deepcopy(result)
        corrupt["arms"]["classifier"]["prediction_unknown_frames"] = 0
        with self.assertRaises(audit.AuditFailure):
            audit.same_tree(corrupt, result)


if __name__ == "__main__":
    unittest.main()
