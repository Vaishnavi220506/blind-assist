"""Source-only query-occupancy checks; no UE, image or native-depth access."""
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from data_coverage_spec import specification as old_coverage
from query_occupancy_spec import (
    specification, check_spec, bounds, classify, FRONT_TRAJECTORY, FRAMES_PER_CLIP,
    RANGES, FAMILIES, SPLIT_GROUPS, MATERIALS,
)
from launch_query_occupancy import validate_cases, validate_capture_output


class QueryOccupancySourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specification()

    def test_fixed_group_split_old_geometry_disjoint_and_deterministic(self):
        report = check_spec(self.spec, [old_coverage()])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["compared_old_cohorts"], 1)
        self.assertEqual(Counter(c["split"] for c in self.spec["cases"]),
                         dict(train=864, dev=288, evaluation=576))
        self.assertEqual(self.spec, specification())
        self.assertEqual(check_spec(json.loads(json.dumps(self.spec)))["status"], "PASS")
        validate_cases(self.spec)

    def test_wide_distance_trajectory_and_full_extent_labels(self):
        groups = defaultdict(list)
        for c in self.spec["cases"]:
            groups[c["base_group_id"]].append(c)
            frame = c["frame_in_clip"]
            front = float(bounds(c)[0][2])
            self.assertAlmostEqual(front, FRONT_TRAJECTORY[frame] + c["trajectory_jitter_m"], places=6)
            self.assertGreaterEqual(front, .4)
            label = classify(*bounds(c))
            self.assertEqual(label["truth"], c["layout_relation"] != "OUTSIDE" and 2 <= frame <= 9)
            if frame in (0, 1, 10, 11):
                self.assertGreater(front, 3.)
            if frame == 7:
                self.assertTrue(.47 - 1e-6 <= front <= .53 + 1e-6)
        for rows in groups.values():
            self.assertEqual(len(rows), 3 * FRAMES_PER_CLIP)
            self.assertEqual(len({c["split"] for c in rows}), 1)
            self.assertEqual(len({c["trajectory_jitter_m"] for c in rows}), 1)

    def test_stratified_geometry_and_training_appearance(self):
        rows = defaultdict(list)
        for c in self.spec["cases"]:
            if c["layout_relation"] == "INSIDE" and c["frame_in_clip"] == 0:
                rows[c["type_id"], c["split"]].append(c)
        for family, *_ in FAMILIES:
            for split, count in SPLIT_GROUPS.items():
                for axis, (lo, hi) in enumerate(RANGES[family]):
                    values = [(c["objects"][0]["size_m"] + [c["objects"][0]["center_m"][2]])[axis]
                              for c in rows[family, split]]
                    self.assertEqual({int((v-lo)/(hi-lo)*count) for v in values}, set(range(count)))
            self.assertEqual({c["objects"][0]["material"].rsplit("/", 1)[-1]
                              for c in rows[family, "train"]}, set(MATERIALS))

    def test_reject_mutated_pair_partition_pose_time_and_geometry(self):
        for field in ("noise", "material", "shape", "split", "time", "pose", "front", "order"):
            changed = copy.deepcopy(self.spec)
            case = changed["cases"][12]
            if field == "noise":
                case["sensor_noise_key"] += "_different"
            elif field == "material":
                case["objects"][0]["material"] += "_different"
            elif field == "shape":
                case["objects"][0]["size_m"][0] += .001
            elif field == "split":
                case["split"] = "dev" if case["split"] != "dev" else "train"
            elif field == "time":
                case["time_s"] += .1
            elif field == "pose":
                case["camera"]["yaw"] = 1.
            elif field == "front":
                for c in changed["cases"][:36]:
                    c["camera"]["x"] += .1
            else:
                changed["cases"][0], changed["cases"][12] = changed["cases"][12], changed["cases"][0]
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_spec(changed)
        with self.assertRaises(AssertionError):
            check_spec(self.spec, [self.spec])

    def test_capture_retains_visibility_and_uses_twelve_frame_clips(self):
        # Capture cannot import outside Unreal. Inspect its AST rather than run it.
        import ast
        capture = Path(__file__).with_name("query_occupancy_capture.py")
        tree = ast.parse(capture.read_text(encoding="utf-8"))
        assertions = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        self.assertFalse(any("same_actor" in value for value in assertions))
        self.assertTrue(any("len(frames) == 1728" in value for value in assertions))
        self.assertTrue(any("range(12)" in value for value in assertions))

    def test_governed_empty_output_requires_running_matching_owner(self):
        # Mock only filesystem boundaries; no test files or native data are used.
        root = (Path(__file__).resolve().parents[4] / "artifacts.local").resolve()
        out = root / "evidence" / "query-occupancy-guard-fixture"
        journal_path = root / "evidence" / "query-occupancy-guard-journal.json"
        valid = dict(schema="blindassist-asset-run-journal-v1", state="running", id="guard-fixture",
                     command=["python", "launch_query_occupancy.py", "--output", str(out)],
                     outputs=[dict(alias="result", path=(out / "launcher-terminal.json").relative_to(root).as_posix())])
        with patch.object(Path, "exists", return_value=False):
            self.assertEqual(validate_capture_output(out, root)["mode"], "NEW_DIRECTORY")
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "iterdir", side_effect=lambda: iter(())), \
             patch.dict(os.environ, {"BLINDASSIST_ASSET_RUN_JOURNAL": str(journal_path)}):
            with patch.object(Path, "read_text", return_value=json.dumps(valid)):
                self.assertEqual(validate_capture_output(out, root)["mode"], "GOVERNED_EMPTY_DIRECTORY")
                with patch.object(Path, "iterdir", side_effect=lambda: iter([out / "old-receipt.json"])), \
                     self.assertRaises(AssertionError):
                    validate_capture_output(out, root)
            for defect in ("finished", "output_parent", "command_output", "schema"):
                changed = copy.deepcopy(valid)
                if defect == "finished":
                    changed["state"] = "failed"
                elif defect == "output_parent":
                    changed["outputs"][0]["path"] = "evidence/other/launcher-terminal.json"
                elif defect == "command_output":
                    changed["command"][-1] = str(root / "evidence" / "other")
                else:
                    changed["schema"] = "unrelated-schema"
                with self.subTest(defect=defect), patch.object(Path, "read_text", return_value=json.dumps(changed)), \
                     self.assertRaises(AssertionError):
                    validate_capture_output(out, root)
            with patch.dict(os.environ, {"BLINDASSIST_ASSET_RUN_JOURNAL": ""}), self.assertRaises(AssertionError):
                validate_capture_output(out, root)


if __name__ == "__main__":
    unittest.main()
