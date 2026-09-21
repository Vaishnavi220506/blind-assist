"""Diagnostic display contracts; synthetic files, never opens serial ports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import ARTIFACTS, Dashboard, RunIndex, cell_view


def tof(stamp, distance=200):
    return {"host_received_monotonic_ns": stamp,
            "sensor": {"seq": 1, "distance_mm": [distance]*16,
                       "target_status": [5]*16, "nb_target": [1]*16},
            "derived": {"hist_normalized": [[1, 2, 3]]*16}}


class DashboardTests(unittest.TestCase):
    def setUp(self):
        root = ARTIFACTS / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root, prefix="dashboard-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, records):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")

    def test_quality_preserves_raw_and_histogram(self):
        for raw, expected in [(0, "UNKNOWN"), (2, "SUSPECT"), (200, "KNOWN")]:
            cell = cell_view(tof(1, raw))[0]
            self.assertEqual((cell["quality"], cell["raw_mm"]), (expected, raw))
            self.assertEqual(cell["hist"], [1, 2, 3])
        frame = tof(1)
        frame["sensor"]["target_status"][0] = 9
        self.assertEqual(cell_view(frame)[0]["quality"], "UNKNOWN")

    def test_conversion_mismatch_and_negative_q2_are_visible(self):
        frame = tof(1)
        frame["derived"]["diagnostic"] = {"distance_conversion_match": [False]*16}
        self.assertEqual(cell_view(frame)[0]["quality"], "SUSPECT")
        frame = tof(1, 0)
        frame["sensor"]["diagnostic"] = {"distance_q2": [-3]*16}
        cell = cell_view(frame)[0]
        self.assertEqual(cell["q2"], -3)
        self.assertTrue(any("负" in r for r in cell["reasons"]))

    def test_partial_lines_wait_for_newline_without_duplication(self):
        path = self.root / "tof/frames.jsonl"
        path.parent.mkdir()
        encoded = json.dumps(tof(100)).encode()
        path.write_bytes(encoded[:25])
        index = RunIndex(self.root)
        index.refresh()
        self.assertEqual(index.rows["tof"], [])
        with path.open("ab") as stream:
            stream.write(encoded[25:]+b"\n")
        index.refresh()
        index.refresh()
        self.assertEqual(len(index.rows["tof"]), 1)

    def test_replay_uses_no_future_frames_and_independent_ages(self):
        self.write("run/tof/frames.jsonl", [tof(1_000_000_000), tof(4_000_000_000)])
        camera = self.root / "run/camera"
        camera.mkdir()
        (camera / "frame.jpg").write_bytes(b"fixture")
        self.write("run/camera/frames.jsonl", [{"host_received_monotonic_ns": 2_000_000_000,
                   "jpeg_validated": True, "filename": "frame.jpg", "header": {"seq": 4}}])
        app = Dashboard(self.root)
        app.replay("run")
        self.assertIsNone(app.state(0)["camera"])
        state = app.state(1200)
        self.assertEqual(state["camera"]["age_ms"], 200)
        self.assertEqual(state["tof"]["age_ms"], 1200)
        self.assertTrue(app.state(3000)["camera"]["stale"])
        self.assertFalse(app.state(3000)["tof"]["stale"])

    def test_unsafe_images_not_served(self):
        (self.root / "outside.jpg").write_bytes(b"fixture")
        self.write("camera/frames.jsonl", [{"host_received_monotonic_ns": 1,
                   "jpeg_validated": True, "filename": "../outside.jpg"}])
        index = RunIndex(self.root)
        index.refresh()
        self.assertEqual(index.rows["camera"], [])
        with self.assertRaises(ValueError):
            index.image_path("../outside.jpg")

    def test_bad_requests_never_start_collectors(self):
        app = Dashboard(self.root)
        with patch("dashboard.ports", return_value=[{"port": p, "vid": 12346} for p in ["COM5", "COM11"]]), patch("dashboard.subprocess.Popen") as launch:
            for payload in [{"camera_port": "COM5", "tof_port": "COM5"},
                            {"camera_port": "COM11", "tof_port": "COM5", "seconds": "nan"}]:
                with self.assertRaises(ValueError):
                    app.start(payload)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
