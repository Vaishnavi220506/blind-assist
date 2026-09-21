import unittest
from unittest.mock import patch

from manual_orientation import plan, mark, status, CNH_GUIDE


class ManualOrientationTests(unittest.TestCase):
    def test_cnh_four_explicit_segments_and_deadline(self):
        schedule = plan(0, CNH_GUIDE)
        self.assertEqual(schedule["schema"], "hardware-bringup.cnh-components.v1")
        self.assertEqual(schedule["session_limit_seconds"], 240)
        for index, phase in enumerate(("background", "foreground", "mixture", "return")):
            now = (20 + index * 40) * 1_000_000_000
            before = status(schedule, now, True)
            self.assertEqual(before["phase"], phase)
            self.assertEqual(before["total"], 4)
            self.assertTrue(before["can_mark"])
            segment = mark(schedule, now)
            self.assertEqual(segment["phase"], phase)
            self.assertEqual(segment["end_host_monotonic_ns"] - now, 6_000_000_000)
            with self.assertRaises(ValueError):
                mark(schedule, now + 1)
        self.assertEqual(status(schedule, 146_000_000_000, True)["phase"], "done")
        self.assertFalse(status(schedule, 146_000_000_000, True)["can_mark"])
        self.assertFalse(status(plan(0, CNH_GUIDE), 235_000_000_000, True)["can_mark"])
        self.assertTrue(status(plan(0, CNH_GUIDE), 190_000_000_000, True)["can_mark"])

    def test_cnh_timer_starts_only_on_fourth_mark(self):
        import tempfile
        import threading
        from pathlib import Path
        from dashboard import Dashboard
        app = object.__new__(Dashboard)
        app.lock = threading.RLock()
        app.manual, app.run_id = plan(0, CNH_GUIDE), "test-cnh"
        with tempfile.TemporaryDirectory() as directory:
            app.controls = Path(directory)
            with patch.object(app, "active", return_value=True), patch.object(app, "state", return_value={"camera": {"age_ms": 1}, "tof": {"age_ms": 1}}), patch("dashboard.time.monotonic_ns", side_effect=[10_000_000_000, 20_000_000_000, 30_000_000_000, 40_000_000_000]), patch("dashboard.threading.Timer") as timer:
                for _ in range(3):
                    app.mark_manual()
                timer.assert_not_called()
                app.mark_manual()
                timer.assert_called_once_with(6, app.finish_manual, args=("test-cnh",))
                timer.return_value.start.assert_called_once()

    def test_waits_for_each_explicit_click(self):
        schedule = plan(0)
        self.assertEqual(status(schedule, 40_000_000_000, True)["phase"], "baseline")
        mark(schedule, 40_000_000_000)
        self.assertFalse(status(schedule, 45_000_000_000, True)["can_mark"])
        self.assertEqual(status(schedule, 46_000_000_000, True)["phase"], "up")
        self.assertEqual(status(schedule, 80_000_000_000, True)["phase"], "up")
        mark(schedule, 80_000_000_000)
        mark(schedule, 95_000_000_000)
        self.assertEqual(status(schedule, 102_000_000_000, True)["phase"], "done")
        with self.assertRaises(ValueError):
            mark(schedule, 103_000_000_000)

    def test_no_overlapping_windows(self):
        schedule = plan(0)
        mark(schedule, 10_000_000_000)
        with self.assertRaises(ValueError):
            mark(schedule, 11_000_000_000)
        self.assertEqual(len(schedule["segments"]), 1)

    def test_stop_and_deadline_do_not_request_more_actions(self):
        schedule = plan(0)
        self.assertFalse(status(schedule, 1, False)["can_mark"])
        self.assertFalse(status(schedule, 175_000_000_000, True)["can_mark"])
        with self.assertRaises(ValueError):
            mark(schedule, 175_000_000_000)

    def test_backend_rejects_stale_before_saving_marker(self):
        from dashboard import Dashboard
        app = object.__new__(Dashboard)
        app.lock = __import__("threading").RLock()
        app.manual = plan(0)
        with patch.object(app, "active", return_value=True), patch.object(app, "state", return_value={"camera": {"age_ms": 1}, "tof": {"age_ms": 501}}):
            with self.assertRaisesRegex(ValueError, "过期"):
                app.mark_manual()
        self.assertEqual(app.manual["segments"], [])

    def test_old_timer_cannot_stop_new_run(self):
        from dashboard import Dashboard
        app = object.__new__(Dashboard)
        app.lock = __import__("threading").RLock()
        app.run_id, app.manual = "new", plan(0)
        with patch.object(app, "stop") as stop:
            app.finish_manual("old")
            stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
