import unittest
from unittest.mock import patch

from manual_orientation import plan, mark, status


class ManualOrientationTests(unittest.TestCase):
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
