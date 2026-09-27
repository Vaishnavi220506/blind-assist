"""Synthetic UI orchestration checks; never opens a device."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from capture import validate_frame
from simple_capture import Guide, Handler, SEGMENTS
from test_capture import frame


PORTS = [{"port": "TOF", "vid": 0x303A, "description": "synthetic", "serial_number": "98:A3:16:F7:88:1C"},
         {"port": "CAM", "vid": 0x303A, "description": "synthetic", "serial_number": "B43A45BD12D8"}]


def fake_process(command, **kwargs):
    root = Path(command[command.index("--output")+1])
    root.mkdir()
    stamp = time.monotonic_ns()
    tof = frame(zones=64, cnh=True)
    for name in ("camera", "tof"):
        (root/name).mkdir()
    (root/"camera/fixture.jpg").write_bytes(b"synthetic-not-served-as-real-camera")
    (root/"camera/frames.jsonl").write_text(json.dumps({"host_received_monotonic_ns": stamp,
                  "filename": "fixture.jpg", "jpeg_validated": True})+'\n')
    (root/"tof/frames.jsonl").write_text(json.dumps({"sensor": tof, "derived": validate_frame(tof),
                  "host_received_monotonic_ns": stamp})+'\n')
    (root/"summary.json").write_text(json.dumps({"result": "BOTH_COLLECTORS_PASSED", "streams": {
        "camera": {"frames": 1, "failure_count": 0},
        "tof": {"frames": 1, "issues": {}, "acquisition_error": None}}}))
    class Process:
        def poll(self): return 0
    return Process()


class GuideTests(unittest.TestCase):
    def test_four_steps_wait_for_operator_and_keep_both_streams(self):
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, fake_process, lambda: PORTS)
            self.assertEqual("idle", guide.state()["phase"])
            self.assertIsNone(guide.output)
            self.assertEqual(150, sum(s["seconds"] for s in SEGMENTS))
            for i in range(4):
                guide.start({"port": "TOF", "camera_port": "CAM", "object": "book"})
                guide.worker.join(3)
                state = guide.state()
                self.assertEqual(i+1, state["step"])
                self.assertEqual("complete" if i == 3 else "ready", state["phase"])
                self.assertEqual(1, state["camera_frames"])
                self.assertEqual(1, state["frames"])
                self.assertEqual(b"synthetic-not-served-as-real-camera", guide.image(str(i+1), "fixture.jpg"))
                with self.assertRaises(ValueError): guide.image(str(i+1), "../session.json")
            guide.close()
            result = json.loads((guide.output/"guide-summary.json").read_text())
            self.assertEqual(4, result["completed_steps"])
            self.assertFalse(result["ended_early"])
            self.assertEqual(4, len(result["attempts"]))

    def test_same_port_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, fake_process, lambda: PORTS)
            with self.assertRaises(ValueError): guide.start({"port": "TOF", "camera_port": "TOF"})
            self.assertIsNone(guide.output)

    def test_early_stop_does_not_claim_four_completed_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, fake_process, lambda: PORTS)
            guide.start({"port": "TOF", "camera_port": "CAM"})
            guide.worker.join(3)
            guide.stop()
            self.assertEqual(1, guide.state()["step"])
            self.assertTrue(json.loads((guide.output/"guide-summary.json").read_text())["ended_early"])

    def test_stop_during_capture_signals_collector_and_waits_for_receipt(self):
        def running(command, **kwargs):
            fake_process(command, **kwargs)
            stop_file = Path(command[command.index("--stop-file")+1])
            class Process:
                def poll(self): return 0 if stop_file.exists() else None
            return Process()
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, running, lambda: PORTS)
            guide.start({"port": "TOF", "camera_port": "CAM"})
            guide.stop()
            guide.worker.join(3)
            self.assertFalse(guide.worker.is_alive())
            self.assertEqual("complete", guide.phase)
            self.assertEqual(0, guide.step)
            self.assertEqual(1, len(guide.results))

    def test_no_summary_is_failure_and_retry_has_new_directory(self):
        class NoOutput:
            def poll(self): return 1
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, lambda *a, **k: NoOutput(), lambda: PORTS)
            guide.start({"port": "TOF", "camera_port": "CAM"})
            guide.worker.join(3)
            self.assertEqual("error", guide.state()["phase"])
            self.assertEqual(0, guide.step)
            guide.process_factory = fake_process
            guide.start({"port": "TOF", "camera_port": "CAM"})
            guide.worker.join(3)
            self.assertEqual(1, guide.step)
            self.assertEqual(2, guide.attempt)

    def test_http_state_does_not_open_device_and_rejects_cross_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            guide = Guide(temp, lambda *a, **k: self.fail("must not start"), lambda: PORTS)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.guide = guide
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url+"/api/state") as response:
                    state = json.load(response)
                self.assertEqual("idle", state["phase"])
                self.assertEqual("TOF", state["suggested_tof"])
                req = Request(url+"/api/start", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.com"})
                with self.assertRaises(HTTPError) as error: urlopen(req)
                self.assertEqual(400, error.exception.code)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(2)


if __name__ == "__main__":
    unittest.main()
