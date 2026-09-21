"""Synthetic serial/JPEG protocol fixtures only; no real ports are opened."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from atom_capture import ProtocolError, ReadTimeout, Wire, acquire, configure_serial_lines, receive, validate_jpeg


def artifact_temp():
    root = Path(__file__).resolve().parents[4]/"artifacts.local"/"hardware-bringup"/"test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="atom-protocol-", dir=root)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def stamp(self):
        return int(self.value*1e9)


class SerialFixture:
    def __init__(self, data, clock, chunk=7):
        self.data = bytearray(data)
        self.clock, self.chunk = clock, chunk
        self.timeout = 0.2
        self.opened = False
        self.closed = False
        self.written = []
        self.read_calls = 0

    @property
    def in_waiting(self):
        return len(self.data)

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def write(self, data):
        self.written.append(data)
        return len(data)

    def read(self, count):
        self.read_calls += 1
        self.clock.value += 0.001 if self.data else self.timeout
        size = min(count, self.chunk, len(self.data))
        result = bytes(self.data[:size])
        del self.data[:size]
        return result


def jpeg_bytes(size=(640, 480)):
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", size, (30, 60, 90)).save(out, format="JPEG")
    return out.getvalue()


def response(jpeg=None, seq=1, **changes):
    jpeg = jpeg_bytes() if jpeg is None else jpeg
    header = {"type": "jpeg", "seq": seq, "device_readout_us": 123456,
              "width": 640, "height": 480, "bytes": len(jpeg), **changes}
    return json.dumps(header).encode()+b"\n"+jpeg+b"\n", header, jpeg


class ProtocolTests(unittest.TestCase):
    def test_multikilobyte_burst_bulk_read_preserves_payload_and_hash(self):
        clock = Clock()
        packet, expected, jpeg = response()
        raw = b'{"type":"boot"}\n'+packet
        self.assertGreater(len(raw), 4096)
        for chunk_size in (65536, 512):
            with self.subTest(chunk_size=chunk_size):
                device = SerialFixture(raw, clock, chunk=chunk_size)
                saved = io.BytesIO()
                wire = Wire(device, saved, clock)
                header, image, _, _ = receive(wire, 10, lambda _: None, clock.stamp)
                self.assertEqual(expected, header)
                self.assertEqual(jpeg, image)
                self.assertEqual(raw, saved.getvalue())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), wire.digest.hexdigest())
                self.assertEqual(b"", wire.pending)
                self.assertLessEqual(device.read_calls, (len(raw)+chunk_size-1)//chunk_size)

    def test_fragmented_header_payload_and_boot_logs_retained(self):
        clock = Clock()
        payload, expected, image = response()
        raw = b"ROM log\r\n"+b'{"type":"boot"}\n'+payload
        device = SerialFixture(raw, clock, chunk=11)
        saved, events = io.BytesIO(), []
        header, decoded, start, end = receive(Wire(device, saved, clock), 10, events.append, clock.stamp)
        self.assertEqual(expected, header)
        self.assertEqual(image, decoded)
        self.assertEqual(raw, saved.getvalue())
        self.assertEqual(2, len(events))
        self.assertLess(start, end)
        validate_jpeg(decoded, header)

    def test_declared_payload_cannot_exceed_bound(self):
        clock = Clock()
        raw, _, _ = response(bytes=5*1024*1024)
        with self.assertRaisesRegex(ProtocolError, "bytes"):
            receive(Wire(SerialFixture(raw, clock), io.BytesIO(), clock), 10, lambda _: None)

    def test_device_error_is_failure_with_evidence(self):
        clock = Clock()
        raw = b'{"type":"error","message":"camera_failed"}\n'
        saved, events = io.BytesIO(), []
        with self.assertRaisesRegex(ProtocolError, "camera_failed"):
            receive(Wire(SerialFixture(raw, clock), saved, clock), 10, events.append)
        self.assertEqual(raw, saved.getvalue())
        self.assertEqual(1, len(events))

    def test_partial_jpeg_times_out_and_keeps_all_bytes(self):
        clock = Clock()
        raw, _, _ = response()
        partial = raw[:-100]
        saved = io.BytesIO()
        with self.assertRaises(ReadTimeout):
            receive(Wire(SerialFixture(partial, clock, chunk=4096), saved, clock), 1, lambda _: None)
        self.assertEqual(partial, saved.getvalue())
        self.assertLessEqual(clock.value, 1.00001)

    def test_missing_newline_and_wrong_jpeg_dimensions(self):
        raw, header, image = response(jpeg_bytes((320, 240)))
        with self.assertRaisesRegex(ProtocolError, "dimensions"):
            validate_jpeg(image, header)
        clock = Clock()
        with self.assertRaisesRegex(ProtocolError, "newline"):
            receive(Wire(SerialFixture(raw[:-1]+b"X", clock, chunk=4096), io.BytesIO(), clock), 10, lambda _: None)


class CaptureTests(unittest.TestCase):
    def test_explicit_usb_otg_line_configuration_and_manifest(self):
        for enabled in (False, True):
            with self.subTest(usb_otg=enabled), artifact_temp() as temp:
                clock = Clock()
                device = SerialFixture(response()[0], clock, 65536)
                configure_serial_lines(device, enabled)
                self.assertIs(device.dtr, enabled)
                self.assertFalse(device.rts)
                self.assertFalse(device.opened)
                out = Path(temp)/"capture"
                acquire(device, out, 1, 1, port="SYNTHETIC_ONLY", clock=clock,
                        stamp=clock.stamp, usb_otg=enabled)
                manifest = json.loads((out/"manifest.json").read_text())
                self.assertEqual("USB_OTG_TINYUSB_CDC" if enabled else "HARDWARE_CDC", manifest["usb_mode"])
                self.assertIs(manifest["dtr_rts"]["dtr_before_open"], enabled)
                self.assertFalse(manifest["dtr_rts"]["rts_before_open"])

    def run_fixture(self, raw, root, seconds=1, timeout=1, chunk=65536):
        clock = Clock()
        device = SerialFixture(raw, clock, chunk)
        result = acquire(device, root, seconds, timeout, port="SYNTHETIC_ONLY", clock=clock, stamp=clock.stamp)
        return result, device

    def test_success_records_hash_jpeg_timing_and_closes(self):
        raw, header, image = response()
        with artifact_temp() as temp:
            out = Path(temp)/"capture"
            result, device = self.run_fixture(raw, out)
            self.assertEqual("FRAMES_RECORDED", result["result"])
            self.assertEqual([b"CAPTURE\n"], device.written)
            self.assertTrue(device.closed)
            self.assertEqual(raw, (out/"serial.bin").read_bytes())
            record = json.loads((out/"frames.jsonl").read_text())
            self.assertEqual(header, record["header"])
            self.assertTrue(record["jpeg_validated"])
            self.assertLessEqual(record["host_request_monotonic_ns"], record["host_received_monotonic_ns"])
            self.assertEqual(image, (out/record["filename"]).read_bytes())

    def test_timeout_summary_closes_and_does_not_claim_success(self):
        with artifact_temp() as temp:
            out = Path(temp)/"capture"
            result, device = self.run_fixture(b"", out)
            self.assertEqual("NO_VALID_FRAMES", result["result"])
            self.assertEqual(1, result["failure_count"])
            self.assertTrue(device.closed)
            self.assertIn("ReadTimeout", (out/"events.jsonl").read_text())
            self.assertTrue((out/"summary.json").exists())

    def test_corrupted_complete_image_kept_and_reported(self):
        raw, _, image = response(b"\xff\xd8invalid\xff\xd9")
        with artifact_temp() as temp:
            out = Path(temp)/"capture"
            result, device = self.run_fixture(raw, out)
            record = json.loads((out/"frames.jsonl").read_text())
            self.assertFalse(record["jpeg_validated"])
            self.assertEqual(image, (out/record["filename"]).read_bytes())
            self.assertEqual(1, result["failure_count"])
            self.assertTrue(device.closed)

    def test_sequence_gap_retained_and_reported(self):
        # Two fragmented transactions consume enough time to reach the scheduling cutoff.
        raw = response(seq=1)[0]+response(seq=4)[0]
        with artifact_temp() as temp:
            result, device = self.run_fixture(raw, Path(temp)/"capture", seconds=1.3, timeout=1)
            self.assertEqual(2, result["frames"])
            self.assertEqual(2, result["missing_sequences"])
            self.assertEqual("FRAMES_WITH_FAILURES", result["result"])
            self.assertTrue(device.closed)

    def test_existing_evidence_is_untouched_and_device_not_opened(self):
        clock = Clock()
        device = SerialFixture(b"", clock)
        with artifact_temp() as temp:
            with self.assertRaises(FileExistsError):
                acquire(device, Path(temp), 1, 1, port="SYNTHETIC_ONLY", clock=clock)
        self.assertFalse(device.opened)
        self.assertTrue(device.closed)

    def test_short_session_still_requests_first_frame_after_open_overhead(self):
        clock = Clock()
        device = SerialFixture(response()[0], clock, 65536)
        def delayed_open():
            device.opened = True
            clock.value += 0.05
        device.open = delayed_open
        with artifact_temp() as temp:
            result = acquire(device, Path(temp)/"capture", 0.5, 1,
                             port="SYNTHETIC_ONLY", clock=clock, stamp=clock.stamp)
        self.assertEqual("FRAMES_RECORDED", result["result"])
        self.assertEqual([b"CAPTURE\n"], device.written)
        self.assertLessEqual(result["elapsed_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
