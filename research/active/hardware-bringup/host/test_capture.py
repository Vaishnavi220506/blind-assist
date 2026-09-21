"""Synthetic protocol regression fixtures only, never device evidence."""
import copy
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from capture import Decoder, bounded_seconds, capture_arguments, firmware_metadata, live_chunks, run_session, select_port, validate_frame


def frame(seq=1, ms=100, zones=16, cnh=False):
    side = math.isqrt(zones)
    item = {"type": "cnh_frame" if cnh else "frame", "seq": seq, "ms": ms,
            "rows": side, "cols": side, "distance_mm": [600]*zones,
            "target_status": [5]*zones, "nb_target": [1]*zones}
    if cnh:
        item.update(bins=24, hist_raw=[[16]*24 for _ in range(16)],
                    hist_scaler=[[0]*24 for _ in range(16)],
                    ambient_raw=[8]*16, ambient_scaler=[1]*16)
    return item


def encoded(*items):
    return b"".join(json.dumps(item).encode() + b"\n" for item in items)


def diagnostic_frame(zones=16):
    item = frame(zones=zones)
    item["diagnostic"] = {
        "schema": 1, "distance_q2": [2400]*zones,
        "signal_kcps_spad": [1200]*zones, "ambient_kcps_spad": [40]*zones,
        "range_sigma_mm": [8]*zones, "reflectance_percent": [30]*zones,
        "nb_spads_enabled": [25000]*zones, "silicon_temp_degc": 25,
    }
    return item


class ParserTests(unittest.TestCase):
    def test_diagnostic_signed_conversion_and_shape(self):
        for zones in (16, 64):
            item = diagnostic_frame(zones)
            item["diagnostic"]["distance_q2"][:9] = [-32768, -7, -4, -1, 0, 1, 7, 2403, 32767]
            item["distance_mm"][:9] = [0, 0, 0, 0, 0, 0, 1, 600, 8191]
            original = copy.deepcopy(item)
            result = validate_frame(item)
            self.assertEqual(item["distance_mm"], result["diagnostic"]["uld_expected_distance_mm"])
            self.assertEqual([True]*zones, result["diagnostic"]["distance_conversion_match"])
            self.assertEqual(0, result["diagnostic"]["distance_conversion_mismatch_count"])
            self.assertEqual([None]*6, result["distance_known_mm"][:6])
            self.assertEqual(original, item)

    def test_diagnostic_mismatch_is_retained_without_changing_validity(self):
        records = []
        decoder = Decoder(lambda kind, record: records.append((kind, record)))
        item = diagnostic_frame()
        item["distance_mm"][0] = 2
        decoder.feed(encoded(item), 123)
        self.assertEqual(1, decoder.frames)
        self.assertFalse(decoder.issues)
        kept = next(record for kind, record in records if kind == "frames")
        self.assertEqual(item, kept["sensor"])
        self.assertTrue(kept["derived"]["range_valid"][0])
        self.assertFalse(kept["derived"]["diagnostic"]["distance_conversion_match"][0])
        self.assertEqual(1, decoder.summary()["distance_conversion_mismatch_cells"])
        self.assertEqual(1, decoder.summary()["diagnostic_frames"])

    def test_diagnostic_optional_for_legacy_frames(self):
        for cnh in (False, True):
            self.assertNotIn("diagnostic", validate_frame(frame(cnh=cnh)))
        self.assertEqual(0, Decoder().summary()["diagnostic_frames"])

    def test_malformed_diagnostic_rejected(self):
        cases = [("schema", True), ("schema", 2), ("distance_q2", [0]*15),
                 ("distance_q2", [32768]*16), ("signal_kcps_spad", [-1]*16),
                 ("ambient_kcps_spad", [2**32]*16), ("range_sigma_mm", [65536]*16),
                 ("reflectance_percent", [256]*16), ("nb_spads_enabled", [1.0]*16),
                 ("silicon_temp_degc", -129)]
        for key, value in cases:
            item = diagnostic_frame()
            item["diagnostic"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frame(item)
        for value in (None, [], 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_frame({**frame(), "diagnostic": value})
        item = diagnostic_frame()
        del item["diagnostic"]["signal_kcps_spad"]
        decoder = Decoder()
        decoder.feed(encoded(item))
        self.assertEqual(0, decoder.frames)
        self.assertEqual(1, decoder.issues["invalid_frame"])

    def test_config_readback_events_preserved_and_error_counted(self):
        records = []
        decoder = Decoder(lambda kind, record: records.append((kind, record)))
        decoder.feed(encoded({"type": "config", "status": 0}, {"type": "config", "status": 1}))
        self.assertEqual(2, decoder.events)
        self.assertEqual(1, decoder.errors)
        self.assertEqual(["events", "events"], [kind for kind, _ in records])

    def test_unknown_is_not_zero_or_negative(self):
        item = frame()
        item["distance_mm"][:3] = [0, -1, 150]
        item["target_status"][2] = 9
        item["nb_target"][3] = 0
        before = copy.deepcopy(item)
        data = validate_frame(item)
        self.assertEqual([None]*4, data["distance_known_mm"][:4])
        self.assertEqual(600, data["distance_known_mm"][4])
        self.assertEqual(item, before)

    def test_both_shapes_and_legacy_no_dimensions(self):
        for zones in (16, 64):
            item = frame(zones=zones)
            del item["rows"], item["cols"]
            self.assertEqual(zones, len(validate_frame(item)["range_valid"]))

    def test_reject_mismatch_and_noninteger(self):
        for key, value in (("rows", 8), ("target_status", [5]), ("seq", True), ("distance_mm", [float("nan")]*16)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frame({**frame(), key: value})

    def test_cnh_signed_scalers_preserved(self):
        item = frame(cnh=True)
        item["hist_scaler"][0][:5] = [-128, -1, 0, 1, 127]
        original = copy.deepcopy(item)
        data = validate_frame(item)
        self.assertEqual([math.ldexp(16.0, 128), 32.0, 16.0, 8.0, math.ldexp(16.0, -127)], data["hist_normalized"][0][:5])
        self.assertEqual([4.0]*16, data["ambient_normalized"])
        self.assertEqual(original, item)
        self.assertEqual("UNVERIFIED_ON_HARDWARE", data["cnh_hardware_status"])

    def test_cnh_bad_dimensions_and_scalers(self):
        for key, value in (("hist_raw", [[0]*23 for _ in range(16)]), ("ambient_raw", [0]*15), ("bins", 23), ("hist_scaler", [[128]*24 for _ in range(16)])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frame({**frame(cnh=True), key: value})

    def test_cnh_summary_counts_zero_and_unavailable(self):
        decoder = Decoder()
        self.assertIsNone(decoder.summary()["cnh_normalized_finite"])
        item = frame(cnh=True)
        item["hist_raw"][0] = [0]*24
        decoder.feed(encoded(item))
        summary = decoder.summary()
        self.assertEqual(1, summary["cnh_frames"])
        self.assertEqual(16, summary["cnh_histograms"])
        self.assertEqual(1, summary["cnh_all_zero_histograms"])
        self.assertIs(True, summary["cnh_normalized_finite"])

    def test_chunk_boundaries_corruption_and_partial_tail(self):
        records = []
        decoder = Decoder(lambda kind, record: records.append((kind, record)))
        raw = b"\xff\n[]\n" + encoded(frame()) + b'{"type":'
        for chunk in (raw[:13], raw[13:70], raw[70:]):
            decoder.feed(chunk, 123)
        decoder.feed(b"", 124, final=True)
        self.assertEqual(1, decoder.frames)
        self.assertEqual({"malformed_line": 2, "non_object": 1}, dict(decoder.issues))
        self.assertEqual(123, next(r for k, r in records if k == "frames")["host_received_monotonic_ns"])

    def test_nonfinite_json_rejected(self):
        decoder = Decoder()
        decoder.feed(b'{"type":"config","value":NaN}\n')
        self.assertEqual(1, decoder.issues["malformed_line"])

    def test_missing_duplicate_reset_and_clock(self):
        decoder = Decoder()
        decoder.feed(encoded(frame(1, 100), frame(4, 400), frame(4, 400), frame(1, 50)))
        summary = decoder.summary()
        self.assertEqual(2, summary["missing_sequences"])
        self.assertEqual(1, summary["duplicate_sequences"])
        self.assertEqual(1, summary["sequence_resets_or_reorders"])
        self.assertIsNone(summary["observed_sensor_hz"])

    def test_sequence_wrap(self):
        decoder = Decoder()
        decoder.feed(encoded(frame(2**32-1, 100), frame(0, 300)))
        self.assertEqual(1, decoder.wraps)
        self.assertEqual(0, decoder.resets)
        self.assertEqual(5, decoder.summary()["observed_sensor_hz"])

    def test_shape_error_not_counted_as_frame(self):
        decoder = Decoder()
        decoder.feed(encoded({**frame(), "nb_target": []}))
        self.assertEqual(0, decoder.frames)
        self.assertEqual(1, decoder.issues["invalid_frame"])

    def test_port_selection_refuses_ambiguity(self):
        with self.assertRaises(ValueError):
            select_port("auto", [])
        with self.assertRaises(ValueError):
            select_port("auto", [{"port": "A", "vid": 0x303A}, {"port": "B", "vid": 0x303A}])
        self.assertEqual("B", select_port("auto", [{"port": "A", "vid": 0x1234}, {"port": "B", "vid": 0x303A}]))


class EvidenceTests(unittest.TestCase):
    def test_config_command_only_when_explicit_and_after_open(self):
        parser = argparse.ArgumentParser()
        capture_arguments(parser)
        for requested in (False, True):
            argv = ["--output", "synthetic-unused"] + (["--query-config"] if requested else [])
            args = parser.parse_args(argv)
            self.assertEqual(requested, args.query_config)
            device = Mock()
            device.read.return_value = b"fixture"
            device.write.return_value = 7
            with patch.dict(sys.modules, {"serial": SimpleNamespace(Serial=Mock(return_value=device))}), \
                 patch("capture.time.monotonic", side_effect=[0, 0, 0, 2]), \
                 patch("capture.time.monotonic_ns", return_value=123):
                self.assertEqual([(b"fixture", 123)], list(live_chunks("FAKE", 115200, 1, args.query_config)))
            device.open.assert_called_once_with()
            device.close.assert_called_once_with()
            if requested:
                device.write.assert_called_once_with(b"CONFIG\n")
                self.assertLess(device.method_calls.index(unittest.mock.call.open()),
                                device.method_calls.index(unittest.mock.call.write(b"CONFIG\n")))
            else:
                device.write.assert_not_called()
            self.assertFalse(device.dtr)
            self.assertFalse(device.rts)

    def test_firmware_hash_is_reference_not_device_attestation(self):
        self.assertEqual("UNKNOWN", firmware_metadata(None)["identity"])
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp)/"synthetic.bin"
            binary.write_bytes(b"fixture-only")
            metadata = firmware_metadata(binary)
            self.assertEqual(hashlib.sha256(b"fixture-only").hexdigest(), metadata["sha256"])
            self.assertFalse(metadata["device_match_verified"])
            out = Path(temp)/"session"
            run_session(out, iter([]), {"mode": "synthetic_test", "firmware": metadata, "label": "synthetic scene"})
            manifest = json.loads((out/"manifest.json").read_text())
            self.assertEqual(metadata, manifest["firmware"])
            self.assertEqual("synthetic scene", manifest["label"])

    def test_offline_cli_without_site_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"fixture.txt"
            source.write_bytes(encoded(frame()))
            result = subprocess.run([sys.executable, "-S", "-B", str(Path(__file__).with_name("capture.py")),
                                     "replay", "--input", str(source), "--output", str(Path(temp)/"out")],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(1, json.loads(result.stdout)["frames"])

    def test_interruption_not_reported_as_completed_capture(self):
        def interrupted():
            yield encoded(frame()), 1
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as temp:
            result = run_session(Path(temp)/"new", interrupted(), {"mode": "synthetic_test"})
            self.assertEqual("INTERRUPTED", result["result"])
            self.assertTrue(result["interrupted"])

    def test_empty_and_error_only_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = run_session(Path(temp)/"empty", iter([]), {"mode": "synthetic_test"})
            self.assertEqual("NO_VALID_FRAMES", empty["result"])
            error = run_session(Path(temp)/"error", iter([(encoded({"type": "error", "message": "init_failed"}), None)]), {"mode": "synthetic_test"})
            self.assertEqual("NO_VALID_FRAMES", error["result"])
            self.assertEqual(1, error["device_errors"])

    def test_raw_preserved_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"new"
            raw = encoded(frame(zones=64)) + b"\xff"
            result = run_session(out, iter([(raw, None)]), {"mode": "synthetic_test"})
            self.assertEqual(raw, (out/"raw.bin").read_bytes())
            self.assertEqual("FRAMES_WITH_ISSUES", result["result"])
            with self.assertRaises(FileExistsError):
                run_session(out, iter([]), {})

    def test_acquisition_failure_finalizes_and_closes(self):
        closed = []
        def broken():
            try:
                yield encoded(frame()), 1
                raise OSError("device disconnected")
            finally:
                closed.append(True)
        with tempfile.TemporaryDirectory() as temp:
            result = run_session(Path(temp)/"new", broken(), {"mode": "synthetic_test"})
            self.assertEqual([True], closed)
            self.assertEqual("FRAMES_WITH_ISSUES", result["result"])
            self.assertIn("device disconnected", result["acquisition_error"])


if __name__ == "__main__":
    unittest.main()
