"""Isolated ToF serial acquisition and offline replay; no alert integration."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


CNH_STATUS = "UNVERIFIED_ON_HARDWARE"


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected integer in [{low}, {high}]")
    return value


def vector(obj, key, length, low, high):
    values = obj.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{key}: expected {length} values")
    for value in values:
        integer(value, key, low, high)
    return values


def validate_frame(obj):
    """Keep raw fields; derive conservative range validity and CNH normalization."""
    if obj.get("type") not in ("frame", "cnh_frame"):
        raise ValueError("not a sensor frame")
    integer(obj.get("seq"), "seq", 0, 2**32 - 1)
    integer(obj.get("ms"), "ms", 0, 2**32 - 1)
    count = len(obj.get("distance_mm", [])) if isinstance(obj.get("distance_mm"), list) else 0
    if count not in (16, 64):
        raise ValueError("distance_mm: expected 4x4 or 8x8")
    side = math.isqrt(count)
    if obj.get("rows", side) != side or obj.get("cols", side) != side:
        raise ValueError("rows/cols disagree with range array")
    distance = vector(obj, "distance_mm", count, -32768, 32767)
    status = vector(obj, "target_status", count, 0, 255)
    targets = vector(obj, "nb_target", count, 0, 255)
    valid = [d > 0 and s == 5 and n > 0 for d, s, n in zip(distance, status, targets)]
    derived = {
        "rows": side, "cols": side,
        "range_valid": valid,
        "distance_known_mm": [d if v else None for d, v in zip(distance, valid)],
        "validity_rule": "distance_mm > 0 AND target_status == 5 AND nb_target > 0; otherwise UNKNOWN",
    }
    if "diagnostic" in obj:
        diagnostic = obj["diagnostic"]
        if not isinstance(diagnostic, dict):
            raise ValueError("diagnostic: expected object")
        integer(diagnostic.get("schema"), "diagnostic.schema", 1, 1)
        for key, low, high in (
            ("distance_q2", -32768, 32767),
            ("signal_kcps_spad", 0, 2**32 - 1),
            ("ambient_kcps_spad", 0, 2**32 - 1),
            ("range_sigma_mm", 0, 65535),
            ("reflectance_percent", 0, 255),
            ("nb_spads_enabled", 0, 2**32 - 1),
        ):
            vector(diagnostic, key, count, low, high)
        integer(diagnostic.get("silicon_temp_degc"), "silicon_temp_degc", -128, 127)
        # C signed integer division truncates toward zero, then ULD clamps negatives.
        expected = [max(0, math.trunc(q2 / 4)) for q2 in diagnostic["distance_q2"]]
        matches = [actual == wanted for actual, wanted in zip(distance, expected)]
        derived["diagnostic"] = {
            "uld_expected_distance_mm": expected,
            "distance_conversion_match": matches,
            "distance_conversion_mismatch_count": matches.count(False),
            "conversion_rule": "max(0, trunc_towards_zero(distance_q2 / 4)); independent of range validity",
        }
    if obj["type"] == "cnh_frame":
        if count != 16 or obj.get("bins", 24) != 24:
            raise ValueError("CNH supports only 16 zones x 24 bins in this bring-up")
        for key, low, high in (("hist_raw", -2**31, 2**31 - 1), ("hist_scaler", -128, 127)):
            matrix = obj.get(key)
            if not isinstance(matrix, list) or len(matrix) != 16:
                raise ValueError(f"{key}: expected 16 rows")
            for row in matrix:
                vector({key: row}, key, 24, low, high)
        ambient = vector(obj, "ambient_raw", 16, -2**31, 2**31 - 1)
        ambient_scaler = vector(obj, "ambient_scaler", 16, -128, 127)
        # ST UM3183 Rev 7, section 5.7: signed raw / (2 ** scaler).
        # The supplied Example_12 uses (2 << scaler), a conflicting factor of 2.
        # Follow the manual; ldexp handles negative signed int8 scalers safely.
        derived["hist_normalized"] = [
            [math.ldexp(float(v), -s) for v, s in zip(row, scales)]
            for row, scales in zip(obj["hist_raw"], obj["hist_scaler"])
        ]
        derived["ambient_normalized"] = [math.ldexp(float(v), -s) for v, s in zip(ambient, ambient_scaler)]
        derived["normalization"] = "ldexp(raw, -scaler), equivalent to raw/(2**scaler); ST UM3183 Rev 7 section 5.7"
        derived["cnh_hardware_status"] = CNH_STATUS
    return derived


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Decoder:
    def __init__(self, emit=None):
        self.emit = emit or (lambda kind, record: None)
        self.pending = bytearray()
        self.line = 0
        self.frames = 0
        self.kinds = Counter()
        self.statuses = Counter()
        self.events = 0
        self.errors = 0
        self.issues = Counter()
        self.known = self.unknown = 0
        self.previous = None
        self.missing = self.resets = self.duplicates = self.wraps = 0
        self.center = []
        self.first_ms = self.last_ms = None
        self.clock_discontinuities = 0
        self.cnh_zero_histograms = 0
        self.cnh_finite = True
        self.diagnostic_frames = self.distance_conversion_mismatches = 0

    def issue(self, reason, detail, host_ns):
        self.issues[reason] += 1
        self.emit("issues", {"line": self.line, "reason": reason, "detail": detail, "host_received_monotonic_ns": host_ns})

    def feed(self, chunk, host_ns=None, final=False):
        self.pending.extend(chunk)
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending = bytearray(rest)
            self.parse(line, host_ns)
        if final and self.pending:
            self.parse(bytes(self.pending), host_ns)
            self.pending.clear()

    def parse(self, line, host_ns):
        self.line += 1
        if not line.strip():
            return
        try:
            text = line.decode("utf-8")
            obj = json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {s}")))
        except (UnicodeError, ValueError) as exc:
            self.issue("malformed_line", str(exc), host_ns)
            return
        if not isinstance(obj, dict):
            self.issue("non_object", "JSON record is not an object", host_ns)
            return
        envelope = {"line": self.line, "host_received_monotonic_ns": host_ns, "sensor": obj}
        if obj.get("type") not in ("frame", "cnh_frame"):
            self.events += 1
            if obj.get("type") == "error" or obj.get("status", 0) != 0:
                self.errors += 1
            self.emit("events", envelope)
            return
        try:
            derived = validate_frame(obj)
        except ValueError as exc:
            self.issue("invalid_frame", str(exc), host_ns)
            return
        seq = obj["seq"]
        if self.previous is not None:
            step = seq - self.previous
            if step < 0 and self.previous > 0xF0000000 and seq < 0x0FFFFFFF:
                step += 2**32
                self.wraps += 1
            if step > 1:
                self.missing += step - 1
                self.issue("sequence_gap", f"{self.previous} -> {seq}; missing {step-1}", host_ns)
            elif step == 0:
                self.duplicates += 1
                self.issue("duplicate_sequence", str(seq), host_ns)
            elif step < 0:
                self.resets += 1
                self.issue("sequence_reset_or_reorder", f"{self.previous} -> {seq}", host_ns)
        if self.last_ms is not None and obj["ms"] <= self.last_ms:
            self.clock_discontinuities += 1
        self.previous = seq
        self.first_ms = obj["ms"] if self.first_ms is None else self.first_ms
        self.last_ms = obj["ms"]
        self.frames += 1
        self.kinds[obj["type"]] += 1
        if "diagnostic" in derived:
            self.diagnostic_frames += 1
            self.distance_conversion_mismatches += derived["diagnostic"]["distance_conversion_mismatch_count"]
        if obj["type"] == "cnh_frame":
            self.cnh_zero_histograms += sum(all(v == 0 for v in row) for row in obj["hist_raw"])
            self.cnh_finite &= all(math.isfinite(v) for row in derived["hist_normalized"] for v in row)
            self.cnh_finite &= all(math.isfinite(v) for v in derived["ambient_normalized"])
        self.statuses.update(map(str, obj["target_status"]))
        self.known += sum(derived["range_valid"])
        self.unknown += len(derived["range_valid"]) - sum(derived["range_valid"])
        side = derived["rows"]
        for r in (side//2 - 1, side//2):
            for c in (side//2 - 1, side//2):
                value = derived["distance_known_mm"][r*side+c]
                if value is not None:
                    self.center.append(value)
        self.emit("frames", {**envelope, "derived": derived})

    def summary(self):
        hz = None
        if self.frames > 1 and not self.clock_discontinuities and not self.resets:
            hz = (self.frames - 1)*1000/(self.last_ms - self.first_ms)
        return {
            "frames": self.frames, "frame_types": dict(self.kinds), "events": self.events,
            "device_errors": self.errors, "issues": dict(self.issues),
            "missing_sequences": self.missing, "duplicate_sequences": self.duplicates,
            "sequence_resets_or_reorders": self.resets, "sequence_wraps": self.wraps,
            "sensor_clock_discontinuities": self.clock_discontinuities,
            "observed_sensor_hz": hz, "status_counts": dict(self.statuses),
            "known_range_cells": self.known, "unknown_range_cells": self.unknown,
            "center_status5_median_mm": statistics.median(self.center) if self.center else None,
            "cnh_frames": self.kinds["cnh_frame"],
            "cnh_histograms": self.kinds["cnh_frame"] * 16,
            "cnh_all_zero_histograms": self.cnh_zero_histograms,
            "cnh_normalized_finite": self.cnh_finite if self.kinds["cnh_frame"] else None,
            "cnh_hardware_status": CNH_STATUS,
            "diagnostic_frames": self.diagnostic_frames,
            "distance_conversion_mismatch_cells": self.distance_conversion_mismatches,
            "evidence_scope": "sensor bring-up only; no alert, synchronization, accuracy or safety claim",
        }


def ports():
    # Offline replay must work without pyserial installed.
    from serial.tools import list_ports
    return [{"port": p.device, "description": p.description, "vid": p.vid, "pid": p.pid, "serial_number": p.serial_number}
            for p in list_ports.comports()]


def select_port(requested, available=None):
    if requested != "auto":
        return requested
    candidates = [p["port"] for p in (ports() if available is None else available) if p["vid"] == 0x303A]
    if len(candidates) != 1:
        raise ValueError(f"auto requires exactly one Espressif VID 303A port; found {candidates}; specify --port")
    return candidates[0]


def live_chunks(port, baud, seconds, query_config=False):
    import serial
    device = serial.Serial(port=None, baudrate=baud, timeout=min(0.2, seconds))
    device.port = port
    device.dtr = device.rts = False
    try:
        device.open()
        if query_config:
            if device.write(b"CONFIG\n") != len(b"CONFIG\n"):
                raise OSError("incomplete CONFIG command write")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            device.timeout = min(0.2, max(0.001, deadline - time.monotonic()))
            chunk = device.read(16384)
            if chunk:
                yield chunk, time.monotonic_ns()
    finally:
        device.close()


def file_chunks(path):
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            yield chunk, None


def run_session(output, chunks, metadata, display=None):
    """Exclusive evidence directory; finalize receipts even after serial failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hardware-bringup.capture.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "host_clock": "monotonic_ns at chunk receipt; not mapped to sensor millis or other devices",
        "frame_receipt": "receipt time of chunk containing final newline; buffered USB latency remains unknown",
        "firmware": firmware_metadata(None), "label": None,
        "cnh_hardware_status": CNH_STATUS, **metadata,
    }
    write_json(output / "manifest.json", manifest)
    handles = {}
    digest = hashlib.sha256()
    total = 0
    error = None
    interrupted = False
    decoder = None
    try:
        for name in ("frames", "events", "issues", "received_chunks"):
            handles[name] = (output / f"{name}.jsonl").open("x", encoding="utf-8")

        def emit(kind, record):
            handles[kind].write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if display and kind == "frames":
                display(record)

        decoder = Decoder(emit)
        with (output / "raw.bin").open("xb") as raw:
            last_ns = None
            try:
                for chunk, host_ns in chunks:
                    raw.write(chunk)
                    digest.update(chunk)
                    handles["received_chunks"].write(json.dumps({"offset": total, "bytes": len(chunk), "host_received_monotonic_ns": host_ns}) + "\n")
                    total += len(chunk)
                    last_ns = host_ns
                    decoder.feed(chunk, host_ns)
            except KeyboardInterrupt:
                interrupted = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                decoder.feed(b"", last_ns, final=True)
    finally:
        # Explicitly close generators so their serial-port finally runs on errors.
        close = getattr(chunks, "close", None)
        if close:
            close()
        for handle in handles.values():
            handle.close()
    summary = decoder.summary() if decoder else {"frames": 0}
    summary.update(bytes=total, raw_sha256=digest.hexdigest(), acquisition_error=error, interrupted=interrupted)
    if not summary["frames"]:
        summary["result"] = "NO_VALID_FRAMES"
    elif interrupted:
        summary["result"] = "INTERRUPTED"
    elif error or summary.get("device_errors") or summary.get("issues"):
        summary["result"] = "FRAMES_WITH_ISSUES"
    else:
        summary["result"] = "FRAMES_RECORDED"
    write_json(output / "summary.json", summary)
    manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(), result=summary["result"])
    write_json(output / "manifest.json", manifest)
    return summary


def bounded_seconds(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 3600:
        raise argparse.ArgumentTypeError("seconds must be > 0 and <= 3600")
    return value


def capture_arguments(parser):
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=bounded_seconds, default=30)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; existing directories rejected")
    parser.add_argument("--firmware", type=Path, help="optional operator-supplied binary to hash; does not prove device firmware identity")
    parser.add_argument("--label", help="optional scene label; operator-provided, not measured ground truth")
    parser.add_argument("--query-config", action="store_true", help="send CONFIG once after opening; request cached boot readback without resetting")


def firmware_metadata(path):
    if path is None:
        return {"identity": "UNKNOWN", "path": None, "sha256": None, "device_match_verified": False}
    path = Path(path)
    with path.open("rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    return {"identity": "OPERATOR_SUPPLIED_BINARY", "path": str(path.resolve()), "sha256": digest,
            "device_match_verified": False}


def capture(args, display=None):
    firmware = firmware_metadata(args.firmware)
    selected = select_port(args.port)
    return run_session(args.output, live_chunks(selected, args.baud, args.seconds, args.query_config),
                       {"mode": "live_serial", "port": selected, "baud": args.baud, "requested_seconds": args.seconds,
                        "firmware": firmware, "label": args.label, "query_config_requested": args.query_config}, display)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-ports", help="enumerate without opening ports")
    capture_arguments(sub.add_parser("capture"))
    replay = sub.add_parser("replay", help="offline parsing of raw.bin or legacy serial.txt, no serial dependency")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "list-ports":
            print(json.dumps(ports(), indent=2))
            return 0
        if args.command == "capture":
            summary = capture(args)
        else:
            if not args.input.is_file():
                raise ValueError(f"input does not exist: {args.input}")
            summary = run_session(args.output, file_chunks(args.input), {"mode": "offline_replay", "source": str(args.input.resolve()), "host_receipts_available": False})
        print(json.dumps(summary, indent=2))
        return 0 if summary["result"] == "FRAMES_RECORDED" else 2
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
