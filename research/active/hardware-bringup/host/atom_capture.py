"""Bounded USB request/JPEG capture for AtomS3R-M12; never flashes firmware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time


class ProtocolError(Exception):
    pass


class ReadTimeout(ProtocolError):
    pass


class Wire:
    """All incoming bytes are retained, including incomplete/invalid records."""
    def __init__(self, device, raw, clock=time.monotonic):
        self.device, self.raw, self.clock = device, raw, clock
        self.bytes = 0
        self.digest = hashlib.sha256()
        self.pending = bytearray()

    def fill(self, minimum, deadline):
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ReadTimeout("serial receive deadline reached")
        self.device.timeout = min(0.2, remaining)
        available = max(0, int(getattr(self.device, "in_waiting", 0)))
        chunk = self.device.read(min(65536, max(1, available, minimum)))
        if chunk:
            # Retain every received byte exactly once, including bytes read ahead.
            self.raw.write(chunk)
            self.digest.update(chunk)
            self.bytes += len(chunk)
            self.pending.extend(chunk)

    def exact(self, count, deadline):
        while len(self.pending) < count:
            try:
                self.fill(count-len(self.pending), deadline)
            except ReadTimeout as exc:
                raise ReadTimeout(f"read timeout after {len(self.pending)}/{count} buffered bytes") from exc
        result = bytes(self.pending[:count])
        del self.pending[:count]
        return result

    def line(self, deadline, limit=4096):
        while True:
            newline = self.pending.find(b"\n")
            if 0 <= newline < limit:
                return self.exact(newline+1, deadline)
            if len(self.pending) >= limit:
                raise ProtocolError(f"header/log line exceeds {limit} bytes")
            self.fill(1, deadline)


def header_valid(header):
    for name, low, high in (("seq", 0, 2**32-1), ("device_readout_us", 0, 2**64-1),
                            ("width", 640, 640), ("height", 480, 480), ("bytes", 4, 4*1024*1024)):
        value = header.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ProtocolError(f"invalid JPEG header {name}: {value!r}")


def receive(wire, deadline, event, stamp=time.monotonic_ns):
    while True:
        line = wire.line(deadline)
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        try:
            header = json.loads(text)
        except ValueError:
            event({"kind": "serial_log", "text": text, "host_received_monotonic_ns": stamp()})
            continue
        if not isinstance(header, dict) or header.get("type") != "jpeg":
            # Preserve line text even when a boot/error record has unexpected fields.
            event({"kind": "device_record", "text": text, "host_received_monotonic_ns": stamp()})
            if isinstance(header, dict) and header.get("type") == "error":
                raise ProtocolError(f"device error: {text}")
            continue
        header_valid(header)
        header_received = stamp()
        jpeg = wire.exact(header["bytes"], deadline)
        jpeg_received = stamp()
        if wire.exact(1, deadline) != b"\n":
            raise ProtocolError("JPEG payload is not followed by newline")
        return header, jpeg, header_received, jpeg_received


def validate_jpeg(jpeg, header):
    from PIL import Image
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ProtocolError("JPEG SOI/EOI marker missing")
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG" or image.size != (header["width"], header["height"]):
                raise ProtocolError(f"decoded image dimensions/format disagree with header: {image.size} / {image.format}")
            image.verify()
        with Image.open(io.BytesIO(jpeg)) as image:
            image.load()
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"JPEG decoding failed: {exc}") from exc


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def acquire(device, output, seconds, frame_timeout, *, port, clock=time.monotonic,
            stamp=time.monotonic_ns, usb_otg=False):
    """Device must be unopened; this function owns and closes it on every outcome."""
    output = Path(output)
    started = clock()
    deadline = started + seconds
    frames = failures = gaps = resets = 0
    previous = None
    first_receipt = last_receipt = None
    wire = None
    interrupted = False
    manifest = {
        "schema": "hardware-bringup.atom-jpeg.v1", "port": port, "baud": 115200,
        "started_utc": datetime.now(timezone.utc).isoformat(), "requested_seconds": seconds,
        "per_frame_timeout_seconds": frame_timeout,
        "clock_boundary": "device_readout_us is MCU time after esp_camera_fb_get; not exposure time; no cross-device clock mapping",
        "host_timing": "monotonic_ns at command request, header receipt and final JPEG byte receipt; includes USB buffering",
        "request_policy": "CAPTURE newline only; one outstanding request; reserve a full frame timeout before next request",
        "firmware_flashing": False,
        "usb_mode": "USB_OTG_TINYUSB_CDC" if usb_otg else "HARDWARE_CDC",
        "dtr_rts": {"dtr_before_open": bool(usb_otg), "rts_before_open": False,
                    "policy": "configured before open; no runtime toggling"},
        "serial_buffer_policy": {"platform": sys.platform, "rx_requested_bytes": 1048576,
                                 "tx_requested_bytes": 65536, "applied": False,
                                 "policy": "Windows set_buffer_size when available; buffered bulk reads capped at 65536 bytes"},
        "evidence_scope": "camera transport and JPEG decode only; not fusion, avoidance or safety evidence",
    }
    try:
        output.mkdir(parents=True, exist_ok=False)
        dump(output/"manifest.json", manifest)
        with (output/"serial.bin").open("xb") as raw, \
                (output/"frames.jsonl").open("x", encoding="utf-8") as records, \
                (output/"events.jsonl").open("x", encoding="utf-8") as events:
            def event(obj):
                events.write(json.dumps(obj, allow_nan=False)+"\n")
                events.flush()

            wire = Wire(device, raw, clock)
            try:
                device.open()
                if sys.platform == "win32" and hasattr(device, "set_buffer_size"):
                    device.set_buffer_size(rx_size=1048576, tx_size=65536)
                    manifest["serial_buffer_policy"]["applied"] = True
                    dump(output/"manifest.json", manifest)
                first_request = True
                while (first_request and clock() < deadline) or deadline-clock() >= frame_timeout:
                    first_request = False
                    requested = stamp()
                    event({"kind": "host_request", "command": "CAPTURE", "host_request_monotonic_ns": requested})
                    if device.write(b"CAPTURE\n") != len(b"CAPTURE\n"):
                        raise ProtocolError("incomplete CAPTURE command write")
                    frame_deadline = min(deadline, clock()+frame_timeout)
                    header, jpeg, header_ns, receipt_ns = receive(wire, frame_deadline, event, stamp)
                    filename = f"frame-{frames:06d}-seq-{header['seq']:010d}.jpg"
                    # Save complete payload even if JPEG verification subsequently fails.
                    with (output/filename).open("xb") as image:
                        image.write(jpeg)
                    record = {
                        "header": header, "host_request_monotonic_ns": requested,
                        "host_header_received_monotonic_ns": header_ns,
                        "host_received_monotonic_ns": receipt_ns,
                        "sha256": hashlib.sha256(jpeg).hexdigest(), "filename": filename,
                        "jpeg_validated": False,
                    }
                    try:
                        validate_jpeg(jpeg, header)
                        record["jpeg_validated"] = True
                    finally:
                        records.write(json.dumps(record, allow_nan=False)+"\n")
                        records.flush()
                    if previous is not None:
                        step = (header["seq"]-previous) % 2**32
                        if step != 1:
                            failures += 1
                            if 1 < step < 2**31:
                                gaps += step-1
                            else:
                                resets += 1
                            event({"kind": "sequence_discontinuity", "previous": previous, "current": header["seq"]})
                    previous = header["seq"]
                    frames += 1
                    first_receipt = receipt_ns if first_receipt is None else first_receipt
                    last_receipt = receipt_ns
            except KeyboardInterrupt:
                interrupted = True
                failures += 1
                event({"kind": "interrupted"})
            except Exception as exc:
                failures += 1
                event({"kind": "capture_failure", "error": f"{type(exc).__name__}: {exc}", "host_monotonic_ns": stamp()})
    finally:
        device.close()
    elapsed = max(0.0, clock()-started)
    result = "NO_VALID_FRAMES" if not frames else "FRAMES_WITH_FAILURES" if failures else "FRAMES_RECORDED"
    summary = {
        "result": result, "frames": frames, "failure_count": failures, "missing_sequences": gaps,
        "sequence_duplicates_resets_or_reorders": resets, "interrupted": interrupted,
        "elapsed_seconds": elapsed, "fps_over_session": frames/elapsed if elapsed else None,
        "receipt_fps": (frames-1)*1e9/(last_receipt-first_receipt) if frames > 1 and last_receipt > first_receipt else None,
        "width": 640 if frames else None, "height": 480 if frames else None,
        "serial_bytes": wire.bytes if wire else 0, "serial_sha256": wire.digest.hexdigest() if wire else None,
    }
    dump(output/"summary.json", summary)
    manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(), result=result)
    dump(output/"manifest.json", manifest)
    return summary


def positive_seconds(value):
    number = float(value)
    if not 0 < number <= 3600:
        raise argparse.ArgumentTypeError("must be >0 and <=3600 seconds")
    return number


def configure_serial_lines(device, usb_otg=False):
    """TinyUSB CDC requires DTR connected; existing HWCDC keeps both inactive."""
    device.dtr = bool(usb_otg)
    device.rts = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="explicit Atom USB serial port; no auto selection")
    parser.add_argument("--seconds", type=positive_seconds, default=20)
    parser.add_argument("--frame-timeout", type=positive_seconds, default=3)
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory; existing output refused")
    parser.add_argument("--usb-otg", action="store_true", help="TinyUSB USB-OTG CDC: assert DTR before open; RTS remains inactive")
    args = parser.parse_args()
    if args.port.lower() == "auto":
        parser.error("--port requires the explicit Atom serial port")
    try:
        import serial
        from PIL import Image  # fail before requesting hardware if dependency is unavailable
        device = serial.Serial(port=None, baudrate=115200, timeout=0.2, write_timeout=min(1, args.frame_timeout))
        device.port = args.port
        configure_serial_lines(device, args.usb_otg)
        result = acquire(device, args.output, args.seconds, args.frame_timeout, port=args.port, usb_otg=args.usb_otg)
        print(json.dumps(result, indent=2))
        return 0 if result["result"] == "FRAMES_RECORDED" else 2
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
