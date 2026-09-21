"""Run bounded camera/ToF collectors together; no clock calibration or fusion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from capture import ports, bounded_seconds


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-port", required=True)
    parser.add_argument("--tof-port", required=True)
    parser.add_argument("--seconds", type=bounded_seconds, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="operator scene not specified")
    parser.add_argument("--stop-file", type=Path, help="shared graceful stop marker passed to both collectors")
    parser.add_argument("--tof-query-config", action="store_true",
                        help="ask the identified ToF diagnostic firmware for cached configuration")
    args = parser.parse_args()
    if args.camera_port.upper() == args.tof_port.upper():
        parser.error("camera and ToF must use different explicitly identified ports")
    available = {p["port"].upper(): p for p in ports()}
    for port in (args.camera_port, args.tof_port):
        if port.upper() not in available or available[port.upper()]["vid"] != 0x303A:
            parser.error(f"{port}: expected an enumerated Espressif USB device")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    common = [sys.executable, "-B"]
    commands = {
        "camera": common + [str(source/"atom_capture.py"), "--port", args.camera_port,
                            "--usb-otg", "--seconds", str(args.seconds), "--output", str(root/"camera")],
        "tof": common + [str(source/"capture.py"), "capture", "--port", args.tof_port,
                         "--seconds", str(args.seconds), "--output", str(root/"tof"), "--label", args.label],
    }
    if args.tof_query_config:
        commands["tof"].append("--query-config")
    if args.stop_file is not None:
        for command in commands.values():
            command.extend(["--stop-file", str(args.stop_file.resolve())])
    manifest = {
        "schema": "hardware-bringup.parallel-capture.v1", "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host_start_monotonic_ns": time.monotonic_ns(), "requested_seconds": args.seconds,
        "devices": {k: available[p.upper()] for k, p in (("camera", args.camera_port), ("tof", args.tof_port))},
        "commands": commands, "status": "STARTING", "firmware_flashing": False,
        "stop_file": str(args.stop_file.resolve()) if args.stop_file else None,
        "clock_boundary": "same-host monotonic receipts only; no exposure alignment, hardware trigger or device-clock mapping",
        "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ("pair_capture.py", "atom_capture.py", "capture.py")},
    }
    write(root/"manifest.json", manifest)
    children, logs = {}, []
    failure = None
    try:
        for name, command in commands.items():
            log = (root/f"{name}-console.log").open("xb")
            logs.append(log)
            children[name] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + args.seconds + 15
        for child in children.values():
            child.wait(timeout=max(.1, deadline-time.monotonic()))
    except (Exception, KeyboardInterrupt) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)
        for log in logs:
            log.close()
    summaries = {}
    for name in commands:
        path = root/name/"summary.json"
        summaries[name] = json.loads(path.read_text()) if path.exists() else None
    success = failure is None and len(children) == 2 and all(p.returncode == 0 for p in children.values())
    result = {"result": "BOTH_COLLECTORS_PASSED" if success else "COLLECTION_HAS_ISSUES",
              "stopped_by_request": any(s and s.get("stopped_by_request") for s in summaries.values()),
              "failure": failure, "child_exit_codes": {k: v.returncode for k, v in children.items()},
              "streams": summaries, "clock_boundary": manifest["clock_boundary"]}
    write(root/"summary.json", result)
    manifest.update(status=result["result"], ended_utc=datetime.now(timezone.utc).isoformat(),
                    host_end_monotonic_ns=time.monotonic_ns(), stopped_by_request=result["stopped_by_request"])
    write(root/"manifest.json", manifest)
    print(json.dumps(result, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
