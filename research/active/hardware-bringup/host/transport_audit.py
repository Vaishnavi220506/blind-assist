"""Read-only transport audit of a completed paired capture; no serial access."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from capture import Decoder


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def intervals(rows):
    times = [r["host_received_monotonic_ns"] for r in rows]
    gaps = [(b-a)/1e6 for a, b in zip(times, times[1:])]
    ordered = sorted(gaps)
    return {"frames": len(times), "span_seconds": (times[-1]-times[0])/1e9 if len(times)>1 else 0,
            "receipt_fps": (len(times)-1)*1e9/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None,
            "gap_p50_ms": statistics.median(gaps) if gaps else None,
            "gap_p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*.95))] if gaps else None,
            "gap_max_ms": max(gaps) if gaps else None,
            "gaps_over_1500ms": sum(g>1500 for g in gaps),
            "backwards_receipts": sum(g<0 for g in gaps)}


def inspect_raw(root):
    raw = (root/"raw.bin").read_bytes()
    frames = records(root/"frames.jsonl")
    issues = records(root/"issues.jsonl")
    chunks = records(root/"received_chunks.jsonl")
    position = 0
    contiguous = True
    for chunk in chunks:
        contiguous &= chunk["offset"] == position
        position = chunk["offset"]+chunk["bytes"]
    replay = []
    for size in (7, 16384, max(1, len(raw))):
        sensors = []
        decoder = Decoder(lambda kind, r: sensors.append(r["sensor"]) if kind == "frames" else None)
        for offset in range(0, len(raw), size):
            decoder.feed(raw[offset:offset+size])
        decoder.feed(b"", final=True)
        summary = decoder.summary()
        replay.append({"chunk_bytes": size, "frames": summary["frames"], "issues": summary["issues"],
                       "missing_sequences": summary["missing_sequences"],
                       "sensors_equal_saved": sensors == [f["sensor"] for f in frames]})
    lines = raw.splitlines(keepends=True)
    malformed = []
    for issue in issues:
        if issue["reason"] != "malformed_line":
            continue
        index = issue["line"]-1
        line = lines[index]
        start = sum(len(x) for x in lines[:index])
        try:
            json.loads(line)
            direct_error = None
        except (ValueError, UnicodeError) as exc:
            direct_error = str(exc)
        malformed.append({"line": index+1, "bytes": len(line), "offset": start,
                          "at_capture_edge": index in (0,len(lines)-1),
                          "newline_terminated": line.endswith(b"\n"), "direct_json_error": direct_error,
                          "intersecting_chunks": [c for c in chunks if c["offset"]<start+len(line) and c["offset"]+c["bytes"]>start]})
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "chunk_receipts_cover_raw": contiguous and position == len(raw),
            "replay": replay, "malformed": malformed, "timing": intervals(frames)}


def audit(root):
    root = Path(root).resolve()
    summary = json.loads((root/"summary.json").read_text())
    manifest = json.loads((root/"manifest.json").read_text())
    camera = records(root/"camera/frames.jsonl")
    bad_images = []
    for frame in camera:
        image = (root/"camera"/frame["filename"]).resolve()
        if not image.is_relative_to(root/"camera") or not image.is_file():
            bad_images.append(frame["filename"])
        elif not frame["jpeg_validated"] or hashlib.sha256(image.read_bytes()).hexdigest()!=frame["sha256"]:
            bad_images.append(frame["filename"])
    raw_camera = root/"camera/serial.bin"
    camera_hash = hashlib.sha256(raw_camera.read_bytes()).hexdigest()
    tof = inspect_raw(root/"tof")
    expected = summary["streams"]
    checks = {
        "collectors_passed": summary["result"] == "BOTH_COLLECTORS_PASSED",
        "natural_completion": not summary.get("stopped_by_request") and not summary.get("failure"),
        "camera_files_verified": not bad_images and bool(camera),
        "camera_raw_hash_matches": camera_hash == expected["camera"]["serial_sha256"],
        "tof_raw_hash_matches": tof["sha256"] == expected["tof"]["raw_sha256"],
        "tof_chunk_receipts_complete": tof["chunk_receipts_cover_raw"],
        "tof_replays_equal": all(r["sensors_equal_saved"] for r in tof["replay"]),
        "camera_count_matches": len(camera) == expected["camera"]["frames"],
        "tof_count_matches": tof["timing"]["frames"] == expected["tof"]["frames"],
        "no_receipt_stalls_over_1500ms": not intervals(camera)["gaps_over_1500ms"] and not tof["timing"]["gaps_over_1500ms"],
    }
    return {"schema": "hardware-bringup.transport-audit.v1", "run": root.name,
            "requested_seconds": manifest["requested_seconds"], "checks": checks,
            "bounded_transport_pass": all(checks.values()), "camera_timing": intervals(camera),
            "camera_bad_images": bad_images, "camera_raw_sha256": camera_hash,
            "tof": tof, "collector_summary": summary,
            "scope": "Observed short transport only; receipt timing is not exposure timing; no range accuracy or long-term reliability claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"run": result["run"], "checks": result["checks"], "bounded_transport_pass": result["bounded_transport_pass"]}))
