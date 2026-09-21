"""Inspect paired captures using host receipt times only; no clock calibration."""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import statistics
import sys


RELATION = "RECEIPT_NEAREST_ONLY_UNCALIBRATED"


def receipt(record):
    value = record.get("host_received_monotonic_ns") if isinstance(record, dict) else None
    return value if type(value) is int and value >= 0 else None


def center_distances(record):
    """Conservative status-5 rule; missing/invalid cells stay None, never zero."""
    sensor = record.get("sensor", {}) if isinstance(record, dict) else {}
    if not isinstance(sensor, dict):
        return [], [None]*4, None, "sensor is not an object"
    distance, status, count = (sensor.get(k) for k in ("distance_mm", "target_status", "nb_target"))
    if not all(isinstance(v, list) for v in (distance, status, count)):
        return [], [None]*4, None, "missing range arrays"
    if len(distance) not in (16, 64) or len(status) != len(distance) or len(count) != len(distance):
        return [], [None]*4, None, "range shape mismatch"
    side = math.isqrt(len(distance))
    zones = [r*side+c for r in (side//2-1, side//2) for c in (side//2-1, side//2)]
    values = []
    for zone in zones:
        d, s, n = distance[zone], status[zone], count[zone]
        valid = type(d) is int and d > 0 and type(s) is int and s == 5 and type(n) is int and n > 0
        values.append(d if valid else None)
    known = [v for v in values if v is not None]
    return zones, values, statistics.median(known) if known else None, None


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {s}")))
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                error = None
            except ValueError as exc:
                value = None
                error = str(exc)
            records.append({"source_line": line_number, "record": value,
                            "parse_error": error, "raw_line_if_invalid": line.rstrip("\r\n") if error else None})
    return records


def safe_camera_path(source, filename):
    if not isinstance(filename, str) or not filename:
        return None
    base = (source/"camera").resolve()
    path = (base/filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return None
    return path.relative_to(source).as_posix()


def write_lines(path, records):
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False)+"\n")


def inspect(source, output, plot=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    # Output cannot replace or be an ancestor of source evidence.
    if output == source or source.is_relative_to(output):
        raise ValueError("output must not equal or contain the input directory")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    camera_rows = load_records(source/"camera"/"frames.jsonl")
    tof_rows = load_records(source/"tof"/"frames.jsonl")
    cameras = []
    for row in camera_rows:
        record = row["record"]
        reason = row["parse_error"]
        stamp = receipt(record)
        path = safe_camera_path(source, record.get("filename")) if record else None
        if not reason and (stamp is None or record.get("jpeg_validated") is not True or path is None):
            reason = "missing receipt, unvalidated JPEG or missing/unsafe image path"
        row["pair_exclusion"] = reason
        if not reason:
            header = record.get("header", {})
            cameras.append({"host_ns": stamp, "source_line": row["source_line"],
                            "seq": header.get("seq") if isinstance(header, dict) else None, "path": path})
    cameras.sort(key=lambda v: (v["host_ns"], v["source_line"]))
    # Duplicate receipt times have one deterministic representative for lookup.
    unique = {}
    for camera in cameras:
        unique.setdefault(camera["host_ns"], camera)
    candidates = list(unique.values())
    camera_times = [v["host_ns"] for v in candidates]
    tof_times = [receipt(row["record"]) for row in tof_rows if receipt(row["record"]) is not None]
    overlap = None
    if camera_times and tof_times:
        low, high = max(min(camera_times), min(tof_times)), min(max(camera_times), max(tof_times))
        if low <= high:
            overlap = (low, high)
    # Both selected observations must be within the shared receipt interval.
    eligible = [v for v in candidates if overlap and overlap[0] <= v["host_ns"] <= overlap[1]]
    eligible_times = [v["host_ns"] for v in eligible]
    pairs, curve = [], []
    missing_center = 0
    for row in tof_rows:
        record = row["record"]
        stamp = receipt(record)
        zones, values, median, range_error = center_distances(record)
        missing_center += median is None
        sensor = record.get("sensor", {}) if record else {}
        seq = sensor.get("seq") if isinstance(sensor, dict) else None
        item = {"source_line": row["source_line"], "tof_seq": seq,
                "host_received_monotonic_ns": stamp, "center_zones": zones,
                "center_known_mm": values, "center_median_known_mm": median,
                "parse_error": row["parse_error"], "range_error": range_error}
        curve.append(item)
        reason = "missing host receipt" if stamp is None else "outside shared receipt interval"
        if stamp is not None and eligible and overlap[0] <= stamp <= overlap[1]:
            index = bisect.bisect_left(eligible_times, stamp)
            nearest = min(eligible[max(0, index-1):min(len(eligible), index+1)],
                          key=lambda c: (abs(c["host_ns"]-stamp), c["host_ns"], c["source_line"]))
            pairs.append({"relation": RELATION, "tof_source_line": row["source_line"], "tof_seq": seq,
                          "camera_seq": nearest["seq"], "camera_source_line": nearest["source_line"],
                          "camera_relative_path": nearest["path"], "tof_host_ns": stamp,
                          "camera_host_ns": nearest["host_ns"], "camera_minus_tof_ns": nearest["host_ns"]-stamp,
                          "tof_center_median_known_mm": median})
            reason = None
        row["pair_exclusion"] = reason
    absolute = [abs(p["camera_minus_tof_ns"])/1e6 for p in pairs]
    summary = {
        "relation": RELATION, "input": str(source), "camera_records": len(camera_rows),
        "camera_pair_eligible_records": len(cameras), "tof_records": len(tof_rows), "pairs": len(pairs),
        "camera_parse_errors": sum(r["parse_error"] is not None for r in camera_rows),
        "tof_parse_errors": sum(r["parse_error"] is not None for r in tof_rows),
        "tof_missing_center_distance_records": missing_center,
        "overlap_start_host_ns": overlap[0] if overlap else None,
        "overlap_end_host_ns": overlap[1] if overlap else None,
        "overlap_seconds": (overlap[1]-overlap[0])/1e9 if overlap else 0,
        "absolute_receipt_delta_ms_median": statistics.median(absolute) if absolute else None,
        "absolute_receipt_delta_ms_max": max(absolute) if absolute else None,
        "limits": ["Host receipt nearest-neighbor only; not true synchronization or spatial calibration.",
                   "ToF host time is chunk receipt containing final newline; unknown USB buffering latency.",
                   "Current CNH firmware uses 4x4 zones; center zones 5,6,9,10. Legacy 8x8 center also supported.",
                   "Missing/nonpositive/non-status-5/no-target distances remain UNKNOWN, not zero.",
                   "No automatic RGB recognition, motion judgement, avoidance or safety conclusion."],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_lines(output/"pairs.jsonl", pairs)
    write_lines(output/"tof_center_curve.jsonl", curve)
    write_lines(output/"camera_records.jsonl", camera_rows)
    write_lines(output/"tof_records.jsonl", tof_rows)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if plot:
        plot_curve(curve, overlap, output/"tof-center-by-host-receipt.png")
    return summary


def plot_curve(curve, overlap, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    timed = [row for row in curve if row["host_received_monotonic_ns"] is not None]
    origin = min((row["host_received_monotonic_ns"] for row in timed), default=0)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for index in range(4):
        ax.plot([(r["host_received_monotonic_ns"]-origin)/1e9 for r in timed],
                [r["center_known_mm"][index] if r["center_known_mm"][index] is not None else math.nan for r in timed],
                marker=".", label=f"Center cell {index+1}")
    if overlap:
        ax.axvspan((overlap[0]-origin)/1e9, (overlap[1]-origin)/1e9, alpha=0.1, color="green", label="Shared host receipt interval")
    ax.set(xlabel="Host receipt time from first ToF receipt (s)", ylabel="Known status-5 distance (mm)",
           title="ToF center distances | RECEIPT NEAREST ONLY, UNCALIBRATED\nMissing distances are gaps; USB delay unknown; no motion judgement")
    ax.grid(alpha=0.2)
    ax.legend()
    try:
        with path.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=140)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.input, args.output, args.plot), indent=2))
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
