"""Offline cue-window review; scheduled directions are never orientation truth."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from capture import validate_frame
from pair_inspect import load_records, receipt, safe_camera_path


PHASE_NAMES = ("baseline", "left", "right", "up", "down")
MAX_CAMERA_DELTA_NS = 500_000_000


def safe_source(run, relative):
    path = (run/relative).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise ValueError(f"missing or unsafe source: {relative}")
    return path


def phase_stats(frames, start_ns, end_ns):
    selected = [r for r in frames if start_ns <= receipt(r) <= end_ns]
    zones = []
    for zone in range(16):
        valid, suspicious, suspicious_valid, unknown = [], 0, 0, 0
        for record in selected:
            sensor = record["sensor"]
            distance = sensor["distance_mm"][zone]
            original_valid = distance > 0 and sensor["target_status"][zone] == 5 and sensor["nb_target"][zone] > 0
            suspicious += int(1 <= distance <= 3)
            suspicious_valid += int(original_valid and distance <= 3)
            unknown += int(not original_valid)
            if original_valid and distance > 3:
                valid.append(distance)
        zones.append({"zone": zone, "row": zone//4, "column": zone % 4,
                      "median_mm": statistics.median(valid) if valid else None,
                      "orientation_valid_count": len(valid), "unknown_count": unknown,
                      "suspicious_1_to_3_mm_count": suspicious,
                      "suspicious_original_valid_count": suspicious_valid})
    return {"frame_count": len(selected), "zones": zones}


def camera_samples(cameras, start_ns, end_ns, origin):
    times = [r["host_ns"] for r in cameras]
    samples = []
    for target in (start_ns, (start_ns+end_ns)//2, end_ns):
        sample = {"target_host_monotonic_ns": target, "target_elapsed_s": (target-origin)/1e9,
                  "status": "MISSING_CAMERA", "path": None}
        position = bisect.bisect_left(times, target)
        candidates = [cameras[i] for i in (position-1, position) if 0 <= i < len(cameras)]
        if candidates:
            nearest = min(candidates, key=lambda r: (abs(r["host_ns"]-target), r["host_ns"]))
            delta = nearest["host_ns"]-target
            sample.update(nearest_elapsed_s=(nearest["host_ns"]-origin)/1e9,
                          receipt_delta_ms=delta/1e6, nearest_source_line=nearest["source_line"])
            if abs(delta) <= MAX_CAMERA_DELTA_NS:
                sample.update(status="RECEIPT_NEAREST_ONLY_UNCALIBRATED", path=nearest["path"],
                              host_received_monotonic_ns=nearest["host_ns"], seq=nearest["seq"])
            else:
                sample["status"] = "MISSING_CAMERA_OVER_500_MS"
        samples.append(sample)
    return samples


def contact_sheet(run, output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    figure, axes = plt.subplots(5, 5, figsize=(19, 17), gridspec_kw={"width_ratios": [1.4, 1.4, 1.4, 1, 1]})
    figure.suptitle("Scheduled cue windows: RGB review required; no orientation assertion\n"
                   "Camera selection uses host receipt only (<=500 ms); ToF cells remain native zone order", fontsize=15)
    image_issues = []
    limits = {key: max([1]+[abs(z[key]) for phase in report["phases"].values()
                            for z in phase["zones"] if z.get(key) is not None])
              for key in ("median_mm", "baseline_minus_phase_mm")}
    for row, name in enumerate(PHASE_NAMES):
        phase = report["phases"][name]
        for column, sample in enumerate(phase["camera_samples"]):
            axis = axes[row, column]
            axis.axis("off")
            if sample["path"]:
                try:
                    path = safe_source(run, sample["path"])
                    with Image.open(path) as image:
                        axis.imshow(image.convert("RGB"))
                except (OSError, ValueError) as exc:
                    sample["status"] = "MISSING_CAMERA_IMAGE_ERROR"
                    image_issues.append({"phase": name, "path": sample["path"], "error": str(exc)})
                    sample["path"] = None
            if not sample["path"]:
                axis.text(.5, .5, sample["status"], ha="center", va="center", wrap=True, fontsize=9)
            title = f"{name.upper()} cue | target {sample['target_elapsed_s']:.2f}s"
            if "nearest_elapsed_s" in sample:
                title += f"\nreceipt {sample['nearest_elapsed_s']:.3f}s; delta {sample['receipt_delta_ms']:+.1f} ms"
            axis.set_title(title, fontsize=9)
        for column, key, title in ((3, "median_mm", "ToF median mm"), (4, "baseline_minus_phase_mm", "Baseline - phase mm")):
            axis = axes[row, column]
            values = [z.get(key) for z in phase["zones"]]
            matrix = np.array([np.nan if v is None else v for v in values], dtype=float).reshape(4, 4)
            limit = limits[key]
            axis.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm" if column == 4 else "viridis",
                        vmin=-limit if column == 4 else 0, vmax=limit)
            for zone, value in enumerate(values):
                axis.text(zone % 4, zone//4, "UNKNOWN" if value is None else f"{value:.0f}",
                          ha="center", va="center", fontsize=8, color="black",
                          bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
            axis.set_xticks(range(4)); axis.set_yticks(range(4))
            axis.set_title(f"{title}\n{phase['frame_count']} ToF frames", fontsize=9)
    figure.text(.02, .012, "Diagnostic statistics exclude 1-3 mm only for this review; global range validity unchanged. Cues are not performed-action ground truth.", fontsize=11)
    figure.tight_layout(rect=(0, .035, 1, .95))
    figure.savefig(output/"contact-sheet.png", dpi=130)
    plt.close(figure)
    return image_issues


def review(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir():
        raise ValueError(f"capture directory missing: {run}")
    if output == run or run.is_relative_to(output):
        raise ValueError("output must not equal or contain the source directory")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    control = (run.parent/".dashboard-control").resolve()
    schedule_path = (control/f"{run.name}.orientation.json").resolve()
    if not schedule_path.is_relative_to(control) or not schedule_path.is_file():
        raise ValueError(f"INSUFFICIENT_EVIDENCE: orientation cue schedule missing: {schedule_path}")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    origin = schedule.get("start_host_monotonic_ns")
    if schedule.get("schema") != "hardware-bringup.orientation-cues.v1" or type(origin) is not int or origin < 0:
        raise ValueError("invalid orientation schedule schema or start timestamp")
    phases = {}
    for phase in schedule.get("phases", []):
        name = phase.get("phase")
        if name not in PHASE_NAMES:
            continue
        start, end = phase.get("start_s"), phase.get("end_s")
        if name in phases or any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or end-start <= 2:
            raise ValueError(f"invalid or duplicate phase window: {name}")
        phases[name] = (origin+round((start+1)*1e9), origin+round((end-1)*1e9))
    if set(phases) != set(PHASE_NAMES):
        raise ValueError("INSUFFICIENT_EVIDENCE: baseline or direction phase missing from schedule")
    sources = {"schedule": schedule_path, "tof": safe_source(run, "tof/frames.jsonl"),
               "camera": safe_source(run, "camera/frames.jsonl")}
    exclusions, frames, cameras = [], [], []
    for kind in ("tof", "camera"):
        for row in load_records(sources[kind]):
            record = row["record"]
            error = row["parse_error"]
            try:
                if error or receipt(record) is None:
                    raise ValueError(error or "missing host receipt")
                if kind == "tof":
                    derived = validate_frame(record.get("sensor", {}))
                    if derived["rows"] != 4:
                        raise ValueError("orientation review requires 4x4 zones")
                    frames.append(record)
                else:
                    path = safe_camera_path(run, record.get("filename"))
                    if not path or record.get("jpeg_validated") is not True:
                        raise ValueError("missing/unsafe image or unvalidated JPEG")
                    cameras.append({"host_ns": receipt(record), "path": path,
                                    "seq": record.get("header", {}).get("seq"), "source_line": row["source_line"]})
            except (ValueError, TypeError, AttributeError) as exc:
                exclusions.append({"source": kind, "source_line": row["source_line"], "reason": str(exc)})
    cameras.sort(key=lambda r: (r["host_ns"], r["source_line"]))
    report = {"schema": "hardware-bringup.orientation-review.v1", "verdict": "NEEDS_RGB_REVIEW",
              "run": str(run), "schedule": schedule, "phases": {}, "exclusions": exclusions,
              "sources": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in sources.items()},
              "statistics_rule": "status == 5 AND targets > 0 AND distance_mm > 3; 1-3 mm excluded only from these diagnostic orientation statistics, NOT a global validity threshold",
              "count_rule": "UNKNOWN follows original status5/targets>0/distance>0 rule; suspicious_1_to_3 counts raw distance regardless of status and may overlap UNKNOWN",
              "scope": "Scheduled cues are not performed-action ground truth. No automatic orientation, calibration, exposure synchronization or accuracy assertion.",
              "camera_max_receipt_delta_ms": 500}
    for name, (start, end) in phases.items():
        report["phases"][name] = {"inner_start_elapsed_s": (start-origin)/1e9, "inner_end_elapsed_s": (end-origin)/1e9,
                                  **phase_stats(frames, start, end), "camera_samples": camera_samples(cameras, start, end, origin)}
    baseline = report["phases"]["baseline"]["zones"]
    for phase in report["phases"].values():
        for zone, reference in zip(phase["zones"], baseline):
            zone["baseline_minus_phase_mm"] = (reference["median_mm"]-zone["median_mm"]
                                                if reference["median_mm"] is not None and zone["median_mm"] is not None else None)
    output.mkdir(parents=True, exist_ok=False)
    report["image_issues"] = contact_sheet(run, output, report)
    if any(not p["frame_count"] or not any(z["median_mm"] is not None for z in p["zones"])
           or any(s["path"] is None for s in p["camera_samples"]) for p in report["phases"].values()):
        report["verdict"] = "INSUFFICIENT_EVIDENCE"
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review(args.run, args.output)
        print(json.dumps({"verdict": result["verdict"], "output": str(args.output.resolve())}))
        return 0 if result["verdict"] == "NEEDS_RGB_REVIEW" else 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
