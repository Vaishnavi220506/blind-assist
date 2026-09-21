"""Review operator-marked baseline/up/down windows without asserting orientation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from capture import validate_frame
from orientation_review import phase_stats, camera_samples, safe_source
from pair_inspect import load_records, receipt, safe_camera_path


PHASES = ("baseline", "up", "down")
SECOND = 1_000_000_000


def marker_windows(schedule):
    """Only accept the complete ordered 6-second marker contract."""
    if not isinstance(schedule, dict) or schedule.get("schema") != "hardware-bringup.manual-vertical.v1":
        raise ValueError("invalid manual marker schema")
    origin = schedule.get("start_host_monotonic_ns")
    if type(origin) is not int or origin < 0:
        raise ValueError("invalid start_host_monotonic_ns")
    segments = schedule.get("segments")
    if not isinstance(segments, list) or len(segments) != 3:
        raise ValueError("baseline, up and down markers must all be present exactly once")
    previous_end = origin
    windows = {}
    for expected, segment in zip(PHASES, segments):
        if not isinstance(segment, dict) or segment.get("phase") != expected:
            raise ValueError("markers must be ordered baseline, up, down without duplicates")
        start, end = segment.get("start_host_monotonic_ns"), segment.get("end_host_monotonic_ns")
        if type(start) is not int or type(end) is not int or end-start != 6*SECOND:
            raise ValueError(f"{expected}: marker duration must be exactly 6 seconds")
        if start < previous_end:
            raise ValueError(f"{expected}: marker overlaps the previous segment or precedes run origin")
        if not isinstance(segment.get("label_source"), str) or not segment["label_source"].strip():
            raise ValueError(f"{expected}: missing label_source")
        previous_end = end
        windows[expected] = (start+SECOND, end-SECOND)
    return origin, windows


def load_streams(run, report):
    frames, cameras = [], []
    for kind in ("tof", "camera"):
        try:
            path = safe_source(run, f"{kind}/frames.jsonl")
            report["sources"][kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            rows = load_records(path)
        except (OSError, ValueError) as exc:
            report["insufficient_reasons"].append(str(exc))
            continue
        for row in rows:
            record = row["record"]
            try:
                if row["parse_error"] or receipt(record) is None:
                    raise ValueError(row["parse_error"] or "missing host receipt")
                if kind == "tof":
                    if validate_frame(record.get("sensor", {}))["rows"] != 4:
                        raise ValueError("manual review requires 4x4 zones")
                    frames.append(record)
                else:
                    image_path = safe_camera_path(run, record.get("filename"))
                    if not image_path or record.get("jpeg_validated") is not True:
                        raise ValueError("missing/unsafe camera path or unvalidated JPEG")
                    cameras.append({"host_ns": receipt(record), "path": image_path,
                                    "seq": record.get("header", {}).get("seq"), "source_line": row["source_line"]})
            except (ValueError, TypeError, AttributeError) as exc:
                report["exclusions"].append({"source": kind, "source_line": row["source_line"], "reason": str(exc)})
    cameras.sort(key=lambda row: (row["host_ns"], row["source_line"]))
    return frames, cameras


def contact_sheet(run, output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    figure, axes = plt.subplots(3, 5, figsize=(19, 12), gridspec_kw={"width_ratios": [1.4, 1.4, 1.4, 1, 1]})
    figure.suptitle("Manual baseline / up / down markers: RGB review required; no orientation assertion\n"
                   "4-second inner windows; host-receipt camera selection <=500 ms; native ToF zone order", fontsize=14)
    limits = {key: max([1]+[abs(zone[key]) for phase in report["phases"].values()
                            for zone in phase["zones"] if zone.get(key) is not None])
              for key in ("median_mm", "baseline_minus_phase_mm")}
    for row, name in enumerate(PHASES):
        phase = report["phases"].get(name)
        if phase is None:
            for axis in axes[row]:
                axis.axis("off")
                axis.text(.5, .5, f"{name.upper()}: NO VALID MARKER WINDOW", ha="center", va="center", fontsize=9, wrap=True)
            continue
        for column, sample in enumerate(phase["camera_samples"]):
            axis = axes[row, column]
            axis.axis("off")
            if sample["path"]:
                try:
                    with Image.open(safe_source(run, sample["path"])) as image:
                        axis.imshow(image.convert("RGB"))
                except (OSError, ValueError) as exc:
                    report["image_issues"].append({"phase": name, "path": sample["path"], "reason": str(exc)})
                    sample.update(path=None, status="MISSING_CAMERA_IMAGE_ERROR")
            if not sample["path"]:
                axis.text(.5, .5, sample["status"], ha="center", va="center", fontsize=9, wrap=True)
            title = f"{name.upper()} marker | target {sample['target_elapsed_s']:.2f}s"
            if "nearest_elapsed_s" in sample:
                title += f"\nreceipt {sample['nearest_elapsed_s']:.3f}s; delta {sample['receipt_delta_ms']:+.1f} ms"
            axis.set_title(title, fontsize=9)
        for column, key, title in ((3, "median_mm", "ToF median mm"), (4, "baseline_minus_phase_mm", "Baseline - phase mm")):
            axis = axes[row, column]
            values = [zone[key] for zone in phase["zones"]]
            matrix = np.array([np.nan if v is None else v for v in values]).reshape(4, 4)
            axis.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm" if column == 4 else "viridis",
                        vmin=-limits[key] if column == 4 else 0, vmax=limits[key])
            for zone, value in enumerate(values):
                axis.text(zone % 4, zone//4, "UNKNOWN" if value is None else f"{value:.0f}", ha="center", va="center", fontsize=8,
                          bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
            axis.set_xticks(range(4)); axis.set_yticks(range(4))
            axis.set_title(f"{title}\n{phase['frame_count']} ToF frames", fontsize=9)
    figure.text(.02, .015, "Clicks identify operator-selected intervals, not verified object positions. Diagnostic 1-3 mm exclusion does not change global validity.", fontsize=10)
    figure.tight_layout(rect=(0, .04, 1, .93), h_pad=3)
    figure.savefig(output/"contact-sheet.png", dpi=130)
    plt.close(figure)


def review(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir():
        raise ValueError(f"INSUFFICIENT_EVIDENCE: capture directory missing: {run}")
    if output == run or run.is_relative_to(output):
        raise ValueError("output must not equal or contain source directory")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    report = {"schema": "hardware-bringup.manual-vertical-review.v1", "verdict": "INSUFFICIENT_EVIDENCE",
              "run": str(run), "sources": {}, "phases": {}, "exclusions": [], "image_issues": [], "insufficient_reasons": [],
              "statistics_rule": "status == 5 AND targets > 0 AND distance_mm > 3; additional 1-3 mm exclusion is diagnostic only, NOT a global validity threshold",
              "count_rule": "UNKNOWN uses original status5/targets>0/distance>0 rule; raw suspicious 1-3 mm counts may overlap UNKNOWN",
              "camera_max_receipt_delta_ms": 500,
              "scope": "Operator clicks label intervals, not verified object positions or motion ground truth. No automatic orientation, calibration or synchronization assertion."}
    control = (run.parent/".dashboard-control").resolve()
    marker = (control/f"{run.name}.manual.json").resolve()
    try:
        if not marker.is_relative_to(control) or not marker.is_file():
            raise ValueError(f"manual marker sidecar missing: {marker}")
        raw = marker.read_bytes()
        report["sources"]["markers"] = {"path": str(marker), "sha256": hashlib.sha256(raw).hexdigest()}
        report["markers"] = json.loads(raw.decode("utf-8"))
        origin, windows = marker_windows(report["markers"])
    except (OSError, ValueError) as exc:
        report["insufficient_reasons"].append(str(exc))
        windows = {}
    if windows:
        frames, cameras = load_streams(run, report)
        for name, (start, end) in windows.items():
            report["phases"][name] = {"inner_start_host_monotonic_ns": start, "inner_end_host_monotonic_ns": end,
                                     "inner_start_elapsed_s": (start-origin)/SECOND, "inner_end_elapsed_s": (end-origin)/SECOND,
                                     **phase_stats(frames, start, end), "camera_samples": camera_samples(cameras, start, end, origin)}
        baseline = report["phases"]["baseline"]["zones"]
        for phase in report["phases"].values():
            for zone, reference in zip(phase["zones"], baseline):
                zone["baseline_minus_phase_mm"] = (reference["median_mm"]-zone["median_mm"]
                                                    if reference["median_mm"] is not None and zone["median_mm"] is not None else None)
    output.mkdir(parents=True, exist_ok=False)
    contact_sheet(run, output, report)
    for name, phase in report["phases"].items():
        if not phase["frame_count"] or not any(zone["median_mm"] is not None for zone in phase["zones"]):
            report["insufficient_reasons"].append(f"{name}: no usable ToF statistics")
        if any(sample["path"] is None for sample in phase["camera_samples"]):
            report["insufficient_reasons"].append(f"{name}: missing camera sample within 500 ms")
    if set(report["phases"]) == set(PHASES) and not report["insufficient_reasons"]:
        report["verdict"] = "NEEDS_RGB_REVIEW"
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = review(args.run, args.output)
        print(json.dumps({"verdict": report["verdict"], "insufficient_reasons": report["insufficient_reasons"],
                          "output": str(args.output.resolve())}))
        return 0 if report["verdict"] == "NEEDS_RGB_REVIEW" else 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
