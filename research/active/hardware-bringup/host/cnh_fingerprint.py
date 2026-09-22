"""Fixed retrospective CNH fingerprint diagnostic; no device I/O or distance output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture import validate_frame
from manual_orientation_review import marker_windows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(rows, reference):
    rows, reference = np.asarray(rows, dtype=float), np.asarray(reference, dtype=float)
    denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(reference)
    result = np.full(len(rows), np.nan)
    np.divide(rows @ reference, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def pairwise_auc(background, changed):
    differences = np.asarray(background)[:, None] - np.asarray(changed)[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def summarize(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"count": len(values), "finite": len(finite),
            "minimum": float(np.min(finite)) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "maximum": float(np.max(finite)) if len(finite) else None}


def evaluate_contrast(background, changed, background_blocks, changed_blocks, protocol):
    blocks = [background[background_blocks == b] for b in range(4)] + [
        changed[changed_blocks == b] for b in range(4)]
    if any(len(v) < protocol["minimum_frames_per_block"] or not np.isfinite(v).all() for v in blocks):
        return {"verdict": "NOT_EVALUABLE", "reason": "missing/nonfinite/underpopulated one-second block"}
    b_medians, c_medians = [float(np.median(v)) for v in blocks[:4]], [float(np.median(v)) for v in blocks[4:]]
    gap = min(b_medians) - max(c_medians)
    auc = pairwise_auc(background, changed)
    passed = gap >= protocol["minimum_block_separation"] and auc >= protocol["minimum_pairwise_auc"]
    return {"verdict": "SUPPORTED_FOR_RETROSPECTIVE_FINGERPRINT" if passed else "NOT_SUPPORTED",
            "pairwise_auc_descriptive": auc, "worst_block_separation": gap,
            "background_block_medians": b_medians, "changed_block_medians": c_medians}


def load_inputs(root, protocol):
    run = root / protocol["run"]
    marker_path, rgb_path = root / protocol["markers"], root / protocol["rgb_adjudication"]
    _, windows = marker_windows(json.loads(marker_path.read_text(encoding="utf-8")))
    rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
    if rgb.get("verdict") != "VERTICAL_AXIS_SUPPORTED_FOR_CURRENT_ASSEMBLY":
        raise ValueError("Required prior RGB review is absent; no scene-label inference permitted")
    phases = {name: [] for name in windows}
    source = run / "tof/frames.jsonl"
    for line, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(text)
        stamp = record["host_received_monotonic_ns"]
        for phase, (start, end) in windows.items():
            if start <= stamp < end:
                sensor = record["sensor"]
                derived = validate_frame(sensor)
                phases[phase].append({"line": line, "stamp": stamp, "seq": sensor["seq"],
                    "block": (stamp-start)//1_000_000_000, "sensor": sensor,
                    "hist": np.asarray(derived["hist_normalized"], dtype=float),
                    "valid": np.asarray(derived["range_valid"], dtype=bool)})
    if any(not rows for rows in phases.values()):
        raise ValueError("Missing fixed phase window")
    paths = [source, marker_path, rgb_path, run / "tof/raw.bin", run / "camera/serial.bin"]
    return phases, {str(path.relative_to(root)): digest(path) for path in paths}


def make_plot(output, phases, report, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [protocol[k] for k in ("background_test_phase", "reference_phase", "changed_scene_phase")]
    colors = {names[0]: "#337c9c", names[1]: "#378459", names[2]: "#c66a36"}
    zone = protocol["primary_zone"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("CNH information under scalar UNKNOWN - retrospective, same-rig diagnostic\n"
                 "Later reference; consumed windows; no metric-distance recovery or online claim", fontsize=13)
    for name in names:
        hist = np.stack([r["hist"][zone] for r in phases[name]])
        median = np.median(hist, axis=0)
        low, high = np.quantile(hist, [.1, .9], axis=0)
        axes[0, 0].plot(range(24), median, label=name, color=colors[name])
        axes[0, 0].fill_between(range(24), low, high, alpha=.13, color=colors[name])
    axes[0, 0].set(title=f"Zone {zone}: signed CNH median, 10-90% band", xlabel="Raw bin (not a calibrated distance)", ylabel="Normalized sensor units")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=.2)
    for ax, arm in zip((axes[0, 1], axes[1, 0]), protocol["arms"]):
        for index, name in enumerate(names):
            values = report["zones"][str(zone)]["arms"][arm]["phases"][name]["scores"]
            ax.scatter(np.full(len(values), index)+np.linspace(-.1, .1, len(values)), values,
                       color=colors[name], s=16, alpha=.6)
        ax.set(xticks=range(3), xticklabels=names, ylim=(-.05, 1.05), ylabel="Cosine similarity to UP reference", title=arm)
        ax.grid(alpha=.2)
    scalar = report["zones"][str(zone)]["scalar"]
    axes[1, 1].axis("off")
    primary = report["zones"][str(zone)]["arms"][protocol["primary_arm"]]["contrast"]
    lines = ["Primary result: " + report["verdict"], "", "Scalar baseline (unchanged rule):"]
    for name in names:
        s = scalar[name]
        lines.append(f"{name}: KNOWN {s['known']}/{s['frames']}, raw median {s['raw_mm_median']:.1f} mm")
    lines += ["", f"Descriptive pairwise AUC: {primary.get('pairwise_auc_descriptive', float('nan')):.4f}",
              f"Worst one-second block gap: {primary.get('worst_block_separation', float('nan')):.4f}",
              "", "Adjacent frames are dependent. No fitted threshold.",
              "RGB labels global book placement, not exact zone occupancy.",
              "UNKNOWN remains UNKNOWN; no output distance is replaced."]
    axes[1, 1].text(0, 1, "\n".join(lines), va="top", fontsize=10)
    fig.savefig(output / "fingerprint.png", dpi=160)
    plt.close(fig)


def run(root, protocol_path, output):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    phases, sources = load_inputs(root, protocol)
    report = {"schema": protocol["schema"], "protocol_sha256": digest(protocol_path),
              "script_sha256": digest(__file__), "sources": sources, "protocol": protocol,
              "compute": "CPU: small scalar/vector diagnostic; no network training",
              "zones": {}, "verdict": "NOT_EVALUABLE"}
    ref, bg, changed = (protocol[k] for k in ("reference_phase", "background_test_phase", "changed_scene_phase"))
    for zone in [protocol["primary_zone"]] + protocol["context_zones"]:
        details = {"scalar": {}, "arms": {}}
        for name, records in phases.items():
            details["scalar"][name] = {"frames": len(records), "known": sum(bool(r["valid"][zone]) for r in records),
                "unknown": sum(not r["valid"][zone] for r in records),
                "raw_mm_median": float(np.median([r["sensor"]["distance_mm"][zone] for r in records])),
                "sequences": [r["seq"] for r in records]}
        template = np.median(np.stack([r["hist"][zone] for r in phases[ref]]), axis=0)
        details["reference_signed_cnh"] = template.tolist()
        for arm, (start, end) in protocol["arms"].items():
            scores = {name: cosine(np.stack([r["hist"][zone, start:end] for r in rows]), template[start:end])
                      for name, rows in phases.items()}
            contrast = evaluate_contrast(scores[bg], scores[changed],
                np.array([r["block"] for r in phases[bg]]), np.array([r["block"] for r in phases[changed]]), protocol)
            details["arms"][arm] = {"phases": {name: {**summarize(values),
                "scores": [float(v) if np.isfinite(v) else None for v in values]} for name, values in scores.items()},
                "contrast": contrast}
        report["zones"][str(zone)] = details
    primary = report["zones"][str(protocol["primary_zone"]) ]
    report["verdict"] = primary["arms"][protocol["primary_arm"]]["contrast"]["verdict"]
    if primary["scalar"][bg]["known"] != 0:
        report["verdict"] = "NOT_EVALUABLE"
        report["reason"] = "Predeclared entirely-UNKNOWN background prerequisite does not hold"
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    make_plot(output, phases, report, protocol)
    return {"verdict": report["verdict"], "primary": primary["arms"], "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(json.dumps({"verdict": result["verdict"], "output": result["output"],
          "contrasts": {name: arm["contrast"] for name, arm in result["primary"].items()}}, indent=2))
