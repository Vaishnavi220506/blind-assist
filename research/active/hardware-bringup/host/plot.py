"""Plot validated CNH capture medians against bin index, not calibrated distance."""
import argparse
import json
from pathlib import Path
import statistics

from capture import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frames.jsonl produced by capture/replay")
    parser.add_argument("--output", type=Path, required=True, help="new PNG file; must not exist")
    parser.add_argument("--label", default="Unlabeled supplied capture", help="visible provenance/scene label; use SYNTHETIC for synthetic fixtures")
    args = parser.parse_args()
    histograms = []
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            sensor = record.get("sensor", record)
            if sensor.get("type") == "cnh_frame":
                histograms.append(validate_frame(sensor)["hist_normalized"])
    if not histograms:
        parser.error("no valid CNH frames; cannot plot range frames or failed initialization")
    if args.output.suffix.lower() != ".png":
        parser.error("output must be a new .png file")
    if args.output.exists():
        parser.error("output exists; refusing to replace evidence")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, layout="constrained")
    for zone, axis in enumerate(axes.flat):
        values = [statistics.median(frame[zone][b] for frame in histograms) for b in range(24)]
        axis.plot(range(24), values)
        axis.set_title(f"Zone {zone}")
        axis.grid(alpha=0.2)
        if zone >= 12:
            axis.set_xlabel("CNH bin index")
        if zone % 4 == 0:
            axis.set_ylabel("Normalized return")
    fig.suptitle(f"{args.label} | CNH median over {len(histograms)} supplied frames\nHardware CNH validation pending; bin axis is not calibrated distance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as image:
            fig.savefig(image, format="png", dpi=150)
    finally:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
