"""Bounded terminal matrix display, while preserving the same capture evidence."""
import argparse
import json
import sys
import time

from capture import capture, capture_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    capture_arguments(parser)
    args = parser.parse_args()
    last = [0.0]

    def show(record):
        if time.monotonic() - last[0] < 0.5:
            return
        last[0] = time.monotonic()
        frame, data = record["sensor"], record["derived"]
        print(f"\n{frame['type']} seq={frame['seq']} sensor_ms={frame['ms']} / mm; -- = UNKNOWN")
        cells = data["distance_known_mm"]
        for start in range(0, len(cells), data["cols"]):
            print(" ".join(f"{v:5d}" if v is not None else "   --" for v in cells[start:start+data["cols"]]))

    try:
        result = capture(args, display=show)
        print(json.dumps(result, indent=2))
        return 0 if result["result"] == "FRAMES_RECORDED" else 2
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
