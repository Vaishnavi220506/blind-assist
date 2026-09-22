"""One consumed saved-score feasibility diagnostic; no model or baseline edits."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from audit_existing_strata_20260920 import tally

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts.local/work/ba-core-workpoint-20260920"
SOURCE = ROOT / "artifacts.local/work/ba-core-transfer-20260920"
T0 = 0.007085703945147101
ARMS = ("raw", "calibrated")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def decide(row, threshold):
    predictions = row["predictions"]
    raw, calibrated = (predictions[a] for a in ARMS)
    return bool(raw["definite_zones"] > 0 or (raw["alert"] and calibrated["score"] >= threshold))


def silence_runs(sequence, flags):
    runs, run = [], []
    for row, flag in zip(sequence, flags):
        if not flag:
            run.append(row["id"])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return {"runs": runs, "max_silent_frames": max(map(len, runs), default=0),
            "max_silent_sampled_s": round(.2 * max(map(len, runs), default=0), 9)}


def core_event_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["layout_relation"] == "INSIDE" and row["truth"]:
            groups[row["clip_id"]].append(row)
    assert len(groups) == 12
    for sequence in groups.values():
        sequence.sort(key=lambda r: r["time_s"])
        assert len(sequence) == 6
        assert all(abs(b["time_s"] - a["time_s"] - .2) < 1e-6
                   for a, b in zip(sequence, sequence[1:]))
    return groups


def required_threshold(rows):
    constrained = [r for r in rows if r["predictions"]["raw"]["definite_zones"] == 0]
    threshold = min((r["predictions"]["calibrated"]["score"] for r in constrained), default=1.)
    assert T0 <= threshold <= 1.
    return {"threshold": threshold, "required_frames": len(rows),
            "bypass_frames": len(rows) - len(constrained),
            "binding_frames": [{"id": r["id"], "clip_id": r["clip_id"],
                                "time_s": r["time_s"], "score": threshold}
                               for r in constrained if r["predictions"]["calibrated"]["score"] == threshold]}


def measure(rows, threshold, include_events=True):
    flag = lambda r: decide(r, threshold)
    masks = {"all432": lambda r: True,
             "core288": lambda r: r["layout_relation"] != "BOUNDARY",
             "outside144": lambda r: r["layout_relation"] == "OUTSIDE",
             "inside144": lambda r: r["layout_relation"] == "INSIDE",
             "inside_negative72": lambda r: r["layout_relation"] == "INSIDE" and not r["truth"],
             "boundary144": lambda r: r["layout_relation"] == "BOUNDARY"}
    metrics = {name: tally(rows, flag, mask, .2, "clip_id") for name, mask in masks.items()}
    events = []
    for clip, sequence in sorted(core_event_rows(rows).items()):
        flags = [flag(r) for r in sequence]
        old = [r["predictions"]["calibrated"]["alert"] for r in sequence]
        first = next((r["time_s"] for r, yes in zip(sequence, flags) if yes), None)
        baseline_first = next((r["time_s"] for r, yes in zip(sequence, old) if yes), None)
        events.append({"clip_id": clip, "positive_frames": len(sequence),
                       "alerted_frames": sum(flags), "alert_fraction": sum(flags) / len(sequence),
                       "first_alert_s": first, "baseline_first_alert_s": baseline_first,
                       "first_alert_retained": first is not None and first == baseline_first,
                       "lost_positive_ids": [r["id"] for r, before, after in zip(sequence, old, flags) if before and not after],
                       **silence_runs(sequence, flags)})
    lost = [r["id"] for r in rows if r["truth"] and r["predictions"]["calibrated"]["alert"] and not flag(r)]
    result = {"threshold": threshold, "metrics": metrics,
              "core_first_alerts_retained": all(e["first_alert_retained"] for e in events),
              "core_max_silent_frames": max(e["max_silent_frames"] for e in events),
              "core_max_silent_sampled_s": max(e["max_silent_sampled_s"] for e in events),
              "core_min_event_alert_fraction": min(e["alert_fraction"] for e in events),
              "core_events": events,
              "all_lost_positive_ids": lost,
              "clip_first_alert_s": {clip: min((r["time_s"] for r in rows if r["clip_id"] == clip and flag(r)), default=None)
                                     for clip in sorted({r["clip_id"] for r in rows})}}
    for name, mask in masks.items():
        result["metrics"][name].pop("zero_return_frames")
        result["metrics"][name]["prediction_unknown"] = sum(r["predictions"]["calibrated"]["unknown"] for r in rows if mask(r))
    if not include_events:
        for value in metrics.values():
            value.pop("events")
        for key in ("core_events", "all_lost_positive_ids", "clip_first_alert_s"):
            result.pop(key)
    return result


def selfcheck():
    def row(i, score, definite=0, raw=True):
        return {"id": str(i), "predictions": {"raw": {"definite_zones": definite, "alert": raw},
                                               "calibrated": {"score": score}}}
    assert decide(row(0, .5), .5) and not decide(row(0, .5), math.nextafter(.5, math.inf))
    assert decide(row(0, 0., 1), 1.)
    assert not decide(row(0, 1., raw=False), .5)
    assert silence_runs([row(i, 0.) for i in range(6)], [True, False, True, False, False, False])["max_silent_sampled_s"] == .6
    assert silence_runs([row(i, 0.) for i in range(6)], [False] * 6)["max_silent_sampled_s"] == 1.2


def main():
    selfcheck()
    protocol = read(OUT / "protocol.json")
    for path, expected in protocol["input_sha256"].items():
        assert sha(ROOT / path) == expected, path
    write_new("implementation-seal.json", {"time_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("audit_existing_strata_20260920.py")]},
        "protocol_sha256": sha(OUT / "protocol.json"), "synthetic_invariants_pass": True})
    rows = read(SOURCE / "frame-results.json")
    pred = {r["id"]: r for r in read(SOURCE / "predictions.json")}
    seal = read(SOURCE / "prediction-seal.json")
    assert sha(SOURCE / "predictions.json") == seal["hashes"]["predictions.json"]
    assert len(rows) == len(pred) == 432
    for row in rows:
        assert row["predictions"] == pred[row["id"]]["predictions"]
        cal = row["predictions"]["calibrated"]
        assert math.isfinite(cal["score"]) and 0 <= cal["score"] <= 1
        assert cal["threshold"] == T0 and decide(row, T0) == cal["alert"]
        assert cal["unknown"] == (row["predictions"]["raw"]["definite_zones"] == 0)
    groups = core_event_rows(rows)
    firsts = [next(r for r in seq if r["predictions"]["calibrated"]["alert"]) for seq in groups.values()]
    positives = [r for seq in groups.values() for r in seq if r["predictions"]["calibrated"]["alert"]]
    constraints = {"event_onset_ceiling": required_threshold(firsts),
                   "full_core_frame_control": required_threshold(positives)}
    roles = {"baseline": measure(rows, T0)}
    for name, constraint in constraints.items():
        roles[name] = measure(rows, constraint["threshold"])
        assert roles[name]["core_first_alerts_retained"]
        if constraint["threshold"] < 1:
            stricter = math.nextafter(constraint["threshold"], math.inf)
            required = firsts if name == "event_onset_ceiling" else positives
            assert any(not decide(r, stricter) for r in required)
    assert roles["full_core_frame_control"]["metrics"]["core288"]["FN"] == 0
    baseline = roles["baseline"]["metrics"]
    assert [baseline["core288"][k] for k in ("TP", "FP", "FN", "false_segments")] == [72, 126, 0, 24]
    bypass = {}
    for layout in ("INSIDE", "OUTSIDE", "BOUNDARY"):
        fp = [r for r in rows if r["layout_relation"] == layout and not r["truth"] and decide(r, T0)]
        definite = [r["id"] for r in fp if r["predictions"]["raw"]["definite_zones"] > 0]
        bypass[layout] = {"baseline_FP": len(fp), "definite_bypass_FP": len(definite),
                          "score_gated_FP": len(fp) - len(definite), "definite_ids": definite}
    thresholds = {T0, 1.}
    for r in rows:
        score = r["predictions"]["calibrated"]["score"]
        for t in (score, math.nextafter(score, math.inf)):
            if T0 <= t <= 1:
                thresholds.add(t)
    curve, seen = [], set()
    for threshold in sorted(thresholds):
        bits = tuple(decide(r, threshold) for r in rows)
        if bits not in seen:
            seen.add(bits)
            curve.append(measure(rows, threshold, include_events=False))
    feasible = [r for r in curve if r["core_first_alerts_retained"]]
    optimum = roles["event_onset_ceiling"]["metrics"]["core288"]["FP"]
    assert min(r["metrics"]["core288"]["FP"] for r in feasible) == optimum
    decision_rows = [{"id": r["id"], "clip_id": r["clip_id"], "layout_relation": r["layout_relation"],
                      "truth": r["truth"], "score": r["predictions"]["calibrated"]["score"],
                      "definite": r["predictions"]["raw"]["definite_zones"] > 0,
                      "flags": {name: decide(r, result["threshold"]) for name, result in roles.items()}}
                     for r in rows]
    for path, expected in protocol["input_sha256"].items():
        assert sha(ROOT / path) == expected, path
    write_new("results.json", {"authority": "CONSUMED_CORE432_WORKPOINT_DIAGNOSTIC_NOT_TRANSFER",
        "constraints": constraints, "bypass": bypass, "roles": roles,
        "distinct_decision_sets": len(curve), "feasible_decision_sets": len(feasible),
        "curve": curve, "original_inputs_unchanged": True,
        "protocol_sha256": sha(OUT / "protocol.json"), "backend": "TASK_NOT_GPU_SUITABLE"})
    write_new("decisions.json", decision_rows)
    for name, result in roles.items():
        print(json.dumps({"role": name, "threshold": result["threshold"],
            "Core": {k: result["metrics"]["core288"][k] for k in ("TP", "FP", "FN", "false_segments", "false_sampled_s")},
            "max_silence_s": result["core_max_silent_sampled_s"]}))
    print(json.dumps({"bypass": bypass, "decision_sets": len(curve)}))


if __name__ == "__main__":
    main()
