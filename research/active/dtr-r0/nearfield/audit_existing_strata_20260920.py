"""Recount saved decisions under explicit reporting strata; never run a model.

This is posthoc, consumed-evidence accounting, not a new experiment or a new
operating point. Only standard-library JSON/hash/scalar work is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "artifacts.local/work"
INPUTS: dict[str, str] = {}
BODY_HEAD = {"substantial_body", "suspended_head"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = WORK / relative
    INPUTS[relative] = digest(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sealed_predictions(folder):
    seal = read(f"{folder}/prediction-seal.json")
    rows = read(f"{folder}/predictions.json")
    expected = seal.get("predictions_sha256") or seal["hashes"]["predictions.json"]
    assert INPUTS[f"{folder}/predictions.json"] == expected, folder
    result = {r["id"]: r for r in rows}
    assert len(result) == len(rows)
    return result


def tally(rows, flag, select, dt, episode_key="episode_id"):
    """Masks retain the original timeline: excluded samples break segments."""
    selected = [r for r in rows if select(r)]
    tp = sum(bool(flag(r)) and r["truth"] for r in selected)
    fp = sum(bool(flag(r)) and not r["truth"] for r in selected)
    pos = sum(r["truth"] for r in selected)
    neg = len(selected) - pos
    groups = defaultdict(list)
    for r in rows:
        groups[r[episode_key]].append(r)
    events, segments = [], 0
    for episode, sequence in sorted(groups.items()):
        sequence.sort(key=lambda r: r["time_s"])
        event, previous_fp, previous_alert = None, False, False
        previous_time = None
        for i, r in enumerate(sequence):
            if previous_time is not None:
                assert abs(r["time_s"] - previous_time - dt) < 1e-6
            active = bool(select(r))
            alert = bool(flag(r))
            false_alert = active and not r["truth"] and alert
            segments += int(false_alert and not previous_fp)
            positive = active and r["truth"]
            if positive and event is None:
                event = {"episode": episode, "entry_s": r["time_s"],
                         "left_censored": i == 0,
                         "preexisting_alert": i > 0 and previous_alert and alert,
                         "first_alert_s": None, "positive_frames": 0}
            if positive:
                event["positive_frames"] += 1
                if alert and event["first_alert_s"] is None:
                    event["first_alert_s"] = r["time_s"]
            elif event is not None:
                events.append(event)
                event = None
            previous_fp, previous_alert, previous_time = false_alert, alert, r["time_s"]
        if event is not None:
            events.append(event)
    for event in events:
        first = event["first_alert_s"]
        event["delay_s"] = None if first is None else round(first - event["entry_s"], 9)
    delays = [e["delay_s"] for e in events if e["delay_s"] is not None]
    return {"frames": len(selected), "positive": pos, "negative": neg,
            "TP": tp, "FP": fp, "FN": pos - tp,
            "silent_negative": neg - fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / pos if pos else None,
            "FPR": fp / neg if neg else None,
            "f1": 2 * tp / (tp + fp + pos) if tp + fp + pos else None,
            "false_segments": segments, "false_sampled_s": round(fp * dt, 9),
            "event_count": len(events), "events_detected": len(delays),
            "first_alert_max_delay_s": max(delays) if delays else None,
            "left_censored_events": sum(e["left_censored"] for e in events),
            "preexisting_alert_events": sum(e["preexisting_alert"] for e in events),
            "zero_return_frames": (sum(r["usable_tof_returns"] == 0 for r in selected)
                                   if selected and all("usable_tof_returns" in r for r in selected)
                                   else None),
            "events": events}


def match_counts(actual, expected):
    for key in ("TP", "FP", "FN"):
        assert actual[key] == expected[key], (key, actual[key], expected[key])


def partition_check(full, parts):
    for key in ("frames", "positive", "negative", "TP", "FP", "FN"):
        assert full[key] == sum(p[key] for p in parts), key


def old_masks():
    return {"all": lambda r: True,
            "clear": lambda r: r["stratum"] != "boundary",
            "body_head_clear": lambda r: r["stratum"] != "boundary" and r["family"] in BODY_HEAD,
            "rod_clear": lambda r: r["stratum"] != "boundary" and r["family"] == "near_rod_farwall",
            "boundary": lambda r: r["stratum"] == "boundary"}


def paired_ids(rows, baseline, candidate, select):
    out = {"TP_lost": [], "FN_rescued": [], "FP_added": [], "FP_removed": []}
    for r in rows:
        if not select(r) or bool(baseline(r)) == bool(candidate(r)):
            continue
        key = ("TP_lost" if baseline(r) else "FN_rescued") if r["truth"] else (
            "FP_removed" if baseline(r) else "FP_added")
        out[key].append(r["id"])
    return out


def audit_singleconfirm():
    folder = "corridor-public-single-20260917/confirmation"
    rows = read(f"{folder}/cases.json")
    assert len(rows) == 288 and len({r["id"] for r in rows}) == 288
    saved = sealed_predictions(folder)
    for r in rows:
        for k, value in saved[r["id"]].items():
            assert r[k] == value, (r["id"], k)
    summaries = {"confirmation": read(f"{folder}/summary.json")}
    flags = {"A": lambda r: r["A"], "A_plus_public": lambda r: r["alert"],
             "A_star": lambda r: r["control"]}
    for source, names in [("corridor-representation-20260918", {"raw": "raw", "single": "single", "multi": "multi"}),
                          ("corridor-ccrl-20260918", {"bce": "bce_residual", "rank": "ranking_residual", "A_star": "ccrl_baseline"})]:
        predictions = sealed_predictions(source)
        cases = {r["id"]: r for r in read(f"{source}/cases.json")}
        assert set(cases) == set(saved) == set(predictions)
        for r in rows:
            for key in ("truth", "stratum", "family", "episode_id", "time_s", "control"):
                assert cases[r["id"]][key] == r[key]
        summaries[source] = read(f"{source}/summary.json")
        for src, dst in names.items():
            flags[dst] = lambda r, p=predictions, k=src: p[r["id"]]["flags"][k]
    masks = old_masks()
    output = {name: {s: tally(rows, flag, mask, .25) for s, mask in masks.items()}
              for name, flag in flags.items()}
    for name, result in output.items():
        partition_check(result["all"], [result[s] for s in ("body_head_clear", "rod_clear", "boundary")])
    for name, src in [("A", "A"), ("A_plus_public", "A_plus_public"), ("A_star", "A_retrained")]:
        for dst, old in [("all", "strict"), ("clear", "clear"), ("boundary", "boundary")]:
            match_counts(output[name][dst], summaries["confirmation"]["methods"][src][old])
    for folder, mapping in [("corridor-representation-20260918", {"raw": "raw", "single": "single", "multi": "multi"}),
                            ("corridor-ccrl-20260918", {"bce_residual": "bce", "ranking_residual": "rank", "ccrl_baseline": "A_star"})]:
        for dst, src in mapping.items():
            old = summaries[folder]["methods"][src]
            for s, key in [("all", "strict"), ("clear", "clear"), ("boundary", "boundary"), ("rod_clear", "near_rod_farwall")]:
                match_counts(output[dst][s], old["metrics"][key])
            assert output[dst]["clear"]["false_segments"] == old["core"]["clear_negative_alert_segments"]
            assert output[dst]["clear"]["events_detected"] == old["core"]["core_events_detected"]
    for r in rows:
        assert flags["A_star"](r) == flags["multi"](r) == flags["ccrl_baseline"](r)
    changes = {name: paired_ids(rows, flags["A_star"], flag, masks["body_head_clear"])
               for name, flag in flags.items()}
    return {"methods": output, "body_head_changes_vs_A_star": changes,
            "unique_groups": len({r["group"] for r in rows}),
            "body_head_groups": len({r["group"] for r in rows if masks["body_head_clear"](r)})}


def audit_v2():
    folder = "corridor-public-positive-v2-20260917/bce"
    rows = read(f"{folder}/cases.json")
    summary = read(f"{folder}/summary.json")
    assert len(rows) == 1152 and len({r["id"] for r in rows}) == 1152
    output = {}
    for cohort, old in summary["cohorts"].items():
        subset = [r for r in rows if r["cohort"] == cohort]
        output[cohort] = {}
        for arm in ("A", "calibrated"):
            result = {s: tally(subset, lambda r, k=arm: r[k], mask, .25)
                      for s, mask in old_masks().items()}
            for new, prev in [("all", "strict"), ("clear", "clear"), ("boundary", "boundary")]:
                match_counts(result[new], old[arm][prev])
            partition_check(result["all"], [result[s] for s in ("body_head_clear", "rod_clear", "boundary")])
            output[cohort][arm] = result
        output[cohort]["body_head_changes"] = paired_ids(subset, lambda r: r["A"],
            lambda r: r["calibrated"], old_masks()["body_head_clear"])
    return output


def audit_camera():
    folder = "ba-core-transfer-20260920"
    rows = read(f"{folder}/frame-results.json")
    summary = read(f"{folder}/results.json")
    saved = sealed_predictions(folder)
    assert len(rows) == 432 and {r["id"] for r in rows} == set(saved)
    for r in rows:
        assert r["predictions"] == saved[r["id"]]["predictions"]
    masks = {"all": lambda r: True,
             "inside_outside_layout": lambda r: r["layout_relation"] != "BOUNDARY",
             "boundary_layout": lambda r: r["layout_relation"] == "BOUNDARY",
             "geometry_nonboundary": lambda r: r["relation"] != "BOUNDARY",
             "geometry_boundary": lambda r: r["relation"] == "BOUNDARY"}
    for layout in ("INSIDE", "BOUNDARY", "OUTSIDE"):
        masks[f"layout_{layout}"] = lambda r, s=layout: r["layout_relation"] == s
    output = {}
    for arm in ("raw", "calibrated", "no_rgb", "rgb"):
        output[arm] = {}
        for name, mask in masks.items():
            result = tally(rows, lambda r, a=arm: r["predictions"][a]["alert"], mask, .2, "clip_id")
            result.pop("zero_return_frames")
            result["prediction_unknown"] = sum(r["predictions"][arm]["unknown"] for r in rows if mask(r))
            output[arm][name] = result
        for new, old in [("all", "all_known"), ("geometry_nonboundary", "interior"), ("geometry_boundary", "boundary")]:
            match_counts(output[arm][new], summary["metrics"]["arms"][arm]["frames"][old])
        old = summary["metrics"]["arms"][arm]
        assert output[arm]["all"]["false_segments"] == old["false_alert_segment_count"]
        prior_events = {(e["clip_id"], e["entry_time_s"]): e for e in old["events"]}
        assert len(prior_events) == output[arm]["all"]["event_count"]
        for event in output[arm]["all"]["events"]:
            prior = prior_events[(event["episode"], event["entry_s"])]
            assert event["first_alert_s"] == prior["first_alert_time_s"]
            assert event["preexisting_alert"] == prior["preexisting_alert_at_entry"]
            assert event["left_censored"] == prior["left_censored"]
        partition_check(output[arm]["all"], [output[arm][s] for s in ("inside_outside_layout", "boundary_layout")])
        partition_check(output[arm]["all"], [output[arm][s] for s in ("geometry_nonboundary", "geometry_boundary")])
    return output


def summary_only_controls():
    output = {}
    for folder, arms in [("corridor-tail-rescue-20260918", ["A_star", "tail_rescue"]),
                         ("corridor-pixel-query-20260918", ["multi", "fixed", "query"])]:
        data = read(f"{folder}/summary.json")
        output[folder] = {}
        for arm in arms:
            metrics = data["methods"][arm]["metrics"]
            core = {k: sum(metrics[f][k] for f in BODY_HEAD) for k in ("frames", "positive", "negative", "TP", "FP", "FN")}
            output[folder][arm] = {"body_head_clear": core, "rod_clear": metrics["near_rod_farwall"], "boundary": metrics["boundary"]}
            partition_check(metrics["strict"], [core, metrics["near_rod_farwall"], metrics["boundary"]])
    return output


def audit_nfo_area():
    area = read("ba-nfo-area-20260919/results.json")
    matched = read("ba-nfo-matched-20260919/results.json")
    records = [r for r in area["records"] if r["return_state"] == "all"]
    for arm in ("depth", "nfo"):
        for distance in ("1.0", "1.5", "2.0", "3.0"):
            bins = [r for r in records if r["arm"] == arm and r["threshold"] == distance]
            assert len(bins) == 5
            original = matched["metrics"][arm][distance]["mixed"]
            for key in ("tp", "fp", "fn", "tn", "pixels"):
                assert sum(r[key] for r in bins) == original[key]
            for row in bins:
                assert sum(row[k] for k in ("tp", "fp", "fn", "tn")) == row["pixels"]
                assert abs(row["iou"] - row["tp"] / (row["tp"] + row["fp"] + row["fn"])) < 1e-12
    return {"area_bins": records, "physical_core_mapping": "NOT_EVALUABLE",
            "reason": "zone near-area fraction is not physical object size or BODY/HEAD event truth"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"authority": "POSTHOC_CONSUMED_SAVED_DECISION_RECOUNT",
              "backend": "TASK_NOT_GPU_SUITABLE: standard-library scalar and hash audit",
              "silent_negative_is_not_clear_space": True,
              "singleconfirm": audit_singleconfirm(), "v2_bce": audit_v2(),
              "camera_transfer": audit_camera(), "summary_only_controls": summary_only_controls(),
              "nfo_area": audit_nfo_area()}
    for relative, before in INPUTS.items():
        assert digest(WORK / relative) == before, relative
    result["source_sha256"] = INPUTS
    result["audit_script_sha256"] = digest(Path(__file__))
    result["checks"] = "PASS: seals, identity joins, partitions, old counts, segment/event parity, unchanged inputs"
    target = args.output.resolve()
    assert target.is_relative_to(WORK.resolve()), "Output belongs under artifacts.local/work"
    assert not target.exists(), "Refuse to overwrite an existing audit receipt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "input_files": len(INPUTS), "output": str(args.output)}))


if __name__ == "__main__":
    main()
