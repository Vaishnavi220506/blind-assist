"""Fixed non-veto union of sealed point/shape predictions; no source or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
DATE = "20260921"
BRIEF = ROOT / f"research/active/dtr-r0/nearfield/SHAPE_SUPPORT_UNION_RESULTS_{DATE}.md"
SOURCE = ROOT / "artifacts.local/work/ba-shape-hypotheses-20260921/run-v1"
STATES = {"IN", "OUT_MODEL_PRIOR", "UNKNOWN"}
RULE = "IN if point_proxy == IN or consensus == IN; else consensus (OUT_MODEL_PRIOR or UNKNOWN)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def union_state(point: str, shape: str) -> str:
    if point not in {"IN", "UNKNOWN"} or shape not in STATES:
        raise ValueError("Unexpected public point or model-prior state")
    return "IN" if point == "IN" or shape == "IN" else shape


def summarize(rows, arm):
    c = Counter((r["truth_inside"], r[arm]) for r in rows)
    tp, fp = c[True, "IN"], c[False, "IN"]
    false_out, correct_out = c[True, "OUT_MODEL_PRIOR"], c[False, "OUT_MODEL_PRIOR"]
    up, un = c[True, "UNKNOWN"], c[False, "UNKNOWN"]
    p, n = tp + false_out + up, fp + correct_out + un
    assert tp + fp + false_out + correct_out + up + un == len(rows)
    return dict(n=len(rows), positive=p, negative=n, TP=tp, FP=fp,
                FN_alert=false_out + up, false_OUT=false_out, correct_OUT=correct_out,
                UNKNOWN_positive=up, UNKNOWN_negative=un, UNKNOWN=up + un,
                coverage=(len(rows) - up - un) / len(rows),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / p if p else None, FPR=fp / n if n else None)


def contrast(rows, base):
    added = [r for r in rows if r["union"] == "IN" and r[base] != "IN"]
    lost = [r for r in rows if r["union"] != "IN" and r[base] == "IN"]
    retained = [r for r in rows if r["union"] == r[base] == "IN"]
    return dict(added_TP=sum(r["truth_inside"] for r in added),
                added_FP=sum(not r["truth_inside"] for r in added),
                retained_TP=sum(r["truth_inside"] for r in retained),
                retained_FP=sum(not r["truth_inside"] for r in retained),
                lost_IN=len(lost), added_ids=[r["case_id"] for r in added],
                lost_ids=[r["case_id"] for r in lost])


def run(output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite the saved-output readout")
    output.mkdir(parents=True)
    started = time.perf_counter()
    old_seal, old_completion = read(SOURCE / "prediction-seal.json"), read(SOURCE / "completion.json")
    assert sha(SOURCE / "predictions.json") == old_seal["predictions_sha256"] == old_completion["predictions_sha256"]
    assert sha(SOURCE / "results.json") == old_completion["results_sha256"]
    files = ["prediction-seal.json", "completion.json", "predictions.json", "results.json"]
    old_hashes = {name: sha(SOURCE / name) for name in files}
    code = Path(__file__).resolve()
    shutil.copyfile(code, output / "source.py")
    shutil.copyfile(BRIEF, output / "brief-before-readout.md")
    write(output / "freeze.json", dict(rule=RULE, source_sha256=sha(code),
          brief_sha256=sha(BRIEF), source_artifact=str(SOURCE.resolve()),
          source_bindings=old_hashes, expected_cases=174,
          backend="CPU Python standard library", backend_reason="TASK_NOT_GPU_SUITABLE",
          scientific_settings="saved outputs only; no fitting, rendering, inference, or policy search",
          consumed_discovery="Six shape-only and eight point-only positives were already disclosed before proposing this readout",
          time_ns=time.time_ns()))
    parent_predictions = read(SOURCE / "predictions.json")
    assert len(parent_predictions) == len({r["case_id"] for r in parent_predictions}) == 174
    predictions = [dict(case_id=r["case_id"], point_proxy=r["point_proxy"],
                        consensus=r["consensus"], union=union_state(r["point_proxy"], r["consensus"]))
                   for r in parent_predictions]
    assert all(r["union"] == "IN" for r in predictions if r["point_proxy"] == "IN")
    write(output / "predictions.json", predictions)
    write(output / "prediction-seal.json", dict(time_ns=time.time_ns(),
          phase="NEW_RULE_PREDICTIONS_SEALED_BEFORE_CONSUMED_EVALUATOR_JOIN",
          predictions_sha256=sha(output / "predictions.json"), freeze_sha256=sha(output / "freeze.json"),
          source_predictions_sha256=old_hashes["predictions.json"]))
    # Only after the new prediction seal, parse the old evaluator's truth/strata.
    old_cases = {r["case_id"]: r for r in read(SOURCE / "results.json")["cases"]}
    assert set(old_cases) == {r["case_id"] for r in predictions}
    rows = []
    for p in predictions:
        prior = old_cases[p["case_id"]]
        assert type(prior["truth_inside"]) is bool
        assert p["point_proxy"] == prior["point_proxy"] and p["consensus"] == prior["consensus"]
        rows.append(dict(**p, truth_inside=prior["truth_inside"], group=prior["group"], slice=prior["slice"]))
    strata = ["all", "closed_world", "appendage_thin", "appendage_visible", "off_grid",
              "closed_near_0p30", "closed_other"]
    grouped = {}
    for name in strata:
        subset = [r for r in rows if name == "all" or name == r["group"] or name == r["slice"]]
        grouped[name] = dict(metrics={arm: summarize(subset, arm) for arm in ["point_proxy", "consensus", "union"]},
                             contrasts={base: contrast(subset, base) for base in ["point_proxy", "consensus"]})
    change = grouped["all"]["contrasts"]["point_proxy"]
    supported = change["added_TP"] > 0 and change["added_FP"] == 0 and change["lost_IN"] == 0
    decision = "COMPONENT_CONSUMED_SYNTHETIC_NONVETO_GAIN" if supported else "NO_USEFUL_INCREMENT_UNDER_FIXED_OR"
    result = dict(decision=decision, strata=grouped, cases=rows,
                  model_prior_false_out_ids=[r["case_id"] for r in rows if r["truth_inside"] and r["union"] == "OUT_MODEL_PRIOR"],
                  classification_boundary="OUT_MODEL_PRIOR is not sensor-certified clear space",
                  shape_out_overridden_ids=[r["case_id"] for r in rows if r["consensus"] == "OUT_MODEL_PRIOR" and r["point_proxy"] == "IN"],
                  seconds=time.perf_counter() - started)
    write(output / "results.json", result)
    write(output / "local-disposition.json", dict(terminal_id=f"ba-shape-support-union-{DATE}",
          role="COMPONENT_OR_CHALLENGER" if supported else "NEGATIVE_CONTROL", mode="COMPONENT",
          decision=decision, scope="Single non-veto OR of consumed toy point/shape predictions; not runtime or independent evidence",
          global_registration="PENDING_SUPPORTED_COMMAND", source_not_rerun=True))
    assert {name: sha(SOURCE / name) for name in files} == old_hashes
    write(output / "completion.json", dict(status="PASS", old_source_hashes_unchanged=True,
          results_sha256=sha(output / "results.json"), prediction_seal_sha256=sha(output / "prediction-seal.json"),
          persistent_resources=[]))
    print(json.dumps(dict(decision=decision, overall=grouped["all"],
                         false_out_ids=result["model_prior_false_out_ids"], seconds=result["seconds"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    allowed = (ROOT / f"artifacts.local/work/ba-shape-support-union-{DATE}").resolve()
    if allowed not in args.output.resolve().parents:
        parser.error("Choose a new child output below the canonical shape-support-union artifact tree")
    run(args.output)
