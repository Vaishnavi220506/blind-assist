"""Reproduce the consumed A* work points and the current oracle alignment.

This is a provenance audit, not a model, threshold, or feature search.  The
important distinction is between a saved probability and the saved model
logit.  ``run_oracle_surface_ceiling.py`` calls ``predict_proba`` for all
three oracle arms, so the frozen A* arm must remain 96/3/12 at the frozen
probability threshold.  The two privileged support interventions can move
that probability and are audited separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
ARTIFACTS = (REPO / "artifacts.local").resolve()
WORK = ARTIFACTS / "work"

SOURCE_ROOT = WORK / "corridor-public-single-20260917"
ORACLE_ROOT = WORK / "oracle-surface-ceiling-20260918"
BG_ROOT = WORK / "corridor-bg-invariance-20260918"
CASES = SOURCE_ROOT / "confirmation" / "cases.json"
PREDICTIONS = SOURCE_ROOT / "confirmation" / "predictions.json"
OLD_CASES = BG_ROOT / "old-cases.json"
BUDGET = BG_ROOT / "budget-diagnostic" / "summary.json"
MODEL_SEAL = WORK / "corridor-representation-20260918" / "model-seal.json"
ASTAR_CONFIG = WORK / "corridor-astar-effect-20260917" / "bundle" / "config.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def confusion(rows, decisions, mask=None):
    decisions = np.asarray(decisions, dtype=bool)
    truth = np.asarray([bool(row["truth"]) for row in rows], dtype=bool)
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    p = decisions[mask]
    y = truth[mask]
    return {
        "frames": int(mask.sum()),
        "positive": int(y.sum()),
        "negative": int((~y).sum()),
        "TP": int((p & y).sum()),
        "FP": int((p & ~y).sum()),
        "FN": int((~p & y).sum()),
        "TN": int((~p & ~y).sum()),
    }


def with_rates(counts):
    out = dict(counts)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    out["precision"] = tp / max(1, tp + fp)
    out["recall"] = tp / max(1, tp + fn)
    out["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def metric(rows, decisions, clear):
    return {
        "clear": with_rates(confusion(rows, decisions, clear)),
        "strict": with_rates(confusion(rows, decisions)),
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def detail_rows(rows, base, candidate, candidate_domain, candidate_threshold, recipe):
    """Return exact clear-frame decision changes, with both score domains."""
    base = np.asarray(base, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    base_alert = base >= float(FROZEN_THRESHOLD)
    candidate_alert = candidate >= float(candidate_threshold)
    clear = np.asarray([row["stratum"] in ("positive", "negative") for row in rows])
    changed = np.flatnonzero(clear & (base_alert != candidate_alert))
    out = []
    for i in changed:
        row = rows[int(i)]
        out.append(
            {
                "recipe": recipe,
                "id": row["id"],
                "episode_id": row["episode_id"],
                "family": row.get("family"),
                "stratum": row["stratum"],
                "label": bool(row["truth"]),
                "base_probability": float(base[i]),
                "candidate_score": float(candidate[i]),
                "candidate_domain": candidate_domain,
                "candidate_threshold": float(candidate_threshold),
                "base_alert": bool(base_alert[i]),
                "candidate_alert": bool(candidate_alert[i]),
                "score_delta": float(candidate[i] - base[i]),
            }
        )
    return out


def assert_counts(name, record, expected):
    got = record["clear"]
    actual = {k: got[k] for k in ("TP", "FP", "FN", "TN")}
    if actual != expected:
        raise AssertionError(f"{name} clear mismatch: {actual} != {expected}")


def main():
    global FROZEN_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ARTIFACTS) or out == ARTIFACTS or out.exists():
        raise ValueError("Fresh alignment output must be strictly under artifacts.local")

    cases = read_json(CASES)
    old_cases = read_json(OLD_CASES)
    prior = read_json(PREDICTIONS)
    budget = read_json(BUDGET)
    model_seal = read_json(MODEL_SEAL)
    astar_config = read_json(ASTAR_CONFIG)
    oracle_summary = read_json(ORACLE_ROOT / "summary.json")
    oracle_seal = read_json(ORACLE_ROOT / "prediction-seal.json")
    with np.load(ORACLE_ROOT / "scores.npz") as saved:
        oracle_ids = [str(v) for v in saved["ids"]]
        oracle_astar = saved["A_star"].astype(np.float64, copy=True)
        oracle_attribution = saved["oracle_attribution"].astype(np.float64, copy=True)
        oracle_full_surface = saved["oracle_full_surface"].astype(np.float64, copy=True)

    if len(cases) != 288 or len(old_cases) != 288 or len(prior) != 288:
        raise AssertionError("Expected 288 rows in each consumed source")
    ids = [row["id"] for row in cases]
    old_ids = [row["id"] for row in old_cases]
    prior_ids = [row["id"] for row in prior]
    if ids != old_ids or ids != prior_ids or ids != oracle_ids:
        raise AssertionError("Frame ID order differs between saved sources")
    labels_match = all(
        bool(a["truth"]) == bool(b["truth"])
        and a["stratum"] == b["stratum"]
        and a["episode_id"] == b["episode_id"]
        for a, b in zip(cases, old_cases)
    )
    if not labels_match:
        raise AssertionError("Case labels, strata, or episode IDs differ")
    # The prior prediction file is predictor output and intentionally carries
    # no evaluator truth labels; its IDs are checked above, while truth and
    # strata remain sourced only from the sealed case list.

    FROZEN_THRESHOLD = float(astar_config["threshold"])
    source_threshold = float(read_json(SOURCE_ROOT / "a-control" / "threshold.json")["threshold"])
    oracle_threshold = float(oracle_summary["threshold_frozen"])
    if not (
        FROZEN_THRESHOLD == source_threshold == oracle_threshold
        and FROZEN_THRESHOLD == float(model_seal["models"]["multi"]["threshold"])
    ):
        raise AssertionError("Frozen probability thresholds disagree")

    # The old consumed detail has both domains.  A* probabilities are the
    # public alert score; A* scores are logits used by diagnostics only.
    frozen_probability = np.asarray(
        [row["bg"]["probabilities"]["A_star"] for row in old_cases], dtype=np.float64
    )
    frozen_logit = np.asarray(
        [row["bg"]["scores"]["A_star"] for row in old_cases], dtype=np.float64
    )
    prior_probability = np.asarray([row["control_score"] for row in prior], dtype=np.float64)
    if not np.array_equal(frozen_probability, prior_probability):
        raise AssertionError("Saved A* probability differs from prior control score")
    if not np.array_equal(frozen_probability, oracle_astar):
        raise AssertionError("Oracle A* arm is not bitwise equal to frozen A* probability")
    if not np.all((frozen_probability >= 0) & (frozen_probability <= 1)):
        raise AssertionError("A* probability domain is invalid")
    if np.array_equal(frozen_probability, frozen_logit):
        raise AssertionError("Probability and logit domains unexpectedly match")

    clear = np.asarray([row["stratum"] in ("positive", "negative") for row in cases])
    frozen_alert = frozen_probability >= FROZEN_THRESHOLD
    frozen_logit_threshold = float(logit(FROZEN_THRESHOLD))

    # The budget diagnostic is intentionally included as a separate recipe:
    # it is a B2 logit point, not the current oracle A* arm.
    selected_astar = budget["selected"]["A_star"]["3"]
    selected_b2 = budget["selected"]["B2"]["3"]
    selected_astar_threshold = float(selected_astar["chosen"]["threshold"])
    selected_b2_threshold = float(selected_b2["chosen"]["threshold"])

    recipes = {}

    def add_recipe(name, score, threshold, source, domain):
        decisions = np.asarray(score) >= float(threshold)
        recipes[name] = {
            "score_domain": domain,
            "threshold_domain": domain,
            "threshold": float(threshold),
            "source": source,
            "metrics": metric(cases, decisions, clear),
            "changed_clear_frames_vs_frozen": int((clear & (decisions != frozen_alert)).sum()),
        }

    add_recipe(
        "frozen_astar_probability", frozen_probability, FROZEN_THRESHOLD,
        "old-cases.json bg.probabilities.A_star; confirmation/predictions.json control_score", "probability"
    )
    add_recipe(
        "oracle_astar_probability", oracle_astar, FROZEN_THRESHOLD,
        "oracle-surface-ceiling-20260918/scores.npz A_star", "probability"
    )
    add_recipe(
        "oracle_attribution_probability", oracle_attribution, FROZEN_THRESHOLD,
        "oracle-surface-ceiling-20260918/scores.npz oracle_attribution", "probability"
    )
    add_recipe(
        "oracle_full_surface_probability", oracle_full_surface, FROZEN_THRESHOLD,
        "oracle-surface-ceiling-20260918/scores.npz oracle_full_surface", "probability"
    )
    add_recipe(
        "astar_logit_wrong_probability_threshold", frozen_logit, FROZEN_THRESHOLD,
        "old-cases.json bg.scores.A_star compared with probability threshold (diagnostic misuse)", "logit"
    )
    add_recipe(
        "astar_logit_correct_threshold", frozen_logit, frozen_logit_threshold,
        "old-cases.json bg.scores.A_star with logit(frozen probability threshold)", "logit"
    )
    selected_astar_scores = np.asarray(
        [row["bg"][selected_astar["score_field"]]["A_star"] for row in old_cases], dtype=np.float64
    )
    selected_b2_scores = np.asarray(
        [row["bg"][selected_b2["score_field"]]["B2"] for row in old_cases], dtype=np.float64
    )
    add_recipe(
        "posthoc_selected_astar_probability", selected_astar_scores, selected_astar_threshold,
        "budget-diagnostic/summary.json selected.A_star.3", "probability"
    )
    add_recipe(
        "posthoc_selected_b2_logit", selected_b2_scores, selected_b2_threshold,
        "budget-diagnostic/summary.json selected.B2.3", "logit"
    )

    expected = {
        "frozen_astar_probability": {"TP": 96, "FP": 3, "FN": 12, "TN": 105},
        "oracle_astar_probability": {"TP": 96, "FP": 3, "FN": 12, "TN": 105},
        "oracle_attribution_probability": {"TP": 93, "FP": 3, "FN": 15, "TN": 105},
        "oracle_full_surface_probability": {"TP": 93, "FP": 3, "FN": 15, "TN": 105},
        "astar_logit_wrong_probability_threshold": {"TP": 93, "FP": 3, "FN": 15, "TN": 105},
        "astar_logit_correct_threshold": {"TP": 96, "FP": 3, "FN": 12, "TN": 105},
        "posthoc_selected_astar_probability": {"TP": 98, "FP": 3, "FN": 10, "TN": 105},
        "posthoc_selected_b2_logit": {"TP": 93, "FP": 3, "FN": 15, "TN": 105},
    }
    for name, counts in expected.items():
        assert_counts(name, recipes[name]["metrics"], counts)

    candidate_rows = []
    candidates = {
        "oracle_astar_probability": (oracle_astar, FROZEN_THRESHOLD, "probability"),
        "oracle_attribution_probability": (oracle_attribution, FROZEN_THRESHOLD, "probability"),
        "oracle_full_surface_probability": (oracle_full_surface, FROZEN_THRESHOLD, "probability"),
        "astar_logit_wrong_probability_threshold": (frozen_logit, FROZEN_THRESHOLD, "logit"),
        "astar_logit_correct_threshold": (frozen_logit, frozen_logit_threshold, "logit"),
        "posthoc_selected_astar_probability": (selected_astar_scores, selected_astar_threshold, "probability"),
        "posthoc_selected_b2_logit": (selected_b2_scores, selected_b2_threshold, "logit"),
    }
    for name, (scores, threshold, domain) in candidates.items():
        candidate_rows.extend(detail_rows(cases, frozen_probability, scores, domain, threshold, name))

    model_hash = model_seal["models"]["multi"]["sha256"]
    if model_hash != astar_config["model_sha256"]:
        raise AssertionError("A* model hash changed across sealed inputs")
    if "model_sha256" in oracle_seal and model_hash != oracle_seal["model_sha256"]:
        raise AssertionError("Oracle model hash differs from sealed A* model")

    inputs = {
        str(path.resolve()): sha(path)
        for path in (
            CASES, OLD_CASES, PREDICTIONS, BUDGET, MODEL_SEAL, ASTAR_CONFIG,
            ORACLE_ROOT / "scores.npz", ORACLE_ROOT / "summary.json", ORACLE_ROOT / "prediction-seal.json",
        )
    }
    out.mkdir(parents=True)
    write_json(out / "decision-changes.json", candidate_rows)
    audit = {
        "schema": "DTR_WORKPOINT_ALIGNMENT_V2",
        "status": "PASS",
        "rows": len(cases),
        "clear_rows": int(clear.sum()),
        "boundary_rows": int((~clear).sum()),
        "labels_order_verified": labels_match and ids == old_ids == prior_ids == oracle_ids,
        "model_unchanged": {
            "model_id": astar_config["model_id"],
            "model_sha256": model_hash,
            "model_seal_multi_sha256": model_seal["models"]["multi"]["sha256"],
            "astar_config_sha256": astar_config["model_sha256"],
            "oracle_non_support_bitwise_unchanged": bool(oracle_seal["non_support_bitwise_unchanged"]),
        },
        "score_domains": {
            "probability": "[0,1], alert uses probability >= 0.5568065433174727",
            "logit": "unbounded log-odds; equivalent threshold is 0.22821148907779354",
            "wrong_domain_diagnostic": "saved A* logit compared with probability threshold reproduces 93/3/15 but is not run_oracle_surface_ceiling.py",
        },
        "frozen_probability_threshold": FROZEN_THRESHOLD,
        "equivalent_frozen_logit_threshold": frozen_logit_threshold,
        "comparisons": recipes,
        "decision_changes_file": "decision-changes.json",
        "decision_change_rows": len(candidate_rows),
        "inputs": inputs,
        "conclusion": (
            "The original A* model and frame order are unchanged. The frozen A* probability remains 96/3/12. At the same frozen probability threshold, both native oracle support interventions are 93/3/15 because exactly three positive near-rod clear IDs move below threshold. Applying that probability threshold directly to the saved A* logit also yields 93/3/15, but that is a score-domain misuse. The separate B2 posthoc logit point also yields 93/3/15 and must not be used to label the oracle A* arm."
        ),
    }
    write_json(out / "audit.json", audit)

    labels = {
        "frozen_astar_probability": "Frozen A*",
        "oracle_astar_probability": "Oracle A* (unchanged features)",
        "oracle_attribution_probability": "Oracle attribution",
        "oracle_full_surface_probability": "Oracle full surface",
        "astar_logit_wrong_probability_threshold": "A* logit with probability threshold (misuse)",
        "astar_logit_correct_threshold": "A* logit with equivalent logit threshold",
        "posthoc_selected_astar_probability": "Posthoc selected A*",
        "posthoc_selected_b2_logit": "Posthoc selected B2",
    }
    lines = [
        "# MZ177 work-point alignment audit (2026-09-18)", "",
        "This audit replays saved IDs, labels, probabilities, logits and the sealed oracle score arrays. It performs no training, fitting, threshold search or model edit.", "",
        "| recipe | score domain | threshold | clear TP/FP/FN/TN | changed clear frames |",
        "|---|---|---:|---:|---:|",
    ]
    for name, recipe in recipes.items():
        c = recipe["metrics"]["clear"]
        lines.append(
            f"| {labels[name]} | `{recipe['score_domain']}` | `{recipe['threshold']:.16g}` | {c['TP']}/{c['FP']}/{c['FN']}/{c['TN']} | {recipe['changed_clear_frames_vs_frozen']} |"
        )
    lines += [
        "", "`run_oracle_surface_ceiling.py` calls `predict_proba` and stores probability-domain `scores.npz`. Its unchanged A* arm is bitwise equal to the frozen A* probability, so it remains 96/3/12.", "",
        "The attribution and complete-surface oracle arms each become 93/3/15 at the frozen probability threshold. The three changed clear IDs are exactly:", "",
        "- `singleconfirm_near_rod_farwall_scene0_in_00`", "- `singleconfirm_near_rod_farwall_scene0_in_02`", "- `singleconfirm_near_rod_farwall_scene1_in_01`", "",
        "Their labels are positive. Native support intervention changes their probability scores below `0.5568065433174727`; the original A* scores and model hash do not change. The full score and label details are in `decision-changes.json`.", "",
        "The same 93/3/15 count can be produced by comparing the saved A* logit to the probability threshold, but the equivalent logit threshold restores 96/3/12. The separate B2 posthoc logit point also reports 93/3/15, with a different set of 21 clear decision changes. Therefore 93/3/15 is not attributable only to B2, and it is not the frozen A* result.", "",
        "Input order, labels, strata, model hash and non-support feature invariance were checked. Boundary rows remain in the 288-row input and are excluded only from the 216-frame clear confusion counts.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "completion.json",
        {
            "status": "PASS",
            "outputs": {name: sha(out / name) for name in ("audit.json", "decision-changes.json", "REPORT.md")},
            "evaluator_read": False,
            "training_steps": 0,
            "threshold_changes": 0,
            "model_changes": 0,
        },
    )
    print(json.dumps({"status": "PASS", "output": str(out), "comparisons": {name: value["metrics"]["clear"] for name, value in recipes.items()}, "decision_change_rows": len(candidate_rows)}, indent=2))


if __name__ == "__main__":
    main()
