"""Independent train-only audit of the two final spatial-fitting states.

No production loss, metric, selector or feasibility implementation is imported.
Model construction only regenerates the frozen initialization; no forward or
training is performed. Original dev/held labels are never opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / "artifacts.local/evidence/ba-query-spatial-fit-20260922"
SOURCE = REPO / "artifacts.local/evidence/ba-query-occupancy-20260922"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def equal(a, b, path="value"):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            equal(a[k], b[k], path + "." + k)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{path}[{i}]")
    elif isinstance(b, float):
        assert math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-9), (path, a, b)
    else:
        assert a == b, (path, a, b)


def summary(margins):
    a = np.asarray(margins, np.float64)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)),
        p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
        wins=int((a > 0).sum()), ties=int((a == 0).sum()), losses=int((a < 0).sum()),
        win_rate=float(((a > 0) + .5 * (a == 0)).mean()))


def counts(score, occupied, valid):
    selected = score >= .5
    pos, neg = occupied & valid, ~occupied & valid
    tp, fp = int((selected & pos).sum()), int((selected & neg).sum())
    fn, tn = int((~selected & pos).sum()), int((~selected & neg).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=tp / (tp + fp) if tp + fp else 0.,
        recall=tp / (tp + fn) if tp + fn else 0., FPR=fp / (fp + tn) if fp + tn else 0.)


def feasibility(labels):
    y, v = labels["classes"] < 6, labels["valid"]
    demand = ((y & v)[:, :, None] & (~y & v)[:, None, :]).sum(0)
    same_height = np.arange(6)[:, None] // 3 == np.arange(6)[None, :] // 3
    maxima = {}
    for name, eligible in (("all", np.ones((6, 6), bool)), ("lateral", same_height), ("height", ~same_height)):
        weight = demand * eligible
        best = 0
        for order in itertools.permutations(range(6)):
            position = np.empty(6, int)
            position[list(order)] = np.arange(6)
            best = max(best, int((weight * (position[:, None] < position[None, :])).sum()))
        maxima[name] = best / int(weight.sum()) if weight.sum() else 0.
    ceilings = []
    for row, q in np.argwhere(v & (labels["mask"].sum((2, 3)) > 0)):
        fg, cover = labels["mask"][row, q].astype(float).ravel(), labels["coverage"][row].astype(float).ravel()
        observed = cover > 0
        fg, cover = fg[observed], cover[observed]
        # Optimizing a fractional set objective reduces to a prefix ordered by fg/cover.
        order = np.argsort(fg / cover)[::-1]
        cumulative_fg = np.cumsum(fg[order])
        candidate = cumulative_fg / (fg.sum() + np.cumsum(cover[order] - fg[order]))
        ceilings.append(float(candidate.max()))
    return dict(fixed_query_order_max_win_rate=maxima, visible_positive_queries=len(ceilings),
        binary_grid_oracle_iou_mean=float(np.mean(ceilings)), binary_grid_oracle_iou_min=min(ceilings),
        evaluable=maxima["all"] < .95 and float(np.mean(ceilings)) >= .5)


def recompute(pred, labels, metadata):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, v = labels["classes"] < 6, labels["valid"]
    score = pred["probability"]
    assert score.shape == (84, 6) and np.isfinite(score).all() and ((score >= 0) & (score <= 1.000002)).all()
    dist = pred["distribution"]
    assert dist.shape == (84, 6, 7) and (dist >= 0).all() and np.isfinite(dist).all()
    assert np.allclose(dist.sum(-1), 1, atol=2e-6, rtol=0)
    assert np.allclose(dist[..., :6].sum(-1), score, atol=2e-6, rtol=0)
    fg = labels["mask"].astype(np.float64)
    coverage = np.broadcast_to(labels["coverage"][:, None], fg.shape).astype(np.float64)
    assert pred["mask"].shape == fg.shape and np.isfinite(pred["mask"]).all()
    mask = pred["mask"] >= .5
    intersection = np.where(mask, fg, 0).sum((2, 3))
    predicted_area = np.where(mask, coverage, 0).sum((2, 3))
    target_area = fg.sum((2, 3))
    union = target_area + predicted_area - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    visible = v & (target_area > 0)
    empty, positive = v & ~y, v & y
    row, pos, neg = np.where((y & v)[:, :, None] & (~y & v)[:, None, :])
    margin = score[row, pos].astype(float) - score[row, neg].astype(float)
    pairs = {name: summary(margin[keep]) for name, keep in (
        ("all", np.ones(len(margin), bool)), ("lateral", pos // 3 == neg // 3), ("height", pos // 3 != neg // 3))}
    query = counts(score, y, v)
    macro_iou = float(iou[visible].mean())
    criteria = dict(precision=query["precision"] >= .95, recall=query["recall"] >= .95,
                    mask_iou=macro_iou >= .5, query_pair_order=pairs["all"]["win_rate"] >= .95)
    tp = float(intersection[v].sum())
    fp = float((predicted_area - intersection)[v].sum())
    fn = float((target_area - intersection)[v].sum())
    family = {}
    for f in sorted({r["type_id"] for r in metadata}):
        keep = np.asarray([r["type_id"] == f for r in metadata])
        family[f] = dict(frames=int(keep.sum()), queries=int(v[keep].sum()),
            query=counts(score[keep], y[keep], v[keep]), visible_positive_queries=int(visible[keep].sum()),
            mask_iou=float(iou[keep][visible[keep]].mean()) if visible[keep].any() else None)
    permutations = Counter(tuple(map(int, order)) for order in np.argsort(-score, axis=1, kind="stable"))
    winner = dist.argmax(-1)
    result = dict(frames=84, known_queries=int(v.sum()), positive_queries=int(positive.sum()),
        visible_positive_queries=int(visible.sum()), geometric_positive_without_visible_pixels=int((positive & ~visible).sum()),
        query=query, query_rank=dict(auc=float(roc_auc_score(y[v], score[v])),
            ap=float(average_precision_score(y[v], score[v])), prevalence=float(y[v].mean())),
        mask_iou=macro_iou, mask_pixel=dict(TP_area=tp, FP_area=fp, FN_area=fn,
            precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.),
        empty_query_mask=dict(queries=int(empty.sum()), any_predicted_area=int(mask.any((2, 3))[empty].sum()),
            false_area_fraction=float(predicted_area[empty].sum() / coverage[empty].sum())),
        same_image_query_order=pairs, distance_bin=dict(accuracy=float((winner[v] == labels["classes"][v]).mean()),
            positive_accuracy=float((winner[positive] == labels["classes"][positive]).mean())),
        query_orderings=[dict(order=list(order), frames=n) for order, n in sorted(permutations.items())],
        strict_order_frames=int(np.all(np.diff(np.sort(score, axis=1), axis=1) > 0, axis=1).sum()),
        per_family=family, criteria=criteria, fit_gate_pass=all(criteria.values()))
    covered_count = int((predicted_area[empty] > 0).sum())
    semantic = dict(empty_queries_with_any_covered_prediction=covered_count,
        reported_any_prediction_count=result["empty_query_mask"]["any_predicted_area"],
        invalid_area_only_prediction_count=result["empty_query_mask"]["any_predicted_area"] - covered_count)
    return result, semantic


def audit(root):
    root = Path(root)
    report, seal, source, pairing = (read(root / name) for name in
        ("result.json", "prediction-seal.json", "source-input-seal.json", "pairing.json"))
    assert report["status"] == "PASS" and report["dev_held_access"] is False and report["threshold_selection"] is False
    assert seal["dev_held_access"] is False and seal["thresholds"] == dict(query=.5, mask=.5)
    assert seal["evaluated_state"] == "final_epoch_100"
    for name, digest in source["code"].items():
        assert sha(HERE / name) == digest, name
    for path, digest in source["inputs"].items():
        assert Path(path).name not in ("evaluation.npz", "dev.npz")
        assert sha(path) == digest, path
    for arm in ("global", "spatial"):
        assert sha(root / f"{arm}.pt") == seal["checkpoints"][arm]
        assert sha(root / f"{arm}-predictions.npz") == seal["predictions"][arm]
    journal = read(source["journal"])
    assert journal["native_exit_code"] == 0 and journal["experiment"]["status"] == "PASS"
    admissions = {r["alias"]: r for r in journal["asset_inputs"]}
    assert set(admissions) == {"observations", "train", "manifest", "plan"}
    assert admissions["train"]["requested_relative_path"].replace("\\", "/") == "labels/train.npz"
    assert admissions["train"]["ue_reuse"]["role"] == "evaluator"
    assert admissions["observations"]["ue_reuse"]["role"] == "observation"
    assert any(x.get("content_id") == "sha256:" + sha(root / "result.json") for x in journal["output_receipts"])
    prepared = SOURCE.with_name(SOURCE.name + "-prepared")
    identities = read(prepared / "observations/identities.json")
    training = [r for r in identities if r["split"] == "train"]
    groups = {f: min(r["base_group_id"] for r in training if r["type_id"] == f) for f in sorted({r["type_id"] for r in training})}
    metadata = sorted([r for r in training if r["base_group_id"] == groups[r["type_id"]]
        and r["frame_in_clip"] in (0, 2, 3, 4, 5, 6, 7)], key=lambda r: r["index"])
    cohort = read(root / "cohort.json")
    equal(cohort["metadata"], metadata, "selected metadata")
    assert len(metadata) == 84 and len(groups) == 4 and cohort["groups"] == groups == report["groups"]
    indices = np.asarray([r["index"] for r in metadata])
    assert cohort["indices"] == indices.tolist() and cohort["selection_uses_labels"] is False
    with np.load(prepared / "labels/train.npz", allow_pickle=False) as data:
        lookup = {int(index): i for i, index in enumerate(data["indices"])}
        rows = [lookup[int(index)] for index in indices]
        original_labels = {name: data[name][rows] for name in data.files}
    with np.load(root / "training-targets.npz", allow_pickle=False) as data:
        labels = dict(data)
    assert set(labels) == set(original_labels)
    for name in labels:
        assert np.array_equal(labels[name], original_labels[name], equal_nan=True), name
    preflight = feasibility(labels)
    equal(read(root / "cohort-feasibility.json"), preflight, "preflight")
    equal(report["feasibility"], preflight, "result preflight")
    assert preflight["evaluable"]
    assert sha(root / "initial.pt") == pairing["initial_sha256"]
    assert sha(root / "batch-orders.npy") == pairing["schedule_sha256"]
    orders = np.load(root / "batch-orders.npy", allow_pickle=False)
    rng = np.random.default_rng(202609224)
    assert np.array_equal(orders, np.stack([rng.permutation(84) for _ in range(100)]))
    assert source["seed"] == 202609224 and source["epochs"] == report["epochs"] == 100
    assert source["batch"] == 12 and source["updates_per_arm"] == report["updates_per_arm"] == 700
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    from query_spatial_fit_model import QuerySpatialFitNet
    torch.set_num_threads(4)
    torch.manual_seed(202609224)
    regenerated = QuerySpatialFitNet(SOURCE / "plan/mobilenet_v3_small.pth", carrier="spatial")
    assert set(initial) == set(regenerated.state_dict())
    assert all(torch.equal(initial[k], v) for k, v in regenerated.state_dict().items()), "Initial state differs from prescribed seed/weights"
    assert sum(p.numel() for p in regenerated.parameters()) == pairing["parameters"]
    branch = "spatial_query_projection.weight"
    assert initial[branch].numel() == pairing["spatial_effective_extra_parameters"] == 768
    assert torch.count_nonzero(initial[branch]) == 0 and pairing["initial_predictions_exactly_equal"] is True
    results, semantic, checkpoints = {}, {}, {}
    for arm in ("global", "spatial"):
        history = read(root / f"{arm}-history.json")
        assert len(history) == 100 and all(r["epoch"] == i and r["updates"] == i * 7
            and math.isfinite(r["loss"]) for i, r in enumerate(history, 1))
        state = torch.load(root / f"{arm}.pt", map_location="cpu", weights_only=True)
        assert set(state) == set(initial) and all(state[k].shape == initial[k].shape for k in initial)
        checkpoints[arm] = dict(final_epoch=100, updates=700,
            projection_nonzero_weights=int(torch.count_nonzero(state[branch])),
            changed_state_tensors=sum(not torch.equal(state[k], initial[k]) for k in initial))
        if arm == "global":
            assert torch.equal(state[branch], initial[branch]), "Bypassed branch changed"
        with np.load(root / f"{arm}-predictions.npz", allow_pickle=False) as data:
            pred = dict(data)
        assert np.array_equal(pred["indices"], indices)
        results[arm], semantic[arm] = recompute(pred, labels, metadata)
        equal(report["arms"][arm], results[arm], arm)
    passed = [results[a]["fit_gate_pass"] for a in ("global", "spatial")]
    decision = {(True, True): "BOTH_FIT_SPATIAL_NOT_NECESSARY", (False, True): "SPATIAL_PACKAGE_FITS_ONLY",
                (True, False): "GLOBAL_FITS_SPATIAL_BRANCH_REJECTED", (False, False): "NEITHER_PACKAGE_FITS"}[tuple(passed)]
    assert report["decision"] == decision
    assert report["unknown_sensor_frames"] == sum(bool(r["baseline"]["unknown"]) for r in metadata)
    return dict(status="PASS", decision=decision, backend="TASK_NOT_GPU_SUITABLE", frames=84, queries=504,
        train_only_cohort_and_targets_match=True, prediction_and_checkpoint_hashes_match=True,
        source_and_input_hashes_match=True, governed_admission_and_output_hash_match=True,
        initialization_exactly_regenerated=True, schedule_exactly_regenerated=True, checkpoint_checks=checkpoints,
        feasibility=preflight, arms=results, covered_area_semantics=semantic,
        input_hashes={name: sha(root / name) for name in ("result.json", "source-input-seal.json", "prediction-seal.json",
            "pairing.json", "cohort.json", "training-targets.npz", "batch-orders.npy", "initial.pt")},
        audit_source_sha256=sha(__file__),
        boundaries=["84 consumed training frames from four layouts; no generalization or alert promotion",
            "No dev/held label arrays or image rows opened; shared observation containers only streamed for integrity hashes",
            "No forward pass, gradient, optimizer, threshold selection or new experiment run by this auditor",
            "Sealed model code uses RGB, ToF and public query/ray geometry only; native labels remain loss/evaluation inputs",
            "Recorded initial forward equality and historical batch use rely on sealed execution records; serialized initialization and schedule independently regenerated"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / "independent-audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit receipt: " + str(output))
    answer = audit(args.root)
    with output.open("x", encoding="utf-8") as f:
        json.dump(answer, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=answer["status"], decision=answer["decision"], output=str(output),
        covered_area_semantics=answer["covered_area_semantics"])))
