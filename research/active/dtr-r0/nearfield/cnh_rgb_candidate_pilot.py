"""Consumed-Development candidate screen with official instance references.

Candidate extraction sees cached RGB-model predictions and public camera rays.
Reference depth, instances and physical bounding boxes are evaluator-only. The
nominal camera-relative volume is not a calibrated wearer collision envelope.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import ndimage

from cnh_rgb_geometry_candidates import extract_candidates, SOURCE_PARAMETERS


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hdf(path):
    with h5py.File(path) as f:
        return f["dataset"][:]


def reference(repo, row, camera, boxes):
    root = repo / "artifacts.local/datasets/hypersim-ba-nfo"
    path = root / row["depth"]
    radial = hdf(path).astype(np.float32)
    instance = hdf(path.with_name(path.name.replace("depth_meters", "semantic_instance")))
    semantic = hdf(path.with_name(path.name.replace("depth_meters", "semantic")))
    assert radial.shape == instance.shape == semantic.shape == (768, 1024)
    height, width = radial.shape
    yy, xx = np.indices(radial.shape)
    uv = np.stack(((xx + .5) / width * 2 - 1, 1 - (yy + .5) / height * 2,
                   np.ones(radial.shape)), axis=-1)
    rays = uv @ np.asarray(camera).T
    norm = np.linalg.norm(rays, axis=-1)
    z = radial * (-rays[..., 2] / norm)
    x = radial * rays[..., 0] / norm
    y = -radial * rays[..., 1] / norm
    # Stuff surfaces such as wall/floor have semantic labels but instance=-1.
    # Their candidates are evaluable nuisance, not missing-label observations.
    structural = np.isin(semantic, (1, 2, 22))
    valid = (np.isfinite(radial) & (radial > 0) & (semantic > 0)
             & ((instance >= 0) | structural))
    objects = valid & (instance >= 0) & ~np.isin(semantic, (1, 2, 22))
    query = objects & (z >= .5) & (z <= 2) & (np.abs(x) <= .30) & (np.abs(y) <= .55)
    object_counts = np.bincount(instance[objects].astype(int))
    query_counts = np.bincount(instance[query].astype(int), minlength=len(object_counts))
    ids = np.flatnonzero(object_counts >= 16)
    slices = ndimage.find_objects(np.where(objects, instance.astype(np.int32) + 1, 0))
    query_ids, query_z = instance[query], z[query]
    targets, all_objects = [], {}
    for identity in ids:
        bounds = slices[identity]
        if bounds is None:
            continue
        ry, rx = bounds
        bbox = [ry.start, rx.start, ry.stop, rx.stop]
        key = (row["scene"], int(identity))
        extent = boxes.get(key)
        info = dict(instance_id=int(identity), bbox=bbox,
                    visible_pixels=int(object_counts[identity]),
                    query_pixels=int(query_counts[identity]), extents_m=extent)
        all_objects[int(identity)] = info
        if query_counts[identity] < 16:
            continue
        distance = float(np.median(query_z[query_ids == identity]))
        primary = extent is not None and max(extent) <= .30
        info.update(distance_m=distance, primary_small=primary,
                    sorted_extents_m=sorted(extent) if extent is not None else None,
                    distance_bin="0.5-1m" if distance < 1 else "1-1.5m" if distance < 1.5 else "1.5-2m")
        targets.append(info)
    return instance, objects, valid, all_objects, targets


def frame_inventory(args):
    repo, row, camera, boxes, split = args
    _, _, valid, all_objects, targets = reference(repo, row, camera, boxes)
    return dict(id=row["id"], scene=row["scene"], split=split,
                valid_label_pixels=int(valid.sum()), frame_pixels=int(valid.size),
                objects=len(all_objects), targets=targets,
                missing_bbox_objects=sum(o["extents_m"] is None for o in all_objects.values()))


def summary_inventory(rows):
    result = {}
    for split in ("calibration", "evaluation"):
        subset = [r for r in rows if r["split"] == split]
        targets = [(r, t) for r in subset for t in r["targets"] if t["primary_small"]]
        unique = {(r["scene"], t["instance_id"]) for r, t in targets}
        scenes = {r["scene"] for r, _ in targets}
        bins = {}
        for name in ("0.5-1m", "1-1.5m", "1.5-2m"):
            group = [(r, t) for r, t in targets if t["distance_bin"] == name]
            count = len({(r["scene"], t["instance_id"]) for r,t in group})
            bins[name] = dict(object_views=len(group), distinct_objects=count, sufficient=count >= 5)
        result[split] = dict(frames=len(subset), primary_object_views=len(targets),
                             distinct_primary_objects=len(unique), primary_scenes=len(scenes), distance_bins=bins,
                             enough=bool(len(unique) >= 10 and len(scenes) >= 3),
                             all_object_views=sum(r["objects"] for r in subset),
                             missing_bbox_object_views=sum(r["missing_bbox_objects"] for r in subset),
                             valid_label_pixel_fraction=sum(r["valid_label_pixels"] for r in subset)/sum(r["frame_pixels"] for r in subset),
                             nonprimary_near_object_views=sum(not t["primary_small"] for r in subset for t in r["targets"]))
    return result


def evaluate_frame(args):
    repo, row, camera, boxes, split, ratios = args
    instance, objects, valid, all_objects, targets = reference(repo, row, camera, boxes)
    path = repo / "artifacts.local/work/ba-nfo-depthpro-20260919/predictions/native" / (row["id"] + ".npz")
    with np.load(path, allow_pickle=False) as p:
        prediction = p["native_depth"]
    target_ids = {t["instance_id"] for t in targets}
    primary_ids = {t["instance_id"] for t in targets if t["primary_small"]}
    outcomes = []
    for ratio in ratios:
        candidates = extract_candidates(prediction, camera, ratio)
        used, matches, nuisance, unknown, duplicate, unmatched = set(), [], 0, 0, 0, 0
        for candidate in sorted(candidates, key=lambda c: c["score"], reverse=True):
            y0, x0, y1, x1 = candidate["bbox"]
            mask = candidate["mask"]
            pix = int(mask.sum())
            local_valid = valid[y0:y1, x0:x1][mask]
            if local_valid.mean() < .80:
                unknown += 1
                continue
            labels = instance[y0:y1, x0:x1][mask]
            foreground = objects[y0:y1, x0:x1][mask]
            identities, counts = np.unique(labels[foreground], return_counts=True)
            eligible = []
            for identity, overlap in zip(identities, counts):
                identity = int(identity)
                if identity not in all_objects:
                    continue
                precision = float(overlap / pix)
                coverage = float(overlap / all_objects[identity]["visible_pixels"])
                if precision >= .50 and coverage >= .25:
                    eligible.append((int(overlap), identity, precision, coverage))
            eligible.sort(reverse=True)
            best = next((e for e in eligible if e[1] not in used), None)
            if best is None:
                nuisance += 1
                unmatched += 1
                duplicate += bool(eligible)
                continue
            overlap, identity, precision, coverage = best
            used.add(identity)
            if identity not in target_ids:
                nuisance += 1
            matches.append(dict(instance_id=identity, bbox=candidate["bbox"],
                                score=candidate["score"], precision=precision, coverage=coverage,
                                primary=identity in primary_ids, relevant=identity in target_ids))
        outcomes.append(dict(ratio=ratio, candidates=len(candidates), task_irrelevant=nuisance,
                             unknown_candidates=unknown, duplicates=duplicate, matches=matches,
                             unmatched=unmatched,
                             primary_hits=sorted(primary_ids & used)))
    return dict(id=row["id"], scene=row["scene"], split=split, targets=targets, outcomes=outcomes,
                camera_matrix=camera)


def aggregate(rows, ratio):
    counts, total, hit, all_targets, all_hit = [], set(), set(), 0, 0
    errors, size_errors, secondary_extents = [], [], []
    secondary_views, secondary_hits = 0, 0
    burden = {key: [] for key in ("candidates", "unmatched", "unknown_candidates", "duplicates")}
    bins = {name: dict(views=0, hits=0, instances=set()) for name in ("0.5-1m", "1-1.5m", "1.5-2m")}
    for row in rows:
        out = next(o for o in row["outcomes"] if o["ratio"] == ratio)
        counts.append(out["task_irrelevant"])
        for key in burden:
            burden[key].append(out[key])
        targets = {t["instance_id"]:t for t in row["targets"] if t["primary_small"]}
        others = [t for t in row["targets"] if not t["primary_small"]]
        secondary_views += len(others)
        secondary_hits += sum(m["relevant"] and not m["primary"] for m in out["matches"])
        secondary_extents.extend(sorted(t["extents_m"]) for t in others if t["extents_m"] is not None)
        all_targets += len(targets)
        all_hit += len(out["primary_hits"])
        total.update((row["scene"], i) for i in targets)
        hit.update((row["scene"], i) for i in out["primary_hits"])
        for identity, target in targets.items():
            group = bins[target["distance_bin"]]
            group["views"] += 1
            group["hits"] += identity in out["primary_hits"]
            group["instances"].add((row["scene"], identity))
        for m in out["matches"]:
            if m["primary"]:
                truth = targets[m["instance_id"]]["bbox"]
                pred = m["bbox"]
                def center_ray(box):
                    uv = [(box[1]+box[3])/1024-1, 1-(box[0]+box[2])/768, 1]
                    ray = np.asarray(row["camera_matrix"]) @ uv
                    return ray / np.linalg.norm(ray)
                errors.append(float(np.degrees(np.arccos(np.clip(center_ray(pred) @ center_ray(truth), -1, 1)))))
                size_errors.append([abs((pred[3]-pred[1])/(truth[3]-truth[1])-1),
                                    abs((pred[2]-pred[0])/(truth[2]-truth[0])-1)])
    def distribution(values):
        return dict(mean=float(np.mean(values)), p95=float(np.percentile(values,95)), max=max(values))
    strata = {name: dict(object_views=g["views"], hit_object_views=g["hits"],
                        recall=g["hits"]/g["views"] if g["views"] else None,
                        distinct_objects=len(g["instances"]), sufficient=len(g["instances"])>=5)
              for name,g in bins.items()}
    return dict(frames=len(rows), object_views=all_targets, hit_object_views=all_hit,
                object_view_recall=all_hit/all_targets if all_targets else None,
                distinct_objects=len(total), hit_distinct_objects=len(hit),
                any_view_object_recall=len(hit)/len(total) if total else None,
                task_irrelevant_mean=float(np.mean(counts)), task_irrelevant_p95=float(np.percentile(counts,95)),
                task_irrelevant_max=max(counts),
                burdens_per_frame={key: distribution(values) for key,values in burden.items()},
                distance_bins=strata,
                secondary_nonprimary_near=dict(object_views=secondary_views, hit_object_views=secondary_hits,
                    sorted_physical_extents_m_median=np.median(secondary_extents,axis=0).tolist() if secondary_extents else None,
                    role="descriptive dimensions; no frozen binary thin/narrow class, no replacement of primary"),
                matched_primary_bbox_center_error_degrees_median=float(np.median(errors)) if errors else None,
                matched_primary_bbox_relative_size_error_xy_median=np.median(size_errors,axis=0).tolist() if size_errors else None)


def disabled_outcome(rows):
    empty = []
    for row in rows:
        copy = dict(row)
        copy["outcomes"] = [dict(ratio=None, candidates=0, unmatched=0,
                                 task_irrelevant=0, unknown_candidates=0,
                                 duplicates=0, matches=[], primary_hits=[])]
        empty.append(copy)
    return aggregate(empty, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output.resolve()
    assert out.is_relative_to((repo/"artifacts.local").resolve())
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/"result.json").exists(), "Preserve completed run; use new output directory"
    root = repo / "artifacts.local/work/cnh-rgb-candidate-supplement-20261001"
    plan = read(root/"CANDIDATE_PLAN.json")
    manifest = repo/"artifacts.local/work/ba-nfo-depthpro-20260919/manifest.json"
    rows = read(manifest)
    observed = read(manifest.with_name("observations.json"))
    camera = {r["id"]: r["camera_matrix"] for r in observed}
    boxfile = root/"hypersim/visible-scene-instances.json"
    boxes = {(r["scene"],r["instance_id"]):r["extents_m"] for r in read(boxfile)}
    scenes = sorted({r["scene"] for r in rows}, key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    calibration = set(scenes[:26])
    inputs = [(repo,r,camera[r["id"]],boxes,"calibration" if r["scene"] in calibration else "evaluation") for r in rows]
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        inventory = list(pool.map(frame_inventory, inputs))
    write(out/"inventory.json", inventory)
    summary = summary_inventory(inventory)
    result = dict(inventory=summary, plan_sha256=sha(root/"CANDIDATE_PLAN.json"),
                  sources={str(p.relative_to(repo)):sha(p) for p in (Path(__file__),Path(__file__).with_name("cnh_rgb_geometry_candidates.py"),manifest,manifest.with_name("observations.json"),boxfile)},
                  source_parameters=SOURCE_PARAMETERS, scene_splits={"calibration":scenes[:26],"evaluation":scenes[26:]},
                  role="consumed synthetic Development; no fusion training or alert evaluation")
    if args.inventory_only or not all(v["enough"] for v in summary.values()):
        result.update(status="DATA_READY" if all(v["enough"] for v in summary.values()) else "INSUFFICIENT_SMALL_OBJECT_OPPORTUNITY",
                      candidate_quality="NOT_RUN", algorithm_verdict=None)
    else:
        ratios = plan["ratios"]
        with ThreadPoolExecutor(args.workers) as pool:
            predictions = list(pool.map(evaluate_frame,[i+(ratios,) for i in inputs]))
        write(out/"candidate-ledger.json", predictions)
        cal_rows = [r for r in predictions if r["split"]=="calibration"]
        eval_rows = [r for r in predictions if r["split"]=="evaluation"]
        summaries = {str(r):aggregate(cal_rows,r) for r in ratios}
        feasible = [r for r in ratios if summaries[str(r)]["task_irrelevant_mean"]<=1.]
        selected = max(feasible,key=lambda r:(summaries[str(r)]["object_view_recall"],-summaries[str(r)]["task_irrelevant_mean"],r)) if feasible else None
        evaluation = aggregate(eval_rows,selected) if selected else disabled_outcome(eval_rows)
        result.update(calibration=summaries, disabled_candidate_baseline={"calibration":disabled_outcome(cal_rows),"evaluation":disabled_outcome(eval_rows)}, selected_ratio=selected, evaluation=evaluation,
                      active_candidate_evaluation="NOT_SELECTED_CALIBRATION_BUDGET" if selected is None else "SELECTED_ON_CALIBRATION_ONLY",
                      nuisance_scope="whole image; no absolute range or corridor filter; not corridor alerts",
                      unknown_rule="candidate valid semantic-and-depth support<80% is UNKNOWN, separately reported; not counted as known nuisance",
                      status="CANDIDATE_SCREEN_PASS_ONLY" if evaluation and evaluation["object_view_recall"]>=.8 and evaluation["task_irrelevant_mean"]<=1 else "NOT_SELECTIVE_ENOUGH_FOR_THIS_SCREEN",
                      retained_ideal_gain="NOT_ESTIMABLE_FROM_CANDIDATE_METRICS",
                      algorithm_verdict="This fixed cached-DepthPro protrusion extractor only")
    result["seconds"] = time.perf_counter()-started
    write(out/"result.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
