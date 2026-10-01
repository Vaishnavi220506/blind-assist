"""Check existing consumed RGB/depth caches before an instance-candidate pilot.

No inference, download, rendering or component-to-instance conversion. Missing
instance annotations make object metrics unavailable, not zero or model failure.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path):
    with zipfile.ZipFile(path) as archive:
        return sorted(Path(n).stem for n in archive.namelist())


def inventory(repo):
    artifacts = repo / "artifacts.local"
    old = artifacts / "work/ba-nfo-depthpro-20260919"
    prepared = artifacts / "work/ba-nfo-20260919"
    hyp = artifacts / "datasets/hypersim-ba-nfo"
    san = artifacts / "datasets/sanpo-synthetic-ba-nfo"
    nfo_manifest = old / "manifest.json"
    san_manifest = prepared / "prepared-manifest.json"
    nfo = read(nfo_manifest)
    sanpo = [r for r in read(san_manifest) if r["source"] == "sanpo"]
    assert len(nfo) == 500 and len({r["id"] for r in nfo}) == 500
    assert all(r["source"] == "hypersim" for r in nfo)
    assert len(sanpo) == 3000 and len({r["id"] for r in sanpo}) == 3000
    nfo_counts = Counter()
    prepared_fields, prediction_fields = Counter(), Counter()
    for row in nfo:
        for key in ("rgb", "depth"):
            nfo_counts[key] += (hyp / row[key]).is_file()
        depth = hyp / row["depth"]
        for label in ("semantic", "semantic_instance", "render_entity_id", "position"):
            path = depth.with_name(depth.name.replace("depth_meters", label))
            nfo_counts[label] += path.is_file()
        cache = prepared / row["prepared"]
        prediction = old / "predictions/native" / (row["id"] + ".npz")
        nfo_counts["prepared"] += cache.is_file()
        nfo_counts["depthpro_native"] += prediction.is_file()
        if cache.is_file():
            prepared_fields.update(fields(cache))
        if prediction.is_file():
            prediction_fields.update(fields(prediction))
    sanpo_counts = Counter()
    for row in sanpo:
        for key in ("rgb", "depth"):
            sanpo_counts[key] += (san / row[key]).is_file()
    session_counts = []
    for session in sorted({r["scene"] for r in sanpo}):
        files = [p for p in (san / session).rglob("*") if p.is_file()]
        labels = [p for p in files if any(w in p.relative_to(san / session).as_posix().lower()
                  for w in ("segmentation", "panoptic", "instance", "labelmap", "annotation"))]
        session_counts.append(dict(session=session, files=len(files), label_like_files=len(labels)))
        sanpo_counts["label_like_files"] += len(labels)
    has_nfo_labels = nfo_counts["semantic_instance"] == 500
    has_sanpo_labels = sanpo_counts["label_like_files"] > 0
    status = "ANNOTATION_REVIEW_REQUIRED" if has_nfo_labels or has_sanpo_labels else "NOT_ESTIMABLE"
    return dict(
        status=status, reason="INSTANCE_REFERENCE_AND_SMALL_OBJECT_DEFINITION_UNAVAILABLE",
        scope="Existing consumed Development only; no additional cohort or label download",
        nfo=dict(frames=len(nfo), scenes=len({r["scene"] for r in nfo}),
                 families=len({r["family"] for r in nfo}), counts=dict(nfo_counts),
                 prepared_fields=dict(prepared_fields), prediction_fields=dict(prediction_fields),
                 instances_0_5_to_2m=None),
        sanpo=dict(frames=len(sanpo), sessions=len(session_counts), counts=dict(sanpo_counts),
                   session_inventory=session_counts, instances_0_5_to_2m=None),
        candidate_recall="NOT_RUN", false_candidates_per_frame="NOT_RUN",
        predicted_retained_gain="NOT_ESTIMABLE_FROM_DOSE_TABLE",
        quality_stage="NOT_RUN_DATA_GATE", algorithm_verdict=None,
        stop="Do not score RGB or train fusion without evaluable instance references",
        missing=["Pixel-to-object instance IDs", "Small-object size/visible-extent definition",
                 "Final corridor geometry and object matching policy",
                 "Sequence/identity evidence for persistence and alerts per minute"],
        limits=["Visible near-depth support is not an object count",
                "Connected depth components are not ground-truth instances",
                "Existing dosage table is conditional, not a universal recall/false-candidate requirement",
                "Missing annotations do not establish NOT_SELECTIVE_ENOUGH or reject RGB"],
        sources={str(p.relative_to(repo)):digest(p) for p in (nfo_manifest, san_manifest, Path(__file__))},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    target = args.output.resolve()
    if not target.is_relative_to((repo / "artifacts.local").resolve()):
        parser.error("Output must remain under the canonical artifacts.local tree")
    if target.exists():
        parser.error("Choose a new output file; previous inventory is preserved")
    started = time.perf_counter()
    result = inventory(repo)
    result["seconds"] = time.perf_counter() - started
    result["backend"] = "CPU: metadata and archive headers only; no numerical/model inference"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    assert read(target) == result
    print(json.dumps({k:result[k] for k in ("status", "quality_stage", "seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
