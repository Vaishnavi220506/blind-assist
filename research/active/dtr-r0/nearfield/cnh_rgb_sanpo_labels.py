"""Supplement only exact consumed SANPO frames with official panoptic references.

No candidate generation, model inference or training. The visible mask extent
is available; full object size, collision geometry and depth convention are not
established by this acquisition. Cached files are reverified before reuse.
"""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PREFIX = "sanpo_dataset/v0/sanpo-synthetic/"
API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
BASE = "https://storage.googleapis.com/gresearch/"


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return s


def contained(root, relative):
    relative = Path(relative)
    target = root / relative
    if relative.is_absolute() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes expected root: {relative}")
    return target


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objects(s, prefix, delimiter=None):
    params = dict(prefix=prefix, maxResults=1000,
                  fields="items(name,size,md5Hash,generation),nextPageToken")
    if delimiter:
        params["delimiter"] = delimiter
    result = []
    while True:
        r = s.get(API, params=params, timeout=90)
        r.raise_for_status()
        data = r.json()
        result.extend(data.get("items", []))
        if not data.get("nextPageToken"):
            return result
        params["pageToken"] = data["nextPageToken"]


def checked(data, item):
    return (len(data) == int(item["size"]) and
            base64.b64encode(hashlib.md5(data).digest()).decode() == item["md5Hash"])


def acquire(s, item, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if not checked(data, item):
            raise ValueError(f"Existing file fails official size/MD5: {path}")
        reused = True
    else:
        partial = path.with_suffix(path.suffix + ".partial")
        start = partial.stat().st_size if partial.exists() else 0
        if start > int(item["size"]):
            raise ValueError(f"Oversized partial: {partial}")
        if start < int(item["size"]):
            headers = {"Range": f"bytes={start}-"} if start else {}
            with s.get(BASE + item["name"], headers=headers, stream=True, timeout=90) as r:
                r.raise_for_status()
                if start and (r.status_code != 206 or not
                              r.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                    raise ValueError("Server failed to honor resumable range")
                with partial.open("ab" if start else "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
        data = partial.read_bytes()
        if not checked(data, item):
            raise ValueError(f"Downloaded file fails official size/MD5: {path}")
        partial.replace(path)
        reused = False
    return dict(path=str(path), object=item["name"], bytes=len(data),
                md5Hash=item["md5Hash"], generation=item.get("generation"),
                sha256=hashlib.sha256(data).hexdigest(), reused=reused)


def supplement(rows, root, work):
    scene = rows[0]["scene"]
    receipt = contained(work, "sessions/" + scene + ".json")
    with session() as s:
        by_name = {i["name"]: i for i in objects(s, PREFIX + scene + "/")}
        save(contained(work, "object-list/" + scene + ".json"), list(by_name.values()))
        files, frames = [], []
        for row in rows:
            relative = row["rgb"].replace("/video_frames/", "/segmentation_masks/")
            name = PREFIX + relative
            if name not in by_name:
                raise ValueError(f"Exact frame has no official mask: {name}")
            files.append(acquire(s, by_name[name], contained(root, relative)))
            frames.append(dict(id=row["id"], scene=scene, split=row["split"],
                               rgb=row["rgb"], depth=row["depth"], mask=relative))
        metadata = PREFIX + str(Path(rows[0]["rgb"]).parent.parent).replace("\\", "/") + "/frame_segmentation_annotation_type.json"
        if metadata in by_name:
            files.append(acquire(s, by_name[metadata], contained(root, metadata[len(PREFIX):])))
        result = dict(scene=scene, frames=frames, files=files)
        save(receipt, result)
        return result


def audit(result, root):
    stats = []
    semantic_pixels = Counter()
    distinct = set()
    for row in result["frames"]:
        with Image.open(contained(root, row["mask"])) as image:
            if image.mode != "RGB":
                raise ValueError(f"Unexpected mask encoding {image.mode}")
            mask = np.asarray(image)
        with Image.open(contained(root, row["rgb"])) as rgb:
            assert rgb.size == (mask.shape[1], mask.shape[0])
        semantic = mask[:, :, 0]
        instance = mask[:, :, 1].astype(np.uint16) * 256 + mask[:, :, 2]
        hist = np.bincount(instance.ravel(), minlength=65536)
        ids = np.flatnonzero(hist[1:]) + 1
        distinct.update((row["scene"], int(i)) for i in ids)
        semhist = np.bincount(semantic.ravel(), minlength=256)
        semantic_pixels.update({int(i): int(semhist[i]) for i in np.flatnonzero(semhist)})
        stats.append(dict(id=row["id"], mask=row["mask"], height=mask.shape[0],
                          width=mask.shape[1], nonzero_instance_ids=len(ids),
                          nonzero_instance_pixels=int(hist[1:].sum()),
                          semantic_ids=[int(i) for i in np.flatnonzero(semhist)]))
    return dict(scene=result["scene"], frames=stats,
                semantic_pixels=dict(semantic_pixels),
                distinct_nonzero_session_instance_ids=len(distinct))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--workers", default=8, type=int, choices=range(1, 9))
    a = p.parse_args()
    art = a.repo / "artifacts.local"
    if art.resolve() != Path("F:/ba-data/blindassist-artifacts-20260805").resolve():
        p.error("Canonical artifact junction target changed")
    if not a.work.resolve().is_relative_to(art.resolve()):
        p.error("Work must be within canonical artifacts")
    root = contained(art, "datasets/sanpo-synthetic-ba-nfo")
    manifest = contained(art, "work/ba-nfo-20260919/prepared-manifest.json")
    rows = [r for r in read(manifest) if r["source"] == "sanpo"]
    assert len(rows) == len({r["id"] for r in rows}) == 3000
    scenes = defaultdict(list)
    for r in rows:
        assert contained(root, r["rgb"]).is_file() and contained(root, r["depth"]).is_file()
        assert any(r["rgb"].startswith(r["scene"] + "/" + camera + "/left/video_frames/")
                   for camera in ("camera_head", "camera_chest"))
        scenes[r["scene"]].append(r)
    assert len(scenes) == 60 and all(len(v) == 50 for v in scenes.values())
    a.work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    taxonomy_receipts = []
    with session() as s:
        top = {i["name"]: i for i in objects(s, "sanpo_dataset/v0/", "/")}
        for filename in ("labelmap.json", "labeltype.json"):
            taxonomy_receipts.append(acquire(s, top["sanpo_dataset/v0/" + filename],
                                            root / "official-taxonomy" / filename))
    results, errors = [], []
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(supplement, r, root, a.work): k for k, r in scenes.items()}
        for job in as_completed(jobs):
            try:
                results.append(job.result())
                print("ACQUIRED", len(results), "/ 60", jobs[job], flush=True)
            except Exception as e:
                errors.append(dict(scene=jobs[job], error=repr(e)))
                print("ERROR", errors[-1], flush=True)
    save(a.work / "acquisition-status.json", dict(errors=errors, sessions=len(results),
         masks=sum(len(r["frames"]) for r in results), taxonomy=taxonomy_receipts,
         manifest_sha256=sha(manifest)))
    if errors:
        raise SystemExit(1)
    audits = []
    with ThreadPoolExecutor(min(a.workers, 4)) as pool:
        jobs = {pool.submit(audit, r, root): r["scene"] for r in results}
        for job in as_completed(jobs):
            value = job.result()
            save(a.work / "audit" / (value["scene"] + ".json"), value)
            audits.append(value)
            print("AUDITED", len(audits), "/ 60", flush=True)
    files = [f for r in results for f in r["files"]] + taxonomy_receipts
    all_frames = [f for v in audits for f in v["frames"]]
    output = dict(status="OFFICIAL_LABELS_COMPLETE", frames=len(all_frames), sessions=60,
        masks=3000, taxonomy={name: read(root / "official-taxonomy" / name)
                              for name in ("labelmap.json", "labeltype.json")},
        frames_with_nonzero_instance_ids=sum(f["nonzero_instance_ids"] > 0 for f in all_frames),
        frame_instance_observations=sum(f["nonzero_instance_ids"] for f in all_frames),
        distinct_nonzero_session_instance_ids=sum(v["distinct_nonzero_session_instance_ids"]
                                                 for v in audits),
        downloaded_bytes=sum(f["bytes"] for f in files if not f["reused"]),
        total_verified_bytes=sum(f["bytes"] for f in files),
        shapes=sorted({(f["height"], f["width"]) for f in all_frames}),
        encoding="semantic=R; instance=G*256+B; zero instance is not an object ID",
        extent_status="Visible instance pixel areas/bboxes are computable from masks; full physical size unavailable",
        unresolved=["Optical-Z versus radial depth", "Pose direction and camera axes for gravity corridor",
                    "Full object physical dimensions and hidden surfaces", "Final object matching/small-visible-extent contract"],
        scope="Same consumed 3000 frames/60 sessions; no new RGB/depth; no inference/training",
        manifest_sha256=sha(manifest), script_sha256=sha(Path(__file__)),
        seconds=time.perf_counter()-started)
    save(a.work / "result.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
