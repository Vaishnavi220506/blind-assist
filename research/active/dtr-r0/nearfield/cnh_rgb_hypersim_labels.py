"""Supplement official Hypersim labels for the exact consumed NFO500 cohort.

Standalone acquisition: no import from untracked historical download helpers.
Fetch only selected masks and scene bounding-box/scale metadata using ZIP ranges.
CRC, hashes, native alignment and direct semantic-instance indexing are checked.
This does not generate instances, model outputs, or new scenes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import zipfile
import zlib

import h5py
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OFFICIAL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/"
BBOX = "metadata_semantic_instance_bounding_box_object_aligned_2d_"
SOURCES = {
    "README.md": "https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md",
    "scene_generate_images_bounding_box.py": "https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools/scene_generate_images_bounding_box.py",
    "scene_generate_bounding_boxes.py": "https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools/scene_generate_bounding_boxes.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    part.replace(path)


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.s, self.pos, self.transferred = url, session(), 0, 0
        response = self.s.head(url, timeout=45)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        if n <= 0:
            return b""
        start = self.pos
        response = self.s.get(self.url, headers={"Range": f"bytes={start}-{start+n-1}"}, timeout=60)
        response.raise_for_status()
        assert response.status_code == 206, "Server did not honor range; do not download whole archive"
        assert response.headers["Content-Range"].startswith(f"bytes {start}-{start+n-1}/")
        assert len(response.content) == n
        self.pos += n
        self.transferred += n
        return response.content

    def member(self, entry):
        self.seek(entry.header_offset)
        header = self.read(30)
        assert header[:4] == b"PK\x03\x04"
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        self.seek(entry.header_offset + 30 + name_len + extra_len)
        payload = self.read(entry.compress_size)
        assert entry.compress_type in (0, 8)
        data = zlib.decompress(payload, -15) if entry.compress_type == 8 else payload
        assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC
        return data


def acquire(scene, rows, root, work):
    remote = RemoteZip(OFFICIAL + scene + ".zip")
    receipts, missing = [], []
    try:
        with zipfile.ZipFile(remote) as archive:
            entries = {entry.filename: entry for entry in archive.infolist()}
        wanted = {scene + "/_detail/metadata_scene.csv"}
        wanted.update(scene + "/_detail/mesh/" + BBOX + key + ".hdf5"
                      for key in ("extents", "positions", "orientations"))
        for row in rows:
            wanted.update(row["depth"].replace("depth_meters", key)
                          for key in ("semantic", "semantic_instance"))
        for name in sorted(wanted):
            if name not in entries:
                missing.append(name)
                continue
            entry = entries[name]
            path = root / name
            assert path.resolve().is_relative_to(root.resolve())
            if path.exists():
                data = path.read_bytes()
                assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC, str(path)
                reused = True
            else:
                data = remote.member(entry)
                path.parent.mkdir(parents=True, exist_ok=True)
                part = path.with_suffix(path.suffix + ".partial")
                part.write_bytes(data)
                part.replace(path)
                reused = False
            receipts.append(dict(path=name, bytes=len(data), sha256=sha(data),
                zip_crc32=entry.CRC, reused=reused))
        value = dict(scene=scene, frames=len(rows), source_url=remote.url,
            archive_bytes=remote.size, transferred_bytes=remote.transferred,
            files=receipts, missing=missing)
        save_json(work / "receipts" / (scene + ".json"), value)
        return {key: value[key] for key in ("scene", "frames", "transferred_bytes", "missing")}
    finally:
        remote.s.close()


def hdf(path):
    with h5py.File(path, "r") as f:
        return f["dataset"][:]


def verify(rows, root, prediction_root, work):
    metadata, observed, frames, shapes = {}, {}, [], Counter()
    for scene in sorted({row["scene"] for row in rows}):
        detail = root / scene / "_detail"
        scale_rows = list(csv.DictReader((detail / "metadata_scene.csv").open(encoding="utf-8")))
        scale = float(next(r["parameter_value"] for r in scale_rows if r["parameter_name"] == "meters_per_asset_unit"))
        assert np.isfinite(scale) and scale > 0
        bbox = {key: hdf(detail / "mesh" / (BBOX + key + ".hdf5"))
                for key in ("extents", "positions", "orientations")}
        n = len(bbox["extents"])
        assert bbox["extents"].shape == bbox["positions"].shape == (n, 3)
        assert bbox["orientations"].shape == (n, 3, 3)
        metadata[scene] = dict(scale=scale, bbox=bbox)
        observed[scene] = set()
    for row in rows:
        instance_path = root / row["depth"].replace("depth_meters", "semantic_instance")
        semantic_path = root / row["depth"].replace("depth_meters", "semantic")
        instance, semantic, depth = hdf(instance_path), hdf(semantic_path), hdf(root / row["depth"])
        with Image.open(root / row["rgb"]) as im:
            rgb_shape = (im.height, im.width)
        with np.load(prediction_root / (row["id"] + ".npz")) as prediction:
            prediction_shape = prediction["native_depth"].shape
        assert instance.shape == semantic.shape == depth.shape == rgb_shape == prediction_shape
        assert np.issubdtype(instance.dtype, np.integer) and np.issubdtype(semantic.dtype, np.integer)
        ids = np.unique(instance)
        assert np.all(ids >= -1)
        valid_ids = ids[ids >= 0]
        bbox = metadata[row["scene"]]["bbox"]
        assert np.all(valid_ids < len(bbox["extents"]))
        assert np.all(np.isfinite(bbox["extents"][valid_ids]))
        assert np.all(np.isfinite(bbox["positions"][valid_ids]))
        assert np.all(np.isfinite(bbox["orientations"][valid_ids]))
        assert np.all(bbox["extents"][valid_ids] >= 0)
        observed[row["scene"]].update(int(i) for i in valid_ids)
        shapes[str(instance.shape)] += 1
        frames.append(dict(id=row["id"], scene=row["scene"], instance_path=str(instance_path),
            semantic_path=str(semantic_path), shape=list(instance.shape),
            instance_dtype=str(instance.dtype), visible_instance_ids=[int(i) for i in valid_ids],
            unannotated_pixels=int(np.count_nonzero(instance == -1))))
    objects = []
    for scene, ids in observed.items():
        data = metadata[scene]
        for iid in sorted(ids):
            extent = data["bbox"]["extents"][iid] * data["scale"]
            objects.append(dict(scene=scene, instance_id=iid, extents_m=extent.tolist(),
                position_m=(data["bbox"]["positions"][iid] * data["scale"]).tolist(),
                orientation_world_from_object=data["bbox"]["orientations"][iid].tolist()))
    save_json(work / "verified-frames.json", frames)
    save_json(work / "visible-scene-instances.json", objects)
    return dict(frames=len(frames), scenes=len(metadata), shapes=dict(shapes),
        unique_visible_scene_instances=len(objects),
        total_visible_frame_instances=sum(len(f["visible_instance_ids"]) for f in frames),
        indexing="semantic_instance pixel ID directly indexes bbox array, no minus-one offset; -1 unannotated",
        bbox_coordinates="asset positions and extents multiplied by meters_per_asset_unit; orientations world_from_object",
        mask_alignment="native RGB, depth, Depth Pro native_depth shapes identical in every frame",
        small_object_instances="NOT_COUNTED_WITHOUT_FROZEN_SIZE_CORRIDOR_AND_VISIBLE_SUPPORT_RULE",
        limits=["Consumed Development synthetic scenes; labels do not grant fresh evidence authority",
                "3D oriented bounding boxes describe semantic instances, not separately annotated constituent parts",
                "Shape agreement plus shared official frame/camera provenance verifies raster alignment; no interpolation performed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    artifacts = args.repo / "artifacts.local"
    assert os.lstat(artifacts).st_reparse_tag == 0xA0000003, "Canonical artifacts.local must remain a junction"
    physical = Path("F:/ba-data/blindassist-artifacts-20260805").resolve()
    assert artifacts.resolve() == physical
    root = artifacts / "datasets/hypersim-ba-nfo"
    work = artifacts / "work/cnh-rgb-candidate-supplement-20261001/hypersim"
    prediction = artifacts / "work/ba-nfo-depthpro-20260919/predictions/native"
    for path in (root, work, prediction):
        assert path.resolve().is_relative_to(physical)
    manifest = artifacts / "work/ba-nfo-depthpro-20260919/manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == 500 and len({r["id"] for r in rows}) == 500
    assert all(r["source"] == "hypersim" for r in rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["scene"], []).append(row)
    assert len(groups) == 52
    work.mkdir(parents=True, exist_ok=True)
    s = session()
    source_receipts = []
    try:
        for name, url in SOURCES.items():
            path = work / "official-sources" / name
            if not path.exists():
                response = s.get(url, timeout=45)
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            source_receipts.append(dict(path=str(path), url=url, sha256=sha(path.read_bytes())))
    finally:
        s.close()
    save_json(work / "selection.json", dict(frames=500, scenes=52, manifest=str(manifest),
        manifest_sha256=sha(manifest.read_bytes()), physical_root=str(physical), official_sources=source_receipts))
    if args.verify_only:
        previous = json.loads((work / "download-status.json").read_text(encoding="utf-8"))
        assert len(previous["completed"]) == 52 and not previous["errors"]
        assert not any(r["missing"] for r in previous["completed"])
        previous["verification"] = verify(rows, root, prediction, work)
        previous["status"] = "VERIFIED"
        previous.pop("verification_error", None)
        save_json(work / "download-status.json", previous)
        save_json(work / "attempts" / (str(time.time_ns()) + ".json"), previous)
        print(json.dumps(previous["verification"]), flush=True)
        return
    groups = dict(sorted(groups.items())[:args.limit_scenes])
    started = time.time()
    errors, done = [], []
    with ThreadPoolExecutor(max_workers=min(8, max(1, args.workers))) as pool:
        jobs = {pool.submit(acquire, scene, selected, root, work): scene
                for scene, selected in groups.items()}
        for future in as_completed(jobs):
            try:
                result = future.result()
                done.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                error = dict(scene=jobs[future], error=repr(exc))
                errors.append(error)
                print(json.dumps(error), flush=True)
    status = dict(status="INCOMPLETE" if errors or any(r["missing"] for r in done) else "DOWNLOADED",
        requested_scenes=len(groups), completed=done, errors=errors, seconds=time.time()-started)
    if status["status"] == "DOWNLOADED" and len(groups) == 52:
        try:
            status["verification"] = verify(rows, root, prediction, work)
            status["status"] = "VERIFIED"
        except Exception as exc:
            status["status"] = "VERIFICATION_FAILED"
            status["verification_error"] = repr(exc)
    save_json(work / "download-status.json", status)
    # Preserve each bounded attempt rather than overwriting its failure evidence.
    save_json(work / "attempts" / (str(time.time_ns()) + ".json"), status)
    print(json.dumps({k: v for k, v in status.items() if k not in ("completed",)}), flush=True)
    if status["status"] in ("INCOMPLETE", "VERIFICATION_FAILED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
