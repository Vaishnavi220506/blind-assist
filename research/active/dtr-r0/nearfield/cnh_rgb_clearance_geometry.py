"""Evaluator-only Hypersim geometry for a consumed RGB clearance probe.

No model prediction is loaded here.  Floor, pose and instance truth are oracle
ROI prerequisites, not deployable ToF gating or wearer collision truth.
"""
import csv
from functools import lru_cache
import json
from pathlib import Path
import re

import h5py
import numpy as np


def hdf(path):
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def _select_pose(indices, positions, orientations, frame, scale):
    matches = np.flatnonzero(np.asarray(indices) == frame)
    assert len(matches) == 1, "Official frame index must match exactly once"
    assert np.isfinite(scale) and scale > 0
    index = int(matches[0])
    position = np.asarray(positions[index], dtype=np.float64) * scale
    orientation = np.asarray(orientations[index], dtype=np.float64)
    assert position.shape == (3,) and orientation.shape == (3, 3)
    assert np.isfinite(position).all() and np.isfinite(orientation).all()
    return position, orientation


@lru_cache(maxsize=128)
def _camera_metadata(detail, camera):
    detail = Path(detail)
    paths = [detail / camera / ("camera_keyframe_" + name + ".hdf5")
             for name in ("frame_indices", "positions", "orientations")]
    scale_path = detail / "metadata_scene.csv"
    with scale_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scale = float(next(row["parameter_value"] for row in rows
                       if row["parameter_name"] == "meters_per_asset_unit"))
    return tuple(hdf(path) for path in paths), scale, paths + [scale_path]


def _ray_factors(matrix, orientation, shape):
    height, width = shape
    yy, xx = np.indices(shape)
    uv = np.stack(((xx + .5) / width * 2 - 1,
                   1 - (yy + .5) / height * 2, np.ones(shape)), axis=-1)
    matrix = np.asarray(matrix, dtype=np.float64)
    assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
    rays = uv @ matrix.T
    assert np.all(rays[..., 2] < 0), "Official camera looks along negative z"
    # Optical-Z unprojection; do not interpret cached optical depth as radial.
    camera_per_z = rays / -rays[..., 2:]
    world_per_z = camera_per_z @ orientation.T
    up = np.array([0., 0., 1.])
    forward = -orientation[:, 2].copy()
    forward[2] = 0
    norm = float(np.linalg.norm(forward))
    heading_valid = np.isfinite(norm) and norm > 1e-8
    if heading_valid:
        forward /= norm
        right = np.cross(forward, up)
        lateral = world_per_z @ right
        longitudinal = world_per_z @ forward
    else:
        forward[:] = np.nan
        right = np.full(3, np.nan)
        lateral = np.full(shape, np.nan)
        longitudinal = np.full(shape, np.nan)
    return dict(lateral_factor=lateral.astype(np.float32),
                forward_factor=longitudinal.astype(np.float32),
                world_up_factor=world_per_z[..., 2].astype(np.float32),
                optical_z_per_radial=(-rays[..., 2] / np.linalg.norm(rays, axis=-1)).astype(np.float32),
                heading_valid=bool(heading_valid),
                heading_horizontal_norm=norm,
                forward_world=forward, right_world=right)


def camera_geometry(repo, row, camera_matrix):
    """Public camera geometry; arrays multiply optical-axis metric Z.

    Original full M_cam_from_uv is retained, including tilt-shift projection.
    Official orientation is used unchanged.  Nonorthogonality is disclosed
    rather than silently repaired by a pose fit.
    """
    root = Path(repo) / "artifacts.local/datasets/hypersim-ba-nfo"
    match = re.search(r"scene_(cam_\d+)_final_preview/frame\.(\d+)\.", row["rgb"])
    assert match, row["rgb"]
    camera, frame = match.group(1), int(match.group(2))
    detail = root / row["scene"] / "_detail"
    arrays, scale, paths = _camera_metadata(str(detail), camera)
    position, orientation = _select_pose(*arrays, frame, scale)
    error = float(np.max(np.abs(orientation.T @ orientation - np.eye(3))))
    determinant = float(np.linalg.det(orientation))
    shape = (768, 1024)
    result = _ray_factors(camera_matrix, orientation, shape)
    result.update(shape=list(shape), camera_position_m=position,
                  camera_world_from_camera=orientation,
                  metadata_paths=[str(path.absolute()) for path in paths],
                  frame_index=frame, camera_name=camera,
                  meters_per_asset_unit=scale,
                  pose_orthogonality_error=error, pose_determinant=determinant,
                  pose_usable_for_cm_probe=bool(error <= 1e-4 and abs(determinant - 1) <= 1e-4),
                  role="public camera calibration; no wearer mounting calibration")
    return result


def reference_frame(repo, row, camera_matrix):
    """Aligned native evaluator truth; instance -1 remains unannotated."""
    result = camera_geometry(repo, row, camera_matrix)
    path = Path(repo) / "artifacts.local/datasets/hypersim-ba-nfo" / row["depth"]
    radial = hdf(path).astype(np.float32)
    semantic_path = path.with_name(path.name.replace("depth_meters", "semantic"))
    instance_path = path.with_name(path.name.replace("depth_meters", "semantic_instance"))
    semantic, instance = hdf(semantic_path), hdf(instance_path)
    assert radial.shape == semantic.shape == instance.shape == tuple(result["shape"])
    assert np.issubdtype(semantic.dtype, np.integer)
    assert np.issubdtype(instance.dtype, np.integer)
    optical = radial * result["optical_z_per_radial"]
    result.update(radial=radial, optical_z=optical, semantic=semantic, instance=instance,
                  lateral=optical * result["lateral_factor"],
                  forward=optical * result["forward_factor"],
                  world_height=result["camera_position_m"][2] + optical * result["world_up_factor"],
                  reference_paths=[str(p.absolute()) for p in (path, semantic_path, instance_path)],
                  reference_role="GT evaluator-only visible surfaces, not complete object collision volume")
    return result


def infer_floor(reference):
    """Known floor truth for nominal body ROI only; failures stay UNKNOWN."""
    shape = tuple(reference["shape"])
    result = dict(floor_height_m=None, pixels=0, coverage_fraction=0.,
                  spread_m=None, reason="OK",
                  role="oracle local floor registration; not a deployable observation")
    if not reference["heading_valid"]:
        result["reason"] = "HEADING_UNDEFINED"
        return result
    if not reference.get("pose_usable_for_cm_probe", True):
        result["reason"] = "POSE_NOT_ORTHONORMAL_FOR_CM_PROBE"
        return result
    valid = ((reference["semantic"] == 2)
             & np.isfinite(reference["radial"]) & (reference["radial"] > 0)
             & np.isfinite(reference["world_height"])
             & (reference["forward"] >= .3) & (reference["forward"] <= 6.))
    values = reference["world_height"][valid]
    result.update(pixels=int(values.size), coverage_fraction=float(values.size / np.prod(shape)))
    if values.size < 256:
        result["reason"] = "INSUFFICIENT_VISIBLE_FLOOR_SUPPORT"
        return result
    q05, q95 = np.percentile(values, (5, 95))
    spread = float(q95 - q05)
    result.update(spread_m=spread, q05_m=float(q05), q95_m=float(q95))
    if spread > .10:
        result["reason"] = "FLOOR_LEVEL_AMBIGUOUS_OR_NONPLANAR"
        return result
    result["floor_height_m"] = float(np.median(values))
    result["camera_height_above_floor_m"] = float(reference["camera_position_m"][2] - result["floor_height_m"])
    return result


def self_check(repo=None, row=None, camera_matrix=None):
    """Focused geometry invariants plus optional one-frame source inspection."""
    matrix = np.diag([.6, .45, -1.])
    # Camera +x is right, +y up, -z forward; world +z is up.
    base = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    theta = .3
    roll = np.array([[np.cos(theta), -np.sin(theta), 0.],
                     [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    for rotation in (base, base @ roll):
        factors = _ray_factors(matrix, rotation, (5, 7))
        radial = np.full((5, 7), 2.)
        z = radial * factors["optical_z_per_radial"]
        reconstructed = np.stack([z * factors[key] for key in
                                  ("lateral_factor", "forward_factor", "world_up_factor")], -1)
        np.testing.assert_allclose(np.linalg.norm(reconstructed, axis=-1), radial, atol=2e-7)
        assert factors["heading_valid"]
        np.testing.assert_allclose(factors["forward_factor"], 1., atol=1e-7)
    assert not _ray_factors(matrix, np.eye(3), (5, 7))["heading_valid"]
    p, r = _select_pose(np.array([9, 3]), np.array([[20., 0., 0.], [100., 200., 300.]]),
                        np.stack([np.eye(3), base]), 3, .01)
    np.testing.assert_allclose(p, [1., 2., 3.])
    np.testing.assert_array_equal(r, base)
    checks = dict(radial_optical_z_and_world_norm="PASS", roll_and_heading="PASS",
                  frame_index_and_asset_scale="PASS")
    if repo is not None:
        root = Path(repo) / "artifacts.local/work/ba-nfo-depthpro-20260919"
        if row is None:
            row = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[0]
        if camera_matrix is None:
            observations = json.loads((root / "observations.json").read_text(encoding="utf-8"))
            camera_matrix = next(x["camera_matrix"] for x in observations if x["id"] == row["id"])
        ref = reference_frame(repo, row, camera_matrix)
        known = np.isfinite(ref["radial"]) & (ref["radial"] > 0)
        xyz = np.stack([ref["lateral"], ref["forward"],
                        ref["world_height"] - ref["camera_position_m"][2]], -1)
        # Public pose is not silently orthonormalized: allow its measured error.
        tolerance = max(2e-6, ref["pose_orthogonality_error"] * 2)
        np.testing.assert_allclose(np.linalg.norm(xyz[known], axis=-1), ref["radial"][known],
                                   rtol=tolerance, atol=2e-5)
        checks["source_frame"] = dict(id=row["id"], shape=ref["shape"],
            pose_orthogonality_error=ref["pose_orthogonality_error"],
            floor=infer_floor(ref), status="PASS")
    return checks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    print(json.dumps(self_check(args.repo), allow_nan=False, indent=2))
