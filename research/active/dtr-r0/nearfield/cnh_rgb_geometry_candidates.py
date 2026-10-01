"""Zero-training protrusion candidates from cached RGB-model optical-Z depth.

This module observes predicted depth and a public camera matrix only. It never
reads semantic, instance, reference-depth or corridor labels, and performs no
model inference. The caller owns independent labels and evaluation. Candidates
are relative local foreground evidence, not detected physical objects or alerts.
They are not filtered by absolute predicted distance or a corridor, so candidate
quality and scale error can be evaluated separately. Native pixels are retained.

Mechanism: close log(depth) at three fixed square scales, subtract log(depth),
then retain the maximum response. Each 4-connected thresholded component with
at least 16 pixels becomes a candidate. Its score is the median component
response, resisting isolated extreme responses. Bounding boxes use exclusive
ends; each returned mask occupies only its bounding box, not the whole image.
Invalid predictions are nearest-valid-filled for morphology and excluded from
components. This is numerical repair, not a prediction of missing geometry.

Run this file directly for two synthetic implementation checks. These checks
are not scientific evidence or source-frame tuning.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy import ndimage


# Plain JSON types allow the evaluator to record the mechanism before scoring.
SOURCE_PARAMETERS = {
    "mechanism": "multiscale_log_depth_black_hat",
    "window_sizes_native_pixels": [33, 65, 129],
    "window_shape": "square",
    "boundary_mode": "nearest",
    "scale_aggregation": "maximum",
    "threshold": "response >= log(ratio)",
    "connectivity": 4,
    "minimum_component_native_pixels": 16,
    "score": "median_component_log_depth_response",
    "invalid_depth": "nearest_valid_fill_for_morphology_exclude_from_components",
    "bbox_order": ["y0", "x0", "y1_exclusive", "x1_exclusive"],
    "mask": "bool_local_to_bbox",
    "resize": False,
    "absolute_distance_filter": False,
    "corridor_filter": False,
    "camera_matrix_role": "public_input_validation_only",
}


def extract_candidates(
    depth: np.ndarray, camera_matrix: np.ndarray, ratio: float
) -> list[dict]:
    """Return ``bbox``, median log-response ``score`` and bbox-local ``mask``.

    ``depth`` is a 2-D native optical-Z prediction, with finite positive values
    considered valid. ``camera_matrix`` must be a finite invertible 3-by-3 public
    matrix; it does not change pixel morphology. ``ratio`` must exceed one.
    Returned boolean masks identify component pixels within their own boxes.
    A candidate therefore has ``mask.shape == (y1-y0, x1-x0)``. Components may
    contain holes and must not be replaced by solid rectangles by the evaluator.
    """
    depth = np.asarray(depth, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a nonempty 2-D native optical-Z array")
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("camera_matrix must be finite and 3-by-3")
    if np.linalg.matrix_rank(camera_matrix) != 3:
        raise ValueError("camera_matrix must be invertible")
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("ratio must be finite and greater than one")

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return []
    if valid.all():
        filled = depth
    else:
        nearest = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = depth[tuple(nearest)]
    log_depth = np.log(filled)
    response = np.zeros(depth.shape, dtype=np.float64)
    for size in SOURCE_PARAMETERS["window_sizes_native_pixels"]:
        closed = ndimage.grey_closing(log_depth, size=(size, size), mode="nearest")
        np.maximum(response, closed - log_depth, out=response)

    foreground = valid & (response >= math.log(ratio))
    components, _ = ndimage.label(
        foreground, structure=ndimage.generate_binary_structure(2, 1)
    )
    candidates = []
    for component_id, bounds in enumerate(ndimage.find_objects(components), 1):
        if bounds is None:
            continue
        local_mask = components[bounds] == component_id
        if int(local_mask.sum()) < SOURCE_PARAMETERS["minimum_component_native_pixels"]:
            continue
        rows, cols = bounds
        candidates.append({
            "bbox": [int(rows.start), int(cols.start), int(rows.stop), int(cols.stop)],
            "score": float(np.median(response[bounds][local_mask])),
            "mask": local_mask,
        })
    return candidates


def synthetic_checks() -> dict:
    """Check background rejection and a narrow foreground's native footprint."""
    camera_matrix = np.array([[700., 0., 512.], [0., 700., 384.], [0., 0., 1.]])
    background = np.full((768, 1024), 3.0)
    assert extract_candidates(background, camera_matrix, 1.15) == []

    y, x = np.indices(background.shape)
    plane = 3.0 + x * 0.0002 + y * 0.0001
    assert extract_candidates(plane, camera_matrix, 1.15) == []
    expected_box = [180, 500, 260, 510]
    plane[180:260, 500:510] = 1.5
    candidates = extract_candidates(plane, camera_matrix, 1.15)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["bbox"] == expected_box
    assert candidate["mask"].shape == (80, 10)
    assert candidate["mask"].all()
    assert candidate["score"] >= math.log(1.15)
    return {
        "status": "PASS",
        "native_shape": [768, 1024],
        "checks": ["flat_and_sloped_background_no_candidates", "narrow_protrusion_exact_footprint"],
        "protrusion_bbox": candidate["bbox"],
        "protrusion_pixels": int(candidate["mask"].sum()),
        "source_parameters": SOURCE_PARAMETERS,
    }


if __name__ == "__main__":
    print(json.dumps(synthetic_checks(), indent=2))
