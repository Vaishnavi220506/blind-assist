"""Frozen bent-path analytic observations and deterministic perturbations.

Simulation metadata is evaluator-only. Inference receives exactly the returned
nominal camera and quantized bins, never actual camera or continuous ranges.
"""
import math

import active_view as av

CONDITIONS = ("nominal", "range_plus2mm", "range_minus2mm", "pose_plus1mm", "pose_minus1mm")


def paths():
    """Thirteen samples on each .12 m path; the two bends end at (.06,.06)."""
    return dict(
        straight_x=tuple((round(i*.01, 10), 0.) for i in range(13)),
        x_then_z=tuple((round(i*.01, 10), 0.) if i <= 6 else (.06, round((i-6)*.01, 10))
                       for i in range(13)),
        z_then_x=tuple((0., round(i*.01, 10)) if i <= 6 else (round((i-6)*.01, 10), .06)
                       for i in range(13)))


def conditions():
    """Fresh metadata dictionaries, no mutable shared settings or sampled noise."""
    return {
        "nominal": dict(range_bias_m=0., pose_bias_m=(0., 0.)),
        "range_plus2mm": dict(range_bias_m=.002, pose_bias_m=(0., 0.)),
        "range_minus2mm": dict(range_bias_m=-.002, pose_bias_m=(0., 0.)),
        "pose_plus1mm": dict(range_bias_m=0., pose_bias_m=(.001, .001)),
        "pose_minus1mm": dict(range_bias_m=0., pose_bias_m=(-.001, -.001)),
    }


def observe(scene, nominalcamera, condition):
    """Original ray law, range bias BEFORE quantization, nominal reported pose."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown frozen observation condition")
    if (not isinstance(nominalcamera, (tuple, list)) or len(nominalcamera) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in nominalcamera)):
        raise ValueError("Expected finite nominal X/Z coordinates")
    camera = tuple(float(v) for v in nominalcamera)
    spec = conditions()[condition]
    actual = tuple(round(v+d, 12) for v, d in zip(camera, spec["pose_bias_m"], strict=True))
    raw = []
    for angle in av.ANGLES:
        wall = (scene.wall_z-actual[1])/math.cos(angle)
        obstacle = min((av.hit_distance(b, actual, angle) for b in scene.boxes), default=math.inf)
        distance = min(wall, obstacle)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Actual camera outside the modeled observation domain")
        raw.append(distance)
    biased = [value+spec["range_bias_m"] for value in raw]
    if any(not math.isfinite(v) or v <= 0 for v in biased):
        raise ValueError("Range bias produced an invalid radial measurement")
    bins = [math.floor(value/av.RANGE_STEP+.5) for value in biased]
    if condition == "nominal":
        assert tuple(bins) == av.observe(scene, camera), "Nominal original-observer identity failed"
    return dict(camera=list(camera), bins=bins, actual_camera=list(actual),
                raw_ranges=raw, biased_ranges=biased, condition=condition,
                range_bias_m=spec["range_bias_m"], pose_bias_m=list(spec["pose_bias_m"]),
                raw_range_kind="RADIAL_METRES", quantization_step_m=av.RANGE_STEP,
                orientation_change_radians=0., metadata_authority="SIMULATOR_EVALUATOR_ONLY")
