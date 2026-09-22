"""Freeze a small continuous approach/near-pass source before capture.

The source describes complete nominal trajectories for the existing
``mz115_zonal_capture.py`` interface.  It contains no model scores or outcome
selection.  Native bounds, target identity and contact labels are evaluator
metadata; the public predictor receives only the captured RGB/sensor records.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = (ROOT / "artifacts.local").resolve()
SEED = 177018
DT_S = 0.1
STEPS = 80
FAMILIES = ("thin_rod", "suspended_bar", "body")
PROCESSES = ("approach", "side_pass", "stop_back", "head_turn")
LAYOUTS = (0, 1)
CONTACT_FORWARD_M = 0.2
CONTACT_LATERAL_M = 0.3
CONTACT_HEIGHT = (0.4, 2.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def intersects(frame, obj, lateral=CONTACT_LATERAL_M,
               forward=(CONTACT_FORWARD_M, 3.6), height=CONTACT_HEIGHT):
    body = frame["body_origin_m"]
    lo = [c - s / 2 - b for c, s, b in zip(obj["center_m"], obj["size_m"], body)]
    hi = [c + s / 2 - b for c, s, b in zip(obj["center_m"], obj["size_m"], body)]
    return (hi[0] >= forward[0] and lo[0] <= forward[1]
            and hi[1] >= -lateral and lo[1] <= lateral
            and hi[2] >= height[0] and lo[2] <= height[1])


def contact(frame, obj):
    """Evaluator-only body-envelope contact proxy, not a model input."""
    body = frame["body_origin_m"]
    # The body envelope is intentionally smaller than the alert corridor.
    lo = [c - s / 2 - b for c, s, b in zip(obj["center_m"], obj["size_m"], body)]
    hi = [c + s / 2 - b for c, s, b in zip(obj["center_m"], obj["size_m"], body)]
    return (hi[0] >= 0.0 and lo[0] <= 0.75 and hi[1] >= -0.22 and lo[1] <= 0.22
            and hi[2] >= 0.45 and lo[2] <= 1.95)


def target_size(family):
    return {
        "thin_rod": [0.055, 0.05, 2.2],
        "suspended_bar": [0.12, 0.62, 0.18],
        "body": [0.35, 1.0, 1.7],
    }[family]


def target_height(family):
    return {"thin_rod": 1.1, "suspended_bar": 1.9, "body": 0.95}[family]


def camera_pose(process, t, layout):
    side = -1.0 if layout == 0 else 1.0
    if process == "approach":
        x, y, yaw = 0.50 * t, 0.0, side * 2.5 * math.sin(0.55 * t)
    elif process == "side_pass":
        x, y, yaw = 0.55 * t, 0.0, side * 2.0 * math.sin(0.45 * t)
    elif process == "stop_back":
        if t < 4.0:
            x = 0.95 * t
        elif t < 5.5:
            x = 3.8
        else:
            x = max(0.0, 3.8 - 1.6 * (t - 5.5))
        y, yaw = 0.0, side * 1.5 * math.sin(0.4 * t)
    elif process == "head_turn":
        x, y, yaw = 1.0, 0.0, side * 30.0 * math.sin(2.0 * math.pi * t / 8.0)
    else:
        raise ValueError(process)
    return dict(x=x, y=y, z=1.7, yaw=yaw, pitch=-3.0, roll=0.0)


def target_center(family, process, t, layout):
    side = -1.0 if layout == 0 else 1.0
    # Keep both approach layouts close enough to reach the body envelope during
    # the finite clip; side-pass deliberately remains outside the lateral tube.
    x = {"approach": 3.95, "side_pass": 3.55,
         # The wearer stops just inside the body-envelope contact proxy and
         # then backs far enough to leave the 3.6 m corridor.
         "stop_back": 3.85, "head_turn": 4.0}[process]
    if process == "side_pass":
        # Keep the largest body half-width (0.50 m) clear of the 0.30 m
        # corridor, with a small geometric margin for native bounds.
        y = side * 0.90
    elif process == "head_turn":
        y = side * 0.10
    else:
        y = side * (0.04 if layout == 0 else 0.11)
    return [x, y, target_height(family)]


def episode(family, process, layout, ordinal):
    episode_id = f"dtr177_{family}_{process}_layout{layout}"
    sensor_seed = SEED + ordinal * 7919
    size = target_size(family)
    frames, truth = [], []
    for index in range(STEPS):
        t = index * DT_S
        camera = camera_pose(process, t, layout)
        target = dict(name="target", center_m=target_center(family, process, t, layout),
                      size_m=size, texture_seed=sensor_seed + 17,
                      texture_grid=[6, 10], tof_reflectance_proxy=0.55,
                      source_role=family)
        # This wall is a context actor in native capture and must be present in
        # evaluator bounds, but its AABB is outside the current corridor.
        background = dict(name="background", center_m=[10.0, 0.0, 1.8],
                          size_m=[0.15, 20.0, 8.0], texture=False,
                          tof_reflectance_proxy=0.5,
                          source_role="distant_context")
        frame = dict(
            id=f"{episode_id}_{index:03d}", episode=episode_id,
            sequence_id=episode_id, family=family, process=process,
            layout=layout, time_s=round(t, 6), camera=camera,
            body_origin_m=[camera["x"], camera["y"], 0.0],
            objects=[target, background], sensor_seed=sensor_seed,
            tof_sensor_seed=sensor_seed + 100003, wearer_speed_mps=None,
            radar_ghost=None,
        )
        if index:
            previous = frames[-1]["camera"]
            frame["wearer_speed_mps"] = math.sqrt(
                ((camera["x"] - previous["x"]) / DT_S) ** 2
                + ((camera["y"] - previous["y"]) / DT_S) ** 2)
        else:
            frame["wearer_speed_mps"] = 0.0
        frame["source_corridor_truth"] = bool(intersects(frame, target))
        frame["source_contact_truth"] = bool(contact(frame, target))
        frames.append(frame)
        truth.append((frame["source_corridor_truth"], frame["source_contact_truth"]))
    return dict(id=episode_id, family=family, process=process, layout=layout,
                sensor_seed=sensor_seed, frames=frames,
                corridor_truth=[x[0] for x in truth], contact_truth=[x[1] for x in truth],
                target_size_m=size)


def source():
    episodes = []
    ordinal = 0
    for family in FAMILIES:
        for process in PROCESSES:
            for layout in LAYOUTS:
                episodes.append(episode(family, process, layout, ordinal))
                ordinal += 1
    frames = [frame for ep in episodes for frame in ep["frames"]]
    return dict(
        schema="dtr-continuous-approach-source-v1", seed=SEED,
        authority="CONTROLLED_UE_CONTINUOUS_DEVELOPMENT_NOT_HARDWARE_NATURAL_OR_SAFETY",
        rig=dict(width=640, height=360, hfov_deg=70.0, tof_hfov_deg=45.0,
                 tof_rows=8, tof_columns=8, rgb_camera_count=1),
        sampling=dict(dt_s=DT_S, nominal_hz=1.0 / DT_S, steps_per_episode=STEPS,
                      duration_s=(STEPS - 1) * DT_S,
                      wall_clock_not_guaranteed=True),
        families=list(FAMILIES), processes=list(PROCESSES), layouts=list(LAYOUTS),
        episode_count=len(episodes), frame_count=len(frames), episodes=episodes,
        # The capture adapter consumes this flattened view; the nested episode
        # copy remains the audit authority for labels and counts.
        frames=frames,
        background=dict(center_m=[10.0, 0.0, 1.8], size_m=[0.15, 20.0, 8.0],
                        texture=False, tof_reflectance_proxy=0.5),
        floor=dict(center_m=[4.0, 0.0, -0.05], size_m=[24.0, 20.0, 0.1],
                   tof_reflectance_proxy=0.3),
        corridor_m=dict(forward=[0.2, 3.6], lateral=[-0.3, 0.3], height=[0.4, 2.05]),
        contact_proxy=dict(forward=[0.0, 0.75], lateral=[-0.22, 0.22],
                           height=[0.45, 1.95], authority="EVALUATOR_ONLY"),
        source_selection="FIXED_ALL_24_EPISODES_BEFORE_CAPTURE_OR_MODEL_OUTPUT",
        predictor_boundary=(
            "Public raw RGB/sensor records only; episode/process/family/layout, "
            "native bounds and source labels are evaluator-only."),
        limitations=[
            "Synthetic UE scene and hypothetical ToF/Radar simulator remain Development evidence.",
            "Nominal 10 Hz timestamps are trajectory timestamps, not a measured transport cadence.",
            "The same renderer and public sensor contract are reused; no natural-distribution claim.",
            "Contact is a body-envelope proxy for timing diagnostics, not a physical safety threshold.",
            "Twenty-four sequences are a mechanism probe; frame counts are not independent events.",
        ],
    )


def check_source(spec):
    assert spec["schema"] == "dtr-continuous-approach-source-v1"
    episodes = spec["episodes"]
    assert len(episodes) == 24 and spec["frame_count"] == 24 * STEPS
    assert Counter((e["family"], e["process"]) for e in episodes) == {
        (family, process): 2 for family in FAMILIES for process in PROCESSES}
    ids = [f["id"] for e in episodes for f in e["frames"]]
    assert len(ids) == len(set(ids)) == 24 * STEPS
    for ep in episodes:
        rows = ep["frames"]
        assert len(rows) == STEPS
        assert [r["time_s"] for r in rows] == [round(i * DT_S, 6) for i in range(STEPS)]
        assert all(r["episode"] == ep["id"] for r in rows)
        assert all(r["objects"][0]["name"] == "target" for r in rows)
        assert all(not intersects(r, r["objects"][1]) for r in rows)
        assert ep["corridor_truth"] == [bool(r["source_corridor_truth"]) for r in rows]
        assert ep["contact_truth"] == [bool(r["source_contact_truth"]) for r in rows]
        if ep["process"] == "side_pass":
            assert not any(ep["corridor_truth"]), "side pass must remain a non-contact control"
        if ep["process"] == "head_turn":
            assert max(abs(r["camera"]["x"] - rows[0]["camera"]["x"]) for r in rows) == 0.0
            assert max(abs(r["camera"]["yaw"]) for r in rows) >= 25.0
        assert all(r["body_origin_m"][:2] == [r["camera"]["x"], r["camera"]["y"]]
                   for r in rows)
    return dict(status="PASS", episodes=24, frames=24 * STEPS,
                family_counts=dict(Counter(e["family"] for e in episodes)),
                process_counts=dict(Counter(e["process"] for e in episodes)),
                corridor_positive_frames=sum(sum(e["corridor_truth"]) for e in episodes),
                contact_positive_frames=sum(sum(e["contact_truth"]) for e in episodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ARTIFACTS) or out == ARTIFACTS or out.exists():
        raise ValueError("Fresh source output must be strictly under artifacts.local")
    spec = source()
    audit = check_source(spec)
    out.mkdir(parents=True)
    write(out / "spec.json", spec)
    write(out / "source-audit.json", audit)
    write(out / "freeze.json", dict(
        status="FROZEN_BEFORE_CAPTURE_OR_MODEL_OUTPUTS",
        source_sha256=sha(Path(__file__)), spec_sha256=sha(out / "spec.json"),
        audit_sha256=sha(out / "source-audit.json"),
        source_selection=spec["source_selection"],
        capture_attempt_budget=1, training_steps=0, threshold_changes=0,
    ))
    print(json.dumps(dict(output=str(out), **audit), indent=2))


if __name__ == "__main__":
    main()
