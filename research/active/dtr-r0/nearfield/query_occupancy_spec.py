"""Independent wide-distance, grouped source for visible query occupancy.

Source geometry is evaluator-only. Posed samples are not a real-time motion
sequence. Visibility failures remain in the cohort for separate reporting.
"""
from collections import Counter, defaultdict
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from core_transfer_spec import MAP_SHA, PROFILE, bounds, classify
from data_coverage_spec import RANGES, MATERIALS, BACKGROUNDS
from spatial_bce_spec import FAMILIES, RELATIONS

SEED = 202609223
FRAMES_PER_CLIP = 12
FRAMES = 1728
SPLIT_GROUPS = dict(train=6, dev=2, evaluation=4)  # Per family.
FRONT_TRAJECTORY = (4.0, 3.35, 2.85, 2.35, 1.85, 1.35, .85, .5, 1.35, 2.35, 3.35, 4.0)
DT_S = .2
CAPTURE_TIMEOUT_S = 2400


def geometry_draws(family_index, split_index, count):
    draws = [[] for _ in range(count)]
    for axis, (low, high) in enumerate(RANGES[FAMILIES[family_index][0]]):
        rng = random.Random(SEED + 100003 * family_index + 1009 * split_index + 71 * axis)
        strata = list(range(count))
        rng.shuffle(strata)
        for rank, stratum in enumerate(strata):
            draws[rank].append(round(low + (high - low) * (stratum + rng.random()) / count, 6))
    return draws


def relative_signature(case):
    return tuple((tuple(o["size_m"]), tuple(round(o["center_m"][j] - case["camera"][k], 6)
        for j, k in enumerate(("x", "y", "z")))) for o in case["objects"])


def specification():
    groups, clips, cases = [], [], []
    for fi, (kind, layer, _size, _height) in enumerate(FAMILIES):
        order = list(range(12))
        random.Random(SEED + fi).shuffle(order)
        partitions, geometry, appearance = {}, {}, {}
        offset = 0
        for si, (split, count) in enumerate(SPLIT_GROUPS.items()):
            for rank, (k, draw) in enumerate(zip(order[offset:offset + count], geometry_draws(fi, si, count))):
                partitions[k], geometry[k] = split, draw
                appearance[k] = ((rank + 2 * fi + si) % len(MATERIALS),
                                 (rank + fi + 2 * si) % len(BACKGROUNDS))
            offset += count
        for k in range(12):
            rng = random.Random(SEED + 1009 * fi + 67 * k)
            group = f"query_occupancy_{kind}_g{k:02d}"
            split, size, height = partitions[k], geometry[k][:3], geometry[k][3]
            side = (-1, 1)[rng.randrange(2)]
            intrusion = round(rng.uniform(.006, .017), 6)
            world_x, camera_y = round(31.7 + rng.uniform(0, .8), 6), round(rng.uniform(-.12, .12), 6)
            jitter = round(rng.uniform(-.03, .03), 6)
            front = [round(z + jitter, 6) for z in FRONT_TRAJECTORY]
            target_material = "/Game/StreetLab/Materials/" + MATERIALS[appearance[k][0]]
            background = dict(name="background", kind="cube",
                center_m=[round(world_x + rng.uniform(3.7, 4.9), 6),
                          round(camera_y + rng.uniform(-.33, .33), 6), round(rng.uniform(2.08, 2.18), 6)],
                size_m=[round(rng.uniform(.22, .31), 6), round(rng.uniform(6.2, 7.3), 6),
                        round(rng.uniform(4.3, 4.7), 6)],
                material="/Game/StreetLab/Materials/" + BACKGROUNDS[appearance[k][1]])
            group_meta = dict(base_group_id=group, split=split, type_id=kind, layer=layer,
                background="background_" + group, boundary_intrusion_m=intrusion,
                trajectory_jitter_m=jitter)
            groups.append(dict(**group_meta, clips=[]))
            inside, outside = rng.uniform(.105, .18), rng.uniform(.065, .15)
            for relation in RELATIONS:
                lateral = side * (.3 + size[1] / 2 +
                    (-inside if relation == "INSIDE" else -intrusion if relation == "BOUNDARY" else outside))
                clip_id = group + "_" + relation.lower()
                groups[-1]["clips"].append(clip_id)
                target = dict(name="target", kind="cube",
                    center_m=[world_x, round(camera_y + lateral, 9), height],
                    size_m=list(size), material=target_material)
                meta = dict(**group_meta, clip_id=clip_id, layout_relation=relation,
                    arrangement_id=clip_id, frames=FRAMES_PER_CLIP)
                clips.append(meta)
                for i, z in enumerate(front):
                    cases.append(dict(**{key: value for key, value in meta.items() if key != "frames"},
                        name=f"{clip_id}_{i:02d}", pair_id=group, frame_in_clip=i,
                        time_s=round(DT_S * i, 6), target_name="target",
                        phase="approach" if i <= 7 else "depart",
                        sensor_noise_key=f"{group}_t{i:02d}",
                        camera=dict(x=round(world_x - size[0] / 2 - z, 9), y=camera_y,
                                    z=1.82, pitch=0., yaw=0., roll=0.),
                        objects=[copy.deepcopy(target), copy.deepcopy(background)]))
    return dict(schema="query-occupancy-source-v1", seed=SEED, frames=FRAMES,
        frames_per_clip=FRAMES_PER_CLIP, dt_s=DT_S, groups=groups, clips=clips, cases=cases,
        profile=copy.deepcopy(PROFILE), expected_map_sha256=MAP_SHA,
        map="/Game/StreetLab/WillowSampleV1",
        calibration=dict(width=640, height=360, hfov_deg=100., camera_world_height_m=1.82,
                         pitch_deg=0., yaw_deg=0., roll_deg=0., depth_axis="OPTICAL_Z_METRES"),
        base_front_trajectory_m=list(FRONT_TRAJECTORY), maximum_group_jitter_m=.03,
        geometry_ranges_m={k: [list(r) for r in v] for k, v in RANGES.items()},
        split_groups_per_family=dict(SPLIT_GROUPS),
        visibility_policy="RETAIN_ALL_FRAMES_REPORT_MISSING_OR_OCCLUDED_SUPPORT_SEPARATELY",
        sampling="POSED_QUASI_STATIC_SAMPLED_TRAJECTORY_NOT_REAL_TIME_SENSOR_OR_HUMAN_MOTION",
        independence="48 new procedural base groups; split24train/8dev/16evaluation. "
            "Lateral pairs share geometry/material/background/camera/noise start; only target camera-X changes. "
            "Four old cuboid families and existing ranges, new seed and wide-distance poses. "
            "Same Willow renderer/profile/sensor assumptions; no natural-source or hardware claim.")


def check_spec(spec, old_specs=()):
    """Source-only checks; no image, native-depth payload or learned outcome."""
    assert spec["schema"] == "query-occupancy-source-v1" and spec["seed"] == SEED
    assert spec["expected_map_sha256"] == MAP_SHA and spec["profile"] == PROFILE
    assert spec["frames"] == FRAMES and len(spec["cases"]) == FRAMES
    assert spec["frames_per_clip"] == FRAMES_PER_CLIP and spec["dt_s"] == DT_S
    assert spec["base_front_trajectory_m"] == list(FRONT_TRAJECTORY)
    assert spec["maximum_group_jitter_m"] == .03
    assert spec["geometry_ranges_m"] == {k: [list(r) for r in v] for k, v in RANGES.items()}
    assert spec["split_groups_per_family"] == SPLIT_GROUPS
    assert spec["visibility_policy"] == "RETAIN_ALL_FRAMES_REPORT_MISSING_OR_OCCLUDED_SUPPORT_SEPARATELY"
    assert len(spec["groups"]) == 48 and len(spec["clips"]) == 144
    assert len({c["name"] for c in spec["cases"]}) == FRAMES
    now = {relative_signature(c) for c in spec["cases"]}
    for old in old_specs:
        assert not now & {relative_signature(c) for c in old["cases"]}, "Old relative geometry reused"
        assert not {c["name"] for c in spec["cases"]} & {c["name"] for c in old["cases"]}
    by_clip, by_group, signatures = defaultdict(list), defaultdict(list), defaultdict(set)
    for case in spec["cases"]:
        by_clip[case["clip_id"]].append(case)
        by_group[case["base_group_id"]].append(case)
        signatures[case["split"]].add(relative_signature(case))
    assert set(signatures) == set(SPLIT_GROUPS)
    assert all(not signatures[a] & signatures[b] for a, b in
               (("train", "dev"), ("train", "evaluation"), ("dev", "evaluation")))
    assert Counter(g["split"] for g in spec["groups"]) == dict(train=24, dev=8, evaluation=16)
    assert Counter(c["split"] for c in spec["cases"]) == dict(train=864, dev=288, evaluation=576)
    for family, *_ in FAMILIES:
        assert Counter(g["split"] for g in spec["groups"] if g["type_id"] == family) == SPLIT_GROUPS
    geometry_seen = set()
    for group in spec["groups"]:
        rows = by_group[group["base_group_id"]]
        assert len(rows) == 36 and {c["split"] for c in rows} == {group["split"]}
        assert {c["type_id"] for c in rows} == {group["type_id"]}
        assert {c["trajectory_jitter_m"] for c in rows} == {group["trajectory_jitter_m"]}
        assert abs(group["trajectory_jitter_m"]) <= .03
        target = rows[0]["objects"][0]
        geometry = (*target["size_m"], target["center_m"][2])
        assert geometry not in geometry_seen, "Base groups must have distinct target geometry"
        geometry_seen.add(geometry)
        assert {c["clip_id"] for c in rows} == set(group["clips"])
        for i in range(FRAMES_PER_CLIP):
            paired = [c for c in rows if c["frame_in_clip"] == i]
            assert len(paired) == 3 and {c["layout_relation"] for c in paired} == set(RELATIONS)
            assert all(c["camera"] == paired[0]["camera"] and c["objects"][1] == paired[0]["objects"][1]
                       and c["sensor_noise_key"] == paired[0]["sensor_noise_key"] for c in paired)
            targets = [copy.deepcopy(c["objects"][0]) for c in paired]
            for target in targets:
                target["center_m"][1] = 0
            assert targets[0] == targets[1] == targets[2], "Lateral pair changes more than target camera-X"
    counts = {}
    for offset in range(0, FRAMES, FRAMES_PER_CLIP):
        seq = spec["cases"][offset:offset + FRAMES_PER_CLIP]
        assert len({c["clip_id"] for c in seq}) == 1, "Clip order changed"
    for clip in spec["clips"]:
        seq = by_clip[clip["clip_id"]]
        assert len(seq) == FRAMES_PER_CLIP and [c["frame_in_clip"] for c in seq] == list(range(FRAMES_PER_CLIP))
        assert all(c["objects"] == seq[0]["objects"] for c in seq)
        assert all(c["time_s"] == round(DT_S * i, 6) for i, c in enumerate(seq))
        assert all(c["phase"] == ("approach" if i <= 7 else "depart") for i, c in enumerate(seq))
        assert all(len(c["objects"]) == 2 and c["target_name"] == "target" for c in seq)
        assert all(o["kind"] == "cube" and "rotation" not in o for c in seq for o in c["objects"])
        assert all(c["camera"]["z"] == 1.82 and all(c["camera"][a] == 0 for a in ("pitch", "yaw", "roll")) for c in seq)
        target = seq[0]["objects"][0]
        for value, (low, high) in zip((*target["size_m"], target["center_m"][2]), RANGES[clip["type_id"]]):
            assert low - 1e-6 <= value <= high + 1e-6
        fronts = [float(bounds(c)[0][2]) for c in seq]
        jitter = clip["trajectory_jitter_m"]
        assert all(abs(z - base - jitter) <= 2e-6 for z, base in zip(fronts, FRONT_TRAJECTORY))
        assert min(fronts) >= .4 and fronts[0] > 3 and fronts[-1] > 3
        labels = [classify(*bounds(c)) for c in seq]
        for i, label in enumerate(labels):
            expected = clip["layout_relation"] != "OUTSIDE" and i in range(2, 10)
            assert label["truth"] == expected
            if expected and clip["layout_relation"] == "BOUNDARY":
                assert label["boundary"] and 0 <= label["signed_boundary_margin_m"] <= .02
    for split in SPLIT_GROUPS:
        appearances = defaultdict(Counter)
        for clip in spec["clips"]:
            if clip["split"] == split:
                appearance = tuple(o["material"] for o in by_clip[clip["clip_id"]][0]["objects"])
                appearances[appearance][clip["layout_relation"]] += 1
        assert all(set(c) == set(RELATIONS) and len(set(c.values())) == 1 for c in appearances.values())
        counts[split] = {}
        for relation in RELATIONS:
            rows = [c for c in spec["cases"] if c["split"] == split and c["layout_relation"] == relation]
            counts[split][relation] = dict(frames=len(rows), controlled_target_positive=sum(classify(*bounds(c))["truth"] for c in rows))
    return dict(status="PASS", groups=48, clips=144, frames=FRAMES, counts=counts,
        compared_old_cohorts=len(old_specs), no_relative_overlap=True, split_unit="BASE_GROUP",
        front_range_m=[min(float(bounds(c)[0][2]) for c in spec["cases"]),
                       max(float(bounds(c)[0][2]) for c in spec["cases"])],
        geometry_scope="CONTROLLED_TARGET_EXTENT_ONLY_NOT_ALL_SCENE_CLEARANCE",
        visibility_scope="NO_CAPTURED_OUTCOMES_INSPECTED_NO_VISIBILITY_EXCLUSIONS")


def freeze(output):
    """Create a new source-only capture protocol; does not launch UE."""
    from data_coverage_spec import specification as old_coverage
    from launch_query_occupancy import MANDATORY_CODE, MANDATORY_INPUTS, verify_protocol
    repo = Path(__file__).resolve().parents[4]
    root, output = (repo / "artifacts.local").resolve(), Path(output).resolve()
    assert output.is_relative_to(root) and output != root and not output.exists(), "New canonical output required"
    spec = specification()
    check = check_spec(spec, [old_coverage()])
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    codes = {name: sha(here / name) for name in sorted(MANDATORY_CODE)}
    inputs = {name: sha(repo / name) for name in sorted(MANDATORY_INPUTS)}
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write("spec.json", spec)
    write("source-check.json", check)
    write("protocol.json", dict(schema="query-occupancy-capture-protocol-v1", id=output.name,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), frames=FRAMES, clips=144,
        frames_per_clip=FRAMES_PER_CLIP, dt_s=DT_S, capture_timeout_s=CAPTURE_TIMEOUT_S,
        spec_sha256=sha(output / "spec.json"), code_hashes=codes, input_hashes=inputs,
        source_check_sha256=sha(output / "source-check.json")))
    verify_protocol(output / "protocol.json", output / "spec.json", repo)
    return dict(status="FROZEN_NOT_CAPTURED", output=str(output), frames=FRAMES, check=check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))
