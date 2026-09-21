"""Frozen fixed-path information diagnostics; no action selector or classifier.

Only signatures and transitions use public quantized views. Truth, raw ranges,
and face identities are evaluator-only attribution and ceiling information.
"""
from __future__ import annotations

from collections import defaultdict
import math

import active_view as av
import active_view_positive as pos

SPOKES = {"plus_x": (1, 0), "minus_x": (-1, 0),
          "plus_z": (0, 1), "minus_z": (0, -1)}
RAW_TOLERANCE = 1e-12
FACE_TOLERANCE = 1e-9


def path_poses(name):
    dx, dz = SPOKES[name]
    return tuple((round(dx*i*.01, 10), round(dz*i*.01, 10)) for i in range(13))


def pose_key(camera):
    return tuple(round(float(x), 10) for x in camera)


def index_views(observations):
    result = {}
    for row in observations:
        if row["id"] in result:
            raise ValueError("Duplicate scene observation ID")
        views = {pose_key(v["camera"]): v for v in row["views"]}
        if len(views) != len(row["views"]):
            raise ValueError("Duplicate camera pose")
        for view in views.values():
            if len(view["bins"]) != 8 or len(view["raw_radial_m"]) != 8:
                raise ValueError("Expected exactly eight rays")
            if any(not isinstance(b, int) for b in view["bins"]):
                raise ValueError("Expected integer quantized bins")
            if any(not math.isfinite(d) or d <= 0 for d in view["raw_radial_m"]):
                raise ValueError("Expected finite positive raw range")
        result[row["id"]] = views
    return result


def public_signature(views, poses):
    """Bins only; no source, raw range, label, or pairing is consulted."""
    return tuple(tuple(views[pose]["bins"]) for pose in poses)


def bin_transitions(views, poses):
    """Observed transitions are brackets between samples, not exact positions."""
    result = []
    for ray in range(8):
        events = []
        for i in range(1, len(poses)):
            before, after = (views[p]["bins"][ray] for p in poses[i-1:i+1])
            if before != after:
                events.append(dict(path_interval_m=[round((i-1)*.01, 10), round(i*.01, 10)],
                                   from_bin=before, to_bin=after))
        result.append(dict(ray=ray, events=events))
    return result


def first_surface(scene, camera, ray):
    """Evaluator identity includes exact face, preventing visibility attribution."""
    angle = av.ANGLES[ray]
    wall = (scene.wall_z-camera[1])/math.cos(angle)
    hits = [av.hit_distance(b, camera, angle) for b in scene.boxes]
    nearest = min([wall, *hits])
    surfaces = []
    if abs(wall-nearest) <= FACE_TOLERANCE:
        surfaces.append(("wall",))
    x = camera[0]+nearest*math.sin(angle)
    z = camera[1]+nearest*math.cos(angle)
    for i, (box, distance) in enumerate(zip(scene.boxes, hits, strict=True)):
        if abs(distance-nearest) > FACE_TOLERANCE:
            continue
        faces = []
        for name, coordinate, boundary in (
            ("x_min", x, box.x-box.width/2), ("x_max", x, box.x+box.width/2),
            ("z_min", z, box.z-box.thickness/2), ("z_max", z, box.z+box.thickness/2)):
            if abs(coordinate-boundary) <= FACE_TOLERANCE:
                faces.append(name)
        surfaces.append(("box", i, tuple(faces)))
    # Ties and corners are rejected conservatively by the attribution rule.
    return tuple(surfaces)


def unambiguous_surface(identity):
    return (len(identity) == 1 and (identity[0] == ("wall",) or
                                  len(identity[0][2]) == 1))


def quantization_witnesses(scene_a, scene_b, views_a, views_b, poses):
    """Strict pure-crossing witnesses; source/raw information never picks actions."""
    if public_signature(views_a, (poses[0], poses[-1])) != public_signature(views_b, (poses[0], poses[-1])):
        return []
    witnesses = []
    for ray in range(8):
        qa = [views_a[p]["bins"][ray] for p in poses]
        qb = [views_b[p]["bins"][ray] for p in poses]
        if qa == qb:
            continue
        delta = [abs(views_a[p]["raw_radial_m"][ray]-views_b[p]["raw_radial_m"][ray]) for p in poses]
        if max(delta[0], delta[-1]) <= RAW_TOLERANCE:
            continue
        ia = [first_surface(scene_a, p, ray) for p in poses]
        ib = [first_surface(scene_b, p, ray) for p in poses]
        if not (unambiguous_surface(ia[0]) and all(v == ia[0] for v in ia+ib)):
            continue
        if not any(qa[i] != qa[i-1] or qb[i] != qb[i-1] for i in range(1, len(poses))):
            continue
        witnesses.append(dict(ray=ray, surface=ia[0], raw_endpoint_difference_m=[delta[0], delta[-1]],
                              different_sample_indices=[i for i in range(13) if qa[i] != qb[i]]))
    return witnesses


def purity(signatures, truth):
    buckets = defaultdict(list)
    for ident, signature in signatures.items():
        buckets[signature].append(ident)
    resolved = sorted(i for ids in buckets.values() if len({truth[j] for j in ids}) == 1 for i in ids)
    resolved_set = set(resolved)
    return dict(resolved_ids=resolved, resolved_scenes=len(resolved),
                resolved_positive=sum(truth[i] for i in resolved),
                resolved_negative=sum(not truth[i] for i in resolved),
                unresolved_ids=sorted(set(truth)-resolved_set),
                mixed_classes=sum(len({truth[i] for i in ids}) == 2 for ids in buckets.values()),
                total_classes=len(buckets))


def summarize_arm(pairs, initial, signatures, truth):
    separated = [p["id"] for p in pairs if signatures[p["members"][0]] != signatures[p["members"][1]]]
    initially_aliased = [p for p in pairs if initial[p["members"][0]] == initial[p["members"][1]]]
    return dict(separated_pairs=len(separated), separated_pair_ids=separated,
                initially_aliased_separated_pairs=sum(p["id"] in separated for p in initially_aliased),
                purity=purity(signatures, truth))


def oracle_ceilings(pairs, initial, signatures_by_spoke, truth):
    """Retrospective finite-cohort ceilings, never an executable choice rule."""
    eligible = [p for p in pairs if initial[p["members"][0]] == initial[p["members"][1]]]
    classes = defaultdict(list)
    for p in eligible:
        classes[initial[p["members"][0]]].append(p)
    counts = []
    for signature, group in sorted(classes.items()):
        separated = {name: [p["id"] for p in group if sig[p["members"][0]] != sig[p["members"][1]]]
                     for name, sig in signatures_by_spoke.items()}
        counts.append(dict(initial_bins=signature, pairs=[p["id"] for p in group],
                           separated_by_spoke=separated, common_spoke_ceiling=max(map(len, separated.values()))))
    scene_classes = defaultdict(list)
    for ident, signature in initial.items():
        scene_classes[signature].append(ident)
    pure = {name: set(purity(sig, truth)["resolved_ids"]) for name, sig in signatures_by_spoke.items()}
    mixed_groups = [ids for ids in scene_classes.values() if len({truth[i] for i in ids}) == 2]
    all_pure = set().union(*pure.values())
    return dict(authority="EVALUATOR_ONLY_NOT_A_POLICY", initially_aliased_pair_denominator=len(eligible),
                pair_oracle_separated=sum(any(sig[p["members"][0]] != sig[p["members"][1]]
                    for sig in signatures_by_spoke.values()) for p in eligible),
                class_common_spoke_separated=sum(c["common_spoke_ceiling"] for c in counts),
                pair_classes=counts,
                initially_mixed_scene_denominator=sum(map(len, mixed_groups)),
                mixed_scene_common_spoke_purity_ceiling=sum(max(len(set(ids)&v) for v in pure.values()) for ids in mixed_groups),
                mixed_scene_per_case_purity_ceiling=sum(len(set(ids)&all_pure) for ids in mixed_groups))


def analyze(pairs, sources, observations):
    views = index_views(observations)
    scenes = {r["id"]: pos.source_scene(r) for r in sources}
    if len(scenes) != len(sources) or set(scenes) != set(views):
        raise ValueError("Source/observation IDs must match exactly and be unique")
    if len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair ID")
    truth = {i: av.intersects_query(s) for i, s in scenes.items()}
    for pair in pairs:
        if len(pair["members"]) != 2 or {truth[i] for i in pair["members"]} != {True, False}:
            raise ValueError("Each designated pair must cross the full-extent query boundary")
    initial = {i: public_signature(v, ((0., 0.),)) for i, v in views.items()}
    endpoints, sampled, arms, attribution = {}, {}, {}, {}
    for name in SPOKES:
        poses = path_poses(name)
        endpoints[name] = {i: public_signature(v, (poses[0], poses[-1])) for i, v in views.items()}
        sampled[name] = {i: public_signature(v, poses) for i, v in views.items()}
        endpoint = summarize_arm(pairs, initial, endpoints[name], truth)
        path = summarize_arm(pairs, initial, sampled[name], truth)
        gains = sorted(set(path["separated_pair_ids"])-set(endpoint["separated_pair_ids"]))
        purity_gains = sorted(set(path["purity"]["resolved_ids"])-set(endpoint["purity"]["resolved_ids"]))
        assert set(endpoint["separated_pair_ids"]) <= set(path["separated_pair_ids"])
        assert set(endpoint["purity"]["resolved_ids"]) <= set(path["purity"]["resolved_ids"])
        arms[name] = dict(endpoint=endpoint, sampled=path, gained_pair_ids=gains,
                          gained_pure_scene_ids=purity_gains,
                          cost=dict(path_length_m=.12, endpoint_views=2, sampled_views=13,
                                    additional_views=11, rays_per_view=8, position_spacing_m=.01))
        rows = []
        for pair in pairs:
            if pair["id"] not in gains:
                continue
            a, b = pair["members"]
            witnesses = quantization_witnesses(scenes[a], scenes[b], views[a], views[b], poses)
            rows.append(dict(pair_id=pair["id"], members=[a, b], pure_crossing_witnesses=witnesses,
                attribution="STRICT_QUANTIZATION_CROSSING" if witnesses else "NOT_STRICTLY_ATTRIBUTED",
                transitions={i: bin_transitions(views[i], poses) for i in (a, b)}))
        attribution[name] = dict(endpoint_aliased_gains=len(rows),
            strict_quantization_gains=sum(bool(r["pure_crossing_witnesses"]) for r in rows), rows=rows)
    primary_path, primary_quant = arms["plus_x"], attribution["plus_x"]
    path_decision = ("PASS_PAIR_AND_COHORT_INFORMATION_GAIN" if primary_path["gained_pair_ids"] and primary_path["gained_pure_scene_ids"]
                     else "PARTIAL_INFORMATION_GAIN" if primary_path["gained_pair_ids"] or primary_path["gained_pure_scene_ids"]
                     else "NO_INCREMENTAL_INFORMATION")
    return dict(schema="path_observation_analysis_v1", authority="INFORMATION_DIAGNOSTIC_NOT_ALERT_ACCURACY",
        denominators=dict(pairs=len(pairs), scenes=len(sources),
            initially_aliased_pairs=sum(initial[p["members"][0]] == initial[p["members"][1]] for p in pairs),
            initial_purity=purity(initial, truth)),
        primary=dict(path_spoke="plus_x", path_decision=path_decision, quantization_spoke="plus_x",
            quantization_decision="PASS_STRICT_CROSSING_INFORMATION_GAIN" if primary_quant["strict_quantization_gains"] else "NO_STRICT_CROSSING_GAIN"),
        fixed_spokes=arms, quantization_attribution=attribution,
        endpoint_oracle=oracle_ceilings(pairs, initial, endpoints, truth),
        sampled_oracle=oracle_ceilings(pairs, initial, sampled, truth))
