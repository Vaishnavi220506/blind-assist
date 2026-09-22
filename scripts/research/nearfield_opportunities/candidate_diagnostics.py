"""Capture unchanged numerical solves and validate a separate IN projection proposal."""
from __future__ import annotations

import math
import threading
from unittest.mock import patch

import numpy as np

import path_constraint_inference as model
import continuous_boundary_witness as base
import active_view as sensor


_COLLECT_LOCK = threading.RLock()


def _structure(candidate):
    if not isinstance(candidate, dict) or set(candidate) != {"boxes", "wall_z"}:
        raise ValueError("Expected a single-box candidate record")
    if candidate["wall_z"] != base.WALL or not isinstance(candidate["boxes"], list) or len(candidate["boxes"]) != 1:
        raise ValueError("Expected one rectangle and the frozen wall")
    b = candidate["boxes"][0]
    if not isinstance(b, dict) or set(b) != {"x", "z", "width", "thickness"}:
        raise ValueError("Unexpected rectangle fields")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in b.values()):
        raise ValueError("Candidate coordinates must be finite")
    if b["thickness"] != base.THICKNESS:
        raise ValueError("Only the frozen thickness is supported")
    return b


def unrounded_candidate(vector):
    """Decode L/R/front without domain projection or decimal rounding."""
    left, right, front = (float(v) for v in vector[:3])
    return dict(wall_z=base.WALL, boxes=[dict(x=(left + right) / 2,
                width=right - left, z=front + base.THICKNESS / 2,
                thickness=base.THICKNESS)])


def validation_detail(candidate, observations, label):
    """Independent failure categories; actual-source labels never enter this API."""
    if label not in ("IN", "OUT"):
        raise ValueError("Expected a hypothetical IN or OUT label")
    failures = []
    result = dict(requested_label=label, valid=False, failures=failures)
    try:
        b = _structure(candidate)
    except (ValueError, TypeError, KeyError) as exc:
        result.update(structure_valid=False, diagnostic_error=str(exc))
        failures.append("STRUCTURE_OR_NONFINITE_COORDINATES")
        return result
    result["structure_valid"] = True
    domain = {"center_x": -.9 <= b["x"] <= .9,
              "width": .03 <= b["width"] <= .6,
              "center_z": .6 <= b["z"] <= 3.4}
    result.update(domain_checks=domain, domain_valid=all(domain.values()))
    if not result["domain_valid"]:
        failures.append("OUTSIDE_DECLARED_DOMAIN")
    left, right = b["x"] - b["width"] / 2, b["x"] + b["width"] / 2
    front, back = b["z"] - b["thickness"] / 2, b["z"] + b["thickness"] / 2
    margins = {"left_face_below_query_right": .3 + sensor.EPS - left,
               "right_face_above_query_left": right - (-.3 - sensor.EPS),
               "front_below_query_far": 3. + sensor.EPS - front,
               "back_above_query_near": back - (.3 - sensor.EPS)}
    scene = model._scene(candidate)
    actual_label = "IN" if sensor.intersects_query(scene) else "OUT"
    result.update(decoded_faces=dict(L=left, R=right, F=front, B=back),
                  query_membership_margins_m=margins, actual_label=actual_label,
                  label_matches=actual_label == label)
    if actual_label != label:
        failures.append("QUERY_LABEL_MISMATCH")
    mismatches, replay = [], []
    try:
        for j, ob in enumerate(observations):
            actual = list(sensor.observe(scene, tuple(ob["camera"])))
            replay.append(dict(camera=list(ob["camera"]), bins=actual))
            for ray, (expected, found) in enumerate(zip(ob["bins"], actual, strict=True)):
                if expected != found:
                    mismatches.append(dict(observation_index=j, camera=list(ob["camera"]),
                                           ray=ray, expected_bin=expected, actual_bin=found))
        result.update(all_bins_match=not mismatches, bin_mismatch_count=len(mismatches),
                      bin_mismatches=mismatches, replay=replay)
        if mismatches:
            failures.append("OBSERVATION_BIN_MISMATCH")
    except (ValueError, TypeError, OverflowError) as exc:
        result.update(all_bins_match=False, replay_error=str(exc))
        failures.append("FORWARD_REPLAY_ERROR")
    result["valid"] = not failures
    # The categorization must agree with the unchanged production validator.
    if result["valid"] != model.validate_witness(candidate, observations, label):
        raise AssertionError("Diagnostic and frozen witness validators disagree")
    return result


def propose_projected_in(candidate):
    """Nearest fixed-width center in the closed query/domain intersection.

    This function proposes geometry only. It cannot certify observations, select
    a label, alter an exclusion receipt, or admit a candidate without validation.
    """
    b = _structure(candidate)
    if not .03 <= b["width"] <= .6:
        raise ValueError("Cannot alter an out-of-domain width")
    lower_x, upper_x = max(-.9, -.3 - b["width"] / 2), min(.9, .3 + b["width"] / 2)
    lower_z, upper_z = max(.6, .3 - b["thickness"] / 2), min(3.4, 3. + b["thickness"] / 2)
    return dict(wall_z=candidate["wall_z"], boxes=[dict(b,
                x=min(upper_x, max(lower_x, b["x"])),
                z=min(upper_z, max(lower_z, b["z"])) )])


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _linear_diagnostics(vector, kwargs, label):
    x = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(x)):
        return dict(error="NONFINITE_SOLVER_VECTOR")
    constraint = kwargs["constraints"]
    if len(x) != len(kwargs["c"]):
        return dict(error="MOCK_OR_INVALID_SOLVER_VECTOR_LENGTH", expected=len(kwargs["c"]), actual=len(x))
    lhs = np.asarray(constraint.A @ x).ravel()
    upper = np.asarray(constraint.ub)
    lower = np.asarray(constraint.lb)
    upper_residual = lhs - upper
    lower_residual = lower - lhs
    bounds = kwargs["bounds"]
    variable_residual = np.maximum(np.asarray(bounds.lb) - x, x - np.asarray(bounds.ub))
    result = dict(max_upper_violation=_number(max(0., float(np.max(upper_residual)))),
                  max_lower_violation=_number(max(0., float(np.max(lower_residual)))),
                  max_variable_bound_violation=_number(max(0., float(np.max(variable_residual)))),
                  max_binary_integrality_residual=_number(max([0., *(abs(v - round(v)) for v in x[4:])])),
                  positive_upper_residuals=[dict(row=i, residual=_number(v))
                                            for i, v in enumerate(upper_residual) if v > 0.],
                  residuals_are_unthresholded_diagnostics_not_new_tolerances=True)
    if label == "IN":
        result["IN_membership_rows"] = [dict(row=i, lhs=_number(lhs[i]), upper=_number(upper[i]),
                                             residual=_number(upper_residual[i])) for i in (4, 5, 6)]
    return result


def collect(observations: list[dict]) -> dict:
    """Run the frozen infer exactly once; instrument, then propose separately.

    Intended for serial batch use. The original module's solver binding is
    restored in finally by the patch context; no original file is edited.
    """
    public = model._inputs(observations)
    captured = []
    with _COLLECT_LOCK:
        original_milp = model.milp

        def capture(*args, **kwargs):
            if args:
                raise AssertionError("Frozen solver unexpectedly changed calling convention")
            if len(captured) >= 2:
                raise AssertionError("Additional optimizer call forbidden")
            record = {"label": ("IN", "OUT")[len(captured)], "kwargs": kwargs}
            captured.append(record)
            try:
                result = original_milp(**kwargs)
                record["result"] = result
                return result
            except Exception as exc:
                record["exception"] = dict(type=type(exc).__name__, message=str(exc))
                raise

        # The actual original infer decides its own status/outcome. Diagnostics
        # cannot introduce another solve or substitute a new candidate into it.
        with patch.object(model, "milp", capture):
            baseline = model.infer(public)
    if len(captured) != 2:
        raise AssertionError("Expected the original exactly two label solves")
    attempts = []
    for record, receipt in zip(captured, baseline["solver_metadata"], strict=True):
        label = record["label"]
        if receipt["requested_label"] != label:
            raise AssertionError("Frozen label solve ordering changed")
        attempt = dict(label=label, original_metadata=receipt,
                       raw_solver_vector=None, unrounded_candidate=None, decoded_candidate=None,
                       raw_validation=None, validation=None, projected_candidate=None,
                       projected_validation=None, projected_valid=False, projection_eligible=False,
                       baseline_candidate_changed=False)
        result = record.get("result")
        if result is None:
            attempt["solver_exception"] = record.get("exception")
        elif result.x is not None:
            vector = [float(v) for v in result.x]
            attempt["raw_solver_vector"] = [_number(v) for v in vector]
            attempt["linear_diagnostics"] = _linear_diagnostics(vector, record["kwargs"], label)
            if all(math.isfinite(v) for v in vector):
                unrounded = unrounded_candidate(vector)
                decoded = base._candidate(vector)
                attempt.update(unrounded_candidate=unrounded, decoded_candidate=decoded,
                               raw_validation=validation_detail(unrounded, public, label),
                               validation=validation_detail(decoded, public, label))
                raw_b, decoded_b = unrounded["boxes"][0], decoded["boxes"][0]
                attempt["decoding_changes"] = {k: decoded_b[k] - raw_b[k] for k in ("x", "z", "width")}
                if receipt["outcome"] == "VALIDATED_WITNESS":
                    if not attempt["validation"]["valid"] or decoded != baseline["witnesses"][label]:
                        raise AssertionError("Original validated candidate changed during collection")
                if label == "IN" and receipt.get("solver_status") == 0 and receipt["outcome"] == "NUMERICAL_CANDIDATE_REJECTED":
                    proposal = propose_projected_in(decoded)
                    validation = validation_detail(proposal, public, label)
                    moved = proposal["boxes"][0]
                    delta = {k: moved[k] - decoded_b[k] for k in ("x", "z", "width")}
                    attempt.update(projection_eligible=True, projected_candidate=proposal,
                                   projected_validation=validation, projected_valid=validation["valid"],
                                   projection_delta=delta, projection_center_distance_m=math.hypot(delta["x"], delta["z"]))
            else:
                attempt["validation"] = dict(valid=False, failures=["NONFINITE_SOLVER_VECTOR"])
        attempts.append(attempt)
    return dict(baseline=baseline, attempts=attempts, solver_calls=len(captured),
                purpose="CONSUMED_NUMERICAL_DIAGNOSTIC_AND_SEPARATE_CLOSED_QUERY_PROPOSAL",
                new_actual_observations=0, true_source_geometry_or_labels_used=False,
                original_solver_settings=dict(base.SOLVER_OPTIONS),
                original_conditional_readout_unchanged=True,
                projected_proposal_is_not_a_decision=True)
