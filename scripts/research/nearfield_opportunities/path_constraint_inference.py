"""Numerical model-conditional readout from continuous saved-path constraints."""
from __future__ import annotations

import math
import time

import scipy
from scipy.optimize import milp

import active_view as sensor
import continuous_boundary_witness as base


AUTHORITY = "NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY"
EPS = sensor.EPS
ALLOWED_POSES = frozenset(
    {(0., 0.)}
    | {(round(dx * i * .01, 10), round(dz * i * .01, 10))
       for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)) for i in range(1, 13)}
    | {(x, z) for x in (-.06, .06) for z in (-.06, .06)}
)
OUTER_DESCRIPTION = "SLACK_ZERO_HALF_OPEN_BIN_CLOSURE_WITH_INHERITED_EPS_CONTACT_ENVELOPE"


def _inputs(observations):
    if not isinstance(observations, list) or not 1 <= len(observations) <= 13:
        raise ValueError("One to thirteen public observations required")
    result = []
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"camera", "bins"}:
            raise ValueError("Only camera and bins accepted; no identity/truth/raw/source")
        camera, bins = row["camera"], row["bins"]
        if not isinstance(camera, (tuple, list)) or len(camera) != 2:
            raise ValueError("Camera must be an X/Z pair")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in camera):
            raise ValueError("Finite camera coordinates required")
        camera = tuple(float(v) for v in camera)
        if camera not in ALLOWED_POSES:
            raise ValueError("Camera not one of the frozen53 diagnostic poses")
        if not isinstance(bins, (tuple, list)) or len(bins) != 8:
            raise ValueError("Eight quantized radial bins required")
        if any(type(v) is not int or v <= 0 for v in bins):
            raise ValueError("Bins must be positive integers")
        result.append(dict(camera=camera, bins=tuple(bins)))
    return result


def _outer_constraints(observations, label):
    """Reuse frozen constraints; only align their outer closure to inherited EPS."""
    if label not in ("IN", "OUT"):
        raise ValueError("Expected a hypothetical IN or OUT class")
    c = base._constraints(observations, label)
    # Deliberately verify the inherited row layout before its narrow adaptation.
    # No original module/global configuration is modified.
    if label == "IN":
        expected = [({0: 1.}, .3), ({1: -1.}, .3), ({2: 1.}, 3.)]
        if c.rows[4:7] != expected:
            raise RuntimeError("Frozen IN constraint layout changed")
        for i in range(4, 7):
            coefficients, bound = c.rows[i]
            c.rows[i] = (coefficients, bound + EPS)
        offset = 7
    else:
        # Four domain rows; three OUT alternative rows and their OR row.
        offset = 8
    hit_rows = 0
    for row in observations:
        cz = row["camera"][1]
        for angle, value in zip(base.ANGLES, row["bins"], strict=True):
            ux, uz = math.sin(angle), math.cos(angle)
            wall_bin = math.floor(((base.WALL - cz) / uz) / base.RANGE_STEP + .5)
            if value == wall_bin:
                offset += 3  # Two miss alternatives and their OR row.
            elif value > wall_bin:
                offset += 1  # Impossible range constraint.
            else:
                for i in (offset, offset + 1):
                    coefficients, bound = c.rows[i]
                    if set(coefficients) - {0, 1, 2, 3} or coefficients.get(3) != 1.:
                        raise RuntimeError("Frozen target-overlap row layout changed")
                    c.rows[i] = (coefficients, bound + abs(ux) * EPS)
                    hit_rows += 1
                offset += 7  # Two overlap, two upper, two lower-OR, one OR row.
    if offset != len(c.rows):
        raise RuntimeError("Frozen ray constraint layout changed")
    return c, dict(query_IN_bounds_expanded=(3 if label == "IN" else 0),
                   target_overlap_rows_expanded=hit_rows, inherited_contact_EPS=EPS,
                   target_overlap_bound_increment="abs(sin(ray_angle))*EPS",
                   settings_fit_or_swept=False)


def _scene(value):
    return sensor.Scene(tuple(sensor.Box(**b) for b in value["boxes"]), wall_z=value["wall_z"])


def validate_witness(value, observations, label):
    """Replay using the unchanged scalar sensor and its inherited EPS conventions."""
    try:
        if not isinstance(value, dict) or set(value) != {"boxes", "wall_z"} or value["wall_z"] != 4.2:
            return False
        if len(value["boxes"]) != 1:
            return False
        b = value["boxes"][0]
        if set(b) != {"x", "z", "width", "thickness"} or b["thickness"] != .04:
            return False
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in b.values()):
            return False
        if not (-.9 <= b["x"] <= .9 and .03 <= b["width"] <= .6 and .6 <= b["z"] <= 3.4):
            return False
        scene = _scene(value)
        if label not in ("IN", "OUT") or sensor.intersects_query(scene) != (label == "IN"):
            return False
        return all(sensor.observe(scene, tuple(o["camera"])) == tuple(o["bins"]) for o in observations)
    except (KeyError, TypeError, ValueError):
        return False


def _solve(observations, label):
    constraints, alignment = _outer_constraints(observations, label)
    started = time.perf_counter()
    receipt = dict(requested_label=label, solver="scipy.optimize.milp / HiGHS",
                   scipy_version=scipy.__version__, budget=dict(base.SOLVER_OPTIONS),
                   continuous_variables=4, binary_variables=len(constraints.lower) - 4,
                   constraints=len(constraints.rows), contact_alignment=alignment,
                   outer_relaxation=OUTER_DESCRIPTION, authority=AUTHORITY,
                   formal_infeasibility_certificate=False, exclusion_supported=False)
    try:
        result = milp(**constraints.scipy())
        receipt.update(solver_status=int(result.status), message=str(result.message),
                       elapsed_seconds=time.perf_counter() - started)
        for key in ("mip_node_count", "mip_gap", "mip_dual_bound"):
            value = getattr(result, key, None)
            receipt[key] = float(value) if value is not None and math.isfinite(value) else None
        if result.status == 1:
            message = str(result.message).lower()
            receipt["limit_kind"] = ("TIME_LIMIT" if "time limit" in message else
                                     "NODE_OR_ITERATION_LIMIT" if "node" in message or "iteration" in message else
                                     "LIMIT_UNSPECIFIED")
        if result.x is None:
            if result.status == 2:
                receipt.update(outcome="RELAXATION_REPORTED_INFEASIBLE", exclusion_supported=True)
            elif result.status == 1:
                receipt["outcome"] = "LIMIT_WITHOUT_CANDIDATE"
            elif result.status == 0:
                receipt["outcome"] = "REPORTED_OPTIMUM_WITHOUT_CANDIDATE"
            else:
                receipt["outcome"] = "SOLVER_FAILURE_WITHOUT_CANDIDATE"
            return None, receipt
        # A candidate combined with infeasible/unbounded/error status is internally
        # inconsistent; it cannot supply the positive half of a conditional readout.
        if result.status not in (0, 1):
            receipt["outcome"] = "SOLVER_STATUS_CANDIDATE_CONFLICT"
            return None, receipt
        candidate = base._candidate(result.x)
        receipt["interior_slack"] = float(result.x[3])
        if not validate_witness(candidate, observations, label):
            receipt["outcome"] = "NUMERICAL_CANDIDATE_REJECTED"
            return None, receipt
        receipt["outcome"] = "VALIDATED_WITNESS"
        return candidate, receipt
    except Exception as exc:
        receipt.update(outcome="SOLVER_EXCEPTION", error_type=type(exc).__name__,
                       message=str(exc), elapsed_seconds=time.perf_counter() - started)
        return None, receipt


def _opposing(witnesses):
    if any(witnesses[k] is None for k in ("IN", "OUT")):
        return None
    views = [dict(action_index=i, camera=list(p),
                  IN_bins=list(sensor.observe(_scene(witnesses["IN"]), p)),
                  OUT_bins=list(sensor.observe(_scene(witnesses["OUT"]), p)))
             for i, p in enumerate(base.ACTIONS)]
    separating = [v["action_index"] for v in views if v["IN_bins"] != v["OUT_bins"]]
    return dict(separating_actions=separating, hypothetical_action_observations=views,
                all_four_actions_indistinguishable=not separating,
                scope="THIS_WITNESS_PAIR_ONLY_NOT_UNIVERSAL_IMPOSSIBILITY")


def infer(observations: list[dict]) -> dict:
    public = _inputs(observations)
    started = time.perf_counter()
    witnesses, receipts = {}, {}
    for label in ("IN", "OUT"):
        witnesses[label], receipts[label] = _solve(public, label)
    opposing = _opposing(witnesses)
    decision = "UNKNOWN"
    if (witnesses["IN"] is not None and receipts["IN"].get("solver_status") == 0
            and receipts["OUT"]["exclusion_supported"]):
        decision = "IN_MODEL_CONDITIONAL"
    elif (witnesses["OUT"] is not None and receipts["OUT"].get("solver_status") == 0
          and receipts["IN"]["exclusion_supported"]):
        decision = "OUT_MODEL_CONDITIONAL"
    if decision != "UNKNOWN":
        reason = "VALIDATED_WITNESS_AND_OPPOSITE_OUTER_RELAXATION_REPORTED_INFEASIBLE"
    elif opposing is not None:
        reason = ("OPPOSING_WITNESSES_SEPARABLE_BY_ALLOWED_ACTION" if opposing["separating_actions"] else
                  "OPPOSING_WITNESS_PAIR_ALIASES_ALL_ALLOWED_ACTIONS")
    elif all(r["exclusion_supported"] for r in receipts.values()):
        reason = "BOTH_CLASSES_REPORTED_INFEASIBLE_MODEL_INCONSISTENCY_OR_NUMERICAL_FAILURE"
    else:
        reason = "UNKNOWN_INCOMPLETE_OR_NUMERICAL_SEARCH"
    return dict(status="MODEL_CONDITIONAL" if decision != "UNKNOWN" else "UNKNOWN",
                decision=decision, authority=AUTHORITY, reason=reason, witnesses=witnesses,
                solver_metadata=[receipts[label] for label in ("IN", "OUT")],
                exclusion_metadata={label: dict(reported_infeasible=r["exclusion_supported"],
                    solver_status=r.get("solver_status"), outcome=r["outcome"], authority=AUTHORITY,
                    is_formal_certificate=False) for label, r in receipts.items()},
                opposing_witness_analysis=opposing, valid_witness_count=sum(v is not None for v in witnesses.values()),
                observations_used=len(public), solver_calls=2, model_domain=dict(base.DOMAIN),
                outer_relaxation=OUTER_DESCRIPTION, elapsed_seconds=time.perf_counter() - started,
                universal_uniqueness_proven=False, sensor_clearance_certified=False,
                no_candidate_alone_supports_exclusion=False, backend="CPU / TASK_NOT_GPU_SUITABLE")
