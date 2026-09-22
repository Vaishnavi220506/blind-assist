"""Bounded continuous rectangle witnesses, never a uniqueness certificate."""
from __future__ import annotations

import math
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp


ANGLES = tuple(math.radians(-22.5 + (i + .5) * 45 / 8) for i in range(8))
ACTIONS = ((.12, 0.), (-.12, 0.), (0., .12), (0., -.12))
POSES = ((0., 0.),) + ACTIONS
THICKNESS = .04
WALL = 4.20
RANGE_STEP = .10
SLACK_MAX = .0001
SOLVER_OPTIONS = {"time_limit": 1.0, "node_limit": 10000,
                  "mip_rel_gap": 0.0, "presolve": True}
DOMAIN = {"center_x": [-.9, .9], "width": [.03, .60],
          "center_z": [.60, 3.40], "thickness": THICKNESS, "wall_z": WALL}


def _inputs(observations):
    if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
        raise ValueError("One to five public observation records required")
    normalized = []
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"camera", "bins"}:
            raise ValueError("Only camera and bins are accepted; no truth or identity")
        camera, bins = row["camera"], row["bins"]
        if not isinstance(camera, (list, tuple)) or len(camera) != 2:
            raise ValueError("Known two-coordinate camera required")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in camera):
            raise ValueError("Camera coordinates must be finite numbers")
        camera = tuple(float(v) for v in camera)
        if camera not in POSES:
            raise ValueError("Camera outside the frozen five poses")
        if not isinstance(bins, (list, tuple)) or len(bins) != 8:
            raise ValueError("Exactly eight radial bins required")
        if any(type(v) is not int or v <= 0 for v in bins):
            raise ValueError("Positive integer radial bins required")
        normalized.append({"camera": camera, "bins": tuple(bins)})
    return normalized


def _truth(scene):
    b = scene["boxes"][0]
    return (b["x"] + b["width"] / 2 >= -.3 - 1e-12
            and b["x"] - b["width"] / 2 <= .3 + 1e-12
            and b["z"] + THICKNESS / 2 >= .3 - 1e-12
            and b["z"] - THICKNESS / 2 <= 3. + 1e-12)


def _bins(scene, camera):
    """Independent face intersections for candidate validation, not slab clipping."""
    b = scene["boxes"][0]
    left, right = b["x"] - b["width"] / 2, b["x"] + b["width"] / 2
    front, back = b["z"] - THICKNESS / 2, b["z"] + THICKNESS / 2
    result = []
    for angle in ANGLES:
        dx, dz = math.sin(angle), math.cos(angle)
        hits = [(WALL - camera[1]) / dz]
        for x in (left, right):
            t = (x - camera[0]) / dx
            z = camera[1] + t * dz
            if t >= 0 and front - 1e-12 <= z <= back + 1e-12:
                hits.append(t)
        for z in (front, back):
            t = (z - camera[1]) / dz
            x = camera[0] + t * dx
            if t >= 0 and left - 1e-12 <= x <= right + 1e-12:
                hits.append(t)
        result.append(math.floor(min(hits) / RANGE_STEP + .5))
    return tuple(result)


def validate_witness(scene, observations, label):
    """Check an explicit witness against the declared domain, supplied bins and label."""
    try:
        if set(scene) != {"boxes", "wall_z"} or scene["wall_z"] != WALL:
            return False
        if len(scene["boxes"]) != 1:
            return False
        b = scene["boxes"][0]
        if set(b) != {"x", "z", "width", "thickness"} or b["thickness"] != THICKNESS:
            return False
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in b.values()):
            return False
        if not (-.9 <= b["x"] <= .9 and .03 <= b["width"] <= .60 and .60 <= b["z"] <= 3.40):
            return False
        if label not in ("IN", "OUT") or _truth(scene) != (label == "IN"):
            return False
        return all(_bins(scene, row["camera"]) == tuple(row["bins"]) for row in observations)
    except (KeyError, TypeError, IndexError):
        return False


class _Constraints:
    def __init__(self):
        # Continuous variables are L, R, F, common numerical interior slack.
        self.lower = [-1.2, -1.2, .58, 0.]
        self.upper = [1.2, 1.2, 3.38, SLACK_MAX]
        self.rows = []

    def le(self, coefficients, bound):
        self.rows.append((dict(coefficients), float(bound)))

    def either(self, alternatives):
        indicators = []
        for coefficients, bound in alternatives:
            i = len(self.lower)
            self.lower.append(0.)
            self.upper.append(1.)
            indicators.append(i)
            maximum = sum(a * (self.upper[j] if a >= 0 else self.lower[j])
                          for j, a in coefficients.items())
            big_m = max(0., maximum - bound) + 1e-6
            self.le({**coefficients, i: big_m}, bound + big_m)
        self.le({i: -1. for i in indicators}, -1.)

    def scipy(self):
        matrix = np.zeros((len(self.rows), len(self.lower)))
        for i, (coefficients, _) in enumerate(self.rows):
            for j, value in coefficients.items():
                matrix[i, j] = value
        objective = np.zeros(len(self.lower))
        objective[3] = -1.
        return dict(c=objective, integrality=np.array([0] * 4 + [1] * (len(self.lower) - 4)),
                    bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(matrix, -np.inf, [b for _, b in self.rows]),
                    options=dict(SOLVER_OPTIONS))


def _constraints(observations, label):
    c = _Constraints()
    c.le({0: 1., 1: -1.}, -.03)  # R-L >= minimum width
    c.le({0: -1., 1: 1.}, .60)
    c.le({0: .5, 1: .5}, .9)
    c.le({0: -.5, 1: -.5}, .9)
    if label == "IN":
        c.le({0: 1.}, .3)
        c.le({1: -1.}, .3)
        c.le({2: 1.}, 3.)
    else:
        # Domain F >= .58 makes the near-depth OUT alternative impossible.
        c.either([({1: 1., 3: 1.}, -.3),
                  ({0: -1., 3: 1.}, -.3),
                  ({2: -1., 3: 1.}, -3.)])
    for row in observations:
        cx, cz = row["camera"]
        for angle, value in zip(ANGLES, row["bins"], strict=True):
            dx, dz = math.sin(angle), math.cos(angle)
            slope = dx / dz
            wall_bin = math.floor(((WALL - cz) / dz) / RANGE_STEP + .5)
            if dx > 0:
                # Miss iff R < x(front) OR L > x(back).
                misses = [({1: 1., 2: -slope, 3: 1.}, cx - slope * cz),
                          ({0: -1., 2: slope, 3: 1.}, -cx + slope * cz - slope * THICKNESS)]
                # Hit iff R >= x(front) AND L <= x(back).
                hits = [({1: -1., 2: slope, 3: 1.}, -cx + slope * cz),
                        ({0: 1., 2: -slope, 3: 1.}, cx - slope * cz + slope * THICKNESS)]
                side = 0
            else:
                misses = [({0: -1., 2: slope, 3: 1.}, -cx + slope * cz),
                          ({1: 1., 2: -slope, 3: 1.}, cx - slope * cz + slope * THICKNESS)]
                hits = [({0: 1., 2: -slope, 3: 1.}, cx - slope * cz),
                        ({1: -1., 2: slope, 3: 1.}, -cx + slope * cz - slope * THICKNESS)]
                side = 1
            if value == wall_bin:
                # Every domain rectangle ends at Z <= 3.42 < wall Z4.20;
                # a target hit cannot share the wall's .10m range bin.
                c.either(misses)
                continue
            if value > wall_bin:
                c.le({}, -1.)  # Explicitly impossible public range under this model.
                continue
            for a, b in hits:
                c.le(a, b)
            low, high = (value - .5) * RANGE_STEP, (value + .5) * RANGE_STEP
            # Entry=max((F-cz)/dz,(near_side-cx)/dx). Both below high;
            # at least one reaches low. The optional slack seeks interior fits.
            c.le({2: 1. / dz, 3: 1.}, high + cz / dz)
            c.le({side: 1. / dx, 3: 1.}, high + cx / dx)
            c.either([({2: -1. / dz, 3: 1.}, -low - cz / dz),
                      ({side: -1. / dx, 3: 1.}, -low - cx / dx)])
    return c


def _candidate(values):
    left, right, front = (float(v) for v in values[:3])
    def normalized(v, lo, hi):
        return min(hi, max(lo, round(v, 12)))
    return {"wall_z": WALL, "boxes": [{"x": normalized((left + right) / 2, -.9, .9),
            "width": normalized(right - left, .03, .60),
            "z": normalized(front + THICKNESS / 2, .60, 3.40), "thickness": THICKNESS}]}


def _solve(observations, label):
    constraints = _constraints(observations, label)
    start = time.perf_counter()
    metadata = {"requested_label": label, "solver": "scipy.optimize.milp / HiGHS",
                "scipy_version": scipy.__version__, "budget": dict(SOLVER_OPTIONS),
                "continuous_variables": 4, "binary_variables": len(constraints.lower) - 4,
                "constraints": len(constraints.rows), "search_exhaustive": False,
                "infeasible_status_is_nonexistence_certificate": False}
    try:
        result = milp(**constraints.scipy())
        metadata.update(solver_status=int(result.status), message=str(result.message),
                        elapsed_seconds=time.perf_counter() - start)
        for key in ("mip_node_count", "mip_gap", "mip_dual_bound"):
            value = getattr(result, key, None)
            metadata[key] = float(value) if value is not None and math.isfinite(value) else None
        if result.x is None:
            metadata["outcome"] = "NO_CANDIDATE_SEARCH_INCONCLUSIVE"
            return None, metadata
        scene = _candidate(result.x)
        metadata["interior_slack"] = float(result.x[3])
        if not validate_witness(scene, observations, label):
            metadata["outcome"] = "CANDIDATE_REJECTED_BY_FORWARD_VALIDATION"
            return None, metadata
        metadata["outcome"] = "VALIDATED_WITNESS"
        return scene, metadata
    except Exception as exc:
        metadata.update(outcome="SOLVER_ERROR_SEARCH_INCONCLUSIVE", error_type=type(exc).__name__,
                        message=str(exc), elapsed_seconds=time.perf_counter() - start)
        return None, metadata


def _opposing_analysis(witnesses):
    if any(witnesses[label] is None for label in ("IN", "OUT")):
        return None
    views = [{"action_index": i, "camera": list(camera),
              "IN_bins": list(_bins(witnesses["IN"], camera)),
              "OUT_bins": list(_bins(witnesses["OUT"], camera))}
             for i, camera in enumerate(ACTIONS)]
    separating = [r["action_index"] for r in views if r["IN_bins"] != r["OUT_bins"]]
    return {"separating_actions": separating, "hypothetical_action_observations": views,
            "all_four_actions_indistinguishable": not separating,
            "scope": "THIS_VALIDATED_WITNESS_PAIR_ONLY_NOT_UNIVERSAL_IMPOSSIBILITY",
            "actual_scene_observations_generated": 0}


def infer(observations: list[dict]) -> dict:
    """Construct two continuous witnesses from public bins; always return UNKNOWN."""
    public = _inputs(observations)
    witnesses, metadata = {}, []
    for label in ("IN", "OUT"):
        witness, receipt = _solve(public, label)
        witnesses[label] = witness
        metadata.append(receipt)
    opposing = _opposing_analysis(witnesses)
    reason = ("SOLVER_INCOMPLETE_OPPOSING_WITNESS_NOT_ESTABLISHED" if opposing is None else
              "OPPOSING_WITNESSES_SEPARABLE_BY_ALLOWED_ACTION" if opposing["separating_actions"] else
              "OPPOSING_WITNESS_PAIR_ALIASES_ALL_ALLOWED_ACTIONS")
    return {"status": "UNKNOWN", "reason": reason, "witnesses": witnesses,
            "opposing_witness_analysis": opposing, "solver_metadata": metadata,
            "model_domain": DOMAIN, "solver_calls": 2, "observations_used": len(public),
            "valid_witness_count": sum(v is not None for v in witnesses.values()),
            "continuous_parameters_not_enumerated_bank": True,
            "unique_label_certified": False, "search_failure_implies_nonexistence": False,
            "backend": "CPU / TASK_NOT_GPU_SUITABLE"}
