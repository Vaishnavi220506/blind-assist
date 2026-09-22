"""Frozen continuous inference for three declared 12 cm nominal paths."""
from __future__ import annotations

import math
import time
from types import FunctionType

import candidate_diagnostics as diagnostic
import path_constraint_inference as model


milp = model.milp
PATHS = {
    'straight_x': tuple((round(i*.01, 10), 0.) for i in range(13)),
    'x_then_z': (tuple((round(i*.01, 10), 0.) for i in range(7))
                 + tuple((.06, round(i*.01, 10)) for i in range(1, 7))),
    'z_then_x': (tuple((0., round(i*.01, 10)) for i in range(7))
                 + tuple((round(i*.01, 10), .06) for i in range(1, 7))),
}
ALLOWED_POSES = frozenset(p for path in PATHS.values() for p in path)


def _inputs(observations):
    if not isinstance(observations, list) or not 1 <= len(observations) <= 13:
        raise ValueError('One to thirteen public observations required')
    normalized = []
    for row in observations:
        if not isinstance(row, dict) or set(row) != {'camera', 'bins'}:
            raise ValueError('Only nominal camera and bins accepted; no identity/truth/stress metadata')
        camera, bins = row['camera'], row['bins']
        if not isinstance(camera, (list, tuple)) or len(camera) != 2:
            raise ValueError('Camera must be an X/Z pair')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in camera):
            raise ValueError('Finite nominal camera coordinates required')
        camera = tuple(float(v) for v in camera)
        if camera not in ALLOWED_POSES:
            raise ValueError('Camera outside the three frozen nominal paths')
        if not isinstance(bins, (list, tuple)) or len(bins) != 8:
            raise ValueError('Eight quantized radial bins required')
        if any(type(v) is not int or v <= 0 for v in bins):
            raise ValueError('Bins must be positive integers')
        normalized.append(dict(camera=camera, bins=tuple(bins)))
    return normalized


def _readout(witnesses, receipts):
    conflicts = [k for k in ('IN', 'OUT') if witnesses[k] is not None
                 and receipts[k]['exclusion_supported']]
    decision = 'UNKNOWN'
    if not conflicts:
        for label, opposite in (('IN', 'OUT'), ('OUT', 'IN')):
            if (witnesses[label] is not None and receipts[label].get('solver_status') == 0
                    and witnesses[opposite] is None
                    and receipts[opposite].get('solver_status') == 2
                    and receipts[opposite]['exclusion_supported']):
                decision = label + '_MODEL_CONDITIONAL'
    if conflicts:
        reason = 'CONSTRUCTIVE_WITNESS_EXCLUSION_CONFLICT'
    elif decision != 'UNKNOWN':
        reason = 'VALIDATED_WITNESS_AND_OPPOSITE_OUTER_RELAXATION_REPORTED_INFEASIBLE'
    elif all(witnesses.values()):
        reason = 'BOTH_LABELS_HAVE_VALIDATED_WITNESSES'
    elif all(r['exclusion_supported'] for r in receipts.values()):
        reason = 'BOTH_CLASSES_REPORTED_INFEASIBLE_MODEL_INCONSISTENCY_OR_NUMERICAL_FAILURE'
    else:
        reason = 'UNKNOWN_INCOMPLETE_OR_NUMERICAL_SEARCH'
    return decision, reason, conflicts


def infer(observations: list[dict]) -> dict:
    public = _inputs(observations)
    started = time.perf_counter()
    captured = []

    def capture(**kwargs):
        if len(captured) >= 2:
            raise AssertionError('Only two original-budget solver calls allowed')
        record = {}
        captured.append(record)
        result = milp(**kwargs)
        record['result'] = result
        return result

    # Execute the original function body with a private solver binding. Neither
    # model._inputs nor any shared module global is changed, even transiently.
    solve = FunctionType(model._solve.__code__, {**model._solve.__globals__, 'milp': capture},
                         name=model._solve.__name__, argdefs=model._solve.__defaults__,
                         closure=model._solve.__closure__)
    witnesses, receipts = {}, {}
    for label in ('IN', 'OUT'):
        witnesses[label], receipts[label] = solve(public, label)
    if len(captured) != 2:
        raise AssertionError('Expected exactly two original label solves')
    original = dict(witnesses)
    attempts = []
    for label, captured_result in zip(('IN', 'OUT'), captured, strict=True):
        receipt = receipts[label]
        attempt = dict(label=label, original_metadata=receipt, raw_solver_vector=None,
                       decoded_candidate=None, validation=None, projected_candidate=None,
                       projected_validation=None, projected_valid=False, projection_eligible=False)
        result = captured_result.get('result')
        if result is not None and result.x is not None:
            vector = [float(v) for v in result.x]
            attempt['raw_solver_vector'] = [diagnostic._number(v) for v in vector]
            if all(math.isfinite(v) for v in vector):
                candidate = model.base._candidate(vector)
                attempt['decoded_candidate'] = candidate
                attempt['validation'] = diagnostic.validation_detail(candidate, public, label)
                if (label == 'IN' and receipt.get('solver_status') == 0
                        and receipt['outcome'] == 'NUMERICAL_CANDIDATE_REJECTED'):
                    proposal = diagnostic.propose_projected_in(candidate)
                    validation = diagnostic.validation_detail(proposal, public, label)
                    attempt.update(projection_eligible=True, projected_candidate=proposal,
                                   projected_validation=validation, projected_valid=validation['valid'])
                    if validation['valid']:
                        witnesses[label] = proposal
        attempts.append(attempt)
    decision, reason, conflicts = _readout(witnesses, receipts)
    return dict(status='UNKNOWN' if decision == 'UNKNOWN' else 'MODEL_CONDITIONAL',
                decision=decision, reason=reason, authority=model.AUTHORITY, witnesses=witnesses,
                original_witnesses=original, attempts=attempts,
                solver_metadata=[receipts[k] for k in ('IN', 'OUT')],
                exclusion_metadata={k: dict(reported_infeasible=r['exclusion_supported'],
                    solver_status=r.get('solver_status'), outcome=r['outcome'],
                    authority=model.AUTHORITY, is_formal_certificate=False) for k,r in receipts.items()},
                recovery_conflicts=conflicts, solver_calls=len(captured), observations_used=len(public),
                valid_witness_count=sum(v is not None for v in witnesses.values()),
                projection_added_labels=[k for k in witnesses if original[k] is None and witnesses[k] is not None],
                elapsed_seconds=time.perf_counter()-started,
                model_domain=dict(model.base.DOMAIN), outer_relaxation=model.OUTER_DESCRIPTION,
                universal_uniqueness_proven=False, sensor_clearance_certified=False,
                robust_to_pose_or_range_error=False,
                stress_interpretation='MIS_SPECIFIED_STRESS_IF_RANGES_OR_POSES_PERTURBED',
                backend='CPU / TASK_NOT_GPU_SUITABLE')
