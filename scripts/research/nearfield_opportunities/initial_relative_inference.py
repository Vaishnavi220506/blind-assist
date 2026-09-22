"""Frozen shared observation model, queried in the actual initial-camera frame."""
from __future__ import annotations

import math
import time

import shared_bias_inference as shared


milp = shared.milp
AUTHORITY = 'NUMERICAL_SINGLE_RECTANGLE_SHARED_BIAS_INITIAL_CAMERA_QUERY_ONLY'
QUERY_FRAME = 'ACTUAL_INITIAL_CAMERA_YAW_ZERO_FIXED_FOR_COMPLETE_HISTORY'
OUTER_DESCRIPTION = shared.OUTER_DESCRIPTION + '_INITIAL_CAMERA_QUERY'
PATH = shared.paths.PATHS['x_then_z']


def _inputs(observations):
    public = shared.paths._inputs(observations)
    if len(public) != 13 or tuple(o['camera'] for o in public) != PATH:
        raise ValueError('Exactly the original ordered 13-view x_then_z path starting at origin is required')
    return public


def _constraints(observations, label):
    """Replace only the query prefix; retain every original observation row."""
    old = shared._constraints(observations, label, 'range_pose')
    domain = [({0:1.,1:-1.},-.03),({0:-1.,1:1.},.60),
              ({0:.5,1:.5},.9),({0:-.5,1:-.5},.9)]
    if old.rows[:4] != domain:
        raise RuntimeError('Frozen world-domain prefix changed')
    new = shared._Constraints('range_pose')
    for coefficients,bound in domain:
        new.le(coefficients,bound)
    e = shared.ROUND_ENVELOPE
    if label == 'IN':
        expected = [({0:1.},.3+shared.sensor.EPS),({1:-1.},.3+shared.sensor.EPS),({2:1.},3.+shared.sensor.EPS)]
        if old.rows[4:7] != expected:
            raise RuntimeError('Frozen IN query prefix changed')
        new.le({0:1.,5:-1.},.3+shared.sensor.EPS+e)
        new.le({1:-1.,5:1.},.3+shared.sensor.EPS+e)
        new.le({2:1.,6:-1.},3.+shared.sensor.EPS+e)
        prefix = 7
    elif label == 'OUT':
        original = shared._Constraints('range_pose')
        for coefficients,bound in domain:
            original.le(coefficients,bound)
        original.either([({1:1.,3:1.},-.3),({0:-1.,3:1.},-.3),({2:-1.,3:1.},-3.)])
        if old.rows[:8] != original.rows:
            raise RuntimeError('Frozen OUT query prefix changed')
        # Regenerate big-M with added nuisance coefficients; never reuse stale M.
        new.either([({1:1.,5:-1.,3:1.},-.3+e),
                    ({0:-1.,5:1.,3:1.},-.3+e),
                    ({2:-1.,6:1.,3:1.},-3.+e)])
        prefix = 8
    else:
        raise ValueError('Expected hypothetical IN or OUT label')
    if (len(new.rows) != prefix or new.lower != old.lower[:len(new.lower)]
            or new.upper != old.upper[:len(new.upper)]):
        raise RuntimeError('Query replacement changed binary variable indexing')
    new.rows.extend(old.rows[prefix:])
    new.lower = list(old.lower)
    new.upper = list(old.upper)
    new.rays = list(old.rays)
    return new


def reference_camera(witness):
    bias = witness['bias']
    return (round(bias['pose_x_m'],12),round(bias['pose_z_m'],12))


def relative_label(witness):
    """Full extent relative to the actual FIRST camera, never the current view."""
    qx,qz = reference_camera(witness)
    boxes = [dict(b,x=b['x']-qx,z=b['z']-qz) for b in witness['scene']['boxes']]
    value = dict(witness['scene'],boxes=boxes)
    return 'IN' if shared.sensor.intersects_query(shared.geometry._scene(value)) else 'OUT'


def validation_detail(witness, observations, label):
    # Original structure, world-domain, nuisance and exact forward-bin checks.
    detail = shared.validation_detail(witness, observations, label, 'range_pose')
    detail['query_frame'] = QUERY_FRAME
    if 'actual_label' not in detail:
        return detail
    detail['fixed_world_query_label'] = detail['actual_label']
    detail['actual_label'] = relative_label(witness)
    detail['query_reference_camera'] = list(reference_camera(witness))
    failures = [s for s in detail['failures'] if s != 'QUERY_LABEL_MISMATCH']
    if detail['actual_label'] != label:
        index = failures.index('OBSERVATION_BIN_MISMATCH') if 'OBSERVATION_BIN_MISMATCH' in failures else len(failures)
        failures.insert(index,'QUERY_LABEL_MISMATCH')
    detail['failures'] = failures
    detail['valid'] = not failures
    return detail


def validate_witness(witness, observations, label):
    return validation_detail(witness,observations,label)['valid']


def propose_projected_in(witness):
    """Nearest fixed-width center in shifted closed query AND world domain."""
    box = shared.diagnostic._structure(witness['scene'])
    if not .03 <= box['width'] <= .6:
        raise ValueError('Cannot alter an out-of-domain width')
    qx,qz = reference_camera(witness)
    lower_x,upper_x = max(-.9,qx-.3-box['width']/2),min(.9,qx+.3+box['width']/2)
    lower_z,upper_z = max(.6,qz+.3-box['thickness']/2),min(3.4,qz+3.+box['thickness']/2)
    if lower_x > upper_x or lower_z > upper_z:
        raise ValueError('Shifted query and fixed world domain do not overlap')
    projected = dict(box,x=min(upper_x,max(lower_x,box['x'])),z=min(upper_z,max(lower_z,box['z'])))
    return dict(scene=dict(witness['scene'],boxes=[projected]),bias=dict(witness['bias']))


def _solve(observations,label):
    """Original shared-model solve lifecycle, with only query validation/projection changed."""
    c=_constraints(observations,label)
    started=time.perf_counter()
    receipt=dict(requested_label=label,mode='range_pose',solver='scipy.optimize.milp / HiGHS',
        scipy_version=shared.scipy.__version__,budget=dict(shared.base.SOLVER_OPTIONS),continuous_variables=7,
        binary_variables=len(c.lower)-7,constraints=len(c.rows),authority=AUTHORITY,
        formal_infeasibility_certificate=False,exclusion_supported=False,query_frame=QUERY_FRAME,
        query_rounding_outer_envelope_m=shared.ROUND_ENVELOPE,outer_relaxation=OUTER_DESCRIPTION)
    attempt=dict(label=label,raw_solver_vector=None,unrounded_candidate=None,raw_validation=None,
        decoded_candidate=None,validation=None,projected_candidate=None,projected_validation=None,
        projected_valid=False,projection_eligible=False)
    witness=None
    try:
        result=milp(**c.scipy())
        receipt.update(solver_status=int(result.status),message=str(result.message))
        for key in ('mip_node_count','mip_gap','mip_dual_bound'):
            value=getattr(result,key,None)
            receipt[key]=float(value) if value is not None and math.isfinite(value) else None
        if result.x is not None:
            values=[float(v) for v in result.x]
            attempt['raw_solver_vector']=[shared.diagnostic._number(v) for v in values]
        if result.x is None:
            if result.status==2:receipt.update(outcome='RELAXATION_REPORTED_INFEASIBLE',exclusion_supported=True)
            elif result.status==1:receipt['outcome']='LIMIT_WITHOUT_CANDIDATE'
            elif result.status==0:receipt['outcome']='REPORTED_OPTIMUM_WITHOUT_CANDIDATE'
            else:receipt['outcome']='SOLVER_FAILURE_WITHOUT_CANDIDATE'
        elif result.status not in (0,1):receipt['outcome']='SOLVER_STATUS_CANDIDATE_CONFLICT'
        elif not all(math.isfinite(v) for v in values):
            receipt['outcome']='NUMERICAL_CANDIDATE_REJECTED'
            attempt['validation']=dict(valid=False,failures=['NONFINITE_SOLVER_VECTOR'])
        else:
            raw=shared._unrounded_candidate(values);candidate=shared._candidate(values,'range_pose')
            detail=validation_detail(candidate,observations,label)
            attempt.update(unrounded_candidate=raw,raw_validation=validation_detail(raw,observations,label),
                           decoded_candidate=candidate,validation=detail)
            receipt['interior_slack']=values[3]
            receipt['outcome']='VALIDATED_WITNESS' if detail['valid'] else 'NUMERICAL_CANDIDATE_REJECTED'
            if detail['valid']:witness=candidate
            elif label=='IN' and result.status==0:
                proposal=propose_projected_in(candidate)
                checked=validation_detail(proposal,observations,label)
                attempt.update(projection_eligible=True,projected_candidate=proposal,
                    projected_validation=checked,projected_valid=checked['valid'])
                if checked['valid']:witness=proposal
    except Exception as exc:
        receipt.update(outcome='SOLVER_EXCEPTION',error_type=type(exc).__name__,message=str(exc),exclusion_supported=False)
        witness=None
    receipt['elapsed_seconds']=time.perf_counter()-started
    attempt['original_metadata']=receipt
    return witness,receipt,attempt


def infer(observations: list[dict]) -> dict:
    public=_inputs(observations)
    started=time.perf_counter();witnesses={};receipts={};attempts=[]
    for label in ('IN','OUT'):
        witnesses[label],receipts[label],attempt=_solve(public,label);attempts.append(attempt)
    decision,reason,conflicts=shared.paths._readout(witnesses,receipts)
    return dict(status='UNKNOWN' if decision=='UNKNOWN' else 'MODEL_CONDITIONAL',
        decision=decision,reason=reason,authority=AUTHORITY,mode='range_pose',query_frame=QUERY_FRAME,
        witnesses=witnesses,solver_metadata=[receipts[k] for k in ('IN','OUT')],attempts=attempts,
        exclusion_metadata={k:dict(reported_infeasible=r['exclusion_supported'],solver_status=r.get('solver_status'),
            outcome=r['outcome'],authority=AUTHORITY,is_formal_certificate=False) for k,r in receipts.items()},
        recovery_conflicts=conflicts,projection_added_labels=[a['label'] for a in attempts if a['projected_valid']],
        valid_witness_count=sum(w is not None for w in witnesses.values()),solver_calls=2,observations_used=13,
        model_domain=dict(shared.base.DOMAIN),nuisance_bounds=dict(range_m=[-.002,.002],pose_x_m=[-.001,.001],pose_z_m=[-.001,.001]),
        biases_shared_across_all_views=True,outer_relaxation=OUTER_DESCRIPTION,
        camera_rounding_outward_envelope_m=shared.ROUND_ENVELOPE,elapsed_seconds=time.perf_counter()-started,
        universal_uniqueness_proven=False,sensor_clearance_certified=False,hardware_error_bounds_calibrated=False,
        parameterization='ORIGINAL_WORLD_GEOMETRY_AND_ANCHORED_WALL_NOT_A_FULL_GAUGE_QUOTIENT',
        backend='CPU / TASK_NOT_GPU_SUITABLE')
