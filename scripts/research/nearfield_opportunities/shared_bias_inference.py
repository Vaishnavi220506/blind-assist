"""Continuous witnesses with one shared bounded range/pose bias per history."""
from __future__ import annotations

import math
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

import active_view as sensor
import bent_path_inference as paths
import candidate_diagnostics as diagnostic
import continuous_boundary_witness as base
import path_constraint_inference as geometry


AUTHORITY = 'NUMERICAL_SINGLE_RECTANGLE_SHARED_BIAS_MODEL_ONLY'
MODES = ('range_only', 'range_pose')
RANGE_BOUND = .002
POSE_BOUND = .001
# Injector rounds each actual coordinate to 12 decimals. One decimal unit
# conservatively covers half-unit rounding plus binary arithmetic representation.
ROUND_ENVELOPE = 1e-12
OUTER_DESCRIPTION = 'INCLUSIVE_BIN_CLOSURE_WITH_SHARED_BIAS_EPS_AND_CAMERA_ROUNDING_ENVELOPE'


def _mode(mode):
    if mode not in MODES:
        raise ValueError('Expected range_only or range_pose')
    return POSE_BOUND if mode == 'range_pose' else 0.


class _Constraints(base._Constraints):
    def __init__(self, mode):
        p = _mode(mode)
        # L, R, front Z, common interior slack, range bias, pose X, pose Z.
        self.lower = [-1.2, -1.2, .58, 0., -RANGE_BOUND, -p, -p]
        self.upper = [1.2, 1.2, 3.38, base.SLACK_MAX, RANGE_BOUND, p, p]
        self.rows = []
        self.rays = []

    def scipy(self):
        matrix = np.zeros((len(self.rows), len(self.lower)))
        for i, (coefficients, _) in enumerate(self.rows):
            for j, value in coefficients.items():
                matrix[i, j] = value
        objective = np.zeros(len(self.lower))
        objective[3] = -1.
        return dict(c=objective,
                    integrality=np.array([0]*7 + [1]*(len(self.lower)-7)),
                    bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(matrix, -np.inf, [b for _, b in self.rows]),
                    options=dict(base.SOLVER_OPTIONS))


def _constraints(observations, label, mode='range_pose'):
    """Sound outer closure of the rounded-camera, shared-nuisance model."""
    p = _mode(mode)
    if label not in ('IN', 'OUT'):
        raise ValueError('Expected hypothetical IN or OUT label')
    c = _Constraints(mode)
    c.le({0:1.,1:-1.},-.03)
    c.le({0:-1.,1:1.},.60)
    c.le({0:.5,1:.5},.9)
    c.le({0:-.5,1:-.5},.9)
    if label == 'IN':
        c.le({0:1.},.3+sensor.EPS)
        c.le({1:-1.},.3+sensor.EPS)
        c.le({2:1.},3.+sensor.EPS)
    else:
        c.either([({1:1.,3:1.},-.3),({0:-1.,3:1.},-.3),({2:-1.,3:1.},-3.)])
    for oi, row in enumerate(observations):
        cx, cz = row['camera']
        for ri, (angle, value) in enumerate(zip(base.ANGLES,row['bins'],strict=True)):
            dx,dz = math.sin(angle), math.cos(angle)
            slope = dx/dz
            low,high = (value-.5)*base.RANGE_STEP,(value+.5)*base.RANGE_STEP
            target_upper = (3.42-cz+p+ROUND_ENVELOPE)/dz + RANGE_BOUND + sensor.EPS
            wall_lower = (base.WALL-cz-p-ROUND_ENVELOPE)/dz - RANGE_BOUND
            wall_upper = (base.WALL-cz+p+ROUND_ENVELOPE)/dz + RANGE_BOUND
            # A whole bin is much narrower than the gap between envelopes.
            if wall_lower-target_upper <= base.RANGE_STEP:
                raise RuntimeError('Declared target and wall bin envelopes overlap')
            target_possible = low <= target_upper
            wall_possible = low <= wall_upper and high >= wall_lower
            if target_possible and wall_possible:
                raise RuntimeError('Unexpected ambiguous target/wall bin branch')
            branch = 'TARGET' if target_possible else 'WALL' if wall_possible else 'IMPOSSIBLE'
            c.rays.append(dict(observation_index=oi,ray=ri,branch=branch,
                               target_max=target_upper,wall_min=wall_lower,wall_max=wall_upper))
            if branch == 'IMPOSSIBLE':
                c.le({},-1.)
                continue
            rounding_overlap = ROUND_ENVELOPE*(1.+abs(slope))
            hit_allowance = rounding_overlap+abs(dx)*sensor.EPS
            if dx > 0:
                misses = [({1:1.,2:-slope,3:1.,5:-1.,6:slope},cx-slope*cz+rounding_overlap),
                          ({0:-1.,2:slope,3:1.,5:1.,6:-slope},-cx+slope*cz-slope*base.THICKNESS+rounding_overlap)]
                hits = [({1:-1.,2:slope,3:1.,5:1.,6:-slope},-cx+slope*cz+hit_allowance),
                        ({0:1.,2:-slope,3:1.,5:-1.,6:slope},cx-slope*cz+slope*base.THICKNESS+hit_allowance)]
                side = 0
            else:
                misses = [({0:-1.,2:slope,3:1.,5:1.,6:-slope},-cx+slope*cz+rounding_overlap),
                          ({1:1.,2:-slope,3:1.,5:-1.,6:slope},cx-slope*cz+slope*base.THICKNESS+rounding_overlap)]
                hits = [({0:1.,2:-slope,3:1.,5:-1.,6:slope},cx-slope*cz+hit_allowance),
                        ({1:-1.,2:slope,3:1.,5:1.,6:-slope},-cx+slope*cz-slope*base.THICKNESS+hit_allowance)]
                side = 1
            if branch == 'WALL':
                c.either(misses)
                c.le({4:1.,6:-1./dz,3:1.},high-(base.WALL-cz)/dz+ROUND_ENVELOPE/dz)
                c.le({4:-1.,6:1./dz,3:1.},-low+(base.WALL-cz)/dz+ROUND_ENVELOPE/dz)
                continue
            for a,b in hits:
                c.le(a,b)
            c.le({2:1./dz,6:-1./dz,4:1.,3:1.},high+cz/dz+ROUND_ENVELOPE/dz)
            c.le({side:1./dx,5:-1./dx,4:1.,3:1.},high+cx/dx+ROUND_ENVELOPE/abs(dx))
            c.either([({2:-1./dz,6:1./dz,4:-1.,3:1.},-low-cz/dz+ROUND_ENVELOPE/dz),
                      ({side:-1./dx,5:1./dx,4:-1.,3:1.},-low-cx/dx+ROUND_ENVELOPE/abs(dx))])
    return c


def _candidate(vector, mode):
    p = _mode(mode)
    def normalized(v, limit):
        return min(limit,max(-limit,round(float(v),12)))
    return dict(scene=base._candidate(vector),
                bias=dict(range_m=normalized(vector[4],RANGE_BOUND),
                          pose_x_m=normalized(vector[5],p),pose_z_m=normalized(vector[6],p)))


def _unrounded_candidate(vector):
    return dict(scene=diagnostic.unrounded_candidate(vector),
                bias=dict(zip(('range_m','pose_x_m','pose_z_m'),map(float,vector[4:7]),strict=True)))


def validation_detail(witness, observations, label, mode='range_pose'):
    p = _mode(mode)
    if label not in ('IN','OUT'):
        raise ValueError('Expected hypothetical IN or OUT label')
    detail = dict(valid=False,failures=[],requested_label=label,mode=mode)
    try:
        if not isinstance(witness,dict) or set(witness) != {'scene','bias'}:
            raise ValueError('Witness must contain scene and bias only')
        box = diagnostic._structure(witness['scene'])
        bias = witness['bias']
        if not isinstance(bias,dict) or set(bias) != {'range_m','pose_x_m','pose_z_m'}:
            raise ValueError('Exactly three shared bias coordinates required')
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in bias.values()):
            raise ValueError('Finite bias coordinates required')
    except (ValueError,KeyError,TypeError) as exc:
        detail['failures'].append('STRUCTURE_OR_NONFINITE_COORDINATES')
        detail['error']=str(exc)
        return detail
    detail['domain_valid'] = -.9<=box['x']<=.9 and .03<=box['width']<=.6 and .6<=box['z']<=3.4
    detail['bias_valid'] = (abs(bias['range_m'])<=RANGE_BOUND and abs(bias['pose_x_m'])<=p
                            and abs(bias['pose_z_m'])<=p)
    if not detail['domain_valid']: detail['failures'].append('OUTSIDE_DECLARED_GEOMETRY_DOMAIN')
    if not detail['bias_valid']: detail['failures'].append('OUTSIDE_DECLARED_BIAS_DOMAIN')
    scene = geometry._scene(witness['scene'])
    detail['actual_label'] = 'IN' if sensor.intersects_query(scene) else 'OUT'
    if detail['actual_label'] != label: detail['failures'].append('QUERY_LABEL_MISMATCH')
    mismatches,replay = [],[]
    for oi,row in enumerate(observations):
        actual = (round(row['camera'][0]+bias['pose_x_m'],12),
                  round(row['camera'][1]+bias['pose_z_m'],12))
        raw=[]
        for angle in sensor.ANGLES:
            wall=(scene.wall_z-actual[1])/math.cos(angle)
            target=min(sensor.hit_distance(b,actual,angle) for b in scene.boxes)
            raw.append(min(wall,target))
        measured=[v+bias['range_m'] for v in raw]
        bins=[math.floor(v/base.RANGE_STEP+.5) for v in measured]
        replay.append(dict(camera=list(row['camera']),actual_camera=list(actual),bins=bins,
                           raw_ranges=raw,biased_ranges=measured))
        for ri,(wanted,found) in enumerate(zip(row['bins'],bins,strict=True)):
            if wanted!=found:
                mismatches.append(dict(observation_index=oi,ray=ri,expected_bin=wanted,actual_bin=found))
    detail.update(bin_mismatches=mismatches,bin_mismatch_count=len(mismatches),replay=replay)
    if mismatches: detail['failures'].append('OBSERVATION_BIN_MISMATCH')
    detail['valid']=not detail['failures']
    return detail


def validate_witness(witness, observations, label, mode='range_pose'):
    return validation_detail(witness,observations,label,mode)['valid']


def _solve(observations,label,mode):
    c=_constraints(observations,label,mode)
    started=time.perf_counter()
    receipt=dict(requested_label=label,mode=mode,solver='scipy.optimize.milp / HiGHS',
        scipy_version=scipy.__version__,budget=dict(base.SOLVER_OPTIONS),continuous_variables=7,
        binary_variables=len(c.lower)-7,constraints=len(c.rows),authority=AUTHORITY,
        formal_infeasibility_certificate=False,exclusion_supported=False,
        outer_relaxation=OUTER_DESCRIPTION)
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
            attempt['raw_solver_vector']=[diagnostic._number(v) for v in values]
        if result.x is None:
            if result.status==2:
                receipt.update(outcome='RELAXATION_REPORTED_INFEASIBLE',exclusion_supported=True)
            elif result.status==1: receipt['outcome']='LIMIT_WITHOUT_CANDIDATE'
            elif result.status==0: receipt['outcome']='REPORTED_OPTIMUM_WITHOUT_CANDIDATE'
            else: receipt['outcome']='SOLVER_FAILURE_WITHOUT_CANDIDATE'
        elif result.status not in (0,1):
            receipt['outcome']='SOLVER_STATUS_CANDIDATE_CONFLICT'
        elif not all(math.isfinite(v) for v in values):
            receipt['outcome']='NUMERICAL_CANDIDATE_REJECTED'
            attempt['validation']=dict(valid=False,failures=['NONFINITE_SOLVER_VECTOR'])
        else:
            raw=_unrounded_candidate(values);candidate=_candidate(values,mode)
            detail=validation_detail(candidate,observations,label,mode)
            attempt.update(unrounded_candidate=raw,raw_validation=validation_detail(raw,observations,label,mode),
                           decoded_candidate=candidate,validation=detail)
            receipt['interior_slack']=values[3]
            receipt['outcome']='VALIDATED_WITNESS' if detail['valid'] else 'NUMERICAL_CANDIDATE_REJECTED'
            if detail['valid']: witness=candidate
            elif label=='IN' and result.status==0:
                proposal=dict(scene=diagnostic.propose_projected_in(candidate['scene']),bias=dict(candidate['bias']))
                checked=validation_detail(proposal,observations,label,mode)
                attempt.update(projection_eligible=True,projected_candidate=proposal,
                               projected_validation=checked,projected_valid=checked['valid'])
                if checked['valid']: witness=proposal
    except Exception as exc:
        receipt.update(outcome='SOLVER_EXCEPTION',error_type=type(exc).__name__,message=str(exc),exclusion_supported=False)
        witness=None
    receipt['elapsed_seconds']=time.perf_counter()-started
    attempt['original_metadata']=receipt
    return witness,receipt,attempt


def infer(observations: list[dict], mode='range_pose') -> dict:
    _mode(mode)
    public=paths._inputs(observations)
    started=time.perf_counter()
    witnesses,receipts,attempts={},{},[]
    for label in ('IN','OUT'):
        witnesses[label],receipts[label],attempt=_solve(public,label,mode)
        attempts.append(attempt)
    decision,reason,conflicts=paths._readout(witnesses,receipts)
    return dict(status='UNKNOWN' if decision=='UNKNOWN' else 'MODEL_CONDITIONAL',
        decision=decision,reason=reason,authority=AUTHORITY,mode=mode,witnesses=witnesses,
        solver_metadata=[receipts[k] for k in ('IN','OUT')],attempts=attempts,
        exclusion_metadata={k:dict(reported_infeasible=r['exclusion_supported'],solver_status=r.get('solver_status'),
            outcome=r['outcome'],authority=AUTHORITY,is_formal_certificate=False) for k,r in receipts.items()},
        recovery_conflicts=conflicts,projection_added_labels=[a['label'] for a in attempts if a['projected_valid']],
        valid_witness_count=sum(w is not None for w in witnesses.values()),solver_calls=2,
        observations_used=len(public),model_domain=dict(base.DOMAIN),
        nuisance_bounds=dict(range_m=[-RANGE_BOUND,RANGE_BOUND],pose_x_m=[-_mode(mode),_mode(mode)],
                             pose_z_m=[-_mode(mode),_mode(mode)]),
        biases_shared_across_all_views=True,outer_relaxation=OUTER_DESCRIPTION,
        camera_rounding_outward_envelope_m=ROUND_ENVELOPE,
        elapsed_seconds=time.perf_counter()-started,universal_uniqueness_proven=False,
        sensor_clearance_certified=False,hardware_error_bounds_calibrated=False,
        backend='CPU / TASK_NOT_GPU_SUITABLE')
