"""Conditional plane outer bounds from finite-zone interval constraints.

Eight closed coefficient-sign octants linearize interval overlap. No surface
identity is proved; infeasible, unbounded or nonpositive inversions abstain.
"""
import copy
import itertools
import math
import numpy as np
from scipy.optimize import linprog
import mz117_surface_intervals as prior
import mz115_spatial_allocation as zonal

LP_EPS=1e-8
LP_OPTIONS=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)


def octants(anchors,intr):
    if len(anchors)<6 or len({a['zone_id']//8 for a in anchors})<2 or len({a['zone_id']%8 for a in anchors})<2:
        return None
    if any(a['range_m']<=prior.RANGE_ERROR_M for a in anchors):return None
    rays=np.array([prior.unit_ray_intervals(a['zone_box'],intr) for a in anchors])
    ranges=np.array([a['range_m'] for a in anchors]);lo=1/(ranges+prior.RANGE_ERROR_M);hi=1/(ranges-prior.RANGE_ERROR_M)
    result=[]
    for signs in itertools.product((-1,1),repeat=3):
        smallest=np.column_stack([rays[:,j,0 if sign>0 else 1] for j,sign in enumerate(signs)])
        largest=np.column_stack([rays[:,j,1 if sign>0 else 0] for j,sign in enumerate(signs)])
        A=np.vstack([smallest,-largest]);b=np.concatenate([hi,-lo])+LP_EPS
        bounds=[(0,None) if sign>0 else (None,0) for sign in signs]
        fit=linprog(np.zeros(3),A_ub=A,b_ub=b,bounds=bounds,method='highs',options=LP_OPTIONS)
        if fit.status==2:continue
        if not fit.success:return None
        result.append(dict(signs=list(signs),A=A.tolist(),b=b.tolist(),bounds=bounds))
    return result or None


def denominator_bounds(polys,roi,intr):
    rays=prior.unit_ray_intervals(roi,intr);lower=math.inf;upper=-math.inf
    for poly in polys:
        smallest=np.array([rays[j][0 if s>0 else 1] for j,s in enumerate(poly['signs'])])
        largest=np.array([rays[j][1 if s>0 else 0] for j,s in enumerate(poly['signs'])])
        common=dict(A_ub=poly['A'],b_ub=poly['b'],bounds=poly['bounds'],method='highs',options=LP_OPTIONS)
        low=linprog(smallest,**common);high=linprog(-largest,**common)
        if not low.success or not high.success:return None
        lower=min(lower,float(low.fun));upper=max(upper,-float(high.fun))
    if not math.isfinite(lower+upper):return None
    return (lower-LP_EPS*max(1,abs(lower)),upper+LP_EPS*max(1,abs(upper)))


def range_bounds(polys,roi,intr):
    denominator=denominator_bounds(polys,roi,intr)
    if denominator is None or denominator[0]<=0:return None
    bounds=(max(.02,1/denominator[1]),min(4.,1/denominator[0]))
    return bounds if bounds[0]<=bounds[1] else None


def refine(row,prediction):
    result=copy.deepcopy(prediction);items=result['spatial_evidence'];boxes=result['proposals'];intr=row['rgb_intrinsics']
    counts={};models=[]
    for item in items:counts[item['zone_id']]=counts.get(item['zone_id'],0)+1
    for j,box in enumerate(boxes):
        inner=[box[0]+2,box[1]+2,box[2]-2,box[3]-2];anchors=[]
        for item in items:
            z=item['zone_box']
            if item['status']!='SIM_VALID' or counts[item['zone_id']]!=1:continue
            if not(inner[0]<=z[0] and inner[1]<=z[1] and inner[2]>=z[2] and inner[3]>=z[3]):continue
            if [k for k,b in enumerate(boxes) if zonal.intersection(z,b)]==[j]:anchors.append(item)
        polys=octants(anchors,intr)
        if polys:models.append(dict(proposal=j,octants=polys,anchor_keys=[[a['zone_id'],a['target_slot']] for a in anchors],
            authority='CONDITIONAL_COMMON_PLANE_INTERVAL_OUTER_RELAXATION_NOT_IDENTITY'))
    result['surface_models']=models
    for item in items:
        if item['status']!='SIM_MERGED':continue
        overlap=[k for k,b in enumerate(boxes) if zonal.intersection(item['zone_box'],b)]
        eligible=[m for m in models if overlap==[m['proposal']]]
        if len(eligible)!=1:continue
        model=eligible[0];j=model['proposal'];b=boxes[j]
        roi=zonal.intersection(item['zone_box'],[b[0]-2,b[1]-2,b[2]+2,b[3]+2])
        bounds=range_bounds(model['octants'],roi,intr)
        if bounds is None or bounds[1]<item['range_m']-.13 or bounds[0]>item['range_m']+.13:continue
        yaw=result['integrated_yaw_deg'];dy=.5+.2*row['time_s'];pitch=row['camera_pitch_deg']
        xyz=zonal.slant_envelope(roi,bounds,intr,(pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
        item.update(original_range_bounds=item['range_bounds'],original_localized_xyz=item['localized_xyz'],original_roi=item['roi'],
            range_bounds=bounds,localized_xyz=xyz,roi=roi,proposal=j,plane_proposal=j,plane_model=model,
            plane_geometry='INTERVAL_PLANE_OCTANT_LP',surface_state='CONDITIONAL_SINGLE_SURFACE_PROXY',sources=['RGB','TOF','IMU'])
    return result
