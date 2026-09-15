"""Fixed observable coarse/fine edge intervention on selected ToF alert supports.

The shared plane is a hypothesis, not a surface association certificate. Raw
packets are immutable. Unmatched, merged and out-of-view supports keep their
incumbent votes; eligible supports have their votes replaced, not ORed back in.
"""
import math
import numpy as np
from mz136_boundary_geometry import camera_to_body, near_cohorts
from mz136_rectified_edges import refine, transform
from mz115_spatial_allocation import possible, certain, intersection
from mz128_zone_weighting import FOUR, column_weights

METHOD = dict(proposal='FROZEN_MZ136_FIRST_CONTRAST_PROPOSAL',
    range_selection='UNIQUE_MATCHING_VALID_TOF_COHORT_MEDIAN_BODY_FORWARD_ZONE_CENTER',
    geometry='BODY_FORWARD_PLANE_AT_MEDIAN_OBSERVED_DEPTH',
    replacements='SINGLE_VALID_SLOT_FULL_ZONE_IN_RGB_INTERSECTS_COARSE_SEED',
    missing='FINE_TO_COARSE_TO_INCUMBENT', distance_m=3.6,
    corridor_y_m=[-.3,.3], corridor_z_m=[.4,2.05], threshold=1.,
    radar='FROZEN_MZ129_RADAR_AND_GUARDS_UNCHANGED', sweeps=0)


def association(row, edge, yaw):
    if 'seed' not in edge or not row['imu_valid']:
        return None
    seed = edge['seed']
    wanted = sorted(seed['zones'])
    groups = [g for g in near_cohorts(row) if sorted(v['zone'] for v in g)==wanted]
    if len(groups)!=1 or len(set(wanted))!=len(wanted):
        return None
    group=groups[0]; intr=row['rgb_intrinsics']
    zones={z['zone_id']:z for z in row['tof_zones']}
    depths=[]; eligible=[]
    for item in group:
        zone=zones[item['zone']]
        a,b=[math.radians(sum(zone[k])/2) for k in ('theta_bounds_deg','phi_bounds_deg')]
        point=camera_to_body(row,yaw)@np.array([item['depth'],item['depth']*math.tan(a),item['depth']*math.tan(b)])
        depths.append(float(point[0]))
        l,t,r,bottom=item['box']
        if (len(zone['targets'])==1 and 0<=l<r<=intr['width']-1 and
                0<=t<bottom<=intr['height']-1 and intersection(item['box'],seed['box'])):
            eligible.append([item['zone'],item['slot']])
    depth=float(np.median(depths))
    if not eligible or not math.isfinite(depth) or depth<=0:
        return None
    l,t,r,b=seed['box']
    corners=transform([[l,t],[r,t],[r,b],[l,b]],np.array(edge['homography']))
    return dict(selected_returns=[[v['zone'],v['slot']] for v in group],
        replaced_returns=eligible, body_forward_depth_m=depth,
        depth_spread_m=float(np.ptp(depths)),
        rectified_vertical_px=[float(corners[:,1].min()),float(corners[:,1].max())],
        assumption='COHORT_AND_VISIBLE_SILHOUETTE_SHARE_ONE_BODY_FORWARD_PLANE')


def plane_support(row, association, edges):
    intr=row['rgb_intrinsics']; d=association['body_forward_depth_m']
    ox,oy,oz=row['camera_in_body_m']; top,bottom=association['rectified_vertical_px']
    return [[ox+d,ox+d],
        [oy+d*(u-intr['cx'])/intr['fx'] for u in edges],
        [oz+d*(intr['cy']-v)/intr['fy'] for v in (bottom,top)]]


def arm_readout(row, cached, incumbent, radar, assoc, edges):
    if assoc is None:
        return dict(candidate=incumbent['candidate'], state='INCUMBENT_FALLBACK',
                    xyz=None, replaced_returns=[], support_bits=[], residual_alert=None)
    replaced={tuple(v) for v in assoc['replaced_returns']}
    xyz=plane_support(row,assoc,edges); weights=column_weights(row,FOUR)
    active=set(); residual=set(); certain_residual=False; bits=[]
    found=set()
    for item in cached['spatial_evidence']:
        key=(item['zone_id'],item['target_slot']); before=possible(item['localized_xyz'])
        changed=key in replaced
        after=possible(xyz) if changed else before
        if changed:found.add(key)
        if after:active.add(item['zone_id'])
        if not changed:
            if before:residual.add(item['zone_id'])
            certain_residual |= certain(item['coarse_xyz'])
        bits.append(dict(zone=key[0],slot=key[1],replaced=changed,
                         before=before,after=after,old_certain=certain(item['coarse_xyz'])))
    assert found==replaced
    radar_vote=bool(radar['candidate'])
    # The frozen Radar branch already includes its raw/flow/resolution guards.
    # cached.common_radar is pre-MZ129 and must not reintroduce its old alerts.
    guards=bool(radar.get('guard_events'))
    residual_alert=bool(radar_vote or certain_residual or sum(weights[z] for z in residual)>=1.)
    flag=bool(radar_vote or certain_residual or certain(xyz) or sum(weights[z] for z in active)>=1.)
    return dict(candidate=flag,state='CONDITIONAL_PLANE_READOUT',xyz=xyz,
        edges_px=list(edges),replaced_returns=assoc['replaced_returns'],support_bits=bits,
        residual_alert=residual_alert,radar_vote=radar_vote,guard_vote=guards,
        tof_score=sum(weights[z] for z in active),plane_corridor=possible(xyz),
        signed_lateral_overlap_m=min(xyz[1][1],.3)-max(xyz[1][0],-.3))


def predict_frame(row,image,cached,incumbent,radar):
    yaw=cached['integrated_yaw_deg']
    edge=refine(row,image,yaw)
    assoc=association(row,edge,yaw)
    coarse=edge.get('rectified_seed_edges')
    fine=edge['candidate']['rectified_edges_px'] if edge['candidate'] else coarse
    arms={name:arm_readout(row,cached,incumbent,radar,assoc,values)
          for name,values in [('coarse',coarse),('fine',fine)]}
    return dict(id=row['id'],edge_state=edge['state'],association=assoc,
        fine_available=edge['candidate'] is not None,coarse_available=coarse is not None,
        edge=edge,**arms)
