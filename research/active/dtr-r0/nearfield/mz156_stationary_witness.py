"""Positive full-zone evidence memory, conditional on a stationary rig and world.

This is an experimental operating condition, not a motion detector. No RGB crop,
surface association, missing-as-free inference, or cross-frame cone intersection.
"""
import math
from mz115_spatial_allocation import zone_box, slant_envelope, certain

METHOD = dict(schema='MZ156_STATIONARY_FULL_ZONE_POSITIVE_MEMORY_V1',
    corridor_m=[[.2,3.6],[-.3,.3],[.4,2.05]],
    range_sigma_multiplier=3.,aggregation_span_allowance_m=.10,quantization_allowance_m=.01,
    initial_yaw_uncertainty_deg=.5,yaw_bias_allowance_deg_per_s=.2,
    increment_noise_sigma_deg=.08,increment_sigma_multiplier=3.,pitch_uncertainty_deg=.5,
    maximum_age_s=1.25,maximum_frames=6,cadence_s=.25,
    independent_positive_or_baseline=True,full_zone_no_rgb_clipping=True,
    bounds_authority='NOMINAL_SIMULATOR_CONTRACT_NOT_HARD_GUARANTEES_OR_HARDWARE_CALIBRATION',
    activation='EXPLICIT_EXPERIMENTAL_STATIONARY_BODY_CAMERA_WORLD_CONDITION',
    absence_semantics='UNKNOWN_NOT_CLEAR_SPACE')


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def pose_valid(row):
    intr=row.get('rgb_intrinsics',{});camera=row.get('camera_in_body_m',[])
    return (row.get('imu_valid') is True and finite(row.get('delta_yaw'))
        and finite(row.get('delta_pitch')) and abs(row['delta_pitch'])<1e-9
        and finite(row.get('camera_pitch_deg')) and len(camera)==3 and all(map(finite,camera))
        and all(finite(intr.get(k)) for k in ('fx','fy','cx','cy'))
        and intr['fx']>0 and intr['fy']>0)


def calibration(row):
    intr=row.get('rgb_intrinsics',{})
    return (row.get('camera_pitch_deg'),tuple(row.get('camera_in_body_m',[])),
        tuple(intr.get(k) for k in ('fx','fy','cx','cy')))


def current_witnesses(row,yaw,elapsed,increments):
    """Keep each qualifying return independently; uncertainty is never fitted."""
    result=[];rejected=[]
    dyaw=.5+.2*elapsed+3*.08*math.sqrt(increments)
    if not row.get('tof_packet_received',False):return result,rejected,dyaw
    for zone in row.get('tof_zones',[]):
        angles=[*zone.get('theta_bounds_deg',[]),*zone.get('phi_bounds_deg',[])]
        valid_angles=(len(angles)==4 and all(map(finite,angles)) and
            -89<angles[0]<angles[1]<89 and -89<angles[2]<angles[3]<89)
        if not valid_angles:
            rejected.append(dict(zone_id=zone.get('zone_id'),reason='INVALID_ANGULAR_INTERVAL'));continue
        box=zone_box(zone,row['rgb_intrinsics'])
        for slot,target in enumerate(zone.get('targets',[])):
            r=target.get('distance_m');sigma=target.get('range_noise_sigma_m')
            if target.get('status')!='SIM_VALID':continue
            if not finite(r) or not finite(sigma) or r<=0 or sigma<0:
                rejected.append(dict(zone_id=zone['zone_id'],slot=slot,reason='INVALID_RANGE'));continue
            half=3*sigma+.10+.01
            ranges=[max(0.,r-half),r+half]
            pitch=row['camera_pitch_deg'];offset=row['camera_in_body_m']
            xyz=slant_envelope(box,ranges,row['rgb_intrinsics'],(pitch-.5,pitch+.5),(yaw-dyaw,yaw+dyaw),offset[2])
            xyz[0]=[v+offset[0] for v in xyz[0]];xyz[1]=[v+offset[1] for v in xyz[1]]
            if certain(xyz,3.6):
                result.append(dict(source_id=row['id'],source_time_s=row['time_s'],zone_id=zone['zone_id'],
                    target_slot=slot,range_bounds_m=ranges,zone_box=box,xyz_m=xyz,
                    public_yaw_deg=yaw,yaw_bounds_deg=[yaw-dyaw,yaw+dyaw],
                    pitch_bounds_deg=[pitch-.5,pitch+.5],evidence_age_s=0.))
    return result,rejected,dyaw


def predict(rows,baseline,*,stationary_world=False):
    """Never re-establish body heading after a missing IMU or an episode gap.

Episode origin must be an explicit forward-aligned time0 sample. Any broken
orientation chain disables this additive branch until the next such episode.
Missing ToF packets alone do not invalidate earlier positive static evidence.
"""
    if len(rows)!=len(baseline):raise ValueError('Matched baseline flags required')
    result=[];previous_episode=None;previous_time=None;origin=None;pose=None
    yaw=0.;increments=0;active=False;memory=[];frame_count=0
    for row,base in zip(rows,baseline):
        if not isinstance(base,bool):raise ValueError('Boolean baseline flags required')
        ep=row['episode_id'];now=row.get('time_s');new=ep!=previous_episode
        reasons=[]
        if new:
            memory=[];yaw=0.;increments=0;frame_count=0;origin=now
            active=bool(stationary_world and finite(now) and abs(now)<1e-9 and pose_valid(row)
                and abs(row['delta_yaw'])<1e-9)
            pose=calibration(row)
            if not active:reasons.append('NO_VALID_FORWARD_ALIGNED_STATIONARY_START')
        else:
            contiguous=(finite(now) and finite(previous_time) and abs(now-previous_time-.25)<1e-6)
            valid=pose_valid(row) and pose==calibration(row)
            if not contiguous or not valid:
                active=False;memory=[];reasons.append('BROKEN_POSE_OR_CADENCE_CHAIN')
            if active:
                yaw+=row['delta_yaw'];increments+=1
        frame_count+=1
        elapsed=now-origin if finite(now) and finite(origin) else None
        if active and (elapsed is None or elapsed>1.25+1e-9 or frame_count>6):
            active=False;memory=[];reasons.append('STATIONARY_EXPERIMENT_WINDOW_ENDED')
        current=[];rejected=[];dyaw=None
        if active:
            current,rejected,dyaw=current_witnesses(row,yaw,elapsed,increments)
            memory=[w for w in memory if 0<=now-w['source_time_s']<=1.25+1e-9]+current
        carried=[dict(w,evidence_age_s=now-w['source_time_s']) for w in memory] if active else []
        current_flag=bool(current);prefix_flag=bool(carried)
        result.append(dict(id=row['id'],episode_id=ep,time_s=now,baseline=base,active=active,
            current_witness=current_flag,prefix_witness=prefix_flag,
            current_candidate=bool(base or current_flag),prefix_candidate=bool(base or prefix_flag),
            current_state='ALERT' if base or current_flag else 'UNKNOWN',
            prefix_state='ALERT' if base or prefix_flag else 'UNKNOWN',
            current_supports=current,prefix_supports=carried,rejected=rejected,reset_reasons=reasons,
            public_yaw_deg=yaw if active else None,yaw_radius_deg=dyaw,
            stationary_condition_is_external=bool(stationary_world)))
        previous_episode=ep;previous_time=now
    return result
