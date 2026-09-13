"""Causal native-displacement Doppler; evaluator truth stays separate from raw.

MZ107 ToF, Radar detection/noise/quantization, ghost and slot rules are retained.
Horizontal radial velocity uses past/current native actor and camera positions;
no source velocity or future position enters observations. First Doppler is missing.
"""
import math
import random
from mz99_angle_information_capture import measure_radar,retained_slots

def basis(c):
    p,y,r=(math.radians(c.get(k,0)) for k in ('pitch','yaw','roll'))
    cp,sp,cy,sy,cr,sr=math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    return ((cp*cy,cp*sy,sp),(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp),(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp))

def relative_radial_velocity(center, camera, previous_center, previous_camera, dt):
    """Past finite-difference relative velocity projected on current horizontal LOS."""
    if previous_center is None or previous_camera is None or dt is None or not math.isfinite(dt) or dt <= 0:
        return None
    values = list(center) + list(camera) + list(previous_center) + list(previous_camera)
    if not all(math.isfinite(v) for v in values):
        return None
    dx, dy = center[0] - camera[0], center[1] - camera[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return None
    vx = ((center[0]-previous_center[0]) - (camera[0]-previous_camera[0])) / dt
    vy = ((center[1]-previous_center[1]) - (camera[1]-previous_camera[1])) / dt
    return (vx*dx + vy*dy) / distance


def sensors(u,world,frame,actors,state):
    ep=frame['episode'];t=frame['time_s'];cam=frame['camera']
    if ep not in state:state[ep]=dict(rng=random.Random(frame['sensor_seed']),previous_yaw=0.)
    st=state[ep];rng=st['rng'];f,r,up=basis(cam)
    previous_time=st.get('previous_time_s');dt=None if previous_time is None else t-previous_time
    camera_m=[cam[k] for k in ('x','y','z')];previous_camera=st.get('previous_camera_m')
    previous_centers=st.get('previous_native_centers_m',{});current_centers={}
    origin=u.Vector(*(cam[k]*100 for k in ('x','y','z')))
    def trace(end):
        hit=u.SystemLibrary.line_trace_single(world,origin,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
        if not hit or not hit.to_tuple()[0]:return None
        fields=hit.to_tuple();return fields[5],fields[10]
    packet=rng.random()>=.1
    row=dict(id=frame['id'],episode_id=ep,time_s=t,camera_in_body_m=[0.,0.,1.7],camera_pitch_deg=-3.,
        tof_packet_received=packet,tof_range_m=[None]*8,tof_theta_deg=[-22.5+(k+.5)*45/8 for k in range(8)],
        tof_status=[255 if packet else 0]*8,tof_range_sigma_m=[.04]*8,
        tof64_range_m=[None]*64,tof64_status=[255 if packet else 0]*64,tof64_theta_deg=[],tof64_phi_deg=[],
        radar_packet_received=True,radar_range_m=[None]*4,radar_angle=[None]*4,radar_velocity=[None]*4,radar_valid=[False]*4,
        delta_yaw=0. if t==0 else cam['yaw']-st['previous_yaw']+.25*.2+rng.gauss(0,.08),delta_pitch=0.,imu_valid=True)
    st['previous_yaw']=cam['yaw'];native_tof=[]
    for iy in range(8):
        for ix in range(8):
            k=iy*8+ix;az=-22.5+(ix+.5)*45/8;el=22.5-(iy+.5)*45/8
            row['tof64_theta_deg'].append(az);row['tof64_phi_deg'].append(el)
            d=[f[j]+math.tan(math.radians(az))*r[j]+math.tan(math.radians(el))*up[j] for j in range(3)]
            norm=math.sqrt(sum(v*v for v in d));hit=trace(origin+u.Vector(*(v/norm*400 for v in d)))
            true_range=None
            if hit:
                p,_=hit;true_range=math.sqrt((p.x-origin.x)**2+(p.y-origin.y)**2+(p.z-origin.z)**2)/100
                pd=(.9 if true_range<=2 else .65 if true_range<=3 else .35)*.85
                if packet and rng.random()<pd:
                    row['tof64_range_m'][k]=max(.05,round((true_range+rng.gauss(0,.04))/.02)*.02);row['tof64_status'][k]=5
            native_tof.append(true_range)
    for ix in range(8):
        values=[row['tof64_range_m'][iy*8+ix] for iy in range(8) if row['tof64_status'][iy*8+ix]==5]
        if values:row['tof_range_m'][ix]=min(values);row['tof_status'][ix]=5
    radar=[];provenance=[];bounds=[]
    for j,(actor,obj) in enumerate(zip(actors,frame['objects'])):
        center,extent=actor.get_actor_bounds(False)
        center_m=[center.x/100,center.y/100,center.z/100]
        current_centers[obj['name']]=center_m
        velocity=relative_radial_velocity(center_m,camera_m,previous_centers.get(obj['name']),previous_camera,dt)
        bounds.append(dict(name=obj['name'],center_m=center_m,extent_m=[extent.x/100,extent.y/100,extent.z/100],
            native_relative_radial_velocity_mps=velocity))
        dx=(center.x-origin.x)/100;dy=(center.y-origin.y)/100
        az=math.degrees(math.atan2(dy,dx))-cam['yaw'];hit=trace(center)
        if abs(az)>60 or not hit or hit[1]!=actor.static_mesh_component:continue
        p,_=hit;rr=math.hypot((p.x-origin.x)/100,(p.y-origin.y)/100)
        # Keep every original noise draw, including a v=0 placeholder when Doppler is missing.
        observed=measure_radar(rr,az,0. if velocity is None else velocity,0.,rng)
        if observed:
            radar.append(observed);provenance.append(dict(kind='real_actor',actor_id=ep+'/'+obj['name'],exact_angle_deg=az,pre_noise_range_m=rr,
                pre_noise_radial_velocity_mps=velocity,doppler_available=velocity is not None,
                doppler_authority='EVALUATOR_ONLY_PAST_NATIVE_CENTER_AND_CAMERA_DISPLACEMENT'))
    ghost=frame.get('radar_ghost')
    if ghost:
        dx=ghost['z']-cam['x'];dy=ghost['x']-cam['y'];rr=math.hypot(dx,dy);az=math.degrees(math.atan2(dy,dx))-cam['yaw']
        ghost_center=[ghost['z'],ghost['x'],cam['z']]
        velocity=relative_radial_velocity(ghost_center,camera_m,ghost_center,previous_camera,dt)
        if abs(az)<=60:
            observed=measure_radar(rr,az,0. if velocity is None else velocity,0.,rng)
            if observed:radar.append(observed);provenance.append(dict(kind='persistent_ghost',actor_id=None,exact_angle_deg=az,pre_noise_range_m=rr,
                pre_noise_radial_velocity_mps=velocity,doppler_available=velocity is not None,
                doppler_authority='EVALUATOR_ONLY_STATIC_HYPOTHETICAL_GHOST_AND_PAST_CAMERA_DISPLACEMENT'))
    if rng.random()<.15:
        radar.append((round(rng.uniform(.4,5)/.05)*.05,round(rng.uniform(-60,60)/10)*10,round(rng.uniform(-1,1)/.1)*.1))
        provenance.append(dict(kind='transient',actor_id=None,doppler_available=dt is not None and dt>0,
            doppler_authority='HYPOTHETICAL_TRANSIENT_RANDOM_VELOCITY_NOT_NATIVE_MOTION'))
    slots=[None]*4
    for k,((rr,az,v),prov) in enumerate(retained_slots(radar,provenance)):
        # Numeric noisy triples retain original stable slot sort; only then expose missing Doppler.
        observed_v=v if prov['doppler_available'] else None
        row['radar_range_m'][k]=rr;row['radar_angle'][k]=az;row['radar_velocity'][k]=observed_v;row['radar_valid'][k]=True
        prov['observed_triple']=[rr,az,observed_v];slots[k]=prov
    st.update(previous_time_s=t,previous_camera_m=camera_m,previous_native_centers_m=current_centers)
    evaluation=dict(id=frame['id'],episode_id=ep,time_s=t,family=frame['family'],camera=cam,body_origin_m=frame['body_origin_m'],
        native_bounds=bounds,tof_native_ranges_m=native_tof,native_motion_dt_s=dt,authority='EVALUATOR_ONLY_ENGINE_BOUNDS_GEOMETRY_AND_PAST_DISPLACEMENT')
    provenance=dict(id=frame['id'],episode_id=ep,time_s=t,radar_slots=slots,authority='EVALUATOR_ONLY_HYPOTHETICAL_RADAR_ORIGIN')
    return row,evaluation,provenance
