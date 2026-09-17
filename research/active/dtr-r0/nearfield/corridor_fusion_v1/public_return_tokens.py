"""MZ161 public return tokens, without RGB reads or dense image tensors."""
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mz143_corridor_features import _public
from mz115_spatial_allocation import slant_envelope, zone_box
from mz136_boundary_geometry import camera_to_body


def encode_tokens(row,yaw):
    """Return float32 tokens[132,21], bool valid[132], and public-only audit.

    Yaw is caller-owned causal public IMU integration. Native annotations,
    scene identifiers, file paths and RGB pixels are never consulted.
    """
    row=_public(row)
    if not math.isfinite(yaw):raise ValueError('Finite public yaw required')
    intr=row['rgb_intrinsics'];h,w=intr['height'],intr['width']
    tokens=[];present=[];outside=0
    rotation=camera_to_body(row,yaw);origin=np.asarray(row['camera_in_body_m'],float)
    packets=(float(row['imu_valid']),float(row['tof_packet_received']),float(row['radar_packet_received']))
    for z in sorted(row['tof_zones'],key=lambda z:z['zone_id']):
        box=zone_box(z,intr)
        fully_visible=0<=box[0]<=box[2]<=w-1 and 0<=box[1]<=box[3]<=h-1
        theta=math.radians(sum(z['theta_bounds_deg'])/2);phi=math.radians(sum(z['phi_bounds_deg'])/2)
        ray=np.array([1.,math.tan(theta),math.tan(phi)]);ray/=np.linalg.norm(ray)
        for slot in range(2):
            target=z['targets'][slot] if slot<len(z['targets']) else None
            usable=bool(target and row['tof_packet_received'] and target['status'] in ('SIM_VALID','SIM_MERGED'))
            values=np.array([target[k] for k in ('distance_m','range_noise_sigma_m','signal_strength_proxy')],float) if target else np.zeros(3)
            usable=usable and np.isfinite(values).all() and values[0]>0 and min(values[1:])>=0
            distance,sigma,strength=values if usable else np.zeros(3)
            merged=bool(usable and target['status']=='SIM_MERGED')
            center=rotation@(distance*ray)+origin if usable else np.zeros(3)
            support=np.zeros((3,2))
            if usable:
                bounds=(.02,4.) if merged else (max(.02,distance-3*sigma),distance+3*sigma)
                support=np.asarray(slant_envelope(box,bounds,intr,(row['camera_pitch_deg'],)*2,(yaw,)*2,0.))+origin[:,None]
            tokens.append([usable,merged,distance/4,sigma/.2,math.log1p(strength*distance**2),
                center[0]/4,center[1]/2,center[2]/3,*list(support[0]/4),*list(support[1]/2),*list(support[2]/3),
                0.,0.,1.,0.,*packets])
            present.append(bool(usable));outside+=int(usable and not fully_visible)
    for r,a,v,valid in zip(row['radar_range_m'],row['radar_angle'],row['radar_velocity'],row['radar_valid']):
        valid=bool(valid and row['radar_packet_received'] and r is not None and a is not None and math.isfinite(r) and math.isfinite(a) and r>0)
        rr=float(r) if valid else 0.;angle=math.radians(float(a)+yaw) if valid else 0.
        x,y=rr*math.cos(angle),rr*math.sin(angle)
        vv=bool(valid and v is not None and math.isfinite(v))
        tokens.append([valid,0.,rr/4,0.,0.,x/4,y/2,0.,x/4,x/4,y/2,y/2,-1.,1.,
            float(v)/3 if vv else 0.,vv,0.,1.,*packets]);present.append(valid)
    tokens=np.asarray(tokens,np.float32);present=np.asarray(present,bool)
    assert tokens.shape==(132,21) and present.shape==(132,)
    if not np.isfinite(tokens).all():raise ValueError('Nonfinite public tensor')
    return dict(tokens=tokens,valid=present,audit=dict(public_tof_slots_encoded=128,
        public_radar_slots_encoded=4,valid_tokens=int(present.sum()),
        outside_or_partial_rgb_tof_tokens=outside,all_slots_have_independent_path=True,
        no_rgb_pixels=True,no_dense_maps=True,no_evaluator_inputs=True))
