"""Observation-only angular adapter for the fixed DVSR transfer check."""
import math

import cv2
import numpy as np

SIZE = 128
DEPTH_SCALE_M = 4.0
METHOD = dict(
    input='PUBLIC_STRONGEST_VALID_ZONE_CENTER_AXIS_DEPTH_HYPOTHESIS',
    rgb='CALIBRATED_ANGULAR_45_DEG_128_SQUARE_RGB_0_1',
    depth_scale_m=DEPTH_SCALE_M,
    missing='ZERO_INPUT_NOT_FREE_SPACE',
    dense='HYPOTHESIS_ONLY_NO_NATIVE_SENSOR_VETO',
    temporal='PAST_AND_CURRENT_EPISODE_PREFIX_MAX6_LAST_OUTPUT',
    comparator='REPEAT_CURRENT_SAME_LENGTH',
    first_frame='DUPLICATE_CURRENT_FOR_TWO_FRAME_FLOW_NO_NEW_INFORMATION',
)


def angular_grid(row):
    zones = row['tof_zones']
    assert len(zones) == 64 and {z['zone_id'] for z in zones} == set(range(64))
    theta_edges = sorted({float(v) for z in zones for v in z['theta_bounds_deg']})
    phi_edges = sorted({float(v) for z in zones for v in z['phi_bounds_deg']})
    assert len(theta_edges) == len(phi_edges) == 9
    assert np.allclose(np.diff(theta_edges), 5.625) and np.allclose(np.diff(phi_edges), 5.625)
    a, b = theta_edges[0], theta_edges[-1]
    c, d = phi_edges[0], phi_edges[-1]
    theta = a + (np.arange(SIZE) + .5) * (b-a) / SIZE
    phi = d - (np.arange(SIZE) + .5) * (d-c) / SIZE
    intr = row['rgb_intrinsics']
    u = intr['cx'] + intr['fx'] * np.tan(np.deg2rad(theta))
    v = intr['cy'] - intr['fy'] * np.tan(np.deg2rad(phi))
    xx, yy = np.meshgrid(u, v)
    visible = (xx >= 0) & (xx <= intr['width']-1) & (yy >= 0) & (yy <= intr['height']-1)
    return xx.astype(np.float32), yy.astype(np.float32), visible, (a,b,c,d)


def make_input(row, bgr):
    """No native identities, masks, geometry, or future frames are accepted."""
    intr = row['rgb_intrinsics']
    assert bgr.shape == (intr['height'], intr['width'], 3)
    xx, yy, visible, bounds = angular_grid(row)
    rgb = cv2.remap(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), xx, yy,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb[~visible] = 0
    depth = np.zeros((8,8), np.float32)
    used, omitted = [], []
    a,b,c,d = bounds
    occupied = set()
    for z in row['tof_zones']:
        theta, phi = sum(z['theta_bounds_deg'])/2, sum(z['phi_bounds_deg'])/2
        col = int(round((theta-a)/(b-a)*8-.5))
        line = int(round((d-phi)/(d-c)*8-.5))
        assert 0 <= line < 8 and 0 <= col < 8 and (line,col) not in occupied
        occupied.add((line,col))
        valid = [(i,t) for i,t in enumerate(z['targets']) if t['status']=='SIM_VALID'
                 and math.isfinite(float(t['distance_m'])) and float(t['distance_m'])>0
                 and math.isfinite(float(t['signal_strength_proxy']))]
        if not row['tof_packet_received'] or not valid:
            omitted.append(dict(zone=z['zone_id'], reason='PACKET_OR_VALID_TARGET_UNAVAILABLE',
                                statuses=[t['status'] for t in z['targets']]))
            continue
        slot,t = max(valid, key=lambda p:(float(p[1]['signal_strength_proxy']),-p[0]))
        axis = float(t['distance_m']) / math.sqrt(1+math.tan(math.radians(theta))**2+math.tan(math.radians(phi))**2)
        depth[line,col] = min(axis/DEPTH_SCALE_M, 1.)
        used.append(dict(zone=z['zone_id'], slot=slot, cell=[line,col], axis_depth_m=axis,
                         clipped=axis>DEPTH_SCALE_M, alternate_targets_retained=len(z['targets'])-1))
    return rgb.transpose(2,0,1).astype(np.float32)/255., depth[None], dict(
        id=row['id'], used=used, omitted=omitted, missing_cells=64-len(used),
        source_rgb_pixels=int(intr['width']*intr['height']), angular_rgb_coverage=float(visible.mean()),
        angular_bounds_deg=list(bounds), source_rgb_shape=list(bgr.shape), method=METHOD)


def full_rgb_depth(row, raw_normalized):
    """Restore original RGB rays, with UNKNOWN outside shared angular coverage."""
    raw = np.asarray(raw_normalized, np.float32)
    assert raw.shape == (SIZE,SIZE)
    intr = row['rgb_intrinsics']
    _,_,_,(a,b,c,d) = angular_grid(row)
    yy,xx = np.mgrid[:intr['height'],:intr['width']]
    theta = np.rad2deg(np.arctan((xx-intr['cx'])/intr['fx']))
    phi = np.rad2deg(np.arctan((intr['cy']-yy)/intr['fy']))
    x = ((theta-a)/(b-a)*SIZE-.5).astype(np.float32)
    y = ((d-phi)/(d-c)*SIZE-.5).astype(np.float32)
    shared = (x>=0)&(x<=SIZE-1)&(y>=0)&(y<=SIZE-1)
    valid_raw = np.isfinite(raw) & (raw>0)
    values = np.where(valid_raw,raw*DEPTH_SCALE_M,0.)
    result = cv2.remap(values,x,y,cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=0.)
    # Use the same interpolation weights for validity. A zero-weight neighbor
    # is irrelevant; any positive-weight UNKNOWN contributor invalidates output.
    valid_weight = cv2.remap(valid_raw.astype(np.float32),x,y,cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0.)
    result[~shared | (valid_weight<1.-1e-6) | ~np.isfinite(result) | (result<=0)] = np.nan
    return result, shared


def prefix_indices(rows, index):
    """Only contiguous past observations within the current episode are used."""
    found=[index]
    while len(found)<6 and found[0]>0:
        k=found[0]
        if rows[k-1]['episode_id']!=rows[index]['episode_id'] or not math.isclose(
                rows[k]['time_s']-rows[k-1]['time_s'],.25,abs_tol=1e-9):
            break
        found.insert(0,k-1)
    return found if len(found)>1 else [index,index]
