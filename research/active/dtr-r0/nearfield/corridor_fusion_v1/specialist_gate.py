"""Cheap observed-structure routing; frozen specialist can only veto A."""
import numpy as np


def geometry_columns(feature_names):
    names=feature_names['sensor_names']+feature_names['geometry_names']
    selected=[]
    for i,n in enumerate(names):
        if n in feature_names['geometry_names'][:10]:selected.append(i)
        elif n.startswith(('nearest.','forward_height.')) and (
            n.endswith('.count') or (n.endswith(('.q00','.q50','.q100')) and any(
                '.'+field+'.' in n for field in ('range_m','side_overlap_m','height_overlap_m',
                'width_m','mask_fraction','left_positive_edge','right_negative_edge','boundary_gradient')))):
            selected.append(i)
    return selected,[names[i] for i in selected]


def features(row,base,a_score,columns):
    """No family, depth, C score, evaluator geometry or identity input."""
    ranges=[];sigmas=[];signals=[];gaps=[];merged=0;present=0;dual=0
    for zone in row['tof_zones']:
        valid=[]
        for t in zone['targets']:
            present+=1;merged+=int(t['status']=='SIM_MERGED')
            if row['tof_packet_received'] and t['status'] in ('SIM_VALID','SIM_MERGED'):
                r=t['distance_m'];s=t['range_noise_sigma_m'];g=t['signal_strength_proxy']
                if np.isfinite([r,s,g]).all() and r>0 and s>=0 and g>=0:
                    valid.append(r);ranges.append(r);sigmas.append(s);signals.append(g*r*r)
        if len(valid)==2:dual+=1;gaps.append(abs(valid[0]-valid[1]))
    scalar=[a_score,float(row['tof_packet_received']),float(row['radar_packet_received']),
            float(row['imu_valid']),present,len(ranges),merged,dual]
    for values in (ranges,sigmas,signals,gaps):
        scalar.extend(np.quantile(values,[0,.5,1]).tolist() if values else [0,0,0])
    output=np.r_[np.asarray(base)[columns],scalar].astype(np.float32)
    assert np.isfinite(output).all()
    return output


EXTRA_NAMES=['A_score','tof_packet','radar_packet','imu_valid','present_returns',
             'usable_returns','merged_returns','dual_return_zones']+[
             f'{n}_{q}' for n in ('range','sigma','signal_r2','dual_gap') for q in ('min','median','max')]


def probability(tree,x):
    classes=list(tree.classes_)
    return tree.predict_proba(x)[:,classes.index(1)] if 1 in classes else np.zeros(len(x))


def policy(a_flags,gate_probability,threshold,c_scores,c_threshold):
    a=np.asarray(a_flags,bool);g=np.asarray(gate_probability,float);c=np.asarray(c_scores,float)
    if a.shape!=g.shape or a.shape!=c.shape:raise ValueError('Matched one-dimensional arrays required')
    invoked=a&np.isfinite(g)&(g>=threshold)
    veto=invoked&np.isfinite(c)&(c<c_threshold)
    return a&~veto,invoked,veto
