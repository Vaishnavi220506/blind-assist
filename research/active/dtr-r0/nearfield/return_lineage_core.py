"""Evaluator-only occupancy and original-bin diagnostics, never a predictor."""
import numpy as np

F=640/(2*np.tan(np.deg2rad(50)))
YY,XX=np.mgrid[:360,:640]
AA,BB=(XX+.5-320)/F,(YY+.5-180)/F


def masks(native,case,geometry):
    p=case['camera']
    assert all(p[k]==0 for k in ('pitch','yaw','roll'))
    known=np.isfinite(native)&(native>0)
    wx,wy,wz=native+p['x'],AA*native+p['y'],p['z']-BB*native
    owned={}
    for obj in geometry['objects']:
        c,e=obj['render_bounds_center_m'],obj['render_bounds_extent_m']
        owned[obj['name']]=known&(wx>=c[0]-e[0]-.02)&(wx<=c[0]+e[0]+.02)&(wy>=c[1]-e[1]-.02)&(wy<=c[1]+e[1]+.02)&(wz>=c[2]-e[2]-.02)&(wz<=c[2]+e[2]+.02)
    assert not (owned['target']&owned['background']).any(),'Overlapping object proxies need unresolved ownership'
    corridor=known&(native>=.3)&(native<=3)&(abs(AA*native)<=.3)&(BB*native>=-.2)&(BB*native<=.9)
    obj=next(o for o in geometry['objects'] if o['name']=='target')
    assert obj['trace']['hit_expected_actor'] and obj['trace']['hit_actor_path']==obj['actor_path']
    c,e=np.array(obj['render_bounds_center_m']),np.array(obj['render_bounds_extent_m'])
    z=(c[0]-e[0]-p['x'],c[0]+e[0]-p['x']);x=(c[1]-e[1]-p['y'],c[1]+e[1]-p['y'])
    y=(p['z']-c[2]-e[2],p['z']-c[2]+e[2]);assert z[0]>0
    xs=[320+F*a/d for a in x for d in z];ys=[180+F*b/d for b in y for d in z]
    return owned['target'],owned['background'],corridor,[min(ys),min(xs),max(ys),max(xs)]


def occupancy(depth,target,background,corridor):
    known=np.isfinite(depth)&(depth>0)
    eligible=known&(depth>=.1)&(depth<8)
    return dict(pixels=depth.size,valid=int(known.sum()),invalid=int((~known).sum()),
        target=int(target.sum()),background=int(background.sum()),other=int((known&~target&~background).sum()),
        target_eligible=int((target&eligible).sum()),target_strict_near=int((target&(depth>=.3)&(depth<=3)).sum()),
        target_corridor=int((target&corridor).sum()),all_strict_near=int((known&(depth>=.3)&(depth<=3)).sum()),
        all_far_gt3=int((known&(depth>3)).sum()),
        target_fraction=float(target.sum()/depth.size),background_fraction=float(background.sum()/depth.size))


def classify(dense,sampled,projected_area,trace,winner,target_depth_max):
    if not trace['observed']:
        return 'C_'+trace['reason']
    if winner['target_count']==winner['count']:
        return 'RETAINED_TARGET'
    if winner['target_count']:
        return 'C_MIXED_WINNER'
    if sampled['target_eligible'] and winner['min_m']>target_depth_max and winner['corridor_count']==0:
        return 'B_RETURN_WINNER_FLIP_CAPABLE'
    if dense['target']==0 and sampled['target']==0 and projected_area==0 and winner['min_m']>3 and winner['corridor_count']==0:
        return 'A_ABSENT'
    if dense['target'] and not sampled['target']:
        return 'C_NATIVE_TARGET_MISSED_BY_SAMPLING'
    if sampled['target'] and not sampled['target_eligible']:
        return 'C_TARGET_OUTSIDE_ELIGIBLE_RANGE'
    if not dense['target']:
        return 'C_PROJECTED_BOUND_WITHOUT_VISIBLE_TARGET'
    return 'C_NON_TARGET_WINNER_NOT_SEPARATED_FAR'


def audit_zone(zone,box,native,target,background,corridor,projected,depth,st,sb,sc,trace):
    y0,x0,y1,x1=map(int,box)
    ylo,yhi=y0*360/192,y1*360/192;xlo,xhi=x0*640/256,x1*640/256
    yi=np.where((np.arange(360)+.5>=ylo)&(np.arange(360)+.5<yhi))[0]
    xi=np.where((np.arange(640)+.5>=xlo)&(np.arange(640)+.5<xhi))[0]
    ix=np.ix_(yi,xi);sl=np.s_[y0:y1,x0:x1]
    dense=occupancy(native[ix],target[ix],background[ix],corridor[ix])
    patch=depth[sl];tm,bm,cm=st[sl],sb[sl],sc[sl]
    sampled=occupancy(patch,tm,bm,cm)
    py0,px0,py1,px1=projected
    area=max(0,min(xhi,px1)-max(xlo,px0))*max(0,min(yhi,py1)-max(ylo,py0))
    eligible=np.isfinite(patch)&(patch>=.1)&(patch<8)
    hits=patch[eligible];t=tm[eligible];b=bm[eligible];c=cm[eligible]
    bins=np.minimum((hits/.1).astype(int),79)
    weights=1/np.maximum(hits,.3)**2
    binweights=np.bincount(bins,weights=weights,minlength=80)
    candidates=[]
    for bid in np.unique(bins):
        use=bins==bid
        candidates.append(dict(bin=int(bid),count=int(use.sum()),mean_m=float(hits[use].mean()),
            min_m=float(hits[use].min()),max_m=float(hits[use].max()),proxy_weight=float(binweights[bid]),
            target_count=int(t[use].sum()),background_count=int(b[use].sum()),other_count=int((~t[use]&~b[use]).sum()),
            corridor_count=int(c[use].sum()),target_corridor_count=int((t[use]&c[use]).sum()),
            target_proxy_weight=float(weights[use&t].sum())))
    winner=next((v for v in candidates if v['bin']==trace['winner_bin']),None)
    if trace['observed']:
        assert winner is not None and trace['winner_bin']==int(np.argmax(binweights))
        selected=bins==trace['winner_bin'];py,px=np.nonzero(eligible)
        indices=((py[selected]+y0)*256+px[selected]+x0).tolist()
        assert indices==trace['pixel_indices']
        assert weights[selected].astype(float).tolist()==trace['weights']
    else:
        assert trace['pixel_indices']==[] and trace['weights']==[] and trace['winner_bin'] is None
    tdmax=float(hits[t].max()) if t.any() else None
    category=classify(dense,sampled,area,trace,winner,tdmax)
    ranked=sorted(candidates,key=lambda q:(-q['proxy_weight'],q['bin']))
    target_ranks=[i+1 for i,q in enumerate(ranked) if q['target_count']]
    target_bins=[q for q in candidates if q['target_count']]
    separation=winner['min_m']-tdmax if winner and tdmax is not None else None
    return dict(zone=zone,native=dense,sampled=sampled,projected_aabb_intersection_pixel_area=area,
        eligible_hits=int(hits.size),candidate_bins=candidates,
        reported={k:v for k,v in trace.items() if k not in ('pixel_indices','weights')},winner=winner,
        category=category,first_target_bin_rank=min(target_ranks) if target_ranks else None,
        max_target_bin_proxy_weight=max((q['proxy_weight'] for q in target_bins),default=None),
        min_target_m=float(hits[t].min()) if t.any() else None,max_target_m=tdmax,
        winner_min_minus_target_max_m=separation,
        sensor_min_separation_600mm_satisfied=separation>=.6 if separation is not None else None)
