"""Observation-only finite upright slab fitting to RGB bounds and regional ToF.

Geometry is proposed first and rendered into measurements. No native/evaluator
input is accepted. Finite cuboid prior, constant reflectance and public 3x3 zone
quadrature are explicit simulation assumptions, not recovered private samples.
"""
import math
import time
import cv2
import numpy as np
import torch
from mz136_boundary_geometry import camera_to_body, near_cohorts
from mz136_rectified_edges import refine
from mz115_spatial_allocation import intersection, area, possible, certain
from mz115_zonal_tof import measure_zone
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD

METHOD=dict(model='UPRIGHT_ORIENTED_FINITE_RECTANGULAR_SLAB_WITH_CONSTANT_REFLECTANCE',
    authority='OBSERVATION_ONLY_PUBLIC_ZONE_QUADRATURE_NOT_PRIVATE_RAYS',
    parameters=['center_x','center_y','center_z','log_half_depth','log_half_width','log_half_height','yaw_rad','log_reflectance'],
    population=192,iterations=64,elite=24,seed=139015,max_proposal_fits=2,
    range_sigma_floor_m=.04,pixel_sigma=2.5,log_signal_sigma=.35,signal_weight=.25,
    plausible_loss_delta=1.,max_loss=4.,max_bbox_error_px=5.,max_range_zscore=3.,
    minimum_zones=2,minimum_explained_zones=2,ray_grid=3,
    unresolved='INCUMBENT_FALLBACK_IF_PLAUSIBLE_SURFACES_DISAGREE_ON_CORRIDOR',
    replacement='EXACT_FORWARD_EXPLAINED_SINGLE_VALID_FULL_RGB_ZONE_ONLY',
    missing='NO_NEGATIVE_EVIDENCE_FROM_MISSING_RETURNS',
    fit_reduction='SINGLE_COMPONENT_INVERSE_SQUARE_EXPECTATION_SURROGATE',
    acceptance_reduction='FROZEN_HISTOGRAM_PEAK_COMPONENT_REDUCTION_WITH_ZERO_ADDED_NOISE',
    temporal='CURRENT_FRAME_ONLY_PUBLIC_INTEGRATED_IMU_YAW',sweeps=0)


class NoNoise:
    def gauss(self,*_):return 0.


def public_rays(row,zone_ids,yaw):
    zones={z['zone_id']:z for z in row['tof_zones']};rays=[]
    for key in zone_ids:
        z=zones[key];a,b=z['theta_bounds_deg'];c,d=z['phi_bounds_deg']
        zone=[]
        for iy in range(3):
            for ix in range(3):
                theta=math.radians(a+(ix+.5)*(b-a)/3)
                phi=math.radians(d-(iy+.5)*(d-c)/3)
                ray=np.array([1.,math.tan(theta),math.tan(phi)])
                zone.append(camera_to_body(row,yaw)@(ray/np.linalg.norm(ray)))
        rays.append(zone)
    return np.array(rays)


def render_population(parameters,rays,origin,camera_rotation,intr):
    """Batched hypothesized first hits, means/strength, and RGB silhouette box."""
    center=parameters[:,:3];half=parameters[:,3:6].exp();angle=parameters[:,6]
    rho=parameters[:,7].exp();co,si=angle.cos(),angle.sin()
    relative=origin[None]-center
    local_origin=torch.stack([co*relative[:,0]+si*relative[:,1],-si*relative[:,0]+co*relative[:,1],relative[:,2]],-1)
    direction=torch.stack([co[:,None,None]*rays[None,:,:,0]+si[:,None,None]*rays[None,:,:,1],
        -si[:,None,None]*rays[None,:,:,0]+co[:,None,None]*rays[None,:,:,1],rays[None,:,:,2].expand(len(parameters),-1,-1)],-1)
    safe=torch.where(direction.abs()<1e-8,torch.full_like(direction,1e-8),direction)
    near=(-half[:,None,None]-local_origin[:,None,None])/safe
    far=(half[:,None,None]-local_origin[:,None,None])/safe
    enter=torch.minimum(near,far).amax(-1);leave=torch.maximum(near,far).amin(-1)
    ranges=torch.where(enter>0,enter,leave)
    valid=(leave>=enter)&(leave>0)&(ranges>0)&(ranges<=4.)
    ranges=torch.where(valid,ranges,torch.zeros_like(ranges))
    weights=valid*rho[:,None,None]/(9*ranges.clamp_min(.2).square())
    strength=weights.sum(-1);mean=(weights*ranges).sum(-1)/strength.clamp_min(1e-12)
    signs=parameters.new_tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    xyz=half[:,None]*signs[None]
    corners=torch.stack([co[:,None]*xyz[:,:,0]-si[:,None]*xyz[:,:,1],si[:,None]*xyz[:,:,0]+co[:,None]*xyz[:,:,1],xyz[:,:,2]],-1)+center[:,None]
    camera=(corners-origin)@camera_rotation
    u=intr['cx']+intr['fx']*camera[:,:,1]/camera[:,:,0].clamp_min(.02)
    v=intr['cy']-intr['fy']*camera[:,:,2]/camera[:,:,0].clamp_min(.02)
    box=torch.stack([u.amin(1),v.amin(1),u.amax(1),v.amax(1)],-1)
    box=torch.maximum(torch.minimum(box,parameters.new_tensor([intr['width']-1,intr['height']-1]*2)),torch.zeros_like(box))
    return dict(ranges=ranges,valid=valid,mean=mean,strength=strength,box=box,corners=corners,
                valid_camera=(camera[:,:,0]>.05).all(1))


def loss_population(parameters,context):
    pred=render_population(parameters,context['rays'],context['origin'],context['rotation'],context['intr'])
    zr=(pred['mean']-context['ranges'])/context['sigma']
    zs=(torch.log(pred['strength'].clamp_min(1e-10))-context['strength'].log())/METHOD['log_signal_sigma']
    zi=(pred['box']-context['box'])/METHOD['pixel_sigma']
    loss=zr.square().clamp_max(100).mean(1)+METHOD['signal_weight']*zs.square().clamp_max(100).mean(1)+zi.square().mean(1)
    loss+=100*(pred['strength']<.01).float().mean(1)+100*(~pred['valid_camera'])
    return loss,pred


def decode(parameter):
    return dict(center=parameter[:3].tolist(),half_extent=np.exp(parameter[3:6]).tolist(),
                yaw_rad=float(parameter[6]),reflectance=float(np.exp(parameter[7])))


def surface_intersects(surface):
    """Exact upright OBB versus walking-corridor AABB separating-axis test."""
    c=np.array(surface['center']);h=np.array(surface['half_extent']);a=surface['yaw_rad']
    if c[2]+h[2]<.4 or c[2]-h[2]>2.05:return False
    co,si=math.cos(a),math.sin(a);axes=[np.array([1.,0]),np.array([0.,1]),np.array([co,si]),np.array([-si,co])]
    delta=c[:2]-np.array([1.9,0.]);basis=np.array([[co,si],[-si,co]])
    return all(abs(delta@axis)<=np.abs(axis)@np.array([1.7,.3])+sum(h[i]*abs(basis[i]@axis) for i in range(2)) for axis in axes)


def surface_certain(surface):
    c=np.array(surface['center']);h=np.array(surface['half_extent']);a=surface['yaw_rad']
    ext=np.array([abs(math.cos(a))*h[0]+abs(math.sin(a))*h[1],abs(math.sin(a))*h[0]+abs(math.cos(a))*h[1],h[2]])
    return certain(np.stack([c-ext,c+ext],-1))


def point_in_surface(point,surface,tolerance=.04):
    d=np.array(point)-surface['center'];a=surface['yaw_rad']
    local=np.array([math.cos(a)*d[0]+math.sin(a)*d[1],-math.sin(a)*d[0]+math.cos(a)*d[1],d[2]])
    return bool(np.all(np.abs(local)<=np.array(surface['half_extent'])+tolerance))


def contexts(row,image,cached,yaw,device):
    if not row['tof_packet_received'] or not row['imu_valid']:return [],'MISSING_TOF_OR_IMU'
    edge=refine(row,image,yaw)
    boxes=[list(map(float,b)) for b in cached.get('recovered_boxes',cached.get('proposals',[]))[:6]]
    if edge.get('seed'):
        b=list(map(float,edge['seed']['box']))
        if edge['candidate']:
            segments=np.array(edge['candidate']['image_segments']);b[0]=float(segments[:2,0].mean());b[2]=float(segments[2:,0].mean())
        boxes.append(b)
    pairs=[];targets={z['zone_id']:z for z in row['tof_zones']}
    for cohort in near_cohorts(row):
        if len({v['zone'] for v in cohort})!=len(cohort):continue
        for box in boxes:
            chosen=[v for v in cohort if intersection(v['box'],box) and len(targets[v['zone']]['targets'])==1]
            if len(chosen)<2:continue
            signal=sum(targets[v['zone']]['targets'][v['slot']]['signal_strength_proxy'] for v in chosen)
            pairs.append((len(chosen)/len(cohort),signal,box,chosen))
    pairs.sort(key=lambda p:(-p[0],-p[1],p[2]))
    selected=[]
    for _,_,box,items in pairs:
        if any(area(intersection(box,b))/max(1.,area(box)+area(b)-area(intersection(box,b)))>.85 for b,_ in selected):continue
        selected.append((box,items))
        if len(selected)>=METHOD['max_proposal_fits']:break
    result=[]
    for box,items in selected:
        observations=[targets[v['zone']]['targets'][v['slot']] for v in items]
        tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
        result.append(dict(box=tensor(box),box_list=box,items=items,observations=observations,
            rays=tensor(public_rays(row,[v['zone'] for v in items],yaw)),origin=tensor(row['camera_in_body_m']),
            rotation=tensor(camera_to_body(row,yaw)),intr=row['rgb_intrinsics'],
            ranges=tensor([o['distance_m'] for o in observations]),
            sigma=tensor([max(METHOD['range_sigma_floor_m'],o['range_noise_sigma_m']) for o in observations]),
            strength=tensor([o['signal_strength_proxy'] for o in observations])))
    return result,edge['state']


def initialize(context):
    intr=context['intr'];l,t,r,b=context['box_list'];d=float(context['ranges'].median().cpu())
    ray=np.array([1.,((l+r)/2-intr['cx'])/intr['fx'],(intr['cy']-(t+b)/2)/intr['fy']]);ray/=np.linalg.norm(ray)
    center=context['origin'].cpu().numpy()+context['rotation'].cpu().numpy()@(ray*d)
    width=max(.03,(r-l)*d/intr['fx']);height=max(.03,(b-t)*d/intr['fy'])
    center[0]+=.075
    low=np.r_[np.maximum([.2,-3.,-.5],center-[.5,.35*d,.35*d]),np.log([.015,max(.008,width*.2),max(.008,height*.2)]),-math.pi/3,math.log(.03)]
    high=np.r_[np.minimum([4.5,3.,4.],center+[.5,.35*d,.35*d]),np.log([.6,max(.025,width*1.3),max(.025,height*1.3)]),math.pi/3,0.]
    initial=np.r_[center,np.log([.075,width/2,height/2]),0.,math.log(.5)]
    return low,high,np.clip(initial,low,high)


def optimize(context):
    low,high,initial=initialize(context);rng=np.random.default_rng(METHOD['seed'])
    device=context['rays'].device;lo=torch.tensor(low,dtype=torch.float32,device=device);span=torch.tensor(high-low,dtype=torch.float32,device=device)
    mean=(initial-low)/(high-low);std=np.ones(8)*.22;bank=None
    for iteration in range(METHOD['iterations']):
        samples=np.clip(rng.normal(mean,std,(METHOD['population'],8)),0,1)
        samples[:METHOD['population']//8]=rng.uniform(0,1,(METHOD['population']//8,8))
        if bank is not None:samples[-len(bank):]=bank
        else:samples[-1]=mean
        params=lo+torch.as_tensor(samples,dtype=torch.float32,device=device)*span
        with torch.no_grad():loss,_=loss_population(params,context)
        order=loss.cpu().numpy().argsort();bank=samples[order[:METHOD['elite']]]
        mean=bank.mean(0);std=np.maximum(bank.std(0)*1.2,np.array([.004,.004,.004,.009,.009,.009,.006,.01]))
    best=low+bank[0]*(high-low)
    # Fixed local alternatives test metric depth, edge-scale extent and heading
    # ambiguity. This is not a calibrated/global confidence region.
    scales=[.08,.02,.03,.2,.1,.1,.10,.15];probe=[best]
    for axis,scale in enumerate(scales):
        for sign in (-1,1):
            p=best.copy();p[axis]+=sign*scale;probe.append(np.clip(p,low,high))
    candidates=np.r_[low+bank*(high-low),probe]
    with torch.no_grad():loss,pred=loss_population(torch.tensor(candidates,dtype=torch.float32,device=device),context)
    losses=loss.cpu().numpy();order=losses.argsort();best_index=int(order[0]);best_loss=float(losses[best_index])
    plausible=[int(i) for i in order if losses[i]<=best_loss+METHOD['plausible_loss_delta']]
    surfaces=[decode(candidates[i]) for i in plausible]
    result=dict(loss=best_loss,bbox_max_error_px=float((pred['box'][best_index]-context['box']).abs().max().cpu()),
        range_max_zscore=float(((pred['mean'][best_index]-context['ranges'])/context['sigma']).abs().max().cpu()),
        surface_candidate=surface_intersects(surfaces[0]),surfaces=surfaces,
        ambiguity_candidate_flags=[surface_intersects(s) for s in surfaces],
        parameters=candidates[best_index].tolist(),observed_box=context['box_list'],
        predicted_box=pred['box'][best_index].cpu().tolist(),plausible_count=len(plausible),
        predicted_ranges=pred['mean'][best_index].cpu().tolist(),observed_ranges=context['ranges'].cpu().tolist(),
        limits='Finite CEM population and fixed perturbations do not certify absence of other solutions')
    ranges=pred['ranges'][best_index].cpu().numpy();valid=pred['valid'][best_index].cpu().numpy()
    rho=surfaces[0]['reflectance'];explained=[];exact=[]
    for item,obs,rr,vv in zip(context['items'],context['observations'],ranges,valid):
        generated=[dict(range_m=float(r) if ok else None,reflectance_proxy=rho) for r,ok in zip(rr,vv)]
        predicted,diagnostic=measure_zone(generated,NoNoise())
        matched=False
        if len(predicted)==1 and predicted[0]['status']==obs['status']=='SIM_VALID':
            component=next(c for c in diagnostic['components'] if c['detected'])
            matched=(abs(component['pre_noise_range_m']-obs['distance_m'])<=3*obs['range_noise_sigma_m']+.01 and
                     abs(math.log(predicted[0]['signal_strength_proxy']/obs['signal_strength_proxy']))<=.7)
        exact.append(dict(zone=item['zone'],slot=item['slot'],predicted=predicted,matched=bool(matched)))
        if matched:explained.append([item['zone'],item['slot']])
    result.update(exact_forward=exact,explained_keys=explained)
    return result


def fit_frame(row,image,cached_corrected,incumbent,radar,device='cuda'):
    started=time.perf_counter();before=repr(row)
    result=dict(id=row['id'],candidate=incumbent['candidate'],candidate_state='ALERT' if incumbent['candidate'] else 'UNKNOWN',
        tof_candidate=bool(incumbent['score']>=1 or incumbent['certain_coarse']),surface_candidate=None,
        replacement_keys=[],surfaces=[],state='INCUMBENT_FALLBACK',fit=dict(device=str(device)))
    choices,edge_state=contexts(row,image,cached_corrected,cached_corrected['integrated_yaw_deg'],device)
    fits=[optimize(c) for c in choices]
    result['fit'].update(edge_state=edge_state,proposal_fits=fits)
    if fits:
        order=sorted(range(len(fits)),key=lambda i:fits[i]['loss']);chosen=fits[order[0]];context=choices[order[0]]
        surfaces=[s for f in fits if f['loss']<=chosen['loss']+METHOD['plausible_loss_delta'] for s in f['surfaces']]
        flags=[surface_intersects(s) for s in surfaces]
        result['fit'].update(best_surface_candidate=chosen['surface_candidate'],ambiguity_candidate_flags=flags,
            loss=chosen['loss'],range_max_zscore=chosen['range_max_zscore'],bbox_max_error_px=chosen['bbox_max_error_px'])
        intr=row['rgb_intrinsics'];eligible=[];explained={tuple(k) for k in chosen['explained_keys']}
        for item in context['items']:
            l,t,r,b=item['box'];key=[item['zone'],item['slot']]
            if tuple(key) in explained and 0<=l<r<=intr['width']-1 and 0<=t<b<=intr['height']-1:eligible.append(key)
        if chosen['loss']>METHOD['max_loss'] or chosen['bbox_max_error_px']>METHOD['max_bbox_error_px'] or chosen['range_max_zscore']>METHOD['max_range_zscore']:
            result['state']='MEASUREMENT_OR_RGB_MISFIT_FALLBACK'
        elif len(set(flags))!=1:result['state']='SURFACE_CORRIDOR_AMBIGUITY_FALLBACK'
        elif len(eligible)<METHOD['minimum_explained_zones']:result['state']='INSUFFICIENT_EXPLAINED_VISIBLE_RETURNS_FALLBACK'
        else:
            selected={tuple(k) for k in eligible};weights=column_weights(row,FOUR);active=set();sure=False
            surface_flag=any(flags);surface_sure=all(surface_certain(s) for s in surfaces)
            for e in cached_corrected['spatial_evidence']:
                replaced=(e['zone_id'],e['target_slot']) in selected
                if surface_flag if replaced else possible(e['localized_xyz']):active.add(e['zone_id'])
                sure |= surface_sure if replaced else certain(e['coarse_xyz'])
            tof=bool(sum(weights[z] for z in active)>=THRESHOLD or sure)
            flag=bool(tof or radar['candidate'])
            result.update(state='OBSERVABLE_SURFACE_ACCEPTED',candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',
                tof_candidate=tof,surface_candidate=surface_flag,surfaces=surfaces,replacement_keys=eligible)
    assert repr(row)==before
    result['fit']['seconds']=time.perf_counter()-started
    return result
