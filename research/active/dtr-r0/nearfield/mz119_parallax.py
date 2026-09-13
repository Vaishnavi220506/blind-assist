"""Conditional ToF-scaled static-scene parallax; no hidden pose or depth inputs."""
import itertools
import math
import cv2
import numpy as np
from scipy.optimize import linprog
from mz115_spatial_allocation import zone_box

MAX_FEATURES=128
MAX_QUERIES=32
MIN_ZONES=8
TOF_ERROR=.23  # working .10m within-zone span + .12m noise + .01m quantization
PIXEL_ERROR=.5
LP_OPTIONS=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)


def basis(pitch,yaw):
    p,y=map(math.radians,(pitch,yaw));cp,sp,cy,sy=math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[-sy,sp*cy,cp*cy],[cy,sp*sy,cp*sy],[0,-cp,sp]])


def ray(pixel,intr):
    q=np.array([(pixel[0]-intr['cx'])/intr['fx'],(pixel[1]-intr['cy'])/intr['fy'],1.])
    return q/np.linalg.norm(q)


def ray_error(pixel,intr):
    q=ray(pixel,intr)
    points=[ray([pixel[0]+du,pixel[1]+dv],intr) for du,dv in itertools.product((-PIXEL_ERROR,PIXEL_ERROR),repeat=2)]
    # Unit-vector derivative <=1 in normalized image coordinates; second-order pad.
    return np.max(np.abs(np.array(points)-q),axis=0)+(PIXEL_ERROR/min(intr['fx'],intr['fy']))**2*2


def rotation(ref,current,yaw_ref,yaw_current):
    R=basis(current['camera_pitch_deg'],yaw_current).T@basis(ref['camera_pitch_deg'],yaw_ref)
    dt=current['time_s']-ref['time_s']
    angle=.2*dt+3*.08*math.sqrt(dt/.25)
    variants=[basis(current['camera_pitch_deg'],yaw_current+d).T@basis(ref['camera_pitch_deg'],yaw_ref) for d in (-angle,angle)]
    error=np.max(np.abs(np.array(variants)-R),axis=0)+math.radians(angle)**2/2
    return R,error


def equation(rows,qs,Rs,es,nvars,left,right):
    """Linear outer constraints with ray uncertainty multiplied by positive depth."""
    q0,q1=qs;R=Rs;e0,e1=es;out=[];rhs=[]
    for j in range(3):
        a=np.zeros(nvars);a[j]=1;a[left]=(R@q0)[j];a[right]=-q1[j]
        for sign in (-1,1):
            v=sign*a;v[left]-=e0[j];v[right]-=e1[j];out.append(v);rhs.append(1e-8)
    rows.extend(out)
    return rhs


def solve(c,poly,bounds=None):
    return linprog(c,A_ub=poly['A'],b_ub=poly['b'],bounds=poly['bounds'] if bounds is None else bounds,method='highs',options=LP_OPTIONS)


def pose_poly(anchors,R,Rerr,intr):
    n=len(anchors);size=3+2*n;A=[];b=[];bounds=[(-.5,.5)]*3
    for k,a in enumerate(anchors):
        q0,q1=ray(a['reference_pixel'],intr),ray(a['pixel'],intr)
        e0=np.abs(R)@ray_error(a['reference_pixel'],intr)+Rerr@(np.abs(q0)+ray_error(a['reference_pixel'],intr))
        e1=ray_error(a['pixel'],intr)
        b+=equation(A,(q0,q1),R,(e0,e1),size,3+2*k,4+2*k)
        bounds.extend([(max(.02,a['range_m']-TOF_ERROR),a['range_m']+TOF_ERROR),(.02,5.)])
    poly=dict(A=np.array(A),b=np.array(b),bounds=bounds)
    if not solve(np.zeros(size),poly).success:return None
    limits=[]
    for j in range(3):
        objective=np.zeros(size);objective[j]=1;lo=solve(objective,poly);hi=solve(-objective,poly)
        if not lo.success or not hi.success:return None
        limits.append([float(lo.fun)-1e-7,-float(hi.fun)+1e-7])
    zero=solve(np.zeros(size),poly,[(0.,0.)]*3+bounds[3:])
    if zero.status not in (0,2):return None
    return poly,limits,bool(zero.success)


def depth_bounds(poly,reference_pixel,pixel,R,Rerr,intr):
    size=len(poly['bounds'])+2;A=[np.pad(a,(0,2)) for a in poly['A']];b=poly['b'].tolist()
    q0,q1=ray(reference_pixel,intr),ray(pixel,intr)
    e0=np.abs(R)@ray_error(reference_pixel,intr)+Rerr@(np.abs(q0)+ray_error(reference_pixel,intr))
    b+=equation(A,(q0,q1),R,(e0,ray_error(pixel,intr)),size,size-2,size-1)
    target=dict(A=np.array(A),b=np.array(b),bounds=poly['bounds']+[(.02,20.),(.02,20.)])
    c=np.zeros(size);c[-1]=1;lo=solve(c,target);hi=solve(-c,target)
    if not lo.success or not hi.success:return None
    return [max(.02,float(lo.fun)-1e-6),-float(hi.fun)+1e-6]


def anchors_for(ref,current,tracks0,tracks1,R):
    if not ref['tof_packet_received']:return []
    intr=ref['rgb_intrinsics'];groups={}
    for z in ref['tof_zones']:
        if len(z['targets'])!=1 or z['targets'][0]['status']!='SIM_VALID':continue
        box=zone_box(z,intr);target=z['targets'][0];choices=[]
        for k,p in tracks0.items():
            if k in tracks1 and box[0]+2<=p[0]<=box[2]-2 and box[1]+2<=p[1]<=box[3]-2:
                choices.append(((p[0]-(box[0]+box[2])/2)**2+(p[1]-(box[1]+box[3])/2)**2,k))
        if choices:
            _,k=min(choices);groups[z['zone_id']]=dict(zone_id=z['zone_id'],feature_id=k,reference_pixel=tracks0[k],pixel=tracks1[k],range_m=target['distance_m'])
    return list(groups.values())


def robust_pose(anchors,R,intr):
    if len(anchors)<MIN_ZONES:return None
    Ps=[];bs=[];points=[];rays=[]
    for a in anchors:
        q=ray(a['pixel'],intr);P=np.eye(3)-np.outer(q,q);point=R@ray(a['reference_pixel'],intr)*a['range_m']
        Ps.append(P);bs.append(-P@point);points.append(point);rays.append(q)
    def estimate(indices):return np.linalg.lstsq(np.vstack([Ps[i] for i in indices]),np.concatenate([bs[i] for i in indices]),rcond=None)[0]
    def inliers(t):
        good=[]
        for i,(q,p) in enumerate(zip(rays,points)):
            depth=float(q@(p+t));error=np.linalg.norm(Ps[i]@(p+t))*intr['fx']/max(depth,.01)
            if depth>.02 and error<=1.5:good.append(i)
        return good
    rng=np.random.default_rng(119013);trials=[list(range(len(anchors)))]+[rng.choice(len(anchors),3,replace=False).tolist() for _ in range(32)]
    best=max((inliers(estimate(indices)) for indices in trials),key=lambda v:(len(v),tuple(-i for i in v)))
    if len(best)<MIN_ZONES or len(best)<.6*len(anchors):return None
    selected=[anchors[i] for i in best]
    if len({a['zone_id']//8 for a in selected})<2 or len({a['zone_id']%8 for a in selected})<2:return None
    A=np.vstack([Ps[i] for i in best])
    if np.linalg.cond(A)>100:return None
    t=estimate(best)
    if not np.isfinite(t).all() or np.max(np.abs(t))>.5:return None
    return selected,t


def predict(rows,image_loader):
    history=[];previous_gray=None;previous_tracks={};next_id=0;yaw=0.;episode=None;output=[]
    for index,row in enumerate(rows):
        if row['episode_id']!=episode:history=[];previous_gray=None;previous_tracks={};next_id=0;yaw=0.
        episode=row['episode_id'];assert row['imu_valid'];yaw+=row['delta_yaw']
        image=image_loader(row);result=dict(poses=[],points=[],state='UNKNOWN',sources=['RGB','TOF','IMU'],no_clear_claim=True)
        if image is None:
            history=[];previous_gray=None;previous_tracks={};output.append(result);continue
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);tracks={};h,w=gray.shape
        if previous_gray is not None and previous_tracks:
            ids=list(previous_tracks);pts=np.float32([previous_tracks[k] for k in ids]).reshape(-1,1,2)
            current,ok,_=cv2.calcOpticalFlowPyrLK(previous_gray,gray,pts,None,winSize=(21,21),maxLevel=3)
            if current is not None:
                back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,previous_gray,current,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    for k,p,p0,pb,a,b in zip(ids,current[:,0],pts[:,0],back[:,0],ok[:,0],back_ok[:,0]):
                        if a and b and np.linalg.norm(pb-p0)<=.5 and 2<=p[0]<w-2 and 2<=p[1]<h-2:tracks[k]=p.tolist()
        mask=np.full(gray.shape,255,np.uint8)
        for p in tracks.values():cv2.circle(mask,tuple(np.round(p).astype(int)),7,0,-1)
        corners=cv2.goodFeaturesToTrack(gray,maxCorners=max(1,MAX_FEATURES-len(tracks)),qualityLevel=.01,minDistance=7,mask=mask,blockSize=7)
        if corners is not None:
            for p in corners[:,0]:
                if len(tracks)>=MAX_FEATURES:break
                tracks[next_id]=p.tolist();next_id+=1
        pairs=[]
        for past in history:
            dt=row['time_s']-past['row']['time_s']
            if not .49<=dt<=.751:continue
            R,Rerr=rotation(past['row'],row,past['yaw'],yaw);anchors=anchors_for(past['row'],row,past['tracks'],tracks,R)
            fit=robust_pose(anchors,R,row['rgb_intrinsics'])
            if fit is None:continue
            selected,t=fit;feasible=pose_poly(selected,R,Rerr,row['rgb_intrinsics'])
            if feasible is None:continue
            poly,bounds,zero=feasible
            result['poses'].append(dict(reference_index=past['index'],translation_m=t.tolist(),translation_bounds=bounds,
                zero_translation_feasible=zero,anchor_zones=[a['zone_id'] for a in selected],anchor_features=selected,
                authority='CONDITIONAL_STATIC_SCENE_AND_TOF_FEATURE_CORRESPONDENCE'))
            if not zero:pairs.append((past,R,Rerr,poly,{a['feature_id'] for a in selected}))
        if len(pairs)>=2:
            used_anchors=set().union(*(v[4] for v in pairs))
            eligible=[k for k in sorted(tracks) if k not in used_anchors and all(k in past['tracks'] for past,_,_,_,_ in pairs)][:MAX_QUERIES]
            for k in eligible:
                ranges=[];references=[]
                for past,R,Rerr,poly,_ in pairs:
                    bounds=depth_bounds(poly,past['tracks'][k],tracks[k],R,Rerr,row['rgb_intrinsics'])
                    if bounds is None:break
                    ranges.append(bounds);references.append(dict(index=past['index'],pixel=past['tracks'][k]))
                if len(ranges)!=len(pairs):continue
                bounds=[max(b[0] for b in ranges),min(b[1] for b in ranges)]
                if bounds[0]<=.2 or bounds[1]>=4. or not 0<bounds[1]-bounds[0]<=2.:continue
                result['points'].append(dict(feature_id=k,pixel=tracks[k],range_bounds_m=bounds,source_references=[v['index'] for v in references],
                    reference_pixels=references,evidence_age_s=0.,state='CONDITIONAL_STATIC_TRACK_RANGE_INTERVAL'))
        if result['points']:result['state']='CONDITIONAL_GEOMETRY'
        result['tracked_features']=len(tracks);output.append(result)
        history.append(dict(index=index,row=row,tracks=tracks,yaw=yaw));history=history[-3:]
        previous_gray=gray;previous_tracks=tracks
    return output
