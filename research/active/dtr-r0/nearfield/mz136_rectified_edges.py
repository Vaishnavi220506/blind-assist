"""One fixed TRAIN diagnostic: rectify verticals, refine signed RGB edges.

Public RGB/intrinsics/IMU and positive-ToF proposal support enter prediction.
Native geometry is read only after predictions have been saved and sealed.
No alarm, depth fit, certified enclosure, or silhouette completeness claim.
"""
import itertools
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from mz136_boundary_geometry import camera_to_body, near_cohorts, proposals

ROOT = Path(__file__).resolve().parents[4]
CAPTURE = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
OUTPUT = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/boundary-rectified-edge-v1'
METHOD = dict(search_radius_px=8, support_padding_px=8, gaussian_sigma_px=.8,
              vertical_trim_fraction=.15, minimum_rows=24, minimum_gradient=1.,
              width_bounds_px=[2,48], vertical_profile='20_PERCENT_TRIMMED_MEAN_SIGNED_SCHARR_X',
              selection='FIRST_ROOT_PROPOSAL_BY_OBSERVED_CONTRAST_SUPPORT',
              edge_selection='MAX_POSITIVE_LEFT_MINUS_NEGATIVE_RIGHT_WITH_FIXED_SEED_PENALTY',
              seed_penalty=.02, retries=0)


def sha(path):
    return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def transform(points, matrix):
    points=np.asarray(points,float)
    result=np.c_[points,np.ones(len(points))]@matrix.T
    return result[:,:2]/result[:,2,None]


def rectification(row,yaw):
    intr=row['rgb_intrinsics'];fx,fy,cx,cy=[intr[k] for k in ('fx','fy','cx','cy')]
    pixel_to_camera=np.array([[0,0,1],[1/fx,0,-cx/fx],[0,-1/fy,cy/fy]])
    body_to_pixel=np.array([[cx,fx,0],[cy,0,-fy],[1,0,0]])
    return body_to_pixel@camera_to_body(row,yaw)@pixel_to_camera


def peak_subpixel(profile,index,sign):
    if not 0<index<len(profile)-1:return float(index)
    a,b,c=sign*profile[index-1:index+2]
    denominator=a-2*b+c
    if not np.isfinite(denominator) or denominator>=-1e-12:return float(index)
    return float(index+np.clip(.5*(a-c)/denominator,-.5,.5))


def refine(row,image,yaw):
    """Observation-only result; first proposal selection is fixed without truth."""
    seeds,_=proposals(row,image,yaw)
    H=rectification(row,yaw)
    result=dict(id=row['id'],yaw_deg=yaw,homography=H.tolist(),candidate=None,
                proposal_count=len(seeds),state='NO_OBSERVABLE_PROPOSAL')
    if not seeds:return result
    seed=seeds[0];l,t,r,b=seed['box'];h,w=image.shape[:2]
    seed_corners=transform([[l,t],[r,t],[r,b],[l,b]],H)
    seed_left=float(seed_corners[[0,3],0].mean());seed_right=float(seed_corners[[1,2],0].mean())
    ymin,ymax=float(seed_corners[:,1].min()),float(seed_corners[:,1].max())
    trim=METHOD['vertical_trim_fraction']*(ymax-ymin)
    top=max(0,int(np.ceil(ymin+trim)));bottom=min(h,int(np.floor(ymax-trim)))
    support=np.zeros((h,w),np.uint8)
    wanted=set(seed['zones'])
    for cohort in near_cohorts(row):
        for item in cohort:
            if item['zone'] not in wanted:continue
            a,c,d,e=item['box']
            support[max(0,int(c)):min(h,int(np.ceil(e))),max(0,int(a)-8):min(w,int(np.ceil(d))+8)]=1
    valid=cv2.warpPerspective(support,H,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    rect=cv2.warpPerspective(gray,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    rect=cv2.GaussianBlur(rect,(0,0),METHOD['gaussian_sigma_px'])
    grad=cv2.Scharr(rect,cv2.CV_32F,1,0)/32.
    profile=np.full(w,np.nan)
    for x in range(max(1,int(np.floor(seed_left))-9),min(w-1,int(np.ceil(seed_right))+10)):
        values=grad[top:bottom,x][valid[top:bottom,x]]
        if len(values)<METHOD['minimum_rows']:continue
        values=np.sort(values);cut=len(values)//5
        profile[x]=float(values[cut:len(values)-cut].mean())
    lefts=[x for x in range(1,w-1) if abs(x-seed_left)<=8 and np.isfinite(profile[x]) and profile[x]>=1.]
    rights=[x for x in range(1,w-1) if abs(x-seed_right)<=8 and np.isfinite(profile[x]) and profile[x]<=-1.]
    pairs=[(profile[a]-profile[z]-.02*(abs(a-seed_left)+abs(z-seed_right)),a,z)
           for a in lefts for z in rights if 2<=z-a<=48]
    result.update(seed=seed,rectified_seed_edges=[seed_left,seed_right],rectified_y_range=[top,bottom],
                  profile=[None if not np.isfinite(v) else float(v) for v in profile])
    if not pairs:
        result['state']='INSUFFICIENT_SIGNED_EDGE_SUPPORT';return result
    _,left,right=max(pairs,key=lambda v:(v[0],-v[1],-v[2]))
    edges=[peak_subpixel(profile,left,1),peak_subpixel(profile,right,-1)]
    endpoints=transform([[edges[0],top],[edges[0],bottom],[edges[1],top],[edges[1],bottom]],np.linalg.inv(H))
    result.update(state='OBSERVABLE_EDGE_HYPOTHESIS',candidate=dict(
        rectified_edges_px=edges,image_segments=endpoints.tolist(),
        signed_peak_gradients=[float(profile[left]),float(profile[right])],
        authority='RGB_SILHOUETTE_HYPOTHESIS_NOT_RANGE_OR_VOLUME_ENCLOSURE'))
    return result


def run():
    assert not OUTPUT.exists()
    spec=json.loads((CAPTURE/'spec.json').read_text())
    selected={f['id'] for f in spec['frames'] if f['split']=='train' and f['family']=='shallow_boundary_stress'}
    rows=[json.loads(s) for s in (CAPTURE/'raw.jsonl').read_text().splitlines() if json.loads(s)['id'] in selected]
    assert len(rows)==len(selected)==48
    receipt=json.loads((CAPTURE/'receipt.json').read_text());assert receipt['status']=='PASS'
    assert sha(CAPTURE/'raw.jsonl')==receipt['hashes']['raw.jsonl']
    assert sha(CAPTURE/'spec.json')==receipt['spec_sha256']
    images={r['id']:CAPTURE/r['rgb_path'] for r in rows}
    for r in rows:assert sha(images[r['id']])==receipt['hashes'][r['rgb_path']]
    inputs={str(p):sha(p) for p in [CAPTURE/'raw.jsonl',CAPTURE/'spec.json',CAPTURE/'receipt.json',CAPTURE/'evaluator.jsonl',*images.values()]}
    OUTPUT.mkdir(parents=True);(OUTPUT/'source-snapshot').mkdir()
    files=[Path(__file__),Path(__file__).with_name('mz136_boundary_geometry.py'),Path(__file__).with_name('mz115_spatial_allocation.py')]
    for f in files:shutil.copyfile(f,OUTPUT/'source-snapshot'/f.name)
    write(OUTPUT/'freeze.json',dict(method=METHOD,frames=48,frame_ids=[r['id'] for r in rows],
        source_hashes={f.name:sha(f) for f in files},inputs=inputs,
        backend=dict(device='CPU',reason='GPU_BACKEND_UNAVAILABLE',opencv=cv2.__version__,numpy=np.__version__),
        scope='ONE_TRAIN_ONLY_OBSERVABLE_RECTIFICATION_DIAGNOSTIC_NO_SWEEP'))
    preds=[];yaw=0.;episode=None;start=time.perf_counter()
    for row in rows:
        if row['episode_id']!=episode:yaw=0.
        assert row['imu_valid'];yaw+=row['delta_yaw'];episode=row['episode_id']
        preds.append(refine(row,cv2.imread(str(images[row['id']])),yaw))
    write(OUTPUT/'predictions.json',preds)
    write(OUTPUT/'prediction-seal.json',dict(predictions_sha256=sha(OUTPUT/'predictions.json'),
        freeze_sha256=sha(OUTPUT/'freeze.json'),seconds=time.perf_counter()-start,
        authority='PREDICTIONS_SAVED_BEFORE_NATIVE_EVALUATOR_PARSE'))
    # Evaluator phase starts only here. No truth-dependent retry is implemented.
    native={}
    for line in (CAPTURE/'evaluator.jsonl').read_text().splitlines():
        value=json.loads(line)
        if value['id'] in selected:native[value['id']]=value
    assert set(native)==selected
    assert sha(CAPTURE/'evaluator.jsonl')==receipt['hashes']['evaluator.jsonl']
    evaluate(rows,preds,native,images)
    assert inputs=={p:sha(p) for p in inputs}


def evaluate(rows,preds,native,images):
    records=[]
    for row,pred in zip(rows,preds):
        evaluation=native[row['id']]
        targets=[o for o in evaluation['native_bounds'] if o['name']=='shape0']
        assert len(targets)==1
        target=targets[0];origin=np.asarray(evaluation['body_origin_m'])+np.asarray(row['camera_in_body_m'])
        low=np.asarray(target['center_m'])-target['extent_m']-origin
        high=np.asarray(target['center_m'])+target['extent_m']-origin
        corners=np.asarray(list(itertools.product(*zip(low,high))))
        intr=row['rgb_intrinsics']
        camera=corners@camera_to_body(row,pred['yaw_deg'])
        uv=np.stack([intr['cx']+intr['fx']*camera[:,1]/camera[:,0],
                     intr['cy']-intr['fy']*camera[:,2]/camera[:,0]],axis=1)
        hull=cv2.convexHull(uv.astype(np.float32))
        viewport=np.array([[0,0],[intr['width']-1,0],[intr['width']-1,intr['height']-1],
                           [0,intr['height']-1]],np.float32)
        area,clipped=cv2.intersectConvexConvex(hull,viewport)
        assert area>0 and clipped is not None
        rectified_visible=transform(clipped.reshape(-1,2),np.asarray(pred['homography']))
        u=rectified_visible[:,0]
        truth_edges=[float(u.min()),float(u.max())]
        inner=0 if np.mean(truth_edges)>=intr['cx'] else 1
        record=dict(id=row['id'],truth_rectified_edges_px=truth_edges,inner_index=inner,
                    available=pred['candidate'] is not None)
        if pred['candidate']:
            estimate=pred['candidate']['rectified_edges_px']
            record.update(predicted_edges_px=estimate,
                inner_error_px=float(estimate[inner]-truth_edges[inner]),
                seed_inner_error_px=float(pred['rectified_seed_edges'][inner]-truth_edges[inner]),
                edge_errors_px=[a-b for a,b in zip(estimate,truth_edges)])
        records.append(record)
    available=[r for r in records if r['available']]
    errors=np.asarray([r['inner_error_px'] for r in available])
    seeds=np.asarray([r['seed_inner_error_px'] for r in available])
    summary=dict(frames=len(rows),coverage=len(available),inner_edge_mae_px=float(np.abs(errors).mean()) if len(errors) else None,
        seed_inner_edge_mae_same_coverage_px=float(np.abs(seeds).mean()) if len(seeds) else None,
        inner_edge_median_absolute_error_px=float(np.median(np.abs(errors))) if len(errors) else None,
        inner_edge_max_absolute_error_px=float(np.abs(errors).max()) if len(errors) else None,
        inner_edge_within_1px=int((np.abs(errors)<1).sum()),
        coordinate_system='BODY_ALIGNED_RECTIFIED_PINHOLE_PIXELS_SAME_FOCAL_LENGTH',
        native_reference='PROJECT_AABB_CONVEX_HULL_CLIP_TO_OBSERVED_RGB_VIEWPORT_THEN_RECTIFY',
        limits='Native AABB projection is a geometric silhouette reference. ToF slant range does not identify front/back depth; no alarm or metric lateral extent claim. Source and cohort are consumed TRAIN only.',
        decision='REPORT_ONE_FIXED_RECTIFICATION_NO_TUNING')
    write(OUTPUT/'cases.json',records);write(OUTPUT/'summary.json',summary)
    # Deterministic representatives: first four available frames; no best-case selection.
    overlay=[]
    for record in available[:4]:
        i=next(k for k,row in enumerate(rows) if row['id']==record['id'])
        pred=preds[i];row=rows[i];image=cv2.imread(str(images[row['id']]))
        rect=cv2.warpPerspective(image,np.asarray(pred['homography']),tuple(image.shape[1::-1]))
        for x in record['truth_rectified_edges_px']:cv2.line(rect,(round(x),0),(round(x),rect.shape[0]-1),(0,255,0),1)
        for x in record['predicted_edges_px']:cv2.line(rect,(round(x),0),(round(x),rect.shape[0]-1),(0,0,255),1)
        cv2.putText(rect,row['id'],(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.putText(rect,'green=native AABB red=observed edge',(5,36),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        overlay.append(rect)
    if overlay:assert cv2.imwrite(str(OUTPUT/'representative-overlay.png'),np.concatenate(overlay,axis=0))
    write(OUTPUT/'completion.json',dict(status='PASS',summary_sha256=sha(OUTPUT/'summary.json'),
        predictions_sha256=sha(OUTPUT/'predictions.json'),resources='No persistent process; diagnostic evidence retained'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
