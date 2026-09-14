"""Fresh finite-footprint analytic cube simulation; no truth in the readout.

Native legacy ray parity admits this geometric proxy. The paired dense lattice
is regrouped into 8 or 32 zones per axis, never interpolated from old packets.
"""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend
from mz115_zonal_tof import peak_bins, measure_zone
from mz113_dynamic_sensors import basis
from mz124_measurement_geometry import zone_box, slant_box, intersects, metrics
from mz115_allocation_audit import native_truth

SOURCE = ROOT/'artifacts.local/work/mz123-frozen-early-20260913/returned-v1'
MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
BRIEF = Path(__file__).with_name('MZ133_ANGULAR_BRIEF_20260914.md')
ARMS = {'legacy8': (8, 3, 1., .04), 'dense8': (8, 12, 1., .04),
        'fine32': (32, 3, 1., .04), 'budget32': (32, 3, 1/16, .16)}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p): return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]


def geometry(grid, sub):
    zones, rays = [], []
    size = 45/grid
    for iy in range(grid):
        for ix in range(grid):
            a, d = -22.5+ix*size, 22.5-iy*size
            zones.append(dict(zone_id=iy*grid+ix, theta_bounds_deg=[a,a+size], phi_bounds_deg=[d-size,d]))
            rays.append([(a+(sx+.5)*size/sub, d-(sy+.5)*size/sub)
                         for sy in range(sub) for sx in range(sub)])
    angles = np.radians(np.asarray(rays))
    return zones, np.stack([np.ones(angles.shape[:2]), np.tan(angles[:,:,0]), np.tan(angles[:,:,1])], axis=-1)


def trace(frame, spec, local_rays):
    """Generator only: finite first-hit ray/AABB slabs, including environment."""
    objects = list(frame['objects']) + [spec[k] for k in ('background','floor') if spec.get(k)]
    centers = np.asarray([o['center_m'] for o in objects])
    half = np.asarray([o['size_m'] for o in objects])/2
    origin = np.array([frame['camera'][k] for k in ('x','y','z')])
    direction = local_rays @ np.asarray(basis(frame['camera']))
    direction /= np.linalg.norm(direction,axis=-1,keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (centers-half-origin)[None,None,:,:] / direction[:,:,None,:]
        t1 = (centers+half-origin)[None,None,:,:] / direction[:,:,None,:]
    # Parallel slabs: explicit containment avoids 0/0 ambiguity.
    parallel = np.abs(direction[:,:,None,:]) < 1e-15
    inside = ((origin >= centers-half) & (origin <= centers+half))[None,None,:,:]
    lower = np.where(parallel, np.where(inside,-np.inf,np.inf), np.minimum(t0,t1)).max(axis=-1)
    upper = np.where(parallel, np.where(inside,np.inf,-np.inf), np.maximum(t0,t1)).min(axis=-1)
    entry = np.maximum(lower, 0.)
    valid = (upper >= entry) & (entry > 0.) & (entry <= 4.)
    ranges = np.where(valid,entry,np.inf)
    owners = ranges.argmin(axis=-1)
    distance = np.take_along_axis(ranges,owners[:,:,None],axis=-1)[:,:,0]
    rho = np.asarray([o.get('tof_reflectance_proxy',1.) for o in objects])[owners]
    return distance, rho, owners, direction, origin


def reduce_zone(distances, reflectances, rng, packet, signal_scale, sigma):
    """MZ115 histogram model generalized only for integration size and stress."""
    n = len(distances); histogram = [0.]*81; contributions = []
    for i, (distance, reflectance) in enumerate(zip(distances, reflectances)):
        if not math.isfinite(distance): continue
        weight = float(reflectance)*signal_scale/(n*max(float(distance),.2)**2)
        if weight <= 0: continue
        b = min(80, int(math.floor(float(distance)/.05+.5)))
        histogram[b] += weight
        contributions.append((i,float(distance),weight))
    smooth = [.25*(histogram[i-1] if i else 0.)+.5*histogram[i]+
              .25*(histogram[i+1] if i<80 else 0.) for i in range(81)]
    peaks = peak_bins(smooth); clusters = []
    for p in peaks:
        if clusters and (p-clusters[-1][-1])*.05 < .60-1e-12: clusters[-1].append(p)
        else: clusters.append([p])
    membership = [[] for _ in clusters]
    for c in contributions:
        p = min(peaks,key=lambda p:(abs(c[1]-p*.05),p))
        membership[next(i for i,cluster in enumerate(clusters) if p in cluster)].append(c)
    components = []
    for cluster,hits in zip(clusters,membership):
        if not hits: continue
        strength = sum(h[2] for h in hits)
        mean = sum(h[1]*h[2] for h in hits)/strength
        merged = len(cluster)>1 or max(h[1] for h in hits)-min(h[1] for h in hits)>=.10-1e-12
        components.append(dict(strength=strength,mean=mean,indices=[h[0] for h in hits],
                               status='SIM_MERGED' if merged else 'SIM_VALID'))
    eligible = sorted([c for c in components if c['strength']>=.01],key=lambda c:(-c['strength'],c['mean']))
    selected = eligible[:2] if packet else []
    targets, lineage = [], []
    for c in selected:
        noisy = c['mean']+rng.gauss(0.,sigma)
        targets.append(dict(distance_m=max(.02,round(noisy/.02)*.02),range_noise_sigma_m=sigma,
                            signal_strength_proxy=c['strength'],status=c['status']))
        lineage.append(c['indices'])
    return targets, lineage, len(components)-len(eligible)


def predict(row, yaw, radar):
    """Public observations only; no scene parameters, labels or private rays."""
    p = row['camera_pitch_deg']; dy = .5+.2*row['time_s']; intr = row['rgb_intrinsics']
    details = []; score = 0; certain = False
    for z in row['tof_zones']:
        active = False
        weighted = z['theta_bounds_deg'][0]>=-11.25 and z['theta_bounds_deg'][1]<=11.25
        box = zone_box(z,intr)
        for slot,t in enumerate(z['targets']):
            bounds = [.02,4.] if t['status']=='SIM_MERGED' else [max(.02,t['distance_m']-3*t['range_noise_sigma_m']),t['distance_m']+3*t['range_noise_sigma_m']]
            xyz = slant_box(box,bounds,intr,(p-.5,p+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
            hit = intersects(xyz)
            sure = all(lo>=a and hi<=b for (lo,hi),(a,b) in zip(xyz,((.2,3.6),(-.3,.3),(.4,2.05))))
            active |= hit; certain |= sure
            details.append(dict(zone_id=z['zone_id'],slot=slot,status=t['status'],xyz=xyz,
                                possible=bool(hit),trigger=bool((hit and weighted) or sure)))
        score += int(active and weighted)
    tof = bool(score>=1 or certain)
    return dict(id=row['id'],tof=tof,full=bool(tof or radar),score=score,certain=bool(certain),
                radar=bool(radar),candidate_state='ALERT' if tof or radar else 'UNKNOWN',returns=details)


def admission(spec, rows, evaluations):
    _, rays = geometry(8,3); count = hits = 0; max_error = max_signal = 0.; states = {}; wrong=[]
    for frame,row,e in zip(spec['frames'],rows,evaluations):
        ds,rs,_,_,_ = trace(frame,spec,rays)
        rng = states.setdefault(frame['episode'],random.Random(frame.get('tof_sensor_seed',frame['sensor_seed']+1150003)))
        for z,native,old in zip(range(64),e['zonal_tof_native'],row['tof_zones']):
            for observed, regenerated in zip(native['private_rays'],ds[z]):
                count += 1; present = observed.get('range_m') is not None
                if present != bool(math.isfinite(regenerated)): wrong.append([row['id'],z,'HIT'])
                if present and math.isfinite(regenerated):
                    hits += 1; err = abs(observed['range_m']-float(regenerated)); max_error=max(max_error,err)
                    if err>1e-5: wrong.append([row['id'],z,'RANGE',err])
            targets,_,_ = reduce_zone(ds[z],rs[z],rng,row['tof_packet_received'],1.,.04)
            if len(targets)!=len(old['targets']): wrong.append([row['id'],z,'COUNT']); continue
            for a,b in zip(targets,old['targets']):
                diff=abs(a['signal_strength_proxy']-b['signal_strength_proxy']); max_signal=max(max_signal,diff)
                if diff>1e-7 or any(a[k]!=b[k] for k in ('status','distance_m','range_noise_sigma_m')):
                    wrong.append([row['id'],z,'PUBLIC_TARGET',a,b])
    return dict(status='PASS' if not wrong else 'FAIL',rays=count,hits=hits,max_range_error_m=max_error,
                max_signal_error=max_signal,failures=wrong,authority='ANALYTIC_REGENERATION_VS_SAVED_NATIVE_UE')


def run(out):
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    raw_path=SOURCE/'capture-v1/raw.jsonl'; spec_path=SOURCE/'capture-v1/spec.json'
    eval_path=SOURCE/'capture-v1/evaluator.jsonl'
    spec=read(spec_path); rows=jsonl(raw_path); evaluations=jsonl(eval_path)
    receipt=read(SOURCE/'capture-v1/receipt.json')
    assert receipt['status']=='PASS' and sha(spec_path)==receipt['spec_sha256']
    assert sha(raw_path)==receipt['hashes']['raw.jsonl'] and sha(eval_path)==receipt['hashes']['evaluator.jsonl']
    cached=read(ROOT/'artifacts.local/work/mz125-observable-correction-20260913/correction-v1r1/predictions.json')
    rp=MZ129/'radar-v1/predictions.json'; rb=read(rp); baseline_path=MZ129/'replay-v1r2/predictions.json'
    baseline=read(baseline_path)['radar']
    inputs=[raw_path,spec_path,eval_path,rp,baseline_path,BRIEF,Path(__file__)]
    for directory in (MZ129/'radar-v1',MZ129/'replay-v1r2'):
        seal=read(directory/'prediction-seal.json'); assert sha(directory/'predictions.json')==seal['predictions_sha256']
        for key in ('input_hashes','dependencies','source_hashes'):
            for path,digest in seal.get(key,{}).items(): assert sha(ROOT/path)==digest,path
        inputs.append(directory/'prediction-seal.json')
    cp=ROOT/'artifacts.local/work/mz125-observable-correction-20260913/correction-v1r1'
    cs=read(cp/'prediction-seal.json'); assert sha(cp/'predictions.json')==cs['predictions_sha256'] and sha(raw_path)==cs['raw_sha256']
    inputs += [cp/'predictions.json',cp/'prediction-seal.json']
    inputs += [Path(__file__).with_name(n) for n in ('mz115_zonal_tof.py','mz113_dynamic_sensors.py','mz124_measurement_geometry.py','mz115_allocation_audit.py','mz109_interval_extent.py')]
    assert len(rows)==len(evaluations)==len(cached)==len(rb)==len(baseline)==288
    assert [r['id'] for r in rows]==[f['id'] for f in spec['frames']]==[r['id'] for r in evaluations]
    frozen_hashes={str(p):sha(p) for p in inputs}
    write(out/'protocol.json',dict(brief_sha256=sha(BRIEF),arms=ARMS,input_hashes=frozen_hashes))
    start=time.perf_counter(); admit=admission(spec,rows,evaluations); write(out/'admission.json',admit)
    assert admit['status']=='PASS', 'Analytic source not admitted; preserve failure before method scoring'
    print(json.dumps({'admission':admit}),flush=True)
    _,probe=geometry(32,3)
    select_backend('batch-tensor',cpu=BackendCandidate('numpy-ray-aabb','cpu',
        lambda:trace(spec['frames'][0],spec,probe),lambda _:DeviceObservation('cpu','host CPU','NumPy '+np.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'python':sys.executable,'numpy':np.__version__,
                      'note':'Torch/CuPy unavailable in inspected local research virtual environments; scalar histogram reduction stays CPU.'})
    # Public context export is independently reloaded by the predictor.
    context=[dict(id=r['id'],yaw=p['integrated_yaw_deg'],radar=bool(b['candidate'])) for r,p,b in zip(rows,cached,rb)]
    write(out/'observable-context.json',context)
    all_predictions={}; timings={}; observation_stats={}; private_audits={}
    lattice8=geometry(8,12)[1].reshape(-1,3); lattice32=geometry(32,3)[1].reshape(-1,3)
    assert np.array_equal(lattice8[np.lexsort(lattice8.T[::-1])],lattice32[np.lexsort(lattice32.T[::-1])])
    for arm,(grid,sub,scale,sigma) in ARMS.items():
        arm_start=time.perf_counter(); zones,local_rays=geometry(grid,sub); states={}; generated=[]; private=[]; stats=Counter()
        for frame,row,e in zip(spec['frames'],rows,evaluations):
            ds,rs,owners,directions,origin=trace(frame,spec,local_rays)
            rng=states.setdefault(frame['episode'],random.Random(frame.get('tof_sensor_seed',frame['sensor_seed']+1150003)))
            newrow={k:copy.deepcopy(row[k]) for k in ('id','episode_id','time_s','camera_pitch_deg','camera_in_body_m','rgb_intrinsics','tof_packet_received')}
            newrow['tof_zones']=[]; records=[]; stats['sample_rays']+=ds.size
            hazard_objects=np.asarray([intersects([(c-s/2-b,c+s/2-b) for c,s,b in zip(o['center_m'],o['size_m'],frame['body_origin_m'])])
                                       for o in frame['objects']]+[False for k in ('background','floor') if spec.get(k)])
            for z,g in enumerate(zones):
                targets,lineages,weak=reduce_zone(ds[z],rs[z],rng,row['tof_packet_received'],scale,sigma)
                newrow['tof_zones'].append(dict(g,target_count=len(targets),targets=targets))
                stats['below_floor_components']+=weak; stats['empty_zones']+=int(not targets)
                stats.update(t['status'] for t in targets)
                for slot,indices in enumerate(lineages):
                    points=origin+directions[z,indices]*ds[z,indices,None]-np.asarray(frame['body_origin_m'])
                    corridor=((points>=np.array([.2,-.3,.4])) & (points<=np.array([3.6,.3,2.05]))).all(axis=1)
                    records.append(dict(zone_id=z,slot=slot,samples=len(indices),corridor_samples=int(corridor.sum()),
                                        hazard_actor_samples=int(hazard_objects[owners[z,indices]].sum()),
                                        body_points=points.tolist()))
            generated.append(newrow); private.append(dict(id=row['id'],returns=records))
        folder=out/arm; folder.mkdir()
        raw=folder/'observations.jsonl'; raw.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in generated))
        # Private first-hit distances and ownership never cross to predict().
        np.savez_compressed(folder/'lattice.npz',local_rays=local_rays)
        write(folder/'private-contributors.json',private)
        generation_seconds=time.perf_counter()-arm_start; predict_start=time.perf_counter()
        public=jsonl(raw); sealed_context=read(out/'observable-context.json')
        predictions=[predict(r,c['yaw'],c['radar']) for r,c in zip(public,sealed_context)]
        write(folder/'predictions.json',predictions)
        write(folder/'seal.json',dict(observations_sha256=sha(raw),predictions_sha256=sha(folder/'predictions.json'),
            private_sha256=sha(folder/'private-contributors.json'),context_sha256=sha(out/'observable-context.json'),
            authority='PUBLIC_OBSERVATIONS_PREDICTIONS_SAVED_BEFORE_SCORING'))
        timings[arm]=dict(generation_s=generation_seconds,prediction_s=time.perf_counter()-predict_start)
        all_predictions[arm]=predictions; private_audits[arm]=private; observation_stats[arm]=dict(stats)
        print(json.dumps({'generated':arm,'seconds':timings[arm],'observation_stats':observation_stats[arm]}),flush=True)
    # Evaluator phase; geometry was used earlier exclusively by the generator/admission.
    truth=[native_truth(e) for e in evaluations]; labels=read(SOURCE/'analysis-v1/frame-report.json')
    assert truth==[r['truth'] for r in labels]
    from mz129_extent_correction import events
    summary=dict(authority='CONSUMED_ANALYTIC_GEOMETRY_CAPABILITY_NOT_HARDWARE',frames=288,arms={},
                 admission=admit,timings=timings,observation_stats=observation_stats,
                 mz129_context=metrics(rows,truth,[p['candidate'] for p in baseline]))
    for arm,preds in all_predictions.items():
        result={}
        for branch in ('tof','full'):
            flags=[p[branch] for p in preds]; m=metrics(rows,truth,flags); m['events']=events(rows,truth,flags)
            m['families']={}
            for family in sorted({r['family'] for r in labels}):
                ix=[i for i,r in enumerate(labels) if r['family']==family]
                m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
            ref=[p[branch] for p in all_predictions['dense8']]
            m['lost_TP_vs_dense8']=[r['id'] for r,t,b,a in zip(rows,truth,ref,flags) if t and b and not a]
            m['new_FP_vs_dense8']=[r['id'] for r,t,b,a in zip(rows,truth,ref,flags) if not t and not b and a]
            old_events=events(rows,truth,ref)
            m['later_or_lost_events_vs_dense8']=[dict(before=b,after=a) for a,b in zip(m['events'],old_events)
                if b['first_alert_s'] is not None and (a['first_alert_s'] is None or a['first_alert_s']>b['first_alert_s'])]
            result[branch]=m
        audit=Counter(); retained_points=0; outside_bounds=0; sample_count=0
        for row,t,p,priv in zip(rows,truth,preds,private_audits[arm]):
            lookup={(r['zone_id'],r['slot']):r for r in p['returns']}
            actual=False; responsible=False
            for r in priv['returns']:
                q=lookup[(r['zone_id'],r['slot'])]; actual |= r['hazard_actor_samples']>0
                responsible |= r['hazard_actor_samples']>0 and q['trigger']
                sample_count+=r['samples']; retained_points+=r['corridor_samples']
                for point in r['body_points']:
                    outside_bounds+=int(not all(lo-1e-7<=v<=hi+1e-7 for v,(lo,hi) in zip(point,q['xyz'])))
            audit['true_frames_with_detected_hazard_actor']+=int(t and actual)
            audit['tof_TP_with_triggering_hazard_actor']+=int(t and p['tof'] and responsible)
            audit['tof_TP_without_triggering_hazard_actor']+=int(t and p['tof'] and not responsible)
            if not t and p['tof']:
                triggers=[q for q in p['returns'] if q['trigger']]
                audit['FP_with_MERGED_trigger']+=int(any(q['status']=='SIM_MERGED' for q in triggers))
                audit['FP_with_VALID_trigger']+=int(any(q['status']=='SIM_VALID' for q in triggers))
        result['native_generated_audit']=dict(audit,returned_samples=sample_count,corridor_samples=retained_points,
            samples_outside_range_pose_enclosure=outside_bounds,angular_support_policy='FULL_ZONE_NO_CLIPPING',
            note='Generated contributor samples, not fresh UE native samples. Range/pose outliers are disclosed, not hidden.')
        summary['arms'][arm]=result
    summary['seconds']=time.perf_counter()-start
    write(out/'summary.json',summary)
    for path,digest in frozen_hashes.items(): assert sha(path)==digest,path
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        decision='REPORT_FIXED_CONTRAST_NO_DEFAULT_PROMOTION',resources='No persistent processes; durable evidence retained'))
    print(json.dumps({a:{b:{k:v for k,v in m[b].items() if k in ('TP','FP','FN','UNKNOWN','false_segments','max_detected_delay_s')}
                        for b in ('tof','full')} for a,m in summary['arms'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); run(a.output)
