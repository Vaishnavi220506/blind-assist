"""One sealed, consumed dev48 comparison; no test scoring or adaptive tuning."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from mz137_edge_corridor import METHOD, predict_frame
from mz136_incumbent import public_observations
from mz136_boundary_geometry import camera_to_body
from mz136_rectified_edges import transform
from mz115_spatial_allocation import possible
from run_mz107_four_sensor import ROOT, sha, write, readrows, truth, metrics
from evaluate_mz136_corridor_pair import score, retention, pair_metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914'
DEFAULT=ROOT/'artifacts.local/work/mz137-edge-corridor-20260915/contrast-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def selected_jsonl(path, ids):
    # Inspect only the join ID before decoding selected evaluator records.
    import re
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        match=re.search(r'"id"\s*:\s*"([^"]+)"',line)
        assert match
        if match[1] in ids:result.append(json.loads(line))
    return result


def native_reference(row,e,pred):
    target=next(o for o in e['native_bounds'] if o['name']=='shape0')
    origin=np.asarray(e['body_origin_m'])
    low=np.asarray(target['center_m'])-target['extent_m']-origin
    high=np.asarray(target['center_m'])+target['extent_m']-origin
    overlap=float(min(high[1],.3)-max(low[1],-.3))
    record=dict(native_target_lateral_overlap_m=overlap,
        boundary_stratum='ABS_LATERAL_GAP_OR_OVERLAP_LT_1CM' if abs(overlap)<.01 else 'AT_LEAST_1CM')
    if not pred['coarse_available']:return record
    corners=np.array(list(itertools.product(*zip(low,high))))-row['camera_in_body_m']
    cam=corners@camera_to_body(row,pred['edge']['yaw_deg']); intr=row['rgb_intrinsics']
    if np.any(cam[:,0]<=0):return record
    uv=np.c_[intr['cx']+intr['fx']*cam[:,1]/cam[:,0],intr['cy']-intr['fy']*cam[:,2]/cam[:,0]]
    viewport=np.array([[0,0],[intr['width']-1,0],[intr['width']-1,intr['height']-1],[0,intr['height']-1]],np.float32)
    area,clipped=cv2.intersectConvexConvex(cv2.convexHull(uv.astype(np.float32)),viewport)
    if area<=0 or clipped is None:return record
    rect=transform(clipped.reshape(-1,2),np.asarray(pred['edge']['homography']))
    bounds=[float(rect[:,0].min()),float(rect[:,0].max())]
    inner=0 if np.mean(bounds)>=intr['cx'] else 1
    record.update(native_visible_edges_px=bounds,inner_index=inner)
    for arm in ('coarse','fine'):
        values=pred[arm].get('edges_px')
        if values:record[arm+'_signed_inner_error_px']=values[inner]-bounds[inner]
    return record


def contributor_audit(row,e,pred,cached):
    records=[]; native={z['zone_id']:z for z in e['zonal_tof_native']}
    for item in cached['spatial_evidence']:
        zone=native[item['zone_id']]
        lineage=next((v for v in zone['returned_lineage'] if v['target_index']==item['target_slot']),None)
        if not lineage:continue
        rays={r['subray']:r for r in zone['private_rays']}
        for index in lineage['hit_indices']:
            ray=rays[index]; point=np.array(ray['hit_point_m'])-e['body_origin_m']
            relative=point-row['camera_in_body_m']
            rec=dict(id=row['id'],zone=item['zone_id'],slot=item['target_slot'],subray=index,
                point_body_m=point.tolist(),actor=ray['actor_id'],corridor=possible([[v,v] for v in point]),
                incumbent_possible=possible(item['localized_xyz']),
                incumbent_contains=all(lo-1e-9<=v<=hi+1e-9 for v,(lo,hi) in zip(point,item['localized_xyz'])))
            for arm in ('coarse','fine'):
                p=pred[arm]; replaced=[item['zone_id'],item['target_slot']] in p['replaced_returns']
                rec[arm+'_replaced']=replaced
                if not replaced:
                    rec[arm+'_angular_retained']=True
                    rec[arm+'_alert_support']=rec['incumbent_possible']
                else:
                    intr=row['rgb_intrinsics']; uv=[intr['cx']+intr['fx']*relative[1]/relative[0],
                        intr['cy']-intr['fy']*relative[2]/relative[0]]
                    l,r=p['edges_px']; t,b=pred['association']['rectified_vertical_px']
                    rec[arm+'_angular_retained']=bool(relative[0]>0 and l<=uv[0]<=r and t<=uv[1]<=b)
                    rec[arm+'_alert_support']=p['plane_corridor']
            records.append(rec)
    return records


def evaluate(rows,es,preds,caches,base,spec,out):
    gt=np.array([truth(e) for e in es],bool); indices=list(range(len(rows)))
    flags={'mz129':np.array([p['candidate'] for p in base],bool)}
    flags.update({a:np.array([p[a]['candidate'] for p in preds],bool) for a in ('coarse','fine')})
    episodes={}
    for i,r in enumerate(rows):episodes.setdefault(r['episode_id'],[]).append(i)
    pairs=[dict(a=i,b=j) for pair in spec['pairs'] if pair['split']=='dev'
           for i,j in zip(*(episodes[ep] for ep in pair['episodes']))]
    results={a:score(rows,es,gt,f,indices,flags['mz129']) for a,f in flags.items()}
    assert results['mz129']['metrics']==dict(TP=24,FP=13,FN=0,TN=11,UNKNOWN=11)
    cases=[]; contributors=[]
    for i,(r,e,p,c) in enumerate(zip(rows,es,preds,caches)):
        coarse,fine=p['coarse'],p['fine']
        rec=dict(id=r['id'],family=e['family'],truth=bool(gt[i]),**native_reference(r,e,p),
            baseline=bool(flags['mz129'][i]),coarse=coarse['candidate'],fine=fine['candidate'],
            coarse_available=p['coarse_available'],fine_available=p['fine_available'],
            association_available=p['association'] is not None,
            edge_changed=coarse.get('edges_px')!=fine.get('edges_px'),
            spatial_support_changed=coarse['xyz']!=fine['xyz'],
            support_bits_changed=coarse['support_bits']!=fine['support_bits'],
            plane_decision_changed=coarse.get('plane_corridor')!=fine.get('plane_corridor'),
            final_changed=coarse['candidate']!=fine['candidate'],
            residual_alert=fine['residual_alert'],
            coarse_overlap_m=coarse.get('signed_lateral_overlap_m'),fine_overlap_m=fine.get('signed_lateral_overlap_m'))
        cases.append(rec); contributors.extend(contributor_audit(r,e,p,c))
    for a,f in flags.items():
        result=results[a]; m=result['metrics']
        result.update(precision=m['TP']/max(1,m['TP']+m['FP']),recall=m['TP']/max(1,m['TP']+m['FN']),
            paired_binary_decision=pair_metrics(gt,f.astype(float),f,pairs),
            retention=retention(result,results['mz129']),strata={})
        result['dev_joint_target_met']=bool(result['retention']['pass_retention'] and m['FP']<=10)
        for label,ix in [('shallow_family',[i for i,e in enumerate(es) if e['family']=='shallow_boundary_stress']),
                         ('other_families',[i for i,e in enumerate(es) if e['family']!='shallow_boundary_stress']),
                         *[(s,[i for i,c in enumerate(cases) if c['boundary_stratum']==s]) for s in sorted({c['boundary_stratum'] for c in cases})]]:
            result['strata'][label]=dict(frames=len(ix),**metrics(gt[ix],f[ix]))
    chain={key:sum(c[key] for c in cases) for key in ('coarse_available','fine_available','association_available',
        'edge_changed','spatial_support_changed','support_bits_changed','plane_decision_changed','final_changed')}
    chain.update(fine_vs_coarse_corrected=int(((flags['fine']==gt)&(flags['coarse']!=gt)).sum()),
        fine_vs_coarse_worsened=int(((flags['fine']!=gt)&(flags['coarse']==gt)).sum()),
        geometry_change_blocked_by_residual=sum(c['plane_decision_changed'] and not c['final_changed'] and c['residual_alert'] for c in cases))
    audit=dict(total_contributor_records=len(contributors),native_corridor=sum(c['corridor'] for c in contributors))
    for arm in ('coarse','fine'):
        audit[arm]=dict(replaced=sum(c[arm+'_replaced'] for c in contributors),
            angular_excluded=sum(c[arm+'_replaced'] and not c[arm+'_angular_retained'] for c in contributors),
            corridor_angular_excluded=sum(c['corridor'] and c[arm+'_replaced'] and not c[arm+'_angular_retained'] for c in contributors),
            previously_contained_corridor_angular_excluded=sum(c['corridor'] and c['incumbent_contains'] and c[arm+'_replaced'] and not c[arm+'_angular_retained'] for c in contributors),
            corridor_alert_support_lost=sum(c['corridor'] and c['incumbent_possible'] and not c[arm+'_alert_support'] for c in contributors))
    fine_increment=retention(results['fine'],results['coarse'])['pass_retention'] and results['fine']['metrics']['FP']<results['coarse']['metrics']['FP']
    summary=dict(frames=48,scope='CONSUMED_DEVELOPMENT_ONLY_NO_TEST',arms=results,chain=chain,
        native_contributors=audit,edge_states=dict(Counter(p['edge_state'] for p in preds)),
        fine_increment_over_coarse=bool(fine_increment),
        decision='DEVELOPMENT_CHALLENGER_REQUIRES_INDEPENDENT_CONFIRMATION' if results['fine']['dev_joint_target_met'] and fine_increment else 'KEEP_MZ129_FIXED_EDGE_PLANE_CHAIN_NO_JOINT_GAIN',
        limits='One conditional plane/readout; does not establish all edge integration ineffective. Raw retention is not geometric retention. Binary pair order is not continuous-score ranking.')
    write(out/'cases.json',cases);write(out/'native-contributors.json',contributors);write(out/'summary.json',summary)
    return summary


def run(out):
    out=out.resolve(); assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    cap=WORK/'source/returned-v1/capture-v1';prep=WORK/'incumbent/fresh-v1'
    spec=read(cap/'spec.json');receipt=read(cap/'receipt.json');assert receipt['status']=='PASS'
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    ids={f['id'] for f in spec['frames'] if f['split']=='dev'};assert len(ids)==48
    rows=public_observations(selected_jsonl(cap/'raw.jsonl',ids));assert len(rows)==48
    done=read(prep/'completion.json');seal=read(prep/'prediction-seal.json')
    assert done['status']=='PASS' and sha(prep/'prediction-seal.json')==done['prediction_seal_sha256']
    assert sha(prep/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
    assert sha(prep/'observation-seal.json')==seal['observation_seal_sha256']
    assert sha(prep/'nominal/raw.jsonl')==read(prep/'observation-seal.json')['hashes']['nominal']
    assert rows==selected_jsonl(prep/'nominal/raw.jsonl',ids)
    cache=read(prep/'nominal/predictions.json');byid={p['id']:i for i,p in enumerate(cache['predictions'])}
    base=[cache['predictions'][byid[r['id']]] for r in rows]
    corrected=[cache['corrected'][byid[r['id']]] for r in rows]
    radar=[cache['radar'][byid[r['id']]] for r in rows]
    paths=[cap/n for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json')]+[prep/'nominal/predictions.json',prep/'prediction-seal.json',prep/'observation-seal.json',prep/'nominal/raw.jsonl']
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/n)==receipt['hashes'][n]
    images=[]
    for r in rows:
        p=(cap/r['rgb_path']).resolve();assert p.is_relative_to(cap.resolve())
        assert sha(p)==receipt['hashes'][r['rgb_path']]; paths.append(p)
        image=cv2.imread(str(p));assert image is not None;images.append(image)
    inputs={str(p):sha(p) for p in paths}
    oldfreeze=read(WORK/'boundary-rectified-edge-v1/freeze.json')
    for name,digest in oldfreeze['source_hashes'].items():assert sha(Path(__file__).with_name(name))==digest
    sources={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in list(sys.modules.values())
             if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(Path(__file__).parent.resolve())}
    sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    protocol=Path(__file__).with_name('MZ137_EDGE_CORRIDOR_20260915.md')
    out.mkdir(parents=True);(out/'source-snapshot').mkdir()
    for p in sources:shutil.copyfile(p,out/'source-snapshot'/Path(p).name)
    shutil.copyfile(protocol,out/'protocol-before-outcomes.md')
    write(out/'freeze.json',dict(method=METHOD,ids=[r['id'] for r in rows],inputs=inputs,sources=sources,
        protocol_sha256=sha(out/'protocol-before-outcomes.md'),python=sys.executable,opencv=cv2.__version__,numpy=np.__version__))
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-numpy-edge','cpu',
        lambda:predict_frame(rows[0],images[0],corrected[0],base[0],radar[0]),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__+' NumPy '+np.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Frozen CPU warp/Scharr/morphology and NumPy profile pipeline has no equivalent implemented GPU backend'})
    start=time.perf_counter()
    preds=[predict_frame(*args) for args in zip(rows,images,corrected,base,radar)]
    write(out/'predictions.json',preds)
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),freeze_sha256=sha(out/'freeze.json'),seconds=time.perf_counter()-start,
        authority='PREDICTIONS_SAVED_BEFORE_SELECTED_EVALUATOR_PARSE'))
    es=selected_jsonl(cap/'evaluator.jsonl',ids);assert [e['id'] for e in es]==[r['id'] for r in rows]
    summary=evaluate(rows,es,preds,corrected,base,spec,out)
    replay=[predict_frame(*args) for args in zip(rows,images,corrected,base,radar)]
    assert json.loads(json.dumps(replay))==read(out/'predictions.json')
    assert inputs=={p:sha(Path(p)) for p in inputs}
    assert sources=={p:sha(Path(p)) for p in sources}
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
        exact_replay=True,inputs_unchanged=True,resources='No persistent worker/allocation; durable evidence retained'))
    print(json.dumps(dict(decision=summary['decision'],arms={a:r['metrics'] for a,r in summary['arms'].items()},chain=summary['chain'],native=summary['native_contributors']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    run(parser.parse_args().output)
