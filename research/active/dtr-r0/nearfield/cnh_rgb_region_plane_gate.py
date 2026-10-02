"""Fixed cached RGB plane support and held-out-return gate, consumed Development.

Only evaluator code sees original object identities and native reference depth.
The predictor receives labels, public rays and the same 64 cached q10 returns.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import cnh_rgb_region_range as C
import cnh_rgb_region_plane as M

P,R,S,A,V,G,ROOT=C.P,C.R,C.S,C.A,C.V,C.G,C.ROOT
OUT=ROOT/'artifacts.local/work/cnh-rgb-region-plane-20261002'
RUN_ID='CNH_RGB_REGION_PLANE_20261002'

def artifact_paths():
    return dict(range_plan=C.OUT/'PLAN.json',range_result=C.OUT/'result.json',
                range_ledger=C.OUT/'frame-ledger.json',inference_audit=R.OUT/'inference-audit.json')

def prepare():
    assert not (OUT/'PLAN.json').exists()
    old=P.read(C.OUT/'PLAN.json')
    for path,digest in old['source_sha256'].items():assert P.sha(ROOT/path)==digest
    paths=[Path(__file__),Path(M.__file__),Path(C.__file__),Path(R.__file__),Path(S.__file__),Path(A.__file__),Path(V.__file__),Path(G.__file__),Path(P.__file__)]
    row=(f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same343 consumed Development; original SAM full-zone anchors>=4, hold highest zone ID, train center rank3 tol1e-10; inverse-Z plane forward native-ray r^-2 q10; one SLSQP maxiter100 ftol1e-12, training-only flat initial; rectangle inverseZ>=1e-6; success+finite+constraints+trainRMS<=.05m+heldout<=.05m; retain train fit/no refit, full region propagation else q10; fixed all-support5/10cm errors | NOT_RUN | rank is support proxy, optimizer failures distinct, GT only evaluator; zero eligible eval contact objects stops fit; no scale repair or alarm claim | `artifacts.local/work/cnh-rgb-region-plane-20261002/REPORT.md` |')
    body=P.RUNS.read_text(encoding='utf8');assert RUN_ID not in body
    OUT.mkdir(parents=True,exist_ok=True)
    P.RUNS.write_text(body.rstrip()+'\n'+row+'\n',encoding='utf8')
    (OUT/'prerun-row.txt').write_text(row+'\n',encoding='utf8')
    P.save(OUT/'PLAN.json',dict(run_id=RUN_ID,frozen_at=datetime.now(timezone.utc).isoformat(),
        role='Consumed Development, same343 previously evaluated frames; no fresh confirmation',inputs=old['inputs'],
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in paths},
        artifacts={k:dict(path=str(p),sha256=P.sha(p)) for k,p in artifact_paths().items()},
        preregistration=row,workers=3,thresholds_m=[.05,.10],
        limits=['Geometry center rank3 is only a support proxy, not proof of forward q10 identifiability.',
                'Piecewise-smooth empirical quantile and local optimizer: numerical failure is not a physical impossibility result.',
                'A single fixed held-out zone checks only that return; no global surface or extrapolation guarantee.',
                'Reference depth generated the cached object-blind q10 proxy; this is not hardware ToF or M3.',
                'No use of GT to choose anchors, fit, accept planes, select arms or tune thresholds.']))
    print('Prepared plane support and fixed held-out gate')

def load():
    plan=P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():assert P.sha(ROOT/path)==digest,path
    for item in plan['artifacts'].values():assert P.sha(item['path'])==item['sha256']
    old={f['id']:f for f in P.read(C.OUT/'frame-ledger.json')}
    audit={f['id']:f for f in P.read(R.OUT/'inference-audit.json')['frames']}
    assert len(old)==len(audit)==len(plan['inputs'])==343
    return plan,old,audit

def observation(row,old,audit):
    path=R.OUT/'predictions'/(row['id']+'.npz');receipt=path.with_suffix('.json')
    assert P.sha(path)==audit['output_sha256'] and P.sha(receipt)==audit['receipt_sha256']
    with np.load(path,allow_pickle=False) as f:labels=f['labels'].copy()
    geo=A.whole_geometry(V.ray_geometry(labels.shape,row['camera_matrix']))
    # Extract only the previously audited simulator interface, never per-pixel depth.
    returns=np.array([np.nan if x is None else x for x in old['range_prediction']['sensor_q10_radial']],dtype=float)
    assert returns.shape==(64,)
    return labels,geo,returns

def support():
    assert not (OUT/'support.json').exists()
    plan,old,audit=load();frames=[]
    for row in plan['inputs']:
        labels,geo,returns=observation(row,old[row['id']],audit[row['id']])
        candidates=M.support(labels,geo,returns)
        eligible={r['region'] for r in candidates if r['status']=='ELIGIBLE'}
        objects=[]
        for o in old[row['id']]['objects']:
            if o['stratum']!='contact':continue
            covered=sum(e['error']['support_pixels'] for e in o['sam_source_evidence']['regions'] if e['region'] in eligible)
            objects.append(dict(instance=o['instance_id'],support=o['support_pixels'],eligible_pixels=covered))
        frames.append(dict(id=row['id'],split=row['split'],family=row['family'],regions=candidates,objects=objects))
    groups={}
    for split in ('cal','eval','all'):
        fs=[f for f in frames if split=='all' or f['split']==split];objects=[o for f in fs for o in f['objects']]
        groups[split]=dict(frames=len(fs),contact_objects=len(objects),eligible_any=sum(o['eligible_pixels']>0 for o in objects),
            eligible_half=sum(o['eligible_pixels']>=.5*o['support'] for o in objects),
            eligible_pixels=sum(o['eligible_pixels'] for o in objects),support_pixels=sum(o['support'] for o in objects),
            region_statuses=dict(Counter(r['status'] for f in fs for r in f['regions'])))
    P.save(OUT/'support.json',dict(groups=groups,frames=frames,decision='FIT' if groups['eval']['eligible_any'] else 'STOP_ZERO_ELIGIBLE_CONTACT'))
    print(json.dumps(groups))

def evaluate_frame(args):
    row,old,audit=args
    labels,geo,returns=observation(row,old,audit)
    predicted=M.predict(labels,geo,returns)
    for name in ('depth','instance'):assert P.sha(row[name+'_path'])==row[name+'_sha256']
    radial=S.hdf(row['depth_path']);inst=S.hdf(row['instance_path']);z=radial/geo['radial_factor']
    contact=np.zeros(z.shape,bool);expanded=contact.copy()
    for q in geo['queries']:
        take=geo['fov_mask']&np.isfinite(z)&(z>=.6)&(z<2.1)&(z*geo['fy']>=q['y_low'])&(z*geo['fy']<=q['y_high'])
        contact|=take&(np.abs(z*geo['fx'])<.3);expanded|=take&(np.abs(z*geo['fx'])<=.4)
    frame=deepcopy(old);frame['plane_diagnostics']=predicted['diagnostics']
    for o in frame['objects']:
        target=(inst==o['instance_id'])&(contact if o['stratum']=='contact' else expanded)
        assert target.sum()==o['support_pixels']
        o['plane_range']=C.error_stats(predicted['predicted_z'],z,target)
        o['plane_range']['accepted_pixels']=int((target&(predicted['region_map']>=0)).sum())
    return frame

def summarize(frames):
    out=dict(frames=len(frames),regions=dict(Counter(r['status'] for f in frames for r in f['plane_diagnostics'])),strata={})
    for stratum in ('contact','pass_adjacent'):
        objects=[o for f in frames for o in f['objects'] if o['stratum']==stratum]
        n=sum(o['support_pixels'] for o in objects);finite=sum(o['plane_range']['finite_pixels'] for o in objects)
        total=sum(o['plane_range']['abs_z_error_sum_m'] for o in objects)
        group=dict(n=len(objects),support_pixels=n,finite_pixels=finite,missing_pixels=n-finite,
            error_sum_m=total,pooled_mae_m=total/finite if finite else None,
            accepted_any=sum(o['plane_range']['accepted_pixels']>0 for o in objects),
            accepted_half=sum(o['plane_range']['accepted_pixels']>=.5*o['support_pixels'] for o in objects),
            accepted_pixels=sum(o['plane_range']['accepted_pixels'] for o in objects),paired={})
        for tag in ('05','10'):
            group['hits_'+tag]=sum(o['plane_range']['hit_'+tag] for o in objects)
            group['within_'+tag+'_pixels']=sum(o['plane_range']['within_'+tag+'_pixels'] for o in objects)
        for arm in C.ARMS:
            p={}
            for tag in ('05','10'):
                p[tag]=dict(plane_only=sum(o['plane_range']['hit_'+tag] and not o['range_arms'][arm]['hit_'+tag] for o in objects),
                    baseline_only=sum(not o['plane_range']['hit_'+tag] and o['range_arms'][arm]['hit_'+tag] for o in objects),
                    both=sum(o['plane_range']['hit_'+tag] and o['range_arms'][arm]['hit_'+tag] for o in objects),
                    neither=sum(not o['plane_range']['hit_'+tag] and not o['range_arms'][arm]['hit_'+tag] for o in objects))
            group['paired'][arm]=p
        out['strata'][stratum]=group
    return out

def evaluate():
    assert not (OUT/'result.json').exists()
    preliminary=P.read(OUT/'support.json');assert preliminary['decision']=='FIT'
    plan,old,audit=load()
    args=[(r,old[r['id']],audit[r['id']]) for r in plan['inputs']]
    with ThreadPoolExecutor(max_workers=3) as pool:frames=list(pool.map(evaluate_frame,args))
    for f in frames:
        original=deepcopy(f);del original['plane_diagnostics']
        for o in original['objects']:del o['plane_range']
        assert original==old[f['id']]
    groups={s:summarize([f for f in frames if s=='all' or f['split']==s]) for s in ('cal','eval','all')}
    families={family:summarize([f for f in frames if f['family']==family]) for family in sorted({f['family'] for f in frames})}
    assert groups['eval']['strata']['contact']['n']==135
    P.save(OUT/'frame-ledger.json',frames)
    P.save(OUT/'result.json',dict(role=plan['role'],groups=groups,families=families,previous_ledger_unchanged=True,limits=plan['limits']))
    print(json.dumps(groups))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','support','evaluate'))
    globals()[parser.parse_args().action]()
