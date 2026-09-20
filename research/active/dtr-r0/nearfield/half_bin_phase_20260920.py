"""Paired proxy phase sensitivity; no best-phase selection or new readout."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from ba_camera_corridor import sample_native
from tof_fov45_core import boxes45
from tof_corridor_calibration import score_frame,decide
from audit_existing_strata_20260920 import tally
from audit_core_workpoint_20260920 import silence_runs

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/ba-core-workpoint-transfer-20260920'
OUT=ROOT/'artifacts.local/work/ba-half-bin-phase-20260920'
T0=.007085703945147101
T=.4071309640537889
PHASES=('zero','half')
ARMS=tuple(p+'_'+r for p in PHASES for r in ('cal','strong','hold'))
IDENTITY=('id','clip_id','time_s','frame_in_clip')
CODE=('half_bin_phase_20260920.py','tof_fov45_core.py','ba_camera_corridor.py',
      'tof_corridor_calibration.py','audit_existing_strata_20260920.py',
      'audit_core_workpoint_20260920.py','return_lineage_core.py')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def bin_ids(hits,phase):
    assert hits.dtype==np.float32
    if phase=='zero':return np.minimum((hits/.1).astype(int),79)
    assert phase=='half'
    bins=np.floor((hits-np.float32(.05))/np.float32(.1)).astype(int)
    assert np.all((bins>=0)&(bins<=79))
    return bins

def simulate(depth,identity,boxes,phase):
    rng=np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:8],16))
    values=[];records=[];indices=[]
    for z,(y0,x0,y1,x1) in enumerate(boxes):
        patch=depth[y0:y1,x0:x1];valid=np.isfinite(patch)&(patch>.001)
        v=patch[valid];hits=v[(v>=.1)&(v<8)]
        value=np.nan;selected=None;idx=np.array([],np.int64)
        reason='INSUFFICIENT_HITS';draw=None;standard=None;center=None
        if hits.size>=4:
            draw=float(rng.random())
            if draw<.05:reason='SIMULATED_DROPOUT'
            else:
                bins=bin_ids(hits,phase)
                weights=np.bincount(bins,weights=1/np.maximum(hits,.3)**2,minlength=80)
                selected=int(np.argmax(weights));chosen=bins==selected;center=np.mean(hits[chosen])
                sigma=.01+.02*center;noise=rng.normal(0,sigma);standard=float(noise)/float(sigma)
                noisy=center+noise;reason='NOISY_RANGE_OUTSIDE_LIMIT'
                eligible=valid&(patch>=.1)&(patch<8);py,px=np.nonzero(eligible)
                idx=((py[chosen]+y0)*depth.shape[1]+px[chosen]+x0).astype(np.int64)
                if .1<=noisy<8:value=noisy;reason='OBSERVED'
        values.append(value);indices.append(idx)
        records.append(dict(zone=z,eligible=int(hits.size),dropout_draw=draw,noise_z=standard,
            selected_bin=selected,center_m=float(center) if center is not None else None,
            candidate_count=len(idx),candidate_set_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
            observed=bool(np.isfinite(value)),range_m=float(np.float32(value)) if np.isfinite(value) else None,reason=reason))
    return np.asarray(values,np.float32),records,indices

def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True)
    protocol=Path(__file__).with_name('HALF_BIN_PHASE_PROTOCOL_20260920.md')
    (OUT/'protocol-before-run.md').write_bytes(protocol.read_bytes())
    files=('protocol.json','observations.json','observation-seal.json','predictions.json','prediction-seal.json',
           'frame-results.json','evaluation-seal.json','private-lineage.json','spec.json','capture/evaluator/geometry.json')
    write(OUT/'protocol.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={n:sha(SOURCE/n) for n in files},code={n:sha(Path(__file__).with_name(n)) for n in CODE},
        protocol_text_sha256=sha(OUT/'protocol-before-run.md'),phase_m=[0,.05],thresholds=[T0,T],
        backend='TASK_NOT_GPU_SUITABLE',scope='CONSUMED_PAIRED_PROXY_SENSITIVITY_NOT_NEW_SENSOR'))
    print('FROZEN: only half-bin shift, complete432')

def verify():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol-before-run.md')==p['protocol_text_sha256']
    for n,h in p['inputs'].items():assert sha(SOURCE/n)==h,n
    for n,h in p['code'].items():assert sha(Path(__file__).with_name(n))==h,n

def seal(name,files):write(OUT/name,dict(status='COMPLETE',frames=432,protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(OUT/n) for n in files}))
def check_seal(name):
    s=read(OUT/name);assert s['status']=='COMPLETE' and s['protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in s['hashes'].items():assert sha(OUT/n)==h,n

def construct():
    verify();write(OUT/'construction-start.json',dict(paired_seed=True,no_labels=True))
    for name in ('observation-seal.json','prediction-seal.json','evaluation-seal.json'):
        for n,h in read(SOURCE/name)['hashes'].items():assert sha(SOURCE/n)==h
    obs=read(SOURCE/'observations.json');lineage=read(SOURCE/'private-lineage.json');geometry=read(SOURCE/'capture/evaluator/geometry.json')
    assert len(obs)==len(lineage)==len(geometry)==432
    vectors={p:[] for p in PHASES};private=[];identities=[]
    for o,old,g in zip(obs,lineage,geometry):
        assert o['id']==old['id'] and o['clip_id']==g['clip_id'] and o['frame_in_clip']==g['frame_in_clip']
        nativepath=SOURCE/'capture/evaluator'/g['native_path'];assert sha(nativepath)==old['native_sha256']==g['native_sha256']
        native=np.load(nativepath,allow_pickle=False);depth=sample_native(native)
        assert sha(SOURCE/o['path'])==o['sha256']
        with np.load(SOURCE/o['path'],allow_pickle=False) as d:boxes,stored=d['boxes'],d['values']
        assert np.array_equal(boxes,boxes45())
        pairs={};all_indices={}
        for phase in PHASES:
            values,records,indices=simulate(depth,old['seed'],boxes,phase)
            vectors[phase].append(values);pairs[phase]=records;all_indices[phase]=indices
        assert np.array_equal(vectors['zero'][-1],stored,equal_nan=True)
        for a,b,trace,idx in zip(pairs['zero'],pairs['half'],old['traces'],all_indices['zero']):
            assert a['reason']==trace['reason'] and a['range_m']==trace['distance_m']
            assert (a['selected_bin'] if a['observed'] else None)==trace['winner_bin']
            assert (idx.tolist() if a['observed'] else [])==trace['pixel_indices']
            if a['observed']:
                weights=(1/np.maximum(depth.ravel()[idx],.3)**2).astype(float).tolist();assert weights==trace['weights']
            assert a['dropout_draw']==b['dropout_draw'] and a['eligible']==b['eligible']
            assert (a['noise_z'] is None)==(b['noise_z'] is None)
            if a['noise_z'] is not None:assert abs(a['noise_z']-b['noise_z'])<1e-14
        identities.append({k:o[k] for k in IDENTITY})
        private.append(dict(id=o['id'],native_sha256=old['native_sha256'],phases=pairs))
    np.savez_compressed(OUT/'phase-observations.npz',boxes=boxes45(),zero=np.stack(vectors['zero']),half=np.stack(vectors['half']))
    write(OUT/'identities.json',identities);write(OUT/'source-pairing-private.json',private)
    seal('observation-seal.json',('phase-observations.npz','identities.json','source-pairing-private.json','construction-start.json'))
    print('OBSERVATIONS_SEALED: zero parity, paired RNG,432x64x2')

def predict():
    verify();check_seal('observation-seal.json');write(OUT/'prediction-start.json',dict(model_calls=0,evaluator_access=False))
    original=read(SOURCE/'predictions.json');ids=read(OUT/'identities.json');arrays=np.load(OUT/'phase-observations.npz',allow_pickle=False)
    rows=[];previous={}
    for i,identity in enumerate(ids):
        scores={};flags={};unknown={};key=identity['clip_id']
        for phase in PHASES:
            s=score_frame(arrays['boxes'],arrays[phase][i]);scores[phase]=s
            strong=decide(s,T);cal=decide(s,T0);prev=previous.get((phase,key))
            held=prev is not None and abs(identity['time_s']-prev[0]-.2)<1e-6 and prev[1]
            flags.update({phase+'_cal':cal['alert'],phase+'_strong':strong['alert'],phase+'_hold':bool(strong['alert'] or held)})
            unknown[phase]=strong['unknown'];previous[phase,key]=(identity['time_s'],strong['alert'])
            if phase=='zero':
                assert identity['id']==original[i]['id']
                for k in ('score','anchors','zone_scores'):assert s[k]==original[i][k]
                assert cal==original[i]['predictions']['baseline'] and strong==original[i]['predictions']['candidate']
        rows.append({**identity,'scores':scores,'flags':flags,'unknown':unknown})
    arrays.close();write(OUT/'predictions.json',rows);seal('prediction-seal.json',('predictions.json','prediction-start.json','observation-seal.json'))
    print('PREDICTIONS_SEALED:432 paired six-arm readouts')

def evaluate():
    verify();check_seal('observation-seal.json');check_seal('prediction-seal.json')
    from return_lineage_core import masks
    rows=read(OUT/'predictions.json');labels=read(SOURCE/'frame-results.json');private=read(OUT/'source-pairing-private.json')
    geometry=read(SOURCE/'capture/evaluator/geometry.json');spec=read(SOURCE/'spec.json')['cases'];boxes=boxes45()
    changes=[];owner_summary=Counter();zone_summary=Counter();ranges=[];candidate_mean_deltas=[]
    for r,label,p,g,case in zip(rows,labels,private,geometry,spec):
        assert r['id']==label['id']==p['id'] and r['clip_id']==label['clip_id']==g['clip_id']==case['clip_id']
        r.update({k:label[k] for k in ('truth','boundary','layout_relation','layer','background','type_id')})
        nativepath=SOURCE/'capture/evaluator'/g['native_path'];assert sha(nativepath)==p['native_sha256']
        native=np.load(nativepath,allow_pickle=False);depth=sample_native(native)
        target,bg,corridor,_=masks(native,case,g);st,sb,sc=map(sample_native,(target,bg,corridor))
        for a,b in zip(p['phases']['zero'],p['phases']['half']):
            z=a['zone'];zone_summary['total']+=1
            changed=a['candidate_set_sha256']!=b['candidate_set_sha256'];zone_summary['candidate_set_changed']+=changed
            observed_changed=(a['observed']!=b['observed'] or (a['observed'] and changed))
            zone_summary['observed_set_changed']+=observed_changed
            zone_summary['observed_both']+=a['observed'] and b['observed']
            zone_summary['missingness_changed']+=a['observed']!=b['observed']
            zone_summary['reason_changed']+=a['reason']!=b['reason']
            if a['center_m'] is not None:
                candidate_mean_deltas.append(abs(a['center_m']-b['center_m']))
            if a['observed'] and b['observed']:
                delta=abs(a['range_m']-b['range_m']);ranges.append(delta)
                zone_summary['range_changed']+=delta>0;zone_summary['range_change_gt1m']+=delta>1
            for phase,q in (('zero',a),('half',b)):
                zone_summary[phase+'_winning_edge_bin']+=q['selected_bin'] in (0,79)
            owners={}
            for phase,q in (('zero',a),('half',b)):
                y0,x0,y1,x1=boxes[z];patch=depth[y0:y1,x0:x1]
                valid=np.isfinite(patch)&(patch>=.1)&(patch<8);hits=patch[valid]
                selected=np.zeros(len(hits),bool) if q['selected_bin'] is None else bin_ids(hits,phase)==q['selected_bin']
                selected &= q['observed']
                t=st[y0:y1,x0:x1][valid];back=sb[y0:y1,x0:x1][valid];co=sc[y0:y1,x0:x1][valid]
                count=int(selected.sum());tc=int((selected&t).sum());bc=int((selected&back).sum());cc=int((selected&t&co).sum())
                owner='NONE' if not count else 'TARGET' if tc==count else 'BACKGROUND' if bc==count else 'MIXED_OR_OTHER'
                owners[phase]=dict(owner=owner,count=count,target=tc,background=bc,target_corridor=cc)
            owner_summary[owners['zero']['owner']+'->'+owners['half']['owner']]+=1
            owner_summary['corridor_contributor_zones_gained']+=owners['zero']['target_corridor']==0 and owners['half']['target_corridor']>0
            owner_summary['corridor_contributor_zones_lost']+=owners['zero']['target_corridor']>0 and owners['half']['target_corridor']==0
            if changed or observed_changed or r['id'] in ('f0004','f0005','f0006','f0292','f0293','f0294'):
                changes.append(dict(id=r['id'],zone=z,truth=r['truth'],layout_relation=r['layout_relation'],
                    candidate_set_changed=changed,observed_set_changed=observed_changed,zero=a,half=b,owners=owners))
    assert len(rows)==432 and zone_summary['total']==27648
    selectors=dict(all432=lambda r:True,core288=lambda r:r['layout_relation']!='BOUNDARY',boundary144=lambda r:r['layout_relation']=='BOUNDARY',
        outside144=lambda r:r['layout_relation']=='OUTSIDE',inside_negative=lambda r:r['layout_relation']=='INSIDE' and not r['truth'])
    for key in ('layer','background'):
        for value in sorted({r[key] for r in rows}):selectors[key+':'+value]=lambda r,k=key,v=value:r['layout_relation']!='BOUNDARY' and r[k]==v
    metrics={a:{n:tally(rows,lambda r,arm=a:r['flags'][arm],s,.2,'clip_id') for n,s in selectors.items()} for a in ARMS}
    for arm in ARMS:
        for n,s in selectors.items():
            metrics[arm][n].pop('zero_return_frames');metrics[arm][n]['prediction_unknown']=sum(r['unknown'][arm.split('_')[0]] for r in rows if s(r))
    paired={}
    for readout in ('cal','strong','hold'):
        pairs={}
        for name,select in selectors.items():
            pairs[name]={k:[r['id'] for r in rows if select(r) and r['truth']==truth and r['flags']['zero_'+readout]==before and r['flags']['half_'+readout]!=before]
                for k,truth,before in (('TP_lost',True,True),('TP_recovered',True,False),('FP_added',False,False),('FP_removed',False,True))}
        paired[readout]=pairs
    groups=defaultdict(list)
    for r in rows:groups[r['clip_id']].append(r)
    events=[]
    for clip,seq in groups.items():
        pos=[r for r in seq if r['truth'] and r['layout_relation']!='BOUNDARY']
        if pos:events.append(dict(clip_id=clip,arms={a:dict(positive_frames=len(pos),alerted_frames=sum(r['flags'][a] for r in pos),first_s=next((r['time_s'] for r in pos if r['flags'][a]),None),**silence_runs(pos,[r['flags'][a] for r in pos])) for a in ARMS}))
    write(OUT/'frame-results.json',rows);write(OUT/'zone-changes-evaluator.json',changes)
    write(OUT/'results.json',dict(metrics=metrics,paired_changes=paired,core_events=events,zone_summary=dict(zone_summary),owner_summary=dict(owner_summary),
        absolute_range_delta=dict(n=len(ranges),median=float(np.median(ranges)),max=max(ranges)),
        absolute_candidate_mean_delta=dict(n=len(candidate_mean_deltas),median=float(np.median(candidate_mean_deltas)),max=max(candidate_mean_deltas)),
        clip_first={a:{clip:next((r['time_s'] for r in seq if r['flags'][a]),None) for clip,seq in groups.items()} for a in ARMS},
        scope='CONSUMED_PAIRED_PROXY_SENSITIVITY_NOT_REPAIR_OR_HARDWARE',phase_choice_frozen=True))
    seal('evaluation-seal.json',('results.json','frame-results.json','zone-changes-evaluator.json','prediction-seal.json'))
    print(json.dumps(dict(zone_summary=dict(zone_summary),core={a:{k:metrics[a]['core288'][k] for k in ('TP','FP','FN','false_segments')} for a in ARMS})))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('freeze','construct','predict','evaluate'));globals()[p.parse_args().stage]()
