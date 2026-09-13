"""Evaluator-only MZ125 attribution; never a predictor or identity assignment.

Reuses sealed MZ116 branch records and native returned-ray lineage. Exact packet
collisions concern explicitly listed non-RGB observations, not information
theoretic separability. Source identities remain evaluator-only diagnostics.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from mz115_allocation_audit import audit_target, native_truth, native_bounds, possible

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def observed_packet(row, include_imu=True):
    """Explicit observation allowlist, excludes evaluator, identity, path, time.

    Invalid or unreceived slots contribute availability, never their stale data.
    RGB content itself is not included: this is a sensor/calibration collision.
    """
    packet={k:row[k] for k in ('tof_packet_received','radar_packet_received','tof_max_targets',
        'tof_target_order','tof_model','rgb_intrinsics','camera_in_body_m','camera_pitch_deg')}
    packet['tof']=[]
    for zone in row['tof_zones']:
        valid=[]
        if row['tof_packet_received']:
            for t in zone['targets']:
                if t['status'] in ('SIM_VALID','SIM_MERGED'):
                    valid.append({k:t[k] for k in ('distance_m','range_noise_sigma_m','signal_strength_proxy','status')})
        packet['tof'].append(dict(zone_id=zone['zone_id'],theta=zone['theta_bounds_deg'],phi=zone['phi_bounds_deg'],targets=valid))
    packet['radar']=[]
    for r,a,v,valid in zip(row['radar_range_m'],row['radar_angle'],row['radar_velocity'],row['radar_valid']):
        valid=bool(row['radar_packet_received'] and valid and r is not None and a is not None)
        packet['radar'].append(dict(valid=valid,range_m=r if valid else None,angle_deg=a if valid else None,velocity=v if valid else None))
    if include_imu:packet.update({k:row[k] for k in ('imu_valid','delta_yaw','delta_pitch')})
    return packet


def collision_groups(packets, frames):
    grouped=defaultdict(list)
    for packet,frame in zip(packets,frames):
        key=json.dumps(packet,sort_keys=True,separators=(',',':'),allow_nan=False)
        grouped[key].append(frame)
    duplicates=[]
    for key,group in grouped.items():
        if len(group)>1:
            duplicates.append(dict(packet_sha256=hashlib.sha256(key.encode()).hexdigest(),
                ids=[f['id'] for f in group],truths=[f['truth'] for f in group],
                mixed_truth=len({f['truth'] for f in group})>1))
    return dict(unique_packets=len(grouped),duplicate_groups=len(duplicates),
        mixed_truth_groups=sum(g['mixed_truth'] for g in duplicates),groups=duplicates)


def run(source,out):
    assert out.resolve().is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);started=time.perf_counter()
    cap=source/'capture-v1';ana=source/'analysis-v1'
    paths=[cap/n for n in ('raw.jsonl','evaluator.jsonl','provenance.jsonl','spec.json')]
    paths += [ana/n for n in ('baseline-predictions.json','prediction-seal.json','frame-report.json')]
    before={p.as_posix():sha(p) for p in paths}
    receipt=read(cap/'receipt.json');seal=read(ana/'prediction-seal.json')
    for name in ('raw.jsonl','evaluator.jsonl','provenance.jsonl'):assert sha(cap/name)==receipt['hashes'][name]
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    assert sha(ana/'baseline-predictions.json')==seal['baseline_sha256']
    raw=rows(cap/'raw.jsonl');ev=rows(cap/'evaluator.jsonl');provenance=rows(cap/'provenance.jsonl')
    cache=read(ana/'baseline-predictions.json');report=read(ana/'frame-report.json');spec=read(cap/'spec.json')
    assert len(raw)==len(ev)==len(provenance)==len(cache)==len(report)==288
    assert [r['id'] for r in raw]==[e['id'] for e in ev]==[p['id'] for p in provenance]==[r['id'] for r in report]
    design={f['id']:f for f in spec['frames']}
    frames=[];returns=[];by_outcome=defaultdict(Counter);by_family=defaultdict(Counter)
    for row,e,prov,p,old in zip(raw,ev,provenance,cache,report):
        gt=native_truth(e);assert gt==old['truth'] and p['candidate']==old['baseline']
        outcome=('TP' if gt else 'FP') if p['candidate'] else ('FN' if gt else 'TN')
        items=[audit_target(row,e,p,item) for item in p['spatial_evidence']]
        tof=any(i['allocation_support'] for i in items);radar=bool(p['common_radar']);guard=bool(p['guard_events'])
        assert bool(p['candidate'])==bool(tof or radar or guard)
        active='+'.join(k for k,v in [('RADAR',radar),('GUARD',guard),('TOF',tof)] if v) or 'NONE'
        by_outcome[outcome][active]+=1;by_family[e['family']+' / '+outcome][active]+=1
        objects={row['episode_id']+'/'+o['name']:o for o in e['native_bounds']}
        roles={row['episode_id']+'/'+o['name']:o['source_role'] for o in design[row['id']]['objects']}
        radar_records=[]
        for slot,(r,a,valid,origin) in enumerate(zip(row['radar_range_m'],row['radar_angle'],row['radar_valid'],prov['radar_slots'])):
            if not (row['radar_packet_received'] and valid and r is not None and a is not None):continue
            ang=math.radians(a+p['integrated_yaw_deg']);x,y=r*math.cos(ang),r*math.sin(ang)
            actor=None if origin is None else origin.get('actor_id');obj=objects.get(actor)
            radar_records.append(dict(slot=slot,range_m=r,angle_deg=a,nominal_xy_m=[x,y],
                nominal_corridor_point=.2<=x<=3.6 and abs(y)<=.3,
                evaluator_origin_kind=None if origin is None else origin.get('kind'),
                evaluator_actor_id=actor,evaluator_role=roles.get(actor),
                actor_intersects_corridor=None if obj is None else possible(native_bounds(obj,e['body_origin_m'])),
                attribution_limit='Common Radar is sealed frame-level aggregate; this slot is not claimed to be its triggering slot'))
        active_items=[i for i in items if i['allocation_support']]
        frame_targets=[]
        for item in items:
            contributing=[c for c in item['contributors']]
            native_actors=sorted({c['actor_id'] for c in contributing if c['actor_id'] is not None})
            ret=dict(id=row['id'],outcome=outcome,family=e['family'],zone_id=item['zone_id'],slot=item['target_slot'],
                status=item['status'],state=item['state'],coarse_support=item['coarse_support'],
                active_support=item['allocation_support'],narrowed=item['narrowed'],
                source_actor_ids=native_actors,source_roles=[roles.get(a,'ENVIRONMENT_OR_UNMATCHED') for a in native_actors],
                multi_actor=len(native_actors)>1,contributor_count=len(contributing),
                hazardous_actor_contributors=sum(c['native_actor_intersects_corridor'] for c in contributing),
                actual_corridor_hit_contributors=sum(c['native_hit_inside_corridor'] for c in contributing),
                unknown_actor_contributors=sum(c['actor_status']!='KNOWN_NATIVE_ACTOR' for c in contributing),
                dropped_hazard_actor_contributors=item['dropped_hazard_actor_contributor_count'],
                range_m=row['tof_zones'][item['zone_id']]['targets'][item['target_slot']]['distance_m'],
                contract_violations=item['contract_violations'])
            assert row['tof_zones'][item['zone_id']]['zone_id']==item['zone_id']
            returns.append(ret);frame_targets.append(ret)
        f=dict(id=row['id'],episode_id=row['episode_id'],time_s=row['time_s'],family=e['family'],truth=gt,
            baseline_alert=bool(p['candidate']),outcome=outcome,active_branches=active,
            common_radar=radar,guard=guard,tof_support=tof,
            active_tof_returns=len(active_items),total_tof_returns=len(items),
            active_tof_states=dict(Counter(i['state'] for i in active_items)),
            active_tof_status=dict(Counter(i['status'] for i in active_items)),
            active_tof_mixed_actor_returns=sum(t['active_support'] and t['multi_actor'] for t in frame_targets),
            active_tof_actual_corridor_hits=sum(t['actual_corridor_hit_contributors'] for t in frame_targets if t['active_support']),
            active_tof_hazardous_actor_hits=sum(t['hazardous_actor_contributors'] for t in frame_targets if t['active_support']),
            active_tof_coarse_only_count=sum(t['coarse_support'] and not t['active_support'] for t in frame_targets),
            active_tof_no_hazard_actor_returns=sum(t['active_support'] and t['hazardous_actor_contributors']==0 for t in frame_targets),
            radar=radar_records,targets=frame_targets)
        frames.append(f)
    pairs=[];indexed={f['id']:f for f in frames}
    for pair in spec['pairs']:
        members=[[f for f in frames if f['episode_id']==ep] for ep in pair['episodes']]
        assert len(members[0])==len(members[1])==12
        for a,b in zip(*members):
            assert a['time_s']==b['time_s'] and a['truth']!=b['truth']
            pos,neg=(a,b) if a['truth'] else (b,a)
            pairs.append(dict(pair_id=pair['pair_id'],time_s=a['time_s'],positive_id=pos['id'],negative_id=neg['id'],
                positive_outcome=pos['outcome'],negative_outcome=neg['outcome'],
                positive_branches=pos['active_branches'],negative_branches=neg['active_branches'],
                positive_tof_returns=pos['total_tof_returns'],negative_tof_returns=neg['total_tof_returns']))
    packets=[observed_packet(r) for r in raw]
    collisions=dict(full_sensor_imu_calibration=collision_groups(packets,frames),
        sensor_calibration_without_imu=collision_groups([observed_packet(r,False) for r in raw],frames),
        radar_packet_only=collision_groups([p['radar'] for p in packets],frames))
    summaries={}
    for outcome in ('TP','FP','FN','TN'):
        fs=[f for f in frames if f['outcome']==outcome];ts=[t for t in returns if t['outcome']==outcome];at=[t for t in ts if t['active_support']]
        summaries[outcome]=dict(frames=len(fs),branches=dict(by_outcome[outcome]),returns=len(ts),active_returns=len(at),
            active_status=dict(Counter(t['status'] for t in at)),active_state=dict(Counter(t['state'] for t in at)),
            mixed_actor_active_returns=sum(t['multi_actor'] for t in at),
            active_returns_with_hazard_actor=sum(t['hazardous_actor_contributors']>0 for t in at),
            active_returns_with_actual_corridor_hit=sum(t['actual_corridor_hit_contributors']>0 for t in at),
            active_returns_without_hazard_actor=sum(t['hazardous_actor_contributors']==0 for t in at),
            no_tof_return_frames=sum(f['total_tof_returns']==0 for f in fs),
            active_returns_with_unknown_actor=sum(t['unknown_actor_contributors']>0 for t in at))
    examples=[]
    for family in sorted({f['family'] for f in frames}):
        eligible=[f for f in frames if f['family']==family and f['outcome']=='FP']
        groups=defaultdict(list)
        for f in eligible:groups[f['active_branches']].append(f)
        for branch,group in groups.items():
            chosen=group[len(group)//2];pair=next(p for p in pairs if p['negative_id']==chosen['id'])
            examples.append(dict(family=family,branch=branch,false_frame=chosen,paired_positive=indexed[pair['positive_id']],
                selection='Middle ordered observed FP in each existing family/branch group; evaluator-only example selection'))
    select_backend('scalar-scoring',cpu=BackendCandidate('python-scalar','cpu',lambda:dict(Counter(f['outcome'] for f in frames)),
        lambda _:DeviceObservation('cpu','host CPU','Python '+sys.version.split()[0])),record_path=out/'backend.json',
        capabilities={'cpu_reason':'TASK_NOT_GPU_SUITABLE'})
    assert {p.as_posix():sha(p) for p in paths}==before
    assert summaries['TP']['frames']==139 and summaries['FP']['frames']==116
    result=dict(status='CONSUMED_EVALUATOR_ATTRIBUTION_COMPLETE',frames=288,returned_tof_targets=len(returns),
        by_outcome=summaries,by_family={k:dict(v) for k,v in by_family.items()},
        collisions={k:{q:v for q,v in x.items() if q!='groups'} for k,x in collisions.items()},
        contracts_violations=sum(len(t['contract_violations']) for t in returns),
        elapsed_seconds=time.perf_counter()-started,cpu_reason='TASK_NOT_GPU_SUITABLE',
        input_hashes=before,code_sha256=sha(Path(__file__)),
        limitation='Native identity/lineage is evaluator authority; absence of exact packet collision does not prove useful observable separability; RGB pixels absent from collision comparison')
    for name,obj in [('frames',frames),('returns',returns),('pairs',pairs),('examples',examples),('collisions',collisions),('summary',result)]:write(out/(name+'.json'),obj)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),resources='Foreground scalar process exits; no worker or GPU'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
