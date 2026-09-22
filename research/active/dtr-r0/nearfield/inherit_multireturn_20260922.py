"""One inherited, frozen hypothetical-sensing Development contrast."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from query_occupancy_data import read, write, sha, new_stage_directory, observation_tokens
from multireturn_pilot_20260920 import simulate_two, readouts, T, MODES
from ba_camera_corridor import sample_native
from tof_fov45_core import boxes45
from tof_corridor_calibration import score_frame
from run_full_event_20260920 import apply_hold

ROOT = Path(__file__).resolve().parents[4]
NEAR = Path(__file__).parent
DEFAULT = ROOT/'artifacts.local/evidence/ba-inherit-multireturn-20260922'
QUERY = ROOT/'artifacts.local/evidence/ba-query-occupancy-20260922'
ARMS = tuple(m+s for m in MODES for s in ('', '_hold'))
CODE = ('inherit_multireturn_20260922.py', 'multireturn_pilot_20260920.py',
        'tof_fov45_core.py', 'tof_corridor_calibration.py', 'ba_camera_corridor.py',
        'run_full_event_20260920.py', 'full_event_metrics_20260920.py',
        'ba_camera_corridor_metrics.py', 'query_occupancy_data.py', 'inherit_multireturn_audit.py')


def stage(root, suffix):
    return root.with_name(root.name+'-'+suffix)


def seal(folder, name, files, **extra):
    write(folder/name, dict(status='PASS', hashes={f: sha(folder/f) for f in files}, **extra))


def verify(folder, name):
    data = read(folder/name)
    assert data['status'] == 'PASS'
    for f, digest in data['hashes'].items():
        assert sha(folder/f) == digest, f
    return data


def check_plan(root):
    p = read(root/'plan/protocol.json')
    for n, digest in p['code'].items():
        assert sha(NEAR/n) == digest, n
    assert sha(root/'plan/protocol-before-run.md') == p['protocol_text_sha256']
    assert p['threshold'] == T
    return p


def plan(root):
    new_stage_directory(root/'plan')
    protocol = NEAR/'INHERIT_MULTIRETURN_PROTOCOL_20260922.md'
    (root/'plan/protocol-before-run.md').write_bytes(protocol.read_bytes())
    write(root/'plan/protocol.json', dict(id='ba-inherit-multireturn-20260922',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), threshold=T,
        frames=576, readouts=ARMS, input_source='ue-query-occupancy-20260922',
        protocol_text_sha256=sha(protocol), code={n:sha(NEAR/n) for n in CODE},
        backend='TASK_NOT_GPU_SUITABLE', native_role='hypothetical_sensor_generation_then_evaluator_only'))
    def inp(alias, path, role):
        return dict(alias=alias, path=str(path), role=role, purpose='consumed-development-'+alias)
    shared = [inp('plan',root/'plan','configuration')]
    inputs = dict(
        construct=shared+[
            inp('source_plan',QUERY/'plan','configuration'),
            inp('source_native',stage(QUERY,'capture')/'evaluator','evaluator'),
            inp('source_observations',stage(QUERY,'prepared')/'observations','observation'),
            inp('source_seal',stage(QUERY,'prepared')/'materialization.json','configuration')],
        predict=shared+[inp('observations',stage(root,'observations')/'public','observation')],
        evaluate=shared+[
            inp('observations',stage(root,'observations')/'public','observation'),
            inp('lineage',stage(root,'observations')/'private','evaluator'),
            inp('predictions',stage(root,'predictions')/'public','observation'),
            inp('source_plan',QUERY/'plan','configuration'),
            inp('source_native',stage(QUERY,'capture')/'evaluator','evaluator'),
            inp('source_observations',stage(QUERY,'prepared')/'observations','observation'),
            inp('source_labels',stage(QUERY,'prepared')/'labels','evaluator'),
            inp('source_seal',stage(QUERY,'prepared')/'materialization.json','configuration')])
    for name in inputs:
        dest = stage(root,dict(construct='observations',predict='predictions',evaluate='evaluated')[name])
        spec = dict(schema='blindassist-asset-run-v1',id='inherit-multireturn-20260922-'+name,
            route='ue-inherit-multireturn',question='Does fixed extra return sensing improve broader-distance current and event support?',
            evaluator='research/active/dtr-r0/nearfield/inherit_multireturn_20260922.py',
            evidence_boundary='Consumed controlled Development; hypothetical extra sensing, not same-input or hardware gain',
            reuse=dict(mode='development',query='BODY HEAD native depth multi return query occupancy complete event'),
            command=[sys.executable,str(Path(__file__)),name,'--root',str(root),'--result','{{output:result}}'],
            inputs=inputs[name], outputs=[dict(alias='result',path=str(dest/'result.json'),role='result',required=True)],
            result_output='result',parameters=dict(stage=name))
        write(root/f'{name}-run-spec.json',spec)
    return dict(status='PASS',stage='plan',frames=576)


def construct(root):
    check_plan(root)
    out = stage(root,'observations')
    new_stage_directory(out)
    (out/'public').mkdir(); (out/'private').mkdir()
    prepared, cap = stage(QUERY,'prepared'), stage(QUERY,'capture')
    source_seal = read(prepared/'materialization.json')
    assert source_seal['status'] == 'PASS' and source_seal['frames'] == 1728
    for n in ('observations/tof.npy','observations/identities.json'):
        assert sha(prepared/n) == source_seal['hashes'][n]
    spec = read(QUERY/'plan/spec.json')
    assert sha(QUERY/'plan/spec.json') == source_seal['source_spec_sha256']
    geo = read(cap/'evaluator/geometry.json')
    ids = read(prepared/'observations/identities.json')
    tof = np.load(prepared/'observations/tof.npy',mmap_mode='r',allow_pickle=False)
    chosen = [r for r in ids if r['split']=='evaluation']
    assert len(chosen)==576 and len({r['base_group_id'] for r in chosen})==16
    public, private, identities = defaultdict(list), [], []
    for k, row in enumerate(chosen):
        i = row['index']; case, g = spec['cases'][i], geo[i]
        assert row['id']==case['name']==g['id'] and g['sample_index']==i
        path = cap/'evaluator'/g['native_path']
        assert sha(path)==g['native_sha256']
        native = np.load(path,allow_pickle=False)
        assert native.shape==(360,640)
        seed = 'query-occupancy/'+case['sensor_noise_key']
        packet, lineage = simulate_two(sample_native(native),seed,boxes45())
        assert np.array_equal(observation_tokens(packet['ranges'][:,0],boxes45()),tof[i]), row['id']
        for key, value in packet.items():
            public[key].append(value)
        identities.append({key:row[key] for key in ('index','id','clip_id','frame_in_clip','time_s')})
        private.append(dict(id=row['id'],seed=seed,native_sha256=g['native_sha256'],zones=lineage))
        if k%144==0:
            print('CONSTRUCT',k,'/576',flush=True)
    np.savez_compressed(out/'public/observations.npz',boxes=boxes45(),**{k:np.stack(v) for k,v in public.items()})
    write(out/'public/identities.json',identities)
    write(out/'private/lineage.json',private)
    write(out/'private/input-hashes.json',{str(p):sha(p) for p in (
        QUERY/'plan/spec.json',cap/'evaluator/geometry.json',prepared/'materialization.json',
        prepared/'observations/tof.npy',prepared/'observations/identities.json')})
    seal(out/'public','observation-seal.json',('observations.npz','identities.json'),
         frames=576,protocol_sha256=sha(root/'plan/protocol.json'),truth_opened=False)
    seal(out/'private','lineage-seal.json',('lineage.json','input-hashes.json'))
    return dict(status='PASS',frames=576,first_slot_exact_parity=576,
                observation_seal_sha256=sha(out/'public/observation-seal.json'))


def prediction_rows(identities, boxes, ranges):
    rows, previous = [], {}
    for i, meta in enumerate(identities):
        values = ranges[i]
        decisions, scores = readouts(boxes,values)
        choose_second = np.isfinite(values[:,1]) & (~np.isfinite(values[:,0]) | (values[:,1]<values[:,0]))
        closest_score = score_frame(boxes,np.where(choose_second,values[:,1],values[:,0]))
        def triggers(score, slots):
            factors = {s['zone']:s for s in score['zone_scores']}
            return [dict(zone=a['zone'],slot=int(slots[a['zone']])) for a in score['anchors']
                if a['definite'] or (a['possible'] and factors[a['zone']]['joint']>=T)]
        active = dict(strongest=triggers(scores[0],np.zeros(64,int)),
            closest_exported=triggers(closest_score,choose_second.astype(int)),
            two_returns=triggers(scores[0],np.zeros(64,int))+triggers(scores[1],np.ones(64,int)))
        flags, predictions = {}, {}
        for mode in MODES:
            assert decisions[mode]['alert']==bool(active[mode])
            current = decisions[mode]['alert']
            key = meta['clip_id'],mode
            held, _, previous[key] = apply_hold(current,previous.get(key),meta['time_s'])
            for name, flag in ((mode,current),(mode+'_hold',held)):
                unknown = decisions[mode]['unknown']
                flags[name] = bool(flag)
                predictions[name]=dict(alert=bool(flag),unknown=bool(unknown),ambiguous=bool(flag and unknown))
        rows.append({**meta,'flags':flags,'predictions':predictions,'decisions':decisions,
            'triggers':active,'second_slot_present':int(np.isfinite(values[:,1]).sum())})
    return rows


def predict(root):
    check_plan(root)
    source = stage(root,'observations')/'public'
    obs_seal = verify(source,'observation-seal.json')
    assert obs_seal['protocol_sha256']==sha(root/'plan/protocol.json')
    with np.load(source/'observations.npz',allow_pickle=False) as data:
        rows = prediction_rows(read(source/'identities.json'),data['boxes'],data['ranges'])
    out = stage(root,'predictions')
    new_stage_directory(out); (out/'public').mkdir()
    write(out/'public/predictions.json',rows)
    seal(out/'public','prediction-seal.json',('predictions.json',),
         protocol_sha256=sha(root/'plan/protocol.json'),
         observation_seal_sha256=sha(source/'observation-seal.json'),truth_opened=False,frames=len(rows))
    return dict(status='PASS',frames=len(rows),prediction_seal_sha256=sha(out/'public/prediction-seal.json'))


def contributor_mask(native, camera, objects):
    # Evaluator-only independent camera projection on the original native grid.
    yy,xx=np.mgrid[:360,:640]
    z=np.asarray(native,np.float64)
    focal=640/(2*np.tan(np.deg2rad(50)))
    x=(xx+.5-320)/focal*z; y=(yy+.5-180)/focal*z
    corridor=np.isfinite(z)&(z>=.3)&(z<=3)&(x>=-.3)&(x<=.3)&(y>=-.2)&(y<=.9)
    target=np.zeros(z.shape,bool)
    for obj in objects:
        if obj['name']!='target':
            continue
        cx,cy,cz=obj['render_bounds_center_m']; hx,hy,hz=obj['render_bounds_extent_m']
        target |= ((x>=cy-camera['y']-hy-.02)&(x<=cy-camera['y']+hy+.02)&
            (y>=camera['z']-cz-hz-.02)&(y<=camera['z']-cz+hz+.02)&
            (z>=cx-camera['x']-hx-.02)&(z<=cx-camera['x']+hx+.02))
    return sample_native(target & corridor).ravel()


def summarize(rows):
    from ba_camera_corridor_metrics import evaluate_rows
    from full_event_metrics_20260920 import _clip_metrics
    result=evaluate_rows(rows,arms=ARMS)
    clips=defaultdict(list)
    for r in rows:
        clips[r['clip_id']].append(r)
    for arm in ARMS:
        details=[_clip_metrics(sorted(c,key=lambda r:r['time_s']),arm,.2) for c in clips.values()]
        result['arms'][arm]['complete_clip_details']=details
        events=[d['event'] for d in details if d['event']]
        assert len(events)==result['arms'][arm]['event_count']
        result['arms'][arm]['complete_summary']=dict(
            internal_interruptions=sum(e['internal_interruption_count'] for e in events),
            internal_silent_frames=sum(e['internal_silent_frames'] for e in events),
            exit_carryover_sampled_s=sum(e['postexit_carryover_sampled_s'] or 0 for e in events),
            release_right_censored=sum(e['release_right_censored'] for e in events),
            alert_episode_count=sum(d['alert_episode_count'] for d in details),
            immediate_onsets=sum(e['first_in_event_alert_delay_s']==0 for e in events),
            detected_delay_median_s=float(np.median([e['first_in_event_alert_delay_s'] for e in events if e['detected']])) if any(e['detected'] for e in events) else None,
            detected_delay_max_s=max((e['first_in_event_alert_delay_s'] for e in events if e['detected']),default=None))
    return result


def evaluate(root):
    check_plan(root)
    obs=stage(root,'observations'); pred=stage(root,'predictions')/'public'
    verify(obs/'public','observation-seal.json'); verify(obs/'private','lineage-seal.json')
    ps=verify(pred,'prediction-seal.json')
    assert ps['protocol_sha256']==sha(root/'plan/protocol.json')
    assert ps['observation_seal_sha256']==sha(obs/'public/observation-seal.json')
    for path,digest in read(obs/'private/input-hashes.json').items():
        assert sha(path)==digest
    prepared=stage(QUERY,'prepared'); cap=stage(QUERY,'capture')
    source_seal=read(prepared/'materialization.json')
    assert sha(prepared/'labels/evaluation.npz')==source_seal['hashes']['labels/evaluation.npz']
    with np.load(prepared/'labels/evaluation.npz',allow_pickle=False) as a:
        indices=a['indices']; truth=(a['classes'][:,[1,4]]<6).any(1); valid=a['valid'][:,[1,4]].all(1)
    assert valid.all()
    ids=read(prepared/'observations/identities.json'); spec=read(QUERY/'plan/spec.json')['cases']
    geo=read(cap/'evaluator/geometry.json'); lineage=read(obs/'private/lineage.json')
    rows=read(pred/'predictions.json')
    assert len(rows)==len(indices)==576
    for k,(r,lin) in enumerate(zip(rows,lineage)):
        i=int(indices[k]); meta=ids[i]; g=geo[i]; case=spec[i]
        assert r['index']==i and r['id']==lin['id']==meta['id']==g['id']==case['name']
        assert r['decisions']['strongest']==meta['baseline']
        r.update(truth=bool(truth[k]),boundary=meta['layout_relation']=='BOUNDARY',
                 **{key:meta[key] for key in ('type_id','layer','layout_relation','base_group_id')})
        path=cap/'evaluator'/g['native_path']; assert sha(path)==g['native_sha256']==lin['native_sha256']
        support=contributor_mask(np.load(path,allow_pickle=False),case['camera'],g['objects'])
        r['native_target_corridor']=[sum(int(support[z['indices'][slot]].sum()) for z in lin['zones']) for slot in (0,1)]
        r['native_trigger_contributors']={mode:sum(int(support[lin['zones'][t['zone']]['indices'][t['slot']]].sum()) for t in r['triggers'][mode]) for mode in MODES}
        r['native_second_trigger_contributors']={mode:sum(int(support[lin['zones'][t['zone']]['indices'][1]].sum()) for t in r['triggers'][mode] if t['slot']==1) for mode in MODES}
    metrics=summarize(rows)
    strata={key:{value:summarize([r for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})}
            for key in ('layout_relation','layer','type_id','base_group_id')}
    strata['Core']=summarize([r for r in rows if r['layout_relation']!='BOUNDARY'])
    def changes(group,base,arm):
        return {name:[r['id'] for r in group if r['truth']==positive and r['flags'][base]==before and r['flags'][arm]!=before]
                for name,positive,before in (('TP_lost',True,True),('FN_recovered',True,False),('FP_added',False,False),('FP_removed',False,True))}
    paired={scope:{a:changes(group,'strongest'+('_hold' if a.endswith('_hold') else ''),a) for a in ARMS if not a.startswith('strongest')}
            for scope,group in (('all',rows),('Core',[r for r in rows if r['layout_relation']!='BOUNDARY']),('Boundary',[r for r in rows if r['boundary']]))}
    retention={}
    core=strata['Core']['arms']
    for arm in ARMS:
        if arm.startswith('strongest'): continue
        mode=arm.removesuffix('_hold'); base='strongest'+('_hold' if arm.endswith('_hold') else '')
        b,c=core[base],core[arm]
        be={e['clip_id']:e for e in b['events']}; ce={e['clip_id']:e for e in c['events']}
        earlier=[k for k in be if ce[k]['detected'] and (not be[k]['detected'] or ce[k]['first_alert_time_s']<be[k]['first_alert_time_s']-1e-8)]
        delayed=[k for k in be if be[k]['detected'] and (not ce[k]['detected'] or ce[k]['first_alert_time_s']>be[k]['first_alert_time_s']+1e-8)]
        native=[r['id'] for r in rows if r['id'] in paired['Core'][arm]['FN_recovered'] and r['flags'][mode] and r['native_second_trigger_contributors'][mode]>0]
        checks=dict(useful_gain=bool(native or earlier or c['complete_summary']['internal_interruptions']<b['complete_summary']['internal_interruptions']),
            no_core_tp_loss=not paired['Core'][arm]['TP_lost'],no_core_event_loss=c['detected_events']>=b['detected_events'],
            no_core_delayed_onset=not delayed,no_core_fp_increase=c['frames']['all_known']['FP']<=b['frames']['all_known']['FP'],
            no_core_segment_increase=c['false_alert_segment_count']<=b['false_alert_segment_count'],
            no_core_false_duration_increase=c['false_alert_sampled_duration_s']<=b['false_alert_sampled_duration_s']+1e-8,
            no_core_exit_carryover_increase=c['complete_summary']['exit_carryover_sampled_s']<=b['complete_summary']['exit_carryover_sampled_s']+1e-8)
        retention[arm]=dict(checks=checks,passed=all(checks.values()),native_backed_recovery=native,earlier_events=earlier,delayed_events=delayed)
    report=dict(status='PASS',scope='CONSUMED_SIMULATION_HYPOTHETICAL_EXTRA_SENSING',metrics=metrics,strata=strata,
        paired=paired,retention=retention,availability=dict(zone_frames=576*64,second_returns=sum(r['second_slot_present'] for r in rows),
        frames_with_second=sum(r['second_slot_present']>0 for r in rows),
        positive_new_corridor_support=[r['id'] for r in rows if r['truth'] and r['native_target_corridor'][0]==0 and r['native_target_corridor'][1]>0]),
        closest_two_differences={s:[r['id'] for r in rows if r['flags']['closest_exported'+s]!=r['flags']['two_returns'+s]] for s in ('','_hold')},
        all_native_backed_current_recoveries={m:[r['id'] for r in rows if r['truth'] and not r['flags']['strongest'] and r['flags'][m] and r['native_second_trigger_contributors'][m]>0] for m in MODES[1:]},
        prediction_seal_sha256=sha(pred/'prediction-seal.json'))
    from inherit_multireturn_audit import audit
    with np.load(obs/'public/observations.npz',allow_pickle=False) as public:
        audited=audit(rows,lineage,spec,geo,cap/'evaluator',report,public)
    out=stage(root,'evaluated'); new_stage_directory(out)
    write(out/'frame-results.json',rows); write(out/'metrics.json',report)
    write(out/'independent-audit.json',audited)
    seal(out,'evaluation-seal.json',('frame-results.json','metrics.json','independent-audit.json'))
    return dict(status='PASS',frames=576,retention={k:v['passed'] for k,v in retention.items()},
        arms={a:dict(**metrics['arms'][a]['frames']['all_known'],events=metrics['arms'][a]['detected_events'],
             false_segments=metrics['arms'][a]['false_alert_segment_count']) for a in ARMS},
        evaluation_seal_sha256=sha(out/'evaluation-seal.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('stage',choices=('plan','construct','predict','evaluate'))
    p.add_argument('--root',type=Path,default=DEFAULT); p.add_argument('--result',type=Path)
    args=p.parse_args(); start=time.perf_counter()
    result=globals()[args.stage](args.root)
    result.update(elapsed_s=time.perf_counter()-start,backend='TASK_NOT_GPU_SUITABLE',python=sys.executable)
    if args.result: write(args.result,result)
    import json
    print(json.dumps(result))
