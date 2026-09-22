"""Frozen public history extraction, followed by separate consumed-label ceiling."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys
import time

REPO = Path(__file__).resolve().parents[5]
WORK = REPO / 'artifacts.local/work'
CAP = WORK / 'corridor-public-single-20260917/source/returned-v1/capture-v1'
REP = WORK / 'corridor-representation-20260918'
WINDOWS = (.5, 1., 1.5)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def median(v):
    return st.median(v) if v else None


def slope(pairs):
    if len(pairs) < 2:
        return None
    x, y = zip(*pairs)
    mx, my = st.mean(x), st.mean(y)
    den = sum((v-mx)**2 for v in x)
    return sum((a-mx)*(b-my) for a,b in pairs)/den if den else None


def prepare(rows, predictions):
    result = []
    episode, yaw, previous = None, 0., None
    for r, p in zip(rows, predictions):
        assert r['id'] == p['id']
        if r['episode_id'] != episode:
            episode, yaw, previous = r['episode_id'], 0., None
        assert previous is None or r['time_s'] > previous
        dt = r['time_s']-previous if previous is not None else None
        if not r['imu_valid'] or not finite(r['delta_yaw']):
            yaw = None
        elif yaw is not None:
            yaw += r['delta_yaw']
        radar = []
        if r['radar_packet_received']:
            for d,a,v,ok in zip(r['radar_range_m'],r['radar_angle'],r['radar_velocity'],r['radar_valid']):
                if ok and finite(d) and d>0 and finite(a):
                    radar.append(dict(range=d, bearing=a+yaw if yaw is not None else None,
                                      velocity=v if finite(v) else None))
        targets = [t for z in r['tof_zones'] for t in z['targets']]
        valid = [t for t in targets if t['status']=='SIM_VALID' and finite(t['distance_m']) and t['distance_m']>0]
        result.append(dict(id=r['id'], episode=r['episode_id'], t=r['time_s'],
                           radar=radar, radar_packet=r['radar_packet_received'],
                           tof_packet=r['tof_packet_received'], tof_valid=len(valid) if r['tof_packet_received'] else 0,
                           tof_invalid=len(targets)-len(valid), free_state='UNKNOWN',
                           yaw_delta=r['delta_yaw'] if r['imu_valid'] else None,
                           pitch_delta=r['delta_pitch'] if r['imu_valid'] else None,
                           angular_velocity=math.hypot(r['delta_yaw'],r['delta_pitch'])/dt if dt and r['imu_valid'] else None,
                           pure_head_motion='UNKNOWN', raw_score=p['scores']['raw'], astar_score=p['scores']['multi']))
        previous = r['time_s']
    return result


def radar_features(history, now, sector=False):
    frames = [[p for p in r['radar'] if p['bearing'] is not None and
               (not sector or (abs(p['bearing'])<=15 and p['range']<=3.6))] for r in history]
    pts = [p for frame in frames for p in frame]
    ranges = [p['range'] for p in pts]
    velocities = [p['velocity'] for p in pts if p['velocity'] is not None]
    bearings = [p['bearing'] for p in pts]
    nearest = [(r['t'],min(p['range'] for p in frame)) for r,frame in zip(history,frames) if frame]
    cells = defaultdict(list)
    for i,(r,frame) in enumerate(zip(history,frames)):
        grouped = defaultdict(list)
        for p in frame:
            grouped[(math.floor((p['bearing']+5)/10),math.floor(p['range']/.5))].append(p['range'])
        for cell,values in grouped.items():
            cells[cell].append((i,r['t'],st.median(values)))
    longest = 0
    for observations in cells.values():
        run, last = 0, -2
        for i,_,_ in observations:
            run = run+1 if i==last+1 else 1
            longest, last = max(longest,run), i
    dominant = max(sorted(cells),key=lambda c:len(cells[c])) if cells else None
    track = cells[dominant] if dominant is not None else []
    sign = Counter(0 if v==0 else (1 if v>0 else -1) for v in velocities)
    vals = [v for _,_,v in track]
    return dict(valid_return_count=len(pts) if history else None,
                nearest_range=min(ranges) if ranges else None, median_range=median(ranges),
                range_slope=slope(nearest), doppler_median=median(velocities),
                doppler_sign_consistency=max(sign.values())/len(velocities) if velocities else None,
                bearing_spread=max(bearings)-min(bearings) if bearings else None,
                consecutive_support=longest if history else None,
                time_since_last_support=now-nearest[-1][0] if nearest else None,
                occupancy_density=max(map(len,cells.values()))/len(history) if cells else (0. if history else None),
                cell_range_slope=slope([(t,v) for _,t,v in track]),
                cell_range_mad=median([abs(v-st.median(vals)) for v in vals]) if vals else None)


def features(history, now):
    out = {}
    for prefix,sector in [('radar',False),('sector',True)]:
        out.update({prefix+'_'+k:v for k,v in radar_features(history,now,sector).items()})
    out.update(tof_valid_mean=st.mean(r['tof_valid'] for r in history) if history else None,
               tof_packet_ratio=st.mean(r['tof_packet'] for r in history) if history else None,
               tof_invalid_mean=st.mean(r['tof_invalid'] for r in history) if history else None)
    for name in ('yaw_delta','pitch_delta','angular_velocity'):
        vals=[r[name] for r in history if r[name] is not None]
        out[name+'_abs_mean']=st.mean(abs(v) for v in vals) if vals else None
    for arm in ('raw','astar'):
        vals=[r[arm+'_score'] for r in history]
        out[arm+'_score_mean']=st.mean(vals) if vals else None
        out[arm+'_score_slope']=slope([(r['t'],r[arm+'_score']) for r in history])
    return out


def ceiling(records, names):
    # Descriptive oracle cuts, never transferred into predictions.
    target=tuple(r['truth'] for r in records)
    cuts=[]; missing=[]; distributions={}
    for name in names:
        v=[r['features'].get(name) for r in records]
        distributions[name]=dict(TP=[x for x,r in zip(v,records) if r['truth']],
                                 FP=[x for x,r in zip(v,records) if not r['truth']])
        if not all(finite(x) for x in v):
            missing.append(name); continue
        values=sorted(set(v))
        for a,b in zip(values,values[1:]):
            threshold=(a+b)/2
            for op in ('le','gt'):
                mask=tuple(x<=threshold if op=='le' else x>threshold for x in v)
                cuts.append(dict(feature=name,op=op,threshold=threshold,mask=mask))
    singles=[c for c in cuts if c['mask']==target]
    pairs=[]; n=0
    for a,b in itertools.combinations(cuts,2):
        if a['feature']==b['feature']:continue
        n+=1
        if tuple(x and y for x,y in zip(a['mask'],b['mask']))==target:
            pairs.append([a,b])
    return dict(rows=len(records),single_cuts_tested=len(cuts),pairs_tested=n,
                perfect_single=singles,perfect_pair=pairs,missing_features=missing,distributions=distributions)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();art=(REPO/'artifacts.local').resolve()
    assert out.is_relative_to(art) and out!=art and not out.exists()
    out.mkdir(parents=True)
    start=time.perf_counter()
    rows=[json.loads(x) for x in (CAP/'raw.jsonl').read_text().splitlines() if x.strip()]
    predictions=read(REP/'predictions.json');seal=read(REP/'prediction-seal.json'); models=read(REP/'model-seal.json')['models']
    assert sha(REP/'predictions.json')==seal['predictions_sha256']
    assert sha(REP/'model-seal.json')==seal['model_seal_sha256']
    receipt=read(CAP/'receipt.json')
    assert sha(CAP/'raw.jsonl')==receipt['hashes']['raw.jsonl']
    assert len(rows)==len(predictions)==len({r['id'] for r in rows})==288
    for p in predictions:
        for a in ('raw','multi'):
            assert p['flags'][a]==(p['scores'][a]>=models[a]['threshold'])
    prepared=prepare(rows,predictions)
    selected=[]
    for r,p in zip(prepared,predictions):
        if p['flags']['multi'] or not p['flags']['raw']:continue
        record=dict(id=r['id'],episode=r['episode'],t=r['t'],current=r,features={})
        record['features'].update({'current_'+k:v for k,v in features([r],r['t']).items()})
        record['histories']={}
        for w in WINDOWS:
            history=[h for h in prepared if h['episode']==r['episode'] and r['t']-w<=h['t']<r['t']]
            key=str(w);record['histories'][key]=history
            record['features'].update({key+'_'+k:v for k,v in features(history,r['t']).items()})
        selected.append(record)
    write(out/'public-features.json',selected)
    inputs=[CAP/'raw.jsonl',REP/'predictions.json',REP/'model-seal.json',Path(__file__),Path(__file__).with_name('MZ179_PROTOCOL_20260918.md')]
    write(out/'feature-seal.json',dict(inputs={str(p):sha(p) for p in inputs},feature_sha256=sha(out/'public-features.json')))
    # Only now load consumed labels and privileged source pose for applicability.
    cases=read(REP/'cases.json');spec=read(CAP/'spec.json')
    assert [r['id'] for r in cases]==[r['id'] for r in rows]
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    cm={r['id']:r for r in cases};fm={r['id']:r for r in spec['frames']}
    for r in selected:
        c=cm[r['id']];r.update(truth=c['truth'],stratum=c['stratum'],audit={})
        assert c['ablation']==predictions[[p['id'] for p in predictions].index(r['id'])]
        for w,history in r['histories'].items():
            allr=history+[r['current']];poses=[fm[h['id']]['camera'] for h in allr]
            displacement=max(math.dist([p[k] for k in ('x','y','z')],[q[k] for k in ('x','y','z')]) for p in poses for q in poses)
            support=[h['t'] for h in history if cm[h['id']]['sampled_witness']]
            r['audit'][w]=dict(history_frames=len(history),left_censored=r['t']<float(w),
                history_span_s=history[-1]['t']-history[0]['t'] if history else None,
                native_support_frames=len(support),time_since_native_support=r['t']-max(support) if support else None,
                translation_diameter_m=displacement,pure_head_eligible=displacement<=.02 and len(history)>0,
                yaw_extent_deg=max(p['yaw'] for p in poses)-min(p['yaw'] for p in poses))
    clear=[r for r in selected if r['stratum']!='boundary'];boundary=[r for r in selected if r['stratum']=='boundary']
    assert len(clear)==6 and sum(r['truth'] for r in clear)==3
    names=sorted(clear[0]['features'])
    groups={}
    for w in map(str,WINDOWS):
        groups[w+'_radar']=[n for n in names if n.startswith(w+'_radar') or n.startswith(w+'_sector')]
        groups[w+'_scores']=[n for n in names if n.startswith(w+'_') and '_score_' in n]
        groups[w+'_other_public']=[n for n in names if n.startswith(w+'_') and ('_tof_' in n or '_delta_' in n or '_angular_' in n)]
    groups['current_radar']=[n for n in names if n.startswith('current_radar') or n.startswith('current_sector')]
    groups['current_scores']=[n for n in names if n.startswith('current_') and '_score_' in n]
    results={g:ceiling(clear,n) for g,n in groups.items()}
    write(out/'audited-candidates.json',selected)
    write(out/'ceiling.json',results)
    write(out/'summary.json',dict(status='CONSUMED_DESCRIPTIVE_CEILING_COMPLETE',clear_candidates=6,boundary_candidates=len(boundary),
        groups={g:dict(single=len(v['perfect_single']),pair=len(v['perfect_pair']),cuts=v['single_cuts_tested'],pairs=v['pairs_tested']) for g,v in results.items()},
        H2_eligible={str(w):sum(r['audit'][str(w)]['pure_head_eligible'] for r in clear) for w in WINDOWS},
        native_history_supported_TP=sum(any(v['native_support_frames'] for v in r['audit'].values()) for r in clear if r['truth']),
        elapsed_s=time.perf_counter()-start,labels_sha256=sha(REP/'cases.json'),spec_sha256=sha(CAP/'spec.json')))
    sys.path.insert(0,str(REPO/'tools'))
    from research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('python-scalar','cpu',lambda:features(prepared[:4],1.),
                   lambda _:DeviceObservation('cpu','host CPU','Python standard library')),record_path=out/'backend.json')
    write(out/'completion.json',dict(status='PASS',hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(read(out/'summary.json')))


if __name__=='__main__':main()
