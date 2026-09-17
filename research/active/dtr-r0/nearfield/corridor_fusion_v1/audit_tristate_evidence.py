"""Independent audit of existing-evidence tri-state probes; no helper imports."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'artifacts.local/work'
SRC=WORK/'corridor-surface-oracle-20260917'
OUT=WORK/'corridor-tristate-evidence-20260917'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):
    y=np.asarray(y,bool); p=np.asarray(p,bool)
    return dict(frames=len(y),TP=int((y&p).sum()),FP=int((~y&p).sum()),FN=int((y&~p).sum()),TN=int((~y&~p).sum()))

def inter(piece, width=.3):
    a=np.asarray(piece,float)
    return bool(np.all(a[:,1] >= [.2,-width,.4]) and np.all(a[:,0] <= [3.6,width,2.05]))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=read(SRC/'freeze.json'); recs=read(SRC/'oracle-returns.json'); cases=read(SRC/'cases.json')
    seal=read(SRC/'prediction-seal.json')
    for filename,key in [('freeze.json','freeze_sha256'),('oracle-returns.json','returns_sha256'),('scores.npz','scores_sha256')]:
        assert sha(SRC/filename)==seal[key]
    scores=np.load(SRC/'scores.npz'); ids=list(freeze['ids']); assert [r['id'] for r in recs]==ids==[c['id'] for c in cases]==list(scores['ids'])
    raw_path=next(Path(p) for p in freeze['bindings'] if Path(p).name=='raw.jsonl')
    assert sha(raw_path)==freeze['bindings'][str(raw_path)]
    raw={r['id']:r for r in map(json.loads,raw_path.read_text().splitlines())}
    threshold=float(freeze['threshold']); A=np.asarray(scores['A']>=threshold,bool)
    truth=np.asarray([c['truth'] for c in cases],bool); states=np.asarray([c['state'] for c in cases]);
    assert len(ids)==288 and int((states!='boundary').sum())==216
    pface=[]; ppoints=[]; nface=[]; npoints=[]; unknown_face=[]; unknown_points=[]
    for i,r in enumerate(recs):
        slots=r['slots']; usable=[s for s in slots if s['usable']]
        # Recompute P from exact face union and sampled point records, never labels.
        pf=any(inter(f) for s in usable if s['completed'] for f in s['faces'])
        pp=any(inter(np.stack([p,p],-1)) for s in usable if s['completed'] for p in s['points'])
        assert pf==r['full_face_reachable'] and pp==r['sampled_point_reachable']
        all_done=bool(usable) and all(bool(s['completed']) for s in usable)
        nf=bool(raw[r['id']]['tof_packet_received'] and r['usable_slots']>0 and all_done and not pf)
        np_=bool(raw[r['id']]['tof_packet_received'] and r['usable_slots']>0 and all_done and not pp)
        # zero-return frames remain UNKNOWN: N is false and P is false.
        assert (r['usable_slots']==0) == (not usable)
        pface.append(pf); ppoints.append(pp); nface.append(nf); npoints.append(np_)
        unknown_face.append(not pf and not nf); unknown_points.append(not pp and not np_)
    pface=np.asarray(pface); ppoints=np.asarray(ppoints); nface=np.asarray(nface); npoints=np.asarray(npoints)
    assert np.all(~(pface&nface)) and np.all(~(ppoints&npoints))
    assert np.all((~pface&~nface)==(np.asarray([r['usable_slots']==0 for r in recs])))
    arrays={
      'A':A,
      'A_OR_P_face':A|pface,
      'A_AND_NOT_N_face':A&~nface,
      'A_OR_P_AND_NOT_N_face':(A|pface)&~nface,
      'A_OR_P_points':A|ppoints,
      'A_AND_NOT_N_points':A&~npoints,
      'A_OR_P_AND_NOT_N_points':(A|ppoints)&~npoints,
    }
    for i,c in enumerate(cases):assert bool(A[i])==c['alerts']['A']
    for name,p in arrays.items():
        u=np.asarray(unknown_face if 'face' in name else unknown_points)
        assert np.array_equal(p[u],A[u])
    report={}
    for name,p in arrays.items():
        report[name]=dict(strict=metric(truth,p),clear=metric(truth[states!='boundary'],p[states!='boundary']),boundary=metric(truth[states=='boundary'],p[states=='boundary']),
          changed_from_A=int((p!=A).sum()),unknown_frames=int((~(pface|nface)).sum()) if 'face' in name else int((~(ppoints|npoints)).sum()))
    # Detailed veto loss: A true, N true, and truth true is a lost positive.
    lost=[]
    for i,(c,nf,np_) in enumerate(zip(cases,nface,npoints)):
        for mode,n in [('face',nf),('points',np_)]:
            if A[i] and n and truth[i]:
                lost.append(dict(id=ids[i],mode=mode,family=c['family'],episode=c['episode'],time_s=c['time_s'],state=c['state'],
                    usable_returns=c['usable_returns'],completed_returns=c['completed_returns'],
                    diagnosis=c['diagnosis']['category']))
    # Event accounting: first-alert and lost-positive episodes for each veto arm.
    by_ep=defaultdict(list)
    for i,c in enumerate(cases): by_ep[c['episode']].append(i)
    events={}
    for name,p in arrays.items():
        rows=[]
        for ep,ix in by_ep.items():
            truth_ix=[i for i in ix if truth[i]]
            alert_ix=[i for i in ix if p[i]]
            rows.append(dict(episode=ep,positive_alerted=bool(any(p[i] for i in truth_ix)),first_alert_time_s=float(cases[alert_ix[0]]['time_s']) if alert_ix else None,
                             positive_frames=len(truth_ix),lost_positive_frames=int(sum(truth[i] and not p[i] for i in ix))))
        events[name]=dict(episodes=len(rows),episodes_with_positive_alert=int(sum(r['positive_alerted'] for r in rows)),
                          lost_positive_frames=int(sum(r['lost_positive_frames'] for r in rows)),rows=rows)
    event_changes={}
    for name,p in arrays.items():
        changes=[]
        for convention,event_mask in [('strict_positive',truth),('core_positive',states=='positive')]:
            for ep,ii in by_ep.items():
                intervals=[]; current=[]
                for i in ii:
                    if event_mask[i]:current.append(i)
                    elif current:intervals.append(current);current=[]
                if current:intervals.append(current)
                for interval in intervals:
                    aa=[i for i in interval if A[i]]; pp=[i for i in interval if p[i]]
                    a0=cases[aa[0]]['time_s'] if aa else None
                    p0=cases[pp[0]]['time_s'] if pp else None
                    if a0!=p0:
                        changes.append(dict(convention=convention,episode=ep,start=cases[interval[0]]['time_s'],
                            end=cases[interval[-1]]['time_s'],A_first_in_event=a0,probe_first_in_event=p0,
                            event_lost=bool(aa and not pp),delay_s=p0-a0 if aa and pp else None))
        event_changes[name]=changes
    result=dict(status='PASS',frames=len(ids),threshold=threshold,tri_state_definition=dict(P='existing usable completed exact face union intersects corridor',N='packet received and usable>0 and all usable completed and no P',U='otherwise; zero usable always U'),
      probe=report,face_counts=dict(P=int(pface.sum()),N=int(nface.sum()),U=int((~pface&~nface).sum())),sampled_point_counts=dict(P=int(ppoints.sum()),N=int(npoints.sum()),U=int((~ppoints&~npoints).sum())),
      lost_TP_cases=lost,events=events,event_changes=event_changes,unknown_consistent=bool(np.all((~pface&~nface)==(np.asarray([r['usable_slots']==0 for r in recs])))),
      negative_inference_boundary='N means all measured returned surfaces are outside the corridor; it does not establish that unmeasured or unreturned corridor space is clear. U retains A.',
      source_seals={str(SRC/f):sha(SRC/f) for f in ('freeze.json','oracle-returns.json','cases.json','scores.npz')})
    summary=OUT/'summary.json'
    if summary.exists():
        result['root_summary_sha256']=sha(summary)
        root=read(summary)
        rootfreeze=read(OUT/'freeze.json'); rootseal=read(OUT/'prediction-seal.json')
        assert rootfreeze['ids']==ids
        assert sha(OUT/'freeze.json')==rootseal['freeze_sha256']
        assert sha(OUT/'predictions.json')==rootseal['predictions_sha256']
        for path,h in rootfreeze['bindings'].items():assert sha(path)==h,path
        for path,h in freeze['bindings'].items():assert sha(path)==h,path
        predictions=read(OUT/'predictions.json'); assert [r['id'] for r in predictions]==ids
        methodmap={'A':'A','A_OR_P':'A_OR_P','A_VETO_N':'A_AND_NOT_N','A_POSITIVE_AND_VETO':'A_OR_P_AND_NOT_N'}
        checked=0
        for rep,suffix,pvec,nvec in [('full_faces','face',pface,nface),('sampled_points','points',ppoints,npoints)]:
            for i,item in enumerate(predictions):
                v=item['variants'][rep]
                expected='POSITIVE' if pvec[i] else 'OUTSIDE_ONLY' if nvec[i] else 'UNKNOWN'
                assert v['state']==expected,(rep,ids[i])
                for rootarm,localarm in methodmap.items():
                    key=localarm if localarm=='A' else localarm+'_'+suffix
                    assert v['alerts'][rootarm]==bool(arrays[key][i]),(rep,rootarm,ids[i])
                    checked+=1
            for rootarm,localarm in methodmap.items():
                key=localarm if localarm=='A' else localarm+'_'+suffix
                got=root['representations'][rep]['methods'][rootarm]; p=arrays[key]
                for subset in ['strict','clear','boundary']:
                    for stat,value in report[key][subset].items():assert got[subset][stat]==value,(rep,rootarm,subset,stat)
                core=[]; strict_events=[]
                for ep,ii in by_ep.items():
                    for mask,target in [(states=='positive',core),(truth,strict_events)]:
                        blocks=[]; block=[]
                        for i in ii:
                            if mask[i]:block.append(i)
                            elif block:blocks.append(block);block=[]
                        if block:blocks.append(block)
                        for block in blocks:
                            hits=[i for i in block if p[i]]
                            target.append(dict(episode=ep,start_s=cases[block[0]]['time_s'],end_last_sample_s=cases[block[-1]]['time_s'],
                                frames=len(block),detected_in_core=bool(hits),first_in_core_delay_s=cases[hits[0]]['time_s']-cases[block[0]]['time_s'] if hits else None,
                                core_alert_fraction=sum(p[i] for i in block)/len(block)))
                temporal=got['temporal']; assert temporal['core_events']==len(core)
                assert temporal['core_events_detected']==sum(c['detected_in_core'] for c in core)
                assert len(temporal['events'])==len(core)
                for expected,actual in zip(core,temporal['events']):
                    for field,val in expected.items():assert actual[field]==val,(rep,rootarm,field)
                assert got['strict_events']==dict(events=len(strict_events),detected=sum(c['detected_in_core'] for c in strict_events))
                changes=[c for c in event_changes[key] if c['convention']=='core_positive']
                expected_changes=[dict(episode=c['episode'],A_delay_s=c['A_first_in_event']-c['start'] if c['A_first_in_event'] is not None else None,
                    probe_delay_s=c['probe_first_in_event']-c['start'] if c['probe_first_in_event'] is not None else None) for c in changes]
                assert temporal['changed_core_first_alert']==expected_changes,(rep,rootarm)
        result['root_summary_compare']=dict(status='PASS',per_frame_alert_assertions=checked,state_assertions=576,
            confusion_counts_asserted=True,core_event_rows_and_changes_asserted=True,strict_event_counts_asserted=True,
            root_and_inherited_binding_hashes_asserted=True,prediction_seals_asserted=True)
    (OUT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','probe','face_counts','sampled_point_counts','lost_TP_cases','event_changes')},indent=2))

if __name__=='__main__': main()
