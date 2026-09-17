"""Evaluator-only position tolerance; no sensor availability or object-width gate."""
import numpy as np

TOLERANCES=(.03,.05,.10)

def geometry(e):
    origin=np.asarray(e['body_origin_m'],float)
    eligible=[];full=[]
    for o in e['native_bounds']:
        lo=np.asarray(o['center_m'],float)-np.asarray(o['extent_m'],float)-origin
        hi=np.asarray(o['center_m'],float)+np.asarray(o['extent_m'],float)-origin
        slack=np.minimum(hi-np.array([.2,-.3,.4]),np.array([3.6,.3,2.05])-lo)
        full.append(float(slack.min()))
        if hi[0]>=.2 and lo[0]<=3.6 and hi[2]>=.4 and lo[2]<=2.05:
            eligible.append(dict(name=o['name'],margin_m=float(slack[1]),lo=lo.tolist(),hi=hi.tolist()))
    best=max(eligible,key=lambda x:x['margin_m']) if eligible else None
    return dict(lateral_margin_m=best['margin_m'] if best else None,eligible_objects=len(eligible),
        closest_object=best['name'] if best else None,strict=bool(best and best['margin_m']>=0),
        signed_min_face_slack_m=max(full) if full else None,
        # Positive margin means depth into the nominal side boundary, not
        # overlap length: a thin central rod can have margin > its diameter.
        eligible=eligible)

def classify(g,tolerance):
    if not 0<=tolerance<.3:raise ValueError('Invalid lateral tolerance')
    m=g['lateral_margin_m']
    if m is None:return 'negative'
    if m>=tolerance:return 'positive'
    if m < -tolerance:return 'negative'
    return 'boundary'

def metric(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(frames=len(y),positive=int(y.sum()),negative=int((~y).sum()),TP=tp,FP=fp,FN=fn,TN=tn,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        alert_fraction=float(p.mean()) if len(p) else None)

def temporal(rows,states,p):
    """Episode-local core events, nuisance time and descriptive response times.

    Every 4Hz sample represents a .25s bin. The final bin is an accounting
    convention; transitions beyond the1.25s trajectory are right-censored.
    No temporal filter or new alert rule is applied.
    """
    p=np.asarray(p,bool);s=np.asarray(states);events=[];release=[];transitions=0;boundary_transitions=0
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['episode_id'],[]).append(i)
    negative_segments=0;negative_episode_count=0;negative_episode_alerts=0
    for episode,ii in groups.items():
        a=p[ii];v=s[ii];t=np.array([rows[i]['time_s'] for i in ii]);n=len(ii)
        assert len(ii)==6 and np.allclose(np.diff(t),.25)
        transitions+=int((a[1:]!=a[:-1]).sum())
        boundary_transitions+=int(((a[1:]!=a[:-1])&(v[1:]=='boundary')&(v[:-1]=='boundary')).sum())
        nuisance=a&(v=='negative');negative_segments+=int(sum(nuisance[j] and (j==0 or not nuisance[j-1]) for j in range(n)))
        if np.all(v=='negative'):
            negative_episode_count+=1;negative_episode_alerts+=int(a.any())
        i=0
        while i<n:
            if v[i]!='positive':i+=1;continue
            start=i
            while i<n and v[i]=='positive':i+=1
            end=i;found=np.flatnonzero(a[start:end]);before=start
            while before>0 and v[before-1]=='boundary':before-=1
            pre=np.flatnonzero(a[before:start]);first=before+int(pre[0]) if len(pre) else start+int(found[0]) if len(found) else None
            events.append(dict(episode=episode,start_s=float(t[start]),end_last_sample_s=float(t[end-1]),
                frames=end-start,detected_in_core=bool(len(found)),
                first_in_core_delay_s=float(t[start+found[0]]-t[start]) if len(found) else None,
                first_boundary_or_core_alert_offset_s=float(t[first]-t[start]) if first is not None else None,
                core_alert_fraction=float(a[start:end].mean()),left_censored=start==0,right_censored=end==n))
        for j in range(1,n):
            if v[j]!='negative' or v[j-1]=='negative' or not np.any(v[:j]=='positive'):continue
            end=j
            while end<n and v[end]=='negative':end+=1
            off=np.flatnonzero(~a[j:end]);release.append(dict(episode=episode,start_s=float(t[j]),
                first_off_delay_s=float(t[j+off[0]]-t[j]) if len(off) else None,
                no_off_before_negative_interval_ends=not bool(len(off)),right_censored=end==n))
    negative=s=='negative';boundary=s=='boundary';positive=s=='positive'
    delays=[e['first_in_core_delay_s'] for e in events if e['detected_in_core']]
    return dict(core_events=len(events),core_events_detected=sum(e['detected_in_core'] for e in events),
        core_events_alert_at_first_sample=sum(e['first_in_core_delay_s']==0 for e in events),
        first_in_core_delay_mean_s=float(np.mean(delays)) if delays else None,
        first_in_core_delay_max_s=float(max(delays)) if delays else None,
        core_events_left_censored=sum(e['left_censored'] for e in events),events=events,
        clear_negative_frames=int(negative.sum()),clear_negative_alert_frames=int((negative&p).sum()),
        clear_negative_sampled_s=float(negative.sum()*.25),clear_negative_alert_sampled_s=float((negative&p).sum()*.25),
        clear_negative_alert_fraction=float(p[negative].mean()) if negative.any() else None,
        clear_negative_alert_segments=negative_segments,clear_negative_episodes=negative_episode_count,
        clear_negative_episodes_with_alert=negative_episode_alerts,
        boundary_frames=int(boundary.sum()),boundary_alert_frames=int((boundary&p).sum()),
        core_positive_frames=int(positive.sum()),alert_transitions=transitions,boundary_internal_alert_transitions=boundary_transitions,
        release_opportunities=len(release),release=release,
        sampling_note='4Hz; six-frame1.25s trajectories; .25s final bin accounting; no continuous real-user evidence')
