"""Independent saved-output recount, without inference/generator/metric imports."""
from collections import defaultdict
import numpy as np


def audit(rows, lineage, cases, geometry, native_root, report, public):
    arms=tuple(rows[0]['flags']); checked=0
    subsets={'all':rows,'Core':[r for r in rows if r['layout_relation']!='BOUNDARY']}
    for key in ('layout_relation','layer','type_id','base_group_id'):
        for value in sorted({r[key] for r in rows}):
            subsets[key+':'+value]=[r for r in rows if r[key]==value]
    def run_count(flags):
        return sum(v and (i==0 or not flags[i-1]) for i,v in enumerate(flags))
    for label,group in subsets.items():
        expected=report['metrics'] if label=='all' else report['strata']['Core'] if label=='Core' else report['strata'][label.split(':')[0]][label.split(':')[1]]
        clips=defaultdict(list)
        for r in group: clips[r['clip_id']].append(r)
        for arm in arms:
            counts=dict(TP=0,FP=0,FN=0,TN=0,prediction_unknown=0)
            for r in group:
                y,a,u=r['truth'],r['flags'][arm],r['predictions'][arm]['unknown']
                counts['prediction_unknown']+=u
                if a: counts['TP' if y else 'FP']+=1
                elif y: counts['FN']+=1
                elif not u: counts['TN']+=1
            m=expected['arms'][arm]
            for k,v in counts.items():
                assert m['frames']['all_known'][k]==v,(label,arm,k)
                checked+=1
            ev=det=seg=interruptions=episodes=0; tails=0.; delays=[]
            for clip in clips.values():
                clip.sort(key=lambda r:r['frame_in_clip'])
                a=[r['flags'][arm] for r in clip]; y=[r['truth'] for r in clip]
                episodes+=run_count(a); seg+=run_count([x and not yy for x,yy in zip(a,y)])
                ix=[i for i,v in enumerate(y) if v]
                if not ix: continue
                assert ix==list(range(min(ix),max(ix)+1))
                ev+=1; hits=[i for i in ix if a[i]]
                if hits:
                    det+=1; delays.append((hits[0]-ix[0])*.2)
                    interruptions+=run_count([not v for v in a[hits[0]:hits[-1]+1]])
                if a[ix[-1]]:
                    for v in a[ix[-1]+1:]:
                        if not v: break
                        tails+=.2
            assert (m['event_count'],m['detected_events'],m['false_alert_segment_count'])==(ev,det,seg)
            assert abs(m['false_alert_sampled_duration_s']-counts['FP']*.2)<1e-8
            s=m['complete_summary']
            assert s['internal_interruptions']==interruptions and s['alert_episode_count']==episodes
            assert abs(s['exit_carryover_sampled_s']-tails)<1e-8
            assert s['immediate_onsets']==sum(x==0 for x in delays)
            assert s['detected_delay_max_s']==max(delays,default=None)
            assert s['detected_delay_median_s']==(float(np.median(delays)) if delays else None)
            checked+=9
    # Separate native projection: sample native ray indices directly, rather
    # than calling the evaluator's mask or the original sample_native helper.
    yi=np.floor((np.arange(192)+.5)*360/192).astype(int)
    xi=np.floor((np.arange(256)+.5)*640/256).astype(int)
    focal=320/np.tan(np.pi*50/180)
    previous={}; original_support_zero=0
    for k,(r,lin) in enumerate(zip(rows,lineage)):
        i=r['index']; g=geometry[i]; camera=cases[i]['camera']
        native=np.load(native_root/g['native_path'],allow_pickle=False)
        z=native[yi[:,None],xi[None,:]].astype(np.float64)
        x=(xi[None,:]+.5-320)*z/focal; y=(yi[:,None]+.5-180)*z/focal
        corridor=np.isfinite(z)&(z>=.3)&(z<=3)&(abs(x)<=.3)&(y>=-.2)&(y<=.9)
        target=next(o for o in g['objects'] if o['name']=='target')
        center=np.array(target['render_bounds_center_m']); half=np.array(target['render_bounds_extent_m'])
        xyz=[z+camera['x'],x+camera['y'],camera['z']-y]
        support=corridor.copy()
        for axis in range(3): support &= (xyz[axis]>=center[axis]-half[axis]-.02)&(xyz[axis]<=center[axis]+half[axis]+.02)
        support=support.ravel()
        counts=[sum(sum(bool(support[j]) for j in zone['indices'][slot]) for zone in lin['zones']) for slot in (0,1)]
        assert counts==r['native_target_corridor'],r['id']
        truth=False
        for obj in g['objects']:
            c=np.array(obj['render_bounds_center_m']); h=np.array(obj['render_bounds_extent_m'])
            lo=np.array([c[1]-camera['y']-h[1],camera['z']-c[2]-h[2],c[0]-camera['x']-h[0]])
            hi=np.array([c[1]-camera['y']+h[1],camera['z']-c[2]+h[2],c[0]-camera['x']+h[0]])
            truth |= bool(hi[0]>=-.3 and lo[0]<=.3 and hi[1]>=-.2 and lo[1]<=.9 and hi[2]>=.3 and lo[2]<=3)
        assert truth==r['truth'],r['id']
        for mode,trigger in r['triggers'].items():
            total=sum(sum(bool(support[j]) for j in lin['zones'][t['zone']]['indices'][t['slot']]) for t in trigger)
            second=sum(sum(bool(support[j]) for j in lin['zones'][t['zone']]['indices'][1]) for t in trigger if t['slot']==1)
            assert total==r['native_trigger_contributors'][mode] and second==r['native_second_trigger_contributors'][mode]
            assert bool(trigger)==r['flags'][mode]
            prev=previous.get((r['clip_id'],mode))
            held=r['flags'][mode] or bool(prev and abs(r['time_s']-prev[0]-.2)<1e-8 and prev[1])
            assert held==r['flags'][mode+'_hold']
            previous[r['clip_id'],mode]=(r['time_s'],r['flags'][mode])
            checked+=4
        first,second=public['ranges'][k].T
        assert np.isfinite(second).sum()==r['second_slot_present']
        assert not np.isfinite(second[~np.isfinite(first)]).any()
        # Verify the fixed slot-selection law without rerunning its noise/RNG.
        flat=native[yi[:,None],xi[None,:]].ravel()
        for zone,(y0,x0,y1,x1) in enumerate(public['boxes']):
            if not np.isfinite(first[zone]): continue
            patch=native[yi[y0:y1,None],xi[None,x0:x1]]
            hits=patch[np.isfinite(patch)&(patch>=.1)&(patch<8)]
            bins=np.minimum((hits/.1).astype(int),79)
            weights=np.bincount(bins,weights=1/np.maximum(hits,.3)**2,minlength=80)
            winner=int(np.argmax(weights)); first_mean=np.mean(hits[bins==winner])
            candidates=[int(b) for b in np.flatnonzero(weights) if np.count_nonzero(bins==b)>=4 and weights[b]>=.01*weights[winner] and abs(float(np.mean(hits[bins==b]))-float(first_mean))>=.6]
            if np.isfinite(second[zone]):
                selected=min(candidates,key=lambda b:(-weights[b],b))
                assert lin['zones'][zone]['bins'][1]==selected
                ix=lin['zones'][zone]['indices'][1]
                assert len(ix)==np.count_nonzero(bins==selected)
                assert np.all(np.minimum((flat[ix]/.1).astype(int),79)==selected)
                checked+=3
        checked+=4
    return dict(status='PASS',frames=len(rows),groups=len(subsets),assertions=checked,
        checks=['all-six-arm frame/event/segment/duration/group recount','nonrecursive hold and onset',
                'rendered full-bound truth','independent native-ray contributor projection','second-slot peak law'],
        inference_or_original_metric_functions_called=False)
