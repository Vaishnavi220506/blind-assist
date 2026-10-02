"""Post-result descriptive events only: no model execution or new decision."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-post-history-diagnostic-20261002';WORK=OUT.parent;H=WORK/'cnh-frame-history-20261002';MC=WORK/'cnh-margin-confirm-20261002';G=WORK/'cnh-observed-sequence-20261002'
ARMS=('M3','AGG','HIST');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    result=read(H/'result.json');audit=read(H/'independent_audit.json');assert result['status']=='COMPLETE' and audit['status']=='PASS' and audit['result_sha256']==sha(H/'result.json')
    fields=('unit','config','query','split','covered','ref_category','clear_all','frame_ranges','frame_poses','reference_time_s','crossing_left_frame','crossing_right_frame','censor_reason')
    with np.load(G/'geometry.npz',allow_pickle=False) as z:
        ev=z['split']=='evaluation';g={k:z[k][ev] for k in fields};frames=z['frames']
    rows=[r for r in read(G/'rows.json') if r['split']=='evaluation']
    manifest={(r['unit'],r['config']):r for r in read(MC/'scene_manifest.json') if r['split']=='evaluation'}
    units=np.arange(96000,96096);scores={};raws={};flags={};first={};paths=[H/'result.json',H/'independent_audit.json',G/'geometry.npz',G/'rows.json',MC/'scene_manifest.json']
    for a in ARMS:
        if a=='M3':
            p,q=MC/'frame_scores_M3_early.npz',MC/'frame_scores_M3.npz';paths.extend([p,q])
            with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:raw=np.stack([np.concatenate((x[str(u)],y[str(u)]),axis=1) for u in units])
        else:
            p=H/f'frame_scores_{a}.npz';paths.append(p)
            with np.load(p,allow_pickle=False) as z:raw=np.stack([z[str(u)] for u in units])
        s=np.empty_like(raw,dtype=float)
        for t in range(13):
            lo=max(0,t-4);w=2.**np.arange(t-lo+1);s[:,:,t,:]=(raw[:,:,lo:t+1,:].astype(float)*w[None,None,:,None]).sum(2)/w.sum()
        scores[a]=s.transpose(0,1,3,2).reshape(-1,13);raws[a]=raw.transpose(0,1,3,2).reshape(-1,13)
        hit=scores[a]>=result['thresholds'][a];f=np.where(hit,np.arange(13),13).min(1);stop=f<13
        timely=stop&(g['frame_ranges'][np.arange(len(f)),np.minimum(f,12)]>=.9)
        first[a]=f;flags[a]=(stop,timely)
    shallow=g['covered']&(g['ref_category']=='contact0-2cm')
    same=g['query']==np.array([r['target_group'] for r in rows]);off=np.array([r['target_off'] for r in rows])
    outside=g['clear_all']&same&(off>=-.20)&(off<-.10)
    assert shallow.sum()==31 and outside.sum()==939
    meta=[]
    for i,(u,c,q) in enumerate(zip(g['unit'],g['config'],g['query'])):
        sc=manifest[int(u),int(c)];lo,hi=np.asarray(sc['boxes'][0]['lo']),np.asarray(sc['boxes'][0]['hi'])
        assert (int(u),int(c),int(q))==(rows[i]['unit'],rows[i]['config'],rows[i]['query'])
        meta.append(dict(unit=int(u),config=int(c),query=('HEAD','BODY')[int(q)],condition='none' if sc['cond']=='none' else 'panel',
            motion=int(u%3),side='positive_x' if lo[0]>0 else 'negative_x',target_nominal_intrusion_m=float(sc['off']),
            final_target_range_m=float(g['frame_ranges'][i,-1]),covered=bool(g['covered'][i]),censor_reason=str(g['censor_reason'][i])))
    def summary(mask,timely=False):
        d=dict(n=int(mask.sum()),units=int(len(np.unique(g['unit'][mask]))))
        for a in ARMS:
            d[a]=dict(stops=int((mask&flags[a][0]).sum()),timely=int((mask&flags[a][1]).sum()),never_alarm=int((mask&~flags[a][0]).sum()),
                late=int((mask&flags[a][0]&~flags[a][1]).sum()) if timely else None)
        return d
    def detail(i,include_scores=False):
        d=dict(meta[i],row=int(i),reference_time_from_frame3_s=float(g['reference_time_s'][i]) if g['covered'][i] else None,
            deadline_left_frame=int(g['crossing_left_frame'][i]) if g['covered'][i] else None,deadline_right_frame=int(g['crossing_right_frame'][i]) if g['covered'][i] else None)
        sc=manifest[int(g['unit'][i]),int(g['config'][i])];center=(np.array(sc['boxes'][0]['lo'])+np.array(sc['boxes'][0]['hi']))/2
        pre=g['frame_ranges'][i]>=.9;assert pre.any()
        last=int(np.flatnonzero(pre)[-1]);d.update(last_sample_at_or_before_deadline_frame=int(frames[last]),last_predeadline_range_m=float(g['frame_ranges'][i,last]))
        for a in ARMS:
            f=int(first[a][i]);stopped=f<13;t=result['thresholds'][a];peak=int(np.argmax(np.where(pre,scores[a][i],-np.inf)))
            pose=g['frame_poses'][i,min(f,12)];local=(center-pose[:3,3])@pose[:3,:3]
            v=dict(threshold=t,status='timely' if flags[a][1][i] else 'late' if stopped else 'no_alarm_observed',
                first_alarm_frame=int(frames[f]) if stopped else None,first_alarm_s=float(f*.2) if stopped else None,
                first_alarm_range_m=float(g['frame_ranges'][i,f]) if stopped else None,
                first_alarm_minus_deadline_s=float(f*.2-g['reference_time_s'][i]) if stopped and g['covered'][i] else None,
                predeadline_peak_frame=int(frames[peak]),predeadline_peak_range_m=float(g['frame_ranges'][i,peak]),
                predeadline_max_smoothed=float(scores[a][i,peak]),predeadline_max_minus_threshold=float(scores[a][i,peak]-t),
                predeadline_max_raw=float(raws[a][i,pre].max()),predeadline_any_raw_ge_smoothed_threshold=bool((raws[a][i,pre]>=t).any()),
                whole_window_max_minus_threshold=float(scores[a][i].max()-t),
                first_alarm_target_center_travel_bearing_deg=float(np.degrees(np.arctan2(local[0],local[2]))) if stopped else None,
                still_above_at_last_frame=bool(scores[a][i,-1]>=t))
            if include_scores:v.update(raw=raws[a][i].tolist(),smoothed=scores[a][i].tolist())
            d[a]=v
        if include_scores:d.update(frames=frames.tolist(),target_ranges_m=g['frame_ranges'][i].tolist())
        return d
    groups={}
    for name,mask in [('all_shallow',shallow),('all_outside',outside)]:
        groups[name]={field:{str(v):summary(mask&np.array([m[field]==v for m in meta]),name=='all_shallow') for v in sorted({m[field] for m in meta})}
                      for field in ('motion','side','condition','query')}
    pairs={};changed=np.zeros(7680,bool)
    for a,b in (('AGG','M3'),('HIST','M3'),('HIST','AGG')):
        added=outside&flags[a][0]&~flags[b][0];removed=outside&~flags[a][0]&flags[b][0];changed|=added|removed
        pairs[f'{a}_minus_{b}']=dict(n=939,added=[detail(int(i)) for i in np.flatnonzero(added)],removed=[detail(int(i)) for i in np.flatnonzero(removed)])
    misses=shallow&~flags['HIST'][1];assert misses.sum()==5
    for a in ARMS:assert np.array_equal(misses,shallow&~flags[a][1])
    for a in ARMS:
        assert int((shallow&flags[a][1]).sum())==result['cells'][a]['metrics']['contact0-2cm']['timely_stops']
        assert int((outside&flags[a][0]).sum())==result['cells'][a]['clear_subgroups']['same_height_nominal_outside10_20cm']['first_stops']
    out=dict(status='COMPLETE_DESCRIPTIVE',original_verdict=result['verdict'],thresholds=result['thresholds'],
        overall=dict(shallow=summary(shallow,True),outside=summary(outside)),groups=groups,
        all_shallow=[detail(int(i)) for i in np.flatnonzero(shallow)],shallow_misses=[detail(int(i),True) for i in np.flatnonzero(misses)],
        outside_changed_pairs=pairs,outside_changed_union=[detail(int(i),True) for i in np.flatnonzero(changed)],
        outside_all_939=[dict(meta[i],**{a:bool(flags[a][0][i]) for a in ARMS}) for i in np.flatnonzero(outside)],
        limits=['No threshold/model/truth changes or new gate; metadata is post-result descriptive, not deployable gating.',
            'No alarm is not never seen: cached model scores do not establish physical observability or photon absence.',
            'Raw crossing of a smoothed-score threshold is diagnostic only, not a evaluated alternative deployment rule.',
            'All-object clear does not identify which physical surface caused score; target bearing is travel-relative geometry, not sensor off-axis attribution.',
            'none/panel are not paired interventions; mode/side/query/context overlap, no post-hoc significance tests.',
            'Covered .9 target-front deadline and full13 sampled exposure are proxies; consumed Development, not hardware.'],
        inputs_sha256={str(p):sha(p) for p in paths},script_sha256=sha(__file__))
    (OUT/'diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    lines=['# HIST之后的残留错误：描述性诊断','','原判读 FRAME_HISTORY_NOT_ESTABLISHED_DEV 保留。所有31条浅擦碰和939条外侧清晰查询保留在分母中。',
        '','| unit/config/query | side/mode/context | .9m跨越帧 | 三臂首报帧 M3/AGG/HIST | 截止前余量 M3/AGG/HIST |', '|---|---|---|---|---|']
    for d in out['shallow_misses']:
        alarms='/'.join(str(d[a]['first_alarm_frame']) if d[a]['first_alarm_frame'] is not None else '未报' for a in ARMS)
        margins='/'.join(f"{d[a]['predeadline_max_minus_threshold']:+.4f}" for a in ARMS)
        lines.append(f"|{d['unit']}/{d['config']}/{d['query']}|{d['side']}/{d['motion']}/{d['condition']}|{d['deadline_left_frame']}→{d['deadline_right_frame']}|{alarms}|{margins}|")
    lines+=['','| 外侧清晰比较 | 新增首停 | 移除首停 |','|---|---:|---:|']
    for name,p in pairs.items():lines.append(f"|{name}|{len(p['added'])}|{len(p['removed'])}|")
    lines+=['','## 全分母分组','','|集合/维度/值|n|M3/AGG/HIST首停|M3/AGG/HIST及时|','|---|---:|---|---|']
    for name,ds in groups.items():
        for dim,vs in ds.items():
            for value,r in vs.items():lines.append(f"|{name}/{dim}/{value}|{r['n']}|"+'/'.join(str(r[a]['stops']) for a in ARMS)+'|'+'/'.join(str(r[a]['timely']) for a in ARMS)+'|')
    lines+=['','逐条增删、截止点、首报距离、分数余量及完整13帧分数见 diagnostic.json。未报警不能写成从未观测到；分数也不能归因给目标或背景。所有几何与模式标签仅用于本次事后描述。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(overall=out['overall'],misses=[{k:v for k,v in d.items() if k not in ARMS+('frames','target_ranges_m')} for d in out['shallow_misses']],
        changes={k:[len(p['added']),len(p['removed'])] for k,p in pairs.items()},changed_union=int(changed.sum())),ensure_ascii=False))
if __name__=='__main__':main()
