"""All existing outside-clear episodes; descriptive cache-only diagnostics."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
W=ROOT/'artifacts.local/work'
OUT=W/'cnh-outside-fp-diagnostic-20261002'
RUN=W/'cnh-yaw-source-transfer-20261002';GEOM=W/'cnh-observed-sequence-20261002';MC=W/'cnh-margin-confirm-20261002'
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smooth(raw):
    raw=np.asarray(raw,float);v=np.empty_like(raw)
    for t in range(13):
        start=max(0,t-4);weight=2.**np.arange(t-start+1)
        v[:,:,t,:]=(raw[:,:,start:t+1,:].transpose(0,1,3,2).reshape(-1,len(weight))@(weight/weight.sum())).reshape(96,40,2)
    return v.transpose(0,1,3,2).reshape(-1,13)


def distribution(values):
    values=np.asarray(values,float)
    if len(values)==0:return dict(n=0,quantiles=None,mean=None)
    return dict(n=len(values),quantiles=dict(zip(('min','q25','median','q75','max'),np.quantile(values,[0,.25,.5,.75,1]).tolist())),mean=float(values.mean()))


def main():
    result=read(RUN/'result.json');receipt=read(RUN/'scores_receipt.json');audit=read(RUN/'independent_audit.json')
    assert result['status']=='COMPLETE' and audit['status']=='PASS' and sha(RUN/'result.json')==audit['result_sha256']
    gr=read(GEOM/'geometry_receipt.json')
    assert sha(GEOM/'geometry.npz')==gr['geometry_sha256'] and sha(GEOM/'rows.json')==gr['rows_sha256']
    assert sha(MC/'scene_manifest.json')==gr['inputs']['scene_manifest_sha256']
    with np.load(GEOM/'geometry.npz') as z:
        ev=z['split']=='evaluation'
        g={k:z[k][ev] for k in ('unit','config','query','clear_all','frame_ranges','frame_poses')}
    rows=[r for r in read(GEOM/'rows.json') if r['split']=='evaluation']
    same=np.array([r['query']==r['target_group'] for r in rows]);off=np.array([r['target_off'] for r in rows])
    keep=g['clear_all']&same&(off>=-.20)&(off<-.10);ids=np.flatnonzero(keep);assert len(ids)==939
    man={(x['unit'],x['config']):x for x in read(MC/'scene_manifest.json')}
    with np.load(MC/'frame_scores_M3_early.npz') as early,np.load(MC/'frame_scores_M3.npz') as late:
        scores={0:smooth(np.stack([np.concatenate([early[str(u)],late[str(u)]],axis=1) for u in range(96000,96096)]))[keep]}
    for name,digest in receipt['output_sha256'].items():
        path=RUN/name;assert sha(path)==digest
        with np.load(path) as z:angle=float(z['bias_deg']);raw=z['logit'];assert np.array_equal(z['units'],np.arange(96000,96096))
        scores[angle]=smooth(raw)[keep]
    metadata=[];gaps=[]
    for i in ids:
        r=rows[i];s=man[r['unit'],r['config']];box=s['boxes'][0]
        lo=np.array(box['lo']);hi=np.array(box['hi']);assert lo[0]*hi[0]>0
        nominal_gap=min(abs(lo[0]),abs(hi[0]))-.3
        assert abs(nominal_gap+r['target_off'])<1e-12
        corners=np.array([[lo[j] if not (b>>j)&1 else hi[j] for j in range(3)] for b in range(8)])
        poses=g['frame_poses'][i]
        local=np.einsum('tji,tkj->tki',poses[:,:3,:3],corners[None]-poses[:,None,:3,3])
        xmin=local[:,:,0].min(1);xmax=local[:,:,0].max(1)
        # One-axis projection only: not an XYZ-clipped surface clearance.
        lateral=np.where(xmin>0,xmin-.3,np.where(xmax<0,-xmax-.3,-.3))
        gaps.append(lateral)
        panel=s['cond']!='none'
        metadata.append(dict(row=int(i),unit=r['unit'],config=r['config'],query='HEAD' if r['query']==0 else 'BODY',mode=r['unit']%3,
            side='positive_x' if lo[0]>0 else 'negative_x',context='panel' if panel else 'none',
            nominal_gap_cm=100*nominal_gap,final_target_range_m=r['final_range'],
            nominal_gap_bin='10-12cm' if nominal_gap<.12 else '12-15cm' if nominal_gap<.15 else '15-20cm',
            final_range_bin='0.6-1.2m' if r['final_range']<1.2 else '1.2-1.8m' if r['final_range']<1.8 else '1.8-2.6m',
            target_width_m=float(hi[0]-lo[0]),target_depth_m=float(hi[2]-lo[2]),target_rho=box['rho'],
            panel_rho=s['boxes'][1]['rho'] if panel else None))
    gaps=np.asarray(gaps);ranges=g['frame_ranges'][keep]
    dims=('mode','side','context','query','nominal_gap_bin','final_range_bin')
    cells={};event_ledger={}
    for bias in (-1,0,1):
        for method in ('BASE','ENV'):
            key=f'{bias}|{method}';threshold=result['fixed_thresholds'][method]
            angles=[bias] if method=='BASE' else [bias-1,bias,bias+1]
            stack=np.stack([scores[a] for a in angles]);score=stack.max(0)
            hit=score>=threshold;stop=hit.any(1);first=np.where(hit,np.arange(13),13).min(1);peak=score.argmax(1)
            nstop=int(stop.sum());assert nstop==result['cells'][key]['clear_subgroups']['same_height_nominal_outside10_20cm']['first_stops']
            summary=dict(n=939,alarm=nstop,no_alarm=939-nstop,alarm_fraction=nstop/939,rate_per_proxy_min=nstop/(939*2.6/60),groups={})
            for dim in dims:
                summary['groups'][dim]={}
                for value in sorted(set(x[dim] for x in metadata)):
                    m=np.array([x[dim]==value for x in metadata]);n=int(m.sum());k=int((m&stop).sum())
                    summary['groups'][dim][str(value)]=dict(n=n,alarm=k,no_alarm=n-k,alarm_fraction=k/n,share_of_all_alarms=k/nstop if nstop else None)
            summary['alarm_vs_no_alarm']={}
            for label,select in [('alarm',stop),('no_alarm',~stop)]:
                local_ids=np.flatnonzero(select)
                summary['alarm_vs_no_alarm'][label]={field:distribution([metadata[i][field] for i in local_ids]) for field in ('nominal_gap_cm','final_target_range_m','target_width_m','target_depth_m','target_rho')}
                summary['alarm_vs_no_alarm'][label].update(peak_frame_hist={str(t+3):int(np.sum(peak[select]==t)) for t in range(13)},
                    peak_actual_range_m=distribution(ranges[local_ids,peak[select]]),
                    peak_target_x_projection_gap_cm=distribution(100*gaps[local_ids,peak[select]]),
                    max_smoothed_score=distribution(score.max(1)[select]))
            ix=np.flatnonzero(stop);ft=first[stop]
            summary['temporal']=dict(first_alarm_frame_hist={str(t+3):int(np.sum(ft==t)) for t in range(13)},
                first_alarm_actual_range_m=distribution(ranges[ix,ft]),
                first_alarm_target_x_projection_gap_cm=distribution(100*gaps[ix,ft]),
                alarm_frame_count_hist={str(k):int(np.sum(hit[stop].sum(1)==k)) for k in range(1,14)},
                first_in_initial3_frames=int((stop&(first<=2)).sum()),first_in_final3_frames=int((stop&(first>=10)).sum()),
                final_frame_above_threshold=int((stop&hit[:,-1]).sum()),earlier_alarm_but_final_below=int((stop&~hit[:,-1]).sum()))
            chosen=stack.argmax(0)
            summary['first_alarm_winning_angle_hist']={str(a):int(np.sum(chosen[ix,ft]==j)) for j,a in enumerate(angles)}
            summary['first_alarm_winning_angle_ties']=int(sum(np.sum(stack[:,i,t]==stack[:,i,t].max())>1 for i,t in zip(ix,ft)))
            # Complete side-by-context table; no selected subgroup is omitted.
            summary['side_context']={}
            for side in ('negative_x','positive_x'):
                for ctx in ('none','panel'):
                    m=np.array([x['side']==side and x['context']==ctx for x in metadata]);n=int(m.sum());k=int((m&stop).sum())
                    summary['side_context'][side+'|'+ctx]=dict(n=n,alarm=k,no_alarm=n-k,alarm_fraction=k/n if n else None)
            event_ledger[key]=[dict(metadata[i],alarm=bool(stop[i]),first_alarm_frame=int(first[i]+3) if stop[i] else None,
                peak_frame=int(peak[i]+3),peak_score=float(score[i,peak[i]]),alarm_frame_count=int(hit[i].sum()),
                scores=score[i].tolist(),actual_target_ranges=ranges[i].tolist(),target_x_projection_gap_cm=(100*gaps[i]).tolist()) for i in range(939)]
            cells[key]=summary
    output=dict(status='COMPLETE_DESCRIPTIVE',cells=cells,original_verdict=result['verdict'],
        population='All939 source-evaluation same-height nominal outside10-20cm target queries intersected with all13-frame all-object clear; BASE and ENV at all3 completed biases',
        limits=['No posthoc significance test, selected population or independent causal claim',
            'Scene metadata/true target boxes are evaluator-only descriptions, not deployable inputs',
            'none still includes floor and backwall; panel association is not a paired intervention and cannot attribute alarm origin',
            'All frame truth is clear: none of the physical surfaces should trigger this query, but scores have no object attribution',
            'target_x_projection_gap only projects all8corners onto x; ignores y/z clipping and is not physical swept-body clearance',
            'first alarm distribution conditions on alarm; no-alarm peak is a descriptive comparison, not matched alarm time',
            'Mode/side/context/query/gap/range are correlated; subgroups overlap and cannot be added as separate causes',
            'All model thresholds and original result judgments unchanged; no candidate gate, retraining, new inference or rendering'],
        input_sha256={str(p):sha(p) for p in (RUN/'result.json',RUN/'scores_receipt.json',GEOM/'geometry.npz',GEOM/'rows.json',MC/'scene_manifest.json')},script_sha256=sha(__file__))
    (OUT/'diagnostic.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (OUT/'episode_ledger.json').write_text(json.dumps(event_ledger,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# 身体外10–20cm清晰误报：缓存描述', '', '全体939条，阈值和旧判读不变；不做事后显著性或因果判定。', '']
    for key,s in cells.items():
        lines += [f'## {key}', '', f"报警{s['alarm']}/939，{s['rate_per_proxy_min']:.3f}次/代理分钟。", '', '| 属性/组 | 报警/全组 | 未报 | 报警率 |', '|---|---:|---:|---:|']
        for dim,groups in s['groups'].items():
            for group,v in groups.items():lines.append(f"| {dim}/{group} | {v['alarm']}/{v['n']} | {v['no_alarm']} | {100*v['alarm_fraction']:.2f}% |")
        t=s['temporal'];lines += ['', f"首报初3帧{t['first_in_initial3_frames']}，末3帧{t['first_in_final3_frames']}；较早报警但末帧低于阈值{t['earlier_alarm_but_final_below']}；首报帧直方图{t['first_alarm_frame_hist']}。", '']
    lines += ['## 边界', '', '元信息只用于描述，不能充作部署输入。none仍有地面和后墙；目标和背景没有逐物体分数，因此不能由该表证明由谁导致误报。完整分数/几何行与未报对照见episode_ledger.json和diagnostic.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:{'alarm':s['alarm'],'groups':s['groups'],'temporal':s['temporal']} for k,s in cells.items() if k.endswith('|ENV')},indent=2))


if __name__=='__main__':main()
