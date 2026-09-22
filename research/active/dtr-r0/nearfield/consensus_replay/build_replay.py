"""Package frozen A vs A OR (B_control AND N) outputs and saved RGB offline.

No model, threshold, inference, training, capture, or private geometry is loaded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
DATA=ROOT/'artifacts.local/work/ba-data-coverage-20260921'
COMB=ROOT/'artifacts.local/work/ba-branch-disagreement-20260921'
DEFAULT=ROOT/'artifacts.local/work/ba-consensus-replay-20260922/site'
DT=.2


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify(path,expected,checked):
    actual=sha(path)
    if actual!=expected:raise ValueError('Source hash mismatch: '+str(path))
    checked[str(path.resolve())]=actual


def sealed_read(folder,sealname,names,checked):
    seal=read(folder/sealname)
    checked[str((folder/sealname).resolve())]=sha(folder/sealname)
    if 'protocol_sha256' in seal and (folder/'protocol.json').exists():
        verify(folder/'protocol.json',seal['protocol_sha256'],checked)
    for name in names:verify(folder/name,seal['hashes'][name],checked)
    return {name:read(folder/name) for name in names}


def joined_rows(frames,predictions,combinations):
    """Exact identity join, then current-only one-frame hold reset for each clip."""
    p={r['id']:r for r in predictions};c={r['id']:r for r in combinations}
    ids={r['id'] for r in frames}
    if len(ids)!=len(frames) or len(p)!=len(predictions) or len(c)!=len(combinations) or ids!=set(p) or ids!=set(c):
        raise ValueError('Duplicate or mismatched source identities')
    clips=defaultdict(list)
    for row in frames:
        pr,co=p[row['id']],c[row['id']]
        for key in ('index','clip_id','frame_in_clip','time_s'):
            if row[key]!=pr[key]:raise ValueError('Source identity mismatch: '+key)
        if co['index']!=row['index']:raise ValueError('Combination index mismatch')
        if row['flags']!=pr['flags'] or row['current_unknown']!=pr['current_unknown']:
            raise ValueError('Evaluated flags differ from prediction seal')
        clips[row['clip_id']].append(row)
    result=[]
    for ident,clip in sorted(clips.items()):
        clip.sort(key=lambda r:r['frame_in_clip'])
        previous_a=previous_and=False
        for i,row in enumerate(clip):
            if row['frame_in_clip']!=i or abs(row['time_s']-i*DT)>1e-8:
                raise ValueError('Noncontiguous clip sampling')
            flags=row['flags'];a,b,n=(flags[k] for k in ('A_current','B_control_current','N_current'))
            if any(type(v) is not bool for v in (a,b,n,row['truth'],row['current_unknown']['A_current'])):
                raise ValueError('Expected boolean flags and truth')
            conjunction=a or (b and n)
            ahold,andhold=a or previous_a,conjunction or previous_and
            co=c[row['id']]
            expected={'A_current':a,'AND_current':conjunction,'A_hold':ahold,'AND_hold':andhold}
            if any(co['flags'][k]!=v for k,v in expected.items()):raise ValueError('Frozen combination mismatch')
            if flags['A_hold']!=ahold:raise ValueError('Frozen A hold mismatch')
            if any(co['current_unknown'][k]!=row['current_unknown']['A_current'] for k in expected):
                raise ValueError('UNKNOWN was changed by combination')
            result.append(dict(row,display=dict(id=row['id'],time=row['time_s'],truth=row['truth'],
                unknown=row['current_unknown']['A_current'],a=a,b=b,n=n,andCurrent=conjunction,
                aHold=ahold,andHold=andhold,image='images/'+row['id']+'.jpg')))
            previous_a,previous_and=a,conjunction
    return result


def intervals(frames,flags):
    result=[];start=None
    for i in range(len(flags)+1):
        active=i<len(flags) and flags[i]
        if active and start is None:start=i
        if not active and start is not None:
            result.append(dict(startFrame=start,endFrame=i-1,startTime=frames[start]['time'],
                endTime=frames[i-1]['time'],endExclusive=round(frames[i-1]['time']+DT,10),
                samples=i-start,seconds=round((i-start)*DT,10)))
            start=None
    return result


def clip_metrics(frames,key):
    alerts=[f[key] for f in frames];truth=[f['truth'] for f in frames]
    eventspans=intervals(frames,truth)
    if len(eventspans)>1:raise ValueError('Frozen clips contain at most one event')
    event=eventspans[0] if eventspans else None
    in_event=[i for i,(a,t) in enumerate(zip(alerts,truth)) if a and t]
    first_alert=next((f['time'] for f in frames if f[key]),None)
    first_event=frames[in_event[0]]['time'] if in_event else None
    fps=intervals(frames,[a and not t for a,t in zip(alerts,truth)])
    return dict(TP=sum(a and t for a,t in zip(alerts,truth)),FP=sum(a and not t for a,t in zip(alerts,truth)),
        FN=sum(t and not a for a,t in zip(alerts,truth)),firstAlertTime=first_alert,
        eventEntry=event['startTime'] if event else None,firstEventAlertTime=first_event,
        delay=round(first_event-event['startTime'],10) if first_event is not None else None,
        detected=bool(in_event) if event else None,eventCount=len(eventspans),
        fpSegments=len(fps),fpSeconds=round(sum(s['seconds'] for s in fps),10),
        eventSpans=eventspans,fpIntervals=fps,unknownFrames=sum(f['unknown'] for f in frames))


def payload(joined,authoritative,summary):
    groups=defaultdict(list)
    for r in joined:groups[r['clip_id']].append(r)
    clips=[]
    for ident,rows in sorted(groups.items()):
        first=rows[0];frames=[r['display'] for r in rows]
        stratum='Boundary' if first['layout_relation']=='BOUNDARY' else 'Core'
        metrics={arm:clip_metrics(frames,key) for arm,key in (('A','aHold'),('AND','andHold'))}
        tags=[]
        if any(f['truth'] and f['andHold'] and not f['aHold'] for f in frames):tags.append('rescue')
        if any(not f['truth'] and f['andHold'] and not f['aHold'] for f in frames):tags.append('extra_fp')
        if metrics['AND']['detected'] is False:tags.append('miss')
        if metrics['A']['detected'] is False and metrics['AND']['detected'] is True:tags.append('new_event')
        clip=dict(id=ident,layoutId=first['base_group_id'],typeId=first['type_id'],layer=first['layer'],
            relation=first['layout_relation'],stratum=stratum,frames=frames,metrics=metrics,tags=tags,
            durationSeconds=round(len(frames)*DT,10))
        clips.append(clip)
        # Match frozen event times and FP intervals, not merely aggregate totals.
        for arm,m in metrics.items():
            old=authoritative[stratum]['arms'][arm+'_hold']
            ev=next(c for c in old['clips'] if c['clip_id']==ident)
            assert m['firstAlertTime']==ev['whole_clip_first_alert_time_s']
            if ev['event'] is None:assert m['eventCount']==0 and m['detected'] is None
            else:
                e=ev['event']
                assert m['eventEntry']==e['entry_time_s'] and m['firstEventAlertTime']==e['first_in_event_alert_time_s']
                assert m['detected']==e['detected']
                assert m['delay'] is None if e['first_in_event_alert_delay_s'] is None else abs(m['delay']-e['first_in_event_alert_delay_s'])<1e-8
            sourcefps=[s for s in old['false_alert_segments'] if s['clip_id']==ident]
            assert len(sourcefps)==len(m['fpIntervals'])
            for src,dst in zip(sourcefps,m['fpIntervals']):
                assert (src['start_frame'],src['end_frame'],src['samples'])==(dst['startFrame'],dst['endFrame'],dst['samples'])
    totals={}
    for stratum in ('Core','Boundary'):
        totals[stratum]={}
        selected=[c for c in clips if c['stratum']==stratum]
        for arm in ('A','AND'):
            ms=[c['metrics'][arm] for c in selected]
            out={k:sum(m[k] for m in ms) for k in ('TP','FP','FN','eventCount','fpSegments','unknownFrames')}
            out.update(detectedEvents=sum(m['detected'] is True for m in ms),fpSeconds=round(sum(m['fpSeconds'] for m in ms),10),
                maxDelay=max((m['delay'] for m in ms if m['delay'] is not None),default=None))
            old=summary[arm]['metrics'][stratum+'_hold'];full=authoritative[stratum]['arms'][arm+'_hold']
            for k in ('TP','FP','FN'):assert out[k]==old[k]==full['frames'][k]
            assert out['detectedEvents']==old['detected_events']==full['detected_events']
            assert out['eventCount']==old['event_count']==full['event_count']
            assert out['maxDelay']==old['max_detected_delay_s']
            assert out['fpSegments']==full['false_alert_segment_count']
            assert abs(out['fpSeconds']-full['false_alert_sampled_s'])<1e-8
            assert out['unknownFrames']==old['current_unknown']
            totals[stratum][arm]=out
    return dict(meta=dict(frames=len(joined),layouts=len({c['layoutId'] for c in clips}),clips=len(clips),dt=DT,
        source='ba-data-coverage-20260921 + ba-branch-disagreement-20260921 frozen evaluation replay',
        limits='受控合成、已消费 Development 回放；不代表硬件、部署或安全效果。时长为采样数 × 0.2 秒，不是端到端延迟。',
        rule='A_current OR (B_control_current AND N_current); hold = current OR previous current; reset each clip',
        unknownMeaning='A 的当帧 UNKNOWN 独立保留，可与当前或保持后的告警同时出现。',
        intervalConvention='start/end frames and endTime inclusive; endExclusive includes the last sample duration',
        costs=summary['AND']['costs']),summary=totals,clips=clips)


def match_image(row,manifest):
    entry=manifest[row['index']]
    if (entry['sample_index'],entry['clip_id'],entry['frame_in_clip'],entry['time_s'])!=(row['index'],row['clip_id'],row['frame_in_clip'],row['time_s']):
        raise ValueError('RGB manifest identity mismatch')
    if entry['id']!=row['clip_id']+'_'+str(row['frame_in_clip']).zfill(2):raise ValueError('RGB frame name mismatch')
    return entry


def build(output,copy_frontend=True):
    from PIL import Image
    if output.exists():raise FileExistsError('Refusing to overwrite replay output: '+str(output))
    if copy_frontend and not all((HERE/n).is_file() for n in ('index.html','app.js','styles.css')):
        raise FileNotFoundError('Frontend files must be ready before packaging')
    started=time.perf_counter();checked={}
    protocol=read(COMB/'protocol.json');checked[str((COMB/'protocol.json').resolve())]=sha(COMB/'protocol.json')
    for n in ('predictions.json','prediction-seal.json','frame-results.json','evaluation-seal.json','protocol.json'):
        verify(DATA/n,protocol['source_hashes'][n],checked)
    source=sealed_read(DATA,'evaluation-seal.json',['frame-results.json'],checked)['frame-results.json']
    predictions=sealed_read(DATA,'prediction-seal.json',['predictions.json'],checked)['predictions.json']
    combinations=sealed_read(COMB,'combination-prediction-seal.json',['combination-predictions.json'],checked)['combination-predictions.json']
    sealed=sealed_read(COMB,'diagnostic-seal.json',['combination-metrics.json','combination-summary.json'],checked)
    rgb=sealed_read(DATA,'observation-seal.json',['capture/observations/manifest.json'],checked)['capture/observations/manifest.json']
    manifest={r['sample_index']:r for r in rgb['frames']}
    if len(manifest)!=len(rgb['frames']):raise ValueError('Duplicate RGB sample identity')
    joined=joined_rows(source,predictions,combinations)
    data=payload(joined,sealed['combination-metrics.json'],sealed['combination-summary.json'])
    assert (data['meta']['frames'],data['meta']['layouts'],data['meta']['clips'])==(1152,16,48)
    assert all(len(c['frames'])==24 for c in data['clips']) and all(r['split']=='evaluation' for r in joined)
    assert all({c['relation'] for c in data['clips'] if c['layoutId']==layout}=={'INSIDE','BOUNDARY','OUTSIDE'} for layout in {c['layoutId'] for c in data['clips']})
    output.mkdir(parents=True);(output/'images').mkdir()
    image_receipts=[]
    for row in joined:
        entry=match_image(row,manifest);original=DATA/'capture/observations'/entry['rgb_path']
        if not original.resolve().is_relative_to((DATA/'capture/observations').resolve()):raise ValueError('RGB escapes capture root')
        verify(original,entry['rgb_sha256'],checked)
        destination=output/row['display']['image']
        with Image.open(original) as image:
            image=image.convert('RGB');original_size=list(image.size)
            image.thumbnail((640,360),Image.Resampling.LANCZOS)
            image.save(destination,'JPEG',quality=80,optimize=True)
            size=list(image.size)
        image_receipts.append(dict(id=row['id'],sample_index=row['index'],source=str(original.resolve()),
            source_sha256=entry['rgb_sha256'],display=row['display']['image'],display_sha256=sha(destination),
            original_dimensions=original_size,display_dimensions=size))
    serialized=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (output/'data.js').write_text('window.REPLAY_DATA='+serialized+';\n',encoding='utf-8')
    if copy_frontend:
        for name in ('index.html','app.js','styles.css','launch.cmd','README.md'):shutil.copyfile(HERE/name,output/name)
    receipt=dict(status='PASS',kind='OFFLINE_ENGINEERING_REPLAY_NOT_NEW_EXPERIMENT',new_inference_calls=0,
        backend='CPU / TASK_NOT_GPU_SUITABLE: frozen flag checks, metadata joins, JPEG display encoding',
        images=len(image_receipts),layouts=16,clips=48,frames=1152,exact_recomputed_combinations=1152,
        hold_scope='nonrecursive previous current, reset per clip',interval_metric_clip_arm_checks=96,
        JPEG_quality=80,maximum_display_dimensions=[640,360],source_hashes=checked,image_provenance=image_receipts,
        summary=data['summary'],output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        builder_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-started)
    (output/'build-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','frames','clips','layouts','images','elapsed_seconds')}))
    return data,receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--data-only',action='store_true',help='Build data and images before frontend packaging')
    args=parser.parse_args();build(args.output,copy_frontend=not args.data_only)
