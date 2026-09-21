"""Audit additional saved CNH information; no new CNH scoring or distance recovery."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from cnh_replay import build
from cnh_components import sha, PHASES


def scalar_relation(distance, valid, foreground, background):
    if not valid:
        return 'UNKNOWN', None
    if 0 < distance <= 3:
        return 'SUSPECT_NEAR_ZERO', None
    if foreground is None or background is None or background <= foreground:
        return 'NOT_COMPARABLE', None
    position = (distance-foreground)/(background-foreground)
    df, db = abs(distance-foreground), abs(distance-background)
    return ('FOREGROUND_SIDE' if df < db else 'BACKGROUND_SIDE' if db < df else 'TIE'), position


def consecutive_runs(rows, predicate):
    runs, current = [], []
    for row in rows:
        matches = predicate(row)
        contiguous = not current or (row['phase']==current[-1]['phase'] and row['seq']==current[-1]['seq']+1)
        if current and (not matches or not contiguous):
            runs.append(current)
            current=[]
        if matches:
            current.append(row)
    if current:
        runs.append(current)
    return [{'count':len(r),'first_seq':r[0]['seq'],'last_seq':r[-1]['seq'],
             'phase':r[0]['phase'],'first_elapsed_s':r[0]['elapsed_s'],'last_elapsed_s':r[-1]['elapsed_s'],
             'observed_span_s':round(r[-1]['elapsed_s']-r[0]['elapsed_s'],9)} for r in runs]


def summarize(rows):
    scored=[r for r in rows if r['cnh'] is not None]
    cross=Counter((r['scalar_category'], 'UNSCORED' if r['cnh'] is None else
                   'NEAR_AND_LATE' if r['cnh']['dual'] else 'LATE_ONLY' if r['cnh']['late'] else
                   'NEAR_ONLY' if r['cnh']['near_present'] else 'NEITHER') for r in rows)
    return {'region_frames':len(rows),'scored_region_frames':len(scored),
            'scalar_categories':dict(Counter(r['scalar_category'] for r in rows)),
            'cross_table':[{'scalar':k[0],'cnh':k[1],'count':v} for k,v in sorted(cross.items())],
            'foreground_scalar_with_late':sum(r['scalar_category']=='FOREGROUND_SIDE' and r['cnh']['late'] for r in scored),
            'background_scalar_with_near':sum(r['scalar_category']=='BACKGROUND_SIDE' and r['cnh']['near_present'] for r in scored),
            'unknown_with_dual':sum(r['scalar_category']=='UNKNOWN' and r['cnh']['dual'] for r in scored),
            'saved_dual':sum(r['cnh']['dual'] for r in scored),
            'amplitude_extrapolation':sum(r['cnh']['amplitude_extrapolation'] for r in scored)}


def audit(data, previous):
    if [z['zone'] for z in previous['zones']] != list(range(16)):
        raise ValueError('Missing old scalar references')
    rows=[]
    for f in data['frames']:
        for z in range(16):
            scalar=previous['zones'][z]['scalar']
            fg,bg=(scalar[name]['median_known_mm'] for name in ('foreground','background'))
            category,pos=scalar_relation(f['distance_mm'][z],f['range_valid'][z],fg,bg)
            rows.append({'zone':z,'seq':f['seq'],'phase':f['phase'],'role':f['role'],'elapsed_s':f['elapsed_s'],
                         'distance_mm':f['distance_mm'][z],'scalar_known':bool(f['range_valid'][z]),
                         'scalar_category':category,'normalized_scalar_position':pos,
                         'foreground_reference_mm':fg,'background_reference_mm':bg,
                         'cnh_eligible':data['zones'][z]['eligible'],'cnh':f['scores'][z]})
    zones=[]
    for saved in data['zones']:
        z=saved['zone']
        zr=[r for r in rows if r['zone']==z and r['phase']=='mixture']
        entry={'zone':z,'eligible':saved['eligible'],'prior_verdict':saved['verdict'],
               'prior_passing':saved['verdict']=='EXPLORATORY_TAIL_EXCESS','reasons':saved['reasons'],
               'mixture':summarize(zr),'foreground_reference_mm':zr[0]['foreground_reference_mm'],
               'background_reference_mm':zr[0]['background_reference_mm']}
        d=[r['distance_mm'] for r in zr if r['scalar_known']]
        entry['known_distance_min_median_max_mm']=[min(d),float(np.median(d)),max(d)] if d else None
        entry['dual_runs']=consecutive_runs(zr,lambda r:r['cnh'] is not None and r['cnh']['dual'])
        entry['foreground_plus_late_runs']=consecutive_runs(zr,lambda r:r['scalar_category']=='FOREGROUND_SIDE' and r['cnh'] is not None and r['cnh']['late'])
        entry['background_plus_near_runs']=consecutive_runs(zr,lambda r:r['scalar_category']=='BACKGROUND_SIDE' and r['cnh'] is not None and r['cnh']['near_present'])
        zones.append(entry)
    mixed=[r for r in rows if r['phase']=='mixture']
    eligible=[r for r in mixed if r['cnh_eligible']]
    passed=[r for r in mixed if data['zones'][r['zone']]['verdict']=='EXPLORATORY_TAIL_EXCESS']
    return {'rows':rows,'zones':zones,'all_mixture':summarize(mixed),'eligible_mixture':summarize(eligible),
            'prior_passing_mixture':summarize(passed),
            'scored_controls':{name:summarize([r for r in rows if r['phase']==name and r['cnh'] is not None])
                               for name in ('background','foreground','return')}}


def byte_cost(run):
    raw=(run/'tof/raw.bin').read_bytes()
    actual,projected=[],[]
    for line in raw.splitlines(keepends=True):
        try:
            obj=json.loads(line)
        except (ValueError,UnicodeError):
            continue
        if obj.get('type')!='cnh_frame':
            continue
        actual.append(len(line))
        projection={k:obj[k] for k in ('type','seq','ms','distance_mm','target_status','nb_target')}
        projected.append(len((json.dumps(projection,separators=(',',':'))+'\n').encode()))
    return {'raw_stream_bytes':len(raw),'cnh_json_frames':len(actual),'actual_cnh_line_bytes':sum(actual),
            'actual_mean_line_bytes':float(np.mean(actual)),
            'scalar_projection_total_bytes':sum(projected),'scalar_projection_mean_line_bytes':float(np.mean(projected)),
            'serialization_ratio':sum(actual)/sum(projected),
            'projection_warning':'Counterfactual compact subset JSON; includes original type label, no alternate firmware or physical USB benchmark.'}


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,4,figsize=(15,10),constrained_layout=True)
    positions=[r['normalized_scalar_position'] for r in report['rows']
               if r['phase']=='mixture' and r['normalized_scalar_position'] is not None]
    # Keep every reference-relative value visible, including values beyond B.
    ymin,ymax=min(-.25,min(positions)-.07),max(1.1,max(positions)+.07)
    for ax,z in zip(axs.flat,report['zones']):
        rows=[r for r in report['rows'] if r['zone']==z['zone'] and r['phase']=='mixture']
        origin=rows[0]['elapsed_s']
        for r in rows:
            c=r['cnh']
            color='#067f78' if c and c['dual'] else '#b5773d' if c else '#92989e'
            y=r['normalized_scalar_position']
            if y is None:
                ax.scatter(r['elapsed_s']-origin,-.18,marker='x',color='#bf4343',s=28)
            else:
                ax.scatter(r['elapsed_s']-origin,y,color=color,s=19)
        ax.axhline(0,color='#999999',linewidth=.7);ax.axhline(1,color='#999999',linewidth=.7)
        ax.axhline(.5,color='#cccccc',linewidth=.7,linestyle=':')
        ax.set_ylim(ymin,ymax);ax.set_yticks([0,.5,1],['F','.5','B'])
        label=f"dual {z['mixture']['saved_dual']}/22" if z['eligible'] else 'CNH N/E | 22 unscored'
        ax.set_title(f"Z{z['zone']} | {label}",fontsize=9)
        ax.set_xlabel('seconds from first mixed sample',fontsize=7)
    fig.suptitle('Consumed data audit: scalar position relative to old pure-scene references\nTeal: saved near+late; orange: other saved state; gray: unscored; red x: invalid/suspect/comparison missing\nF/B are scalar prototypes, NOT measured physical target positions; no new CNH scoring',fontsize=11)
    fig.savefig(output/'incremental.png',dpi=140)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','report','previous-report','capture-protocol','audit-protocol','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    start=time.perf_counter()
    protocol=json.loads(args.audit_protocol.read_text(encoding='utf-8'))
    if args.run.name != protocol['source_run'] or sha(args.report)!=protocol['score_report_sha256']:
        raise ValueError('Frozen audit source mismatch')
    data=build(args.run.resolve(),args.report.resolve(),args.previous_report.resolve(),args.capture_protocol.resolve())
    previous=json.loads(args.previous_report.read_text(encoding='utf-8'))
    report=audit(data,previous)
    report.update(schema=protocol['schema'],protocol=protocol,audit_protocol_sha256=sha(args.audit_protocol),
                  script_sha256=sha(__file__),loader_sha256=sha(Path(__file__).with_name('cnh_replay.py')),
                  score_report_sha256=sha(args.report),previous_report_sha256=sha(args.previous_report),
                  provenance=data['provenance'],byte_cost=byte_cost(args.run),
                  compute={'backend':'CPU','reason':'TASK_NOT_GPU_SUITABLE: joins and counts of saved scalar records',
                           'seconds':time.perf_counter()-start})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    plot(report,args.output)
    print(json.dumps({k:report[k] for k in ('all_mixture','eligible_mixture','prior_passing_mixture','byte_cost','compute')}))


if __name__=='__main__':
    main()
