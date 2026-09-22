"""Posthoc binary disagreement diagnosis; never fits or selects score cutoffs."""
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
from run_data_coverage import OUT as SOURCE, ROOT, HERE, read, write, sha, verify
from run_spatial_bce import held, seal
from run_corridor_relative import costs

OUT=ROOT/'artifacts.local/work/ba-branch-disagreement-20260921'
PATTERNS=('common','B_only','N_only','neither')
RULES={0:'A',1:'AND',2:'B_only',3:'B',4:'N_only',5:'N',6:'XOR',7:'OR'}


def pattern(b,n):
    return 'common' if b and n else 'B_only' if b else 'N_only' if n else 'neither'


def current_rule(a,b,n,mask):
    """A always retained; mask consumes only two current public decision bits."""
    if mask not in RULES:
        raise ValueError('Three-bit global pattern mask required')
    return [bool(aa or ((mask & (1 if bb and nn else 2 if bb else 4 if nn else 0)) != 0))
            for aa,bb,nn in zip(a,b,n)]


def partition(rows,mode):
    bins={p:dict(positive_ids=[],negative_ids=[]) for p in PATTERNS}
    for row in rows:
        if row['flags']['A_'+mode]:
            continue
        key=pattern(row['flags']['B_control_'+mode],row['flags']['N_'+mode])
        bins[key]['positive_ids' if row['truth'] else 'negative_ids'].append(row['id'])
    for item in bins.values():
        item.update(positive_frames=len(item['positive_ids']),negative_frames=len(item['negative_ids']))
    return bins


def strata(rows):
    subsets={g:[r for r in rows if (r['layout_relation']=='BOUNDARY')==(g=='Boundary')]
             for g in ('Core','Boundary')}
    for g,subset in list(subsets.items()):
        for layer in ('HEAD','BODY'):
            subsets[g+'/'+layer]=[r for r in subset if r['layer']==layer]
        for t in sorted({r['type_id'] for r in subset}):
            subsets[g+'/'+t]=[r for r in subset if r['type_id']==t]
    return subsets


def metric_module():
    s=importlib.util.spec_from_file_location('disagreement_metrics',HERE/'full_event_metrics_20260920.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.ARMS=tuple(n+'_'+mode for mode in ('current','hold') for n in RULES.values())
    return m


def main():
    verify()
    assert not OUT.exists(),'One diagnostic output; no overwrite'
    dependencies={}
    for sn,names in [('prediction-seal.json',('predictions.json',)),
                     ('evaluation-seal.json',('frame-results.json','metrics.json','result.json'))]:
        sealed=read(SOURCE/sn)
        assert sealed['protocol_sha256']==sha(SOURCE/'protocol.json')
        for name in names:
            assert sha(SOURCE/name)==sealed['hashes'][name],name
            dependencies[name]=sha(SOURCE/name)
        dependencies[sn]=sha(SOURCE/sn)
    dependencies.update({n:sha(SOURCE/n) for n in ('protocol.json','selection.json')})
    OUT.mkdir()
    (OUT/'protocol-before-run.md').write_bytes((HERE/'BRANCH_DISAGREEMENT_PROTOCOL_20260921.md').read_bytes())
    code=('branch_disagreement.py','test_branch_disagreement.py','run_spatial_bce.py',
          'run_corridor_relative.py','full_event_metrics_20260920.py')
    write(OUT/'protocol.json',dict(id=OUT.name,scope='CONSUMED_POSTHOC_BINARY_PATTERN_DIAGNOSTIC',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_text_sha256=sha(OUT/'protocol-before-run.md'),source_hashes=dependencies,
        code_hashes={n:sha(HERE/n) for n in code},pattern_masks=RULES,
        runtime_features=['A current flag','B_control current flag','N current flag'],
        selected_rule=None,training=False,threshold_changes=False,automatic_successor=False))
    rows=read(SOURCE/'frame-results.json');predictions=read(SOURCE/'predictions.json')
    assert len(rows)==len(predictions)==1152
    for r,p in zip(rows,predictions):
        assert r['index']==p['index'] and r['id']==p['id'] and r['flags']==p['flags']
        assert r['current_unknown']==p['current_unknown'] and r['split']=='evaluation'
    assert len({r['base_group_id'] for r in rows})==16 and len({r['clip_id'] for r in rows})==48
    partitions={mode:{name:partition(subset,mode) for name,subset in strata(rows).items()}
                for mode in ('current','hold')}
    groups={g:{mode:partition([r for r in rows if r['base_group_id']==g],mode)
               for mode in ('current','hold')} for g in sorted({r['base_group_id'] for r in rows})}
    # Identities reconcile every added output; neither is not an alarm category.
    for mode in ('current','hold'):
        for group in ('Core','Boundary'):
            subset=strata(rows)[group];bins=partitions[mode][group]
            assert sum(v['positive_frames']+v['negative_frames'] for v in bins.values())==sum(not r['flags']['A_'+mode] for r in subset)
            for arm,parts in [('B_control',('common','B_only')),('N',('common','N_only'))]:
                for y,field in [(True,'positive_frames'),(False,'negative_frames')]:
                    assert sum(bins[p][field] for p in parts)==sum(r['truth']==y and r['flags'][arm+'_'+mode] and not r['flags']['A_'+mode] for r in subset)
    a=[r['flags']['A_current'] for r in rows];b=[r['flags']['B_control_current'] for r in rows];n=[r['flags']['N_current'] for r in rows]
    flags={};cost={}
    for mask,name in RULES.items():
        current=current_rule(a,b,n,mask)
        flags[name+'_current']=current;flags[name+'_hold']=held(current,rows).tolist()
        cost[name]=costs(np.array(current,float),.5,[r['truth'] for r in rows],rows,a)
    replay=[dict(**{k:v for k,v in r.items() if k not in ('flags','current_unknown')},
        flags={k:v[i] for k,v in flags.items()},
        current_unknown={k:r['current_unknown']['A_current'] for k in flags}) for i,r in enumerate(rows)]
    for mode in ('current','hold'):
        for name,old in [('A','A'),('B','B_control'),('N','N')]:
            assert all(r['flags'][old+'_'+mode]==flags[name+'_'+mode][i] for i,r in enumerate(rows))
    write(OUT/'pattern-counts.json',partitions);write(OUT/'group-patterns.json',groups)
    write(OUT/'combination-predictions.json',[{k:r[k] for k in ('id','index','flags','current_unknown')} for r in replay])
    seal(OUT,'combination-prediction-seal.json',['combination-predictions.json'])
    module=metric_module();metrics=module.evaluate(replay)
    group_metrics={g:module.evaluate([r for r in replay if r['base_group_id']==g])
                   for g in sorted({r['base_group_id'] for r in replay})}
    summary={}
    for name in RULES.values():
        summary[name]=dict(costs=cost[name],all_cost_caps_pass=all(v['pass_cost'] for v in cost[name].values()),metrics={})
        for region in ('Core','Boundary'):
            for mode in ('current','hold'):
                m=metrics[region]['arms'][name+'_'+mode];base=metrics[region]['arms']['A_'+mode]
                saved=m['frames'];events=m['events']
                summary[name]['metrics'][region+'_'+mode]=dict(
                    **{k:saved[k] for k in ('TP','FP','FN','recall','precision','FPR','current_unknown')},
                    added_TP=saved['TP']-base['frames']['TP'],added_FP=saved['FP']-base['frames']['FP'],
                    detected_events=m['detected_events'],event_count=m['event_count'],
                    max_detected_delay_s=max((e['first_in_event_alert_delay_s'] for e in events if e['detected']),default=None),
                    missed_event_ids=[e['clip_id'] for e in events if not e['detected']],
                    newly_detected_vs_A=[e['clip_id'] for e,ae in zip(events,base['events']) if e['detected'] and not ae['detected']])
                assert all(not r['flags']['A_'+mode] or r['flags'][name+'_'+mode] for r in replay)
                assert all(e['clip_id']==ae['clip_id'] and (not ae['detected'] or (e['detected'] and e['first_in_event_alert_delay_s']<=ae['first_in_event_alert_delay_s'])) for e,ae in zip(events,base['events']))
    direct=[r['flags']['B_control_hold'] and r['flags']['N_hold'] for r in rows]
    temporal_difference=[r['id'] for i,r in enumerate(rows) if direct[i]!=flags['AND_hold'][i]]
    splice={k:sum(int(r['truth']==y and r['flags'][('N' if r['layer']=='HEAD' else 'B_control')+'_hold']==flag)
                   for r in rows if r['layout_relation']=='BOUNDARY')
            for k,y,flag in [('TP',True,True),('FP',False,True),('FN',True,False)]}
    admissible=[name for name,v in summary.items() if v['all_cost_caps_pass']]
    result=dict(scope='CONSUMED_POSTHOC_DIAGNOSTIC',status='COMPLETE',
        all_eight_global_memoryless_retention_masks_reported=True,admissible_masks=admissible,
        nontrivial_admissible_rescue=[name for name in admissible if summary[name]['metrics']['Boundary_hold']['added_TP']>0],
        hold_intersection_not_same_as_hold_of_current_intersection_ids=temporal_difference,
        family_splice_arithmetic=splice,family_splice_is_implemented_router=False,
        selected_mask=None,threshold_changes=False,training=False,automatic_successor=False,
        scope_limit='Not an upper bound on continuous-score, input-conditioned, temporal or new-evidence methods')
    write(OUT/'combination-metrics.json',metrics);write(OUT/'combination-group-metrics.json',group_metrics)
    write(OUT/'combination-summary.json',summary);write(OUT/'result.json',result)
    write(OUT/'local-inheritance.json',dict(terminal_id=OUT.name,inheritance_role='COMPONENT_OR_CHALLENGER',
        inheritance_mode='COMPONENT',role_scope='Consumed binary-disagreement diagnostic only; no algorithm promotion',
        retained_surface='A UNKNOWN and all saved B/N/R dispositions',
        revisit_trigger='Separately justified inference-time evidence and isolated validation',
        automatic_successor=False))
    seal(OUT,'diagnostic-seal.json',['pattern-counts.json','group-patterns.json','combination-metrics.json',
        'combination-group-metrics.json','combination-summary.json','result.json','local-inheritance.json'])
    verify()
    for name,digest in dependencies.items():
        assert sha(SOURCE/name)==digest,name
    print(result,flush=True)


if __name__=='__main__':
    main()
