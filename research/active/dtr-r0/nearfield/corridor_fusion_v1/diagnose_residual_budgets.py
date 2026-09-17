"""Saved-score-only posthoc FP-budget diagnosis; no fitting or calibration writes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tolerance_eval import metric, temporal

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-bg-invariance-20260918'
OUT=HOME/'budget-diagnostic'
ARMS=('raw_hgb','A_star','B0','B1','B2')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def event_changes(current,baseline):
    previous={(e['episode'],e['start_s']):e for e in baseline['events']}
    result=dict(lost=[],gained=[],delayed=[],earlier=[])
    for e in current['events']:
        old=previous[(e['episode'],e['start_s'])]['first_in_core_delay_s']
        new=e['first_in_core_delay_s']
        kind=('lost' if old is not None and new is None else 'gained' if old is None and new is not None
            else 'delayed' if old is not None and new is not None and new>old
            else 'earlier' if old is not None and new is not None and new<old else None)
        if kind:result[kind].append(dict(episode=e['episode'],event_start_s=e['start_s'],baseline_delay_s=old,current_delay_s=new))
    return result

def main():
    assert not OUT.exists(),'Preserve completed diagnostic'
    OUT.mkdir()
    inputs=[HOME/n for n in ('old-predictions.json','old-cases.json','prediction-seal.json','model-seal.json','summary.json')]
    bindings={str(p):sha(p) for p in inputs}
    seal=read(HOME/'prediction-seal.json')
    assert sha(HOME/'old-predictions.json')==seal['outputs']['old-predictions.json']
    assert sha(HOME/'model-seal.json')==seal['model_seal_sha256']
    cases=read(HOME/'old-cases.json');pred=read(HOME/'old-predictions.json')
    assert [c['id'] for c in cases]==[p['id'] for p in pred] and len(cases)==288
    y=np.array([c['truth'] for c in cases],bool);states=np.array([c['stratum'] for c in cases]);clear=states!='boundary'
    strictstates=np.where(y,'positive','negative');native=np.array([c['sampled_witness'] for c in cases])
    def describe(flags):
        return dict(clear=metric(y[clear],flags[clear]),strict=metric(y,flags),boundary=metric(y[~clear],flags[~clear]),
            core=temporal(cases,states,flags),events=temporal(cases,strictstates,flags),
            native_clear=metric(y[clear&native],flags[clear&native]),
            rod=metric(y[clear&np.array([c['family']=='near_rod_farwall' for c in cases])],flags[clear&np.array([c['family']=='near_rod_farwall' for c in cases])]))
    fixed={a:describe(np.array([p['flags'][a] for p in pred])) for a in ARMS}
    curves={};selected={};scores={}
    # Preserve exact HGB probabilities; float32 clipped logits can merge distinct ranks.
    for a in ARMS:
        field='probabilities' if a in ('raw_hgb','A_star') else 'scores'
        s=np.array([p[field][a] for p in pred]);scores[a]=s
        curve=[]
        for t in np.r_[np.nextafter(s.max(),np.inf),np.unique(s)]:
            p=s>=t;desc=describe(p)
            curve.append(dict(threshold=float(t),**desc))
        curves[a]=curve
        selected[a]={}
        for budget in (3,5):
            eligible=[v for v in curve if v['clear']['FP']<=budget]
            chosen=max(eligible,key=lambda v:(v['clear']['TP'],-v['clear']['FP'],v['threshold']))
            selected[a][str(budget)]=dict(chosen=chosen,score_field=field,
                maxTP_plateau=[dict(threshold=v['threshold'],TP=v['clear']['TP'],FP=v['clear']['FP'],
                    core=v['core']['core_events_detected'],strict=v['events']['core_events_detected'])
                    for v in eligible if v['clear']['TP']==chosen['clear']['TP']],
                vs_frozen={b:dict(core=event_changes(chosen['core'],fixed[b]['core']),strict=event_changes(chosen['events'],fixed[b]['events'])) for b in ('raw_hgb','A_star')})
    for a in ARMS:
        for budget in (3,5):
            entry=selected[a][str(budget)];chosen=entry['chosen']
            entry['vs_matched']={b:dict(core=event_changes(chosen['core'],selected[b][str(budget)]['chosen']['core']),
                strict=event_changes(chosen['events'],selected[b][str(budget)]['chosen']['events'])) for b in ('raw_hgb','A_star')}
            # Check every threshold, not only the max-TP tie-break, for joint improvement.
            entry['any_joint_dominance']={}
            for b in ('raw_hgb','A_star'):
                ref=selected[b][str(budget)]['chosen'];witness=[]
                for v in curves[a]:
                    if v['clear']['FP']>ref['clear']['FP'] or v['clear']['TP']<ref['clear']['TP']:continue
                    changes=[event_changes(v[k],ref[k]) for k in ('core','events')]
                    if any(c['lost'] or c['delayed'] for c in changes):continue
                    if v['clear']['TP']>ref['clear']['TP'] or v['clear']['FP']<ref['clear']['FP'] or any(c['gained'] or c['earlier'] for c in changes):
                        witness.append(v['threshold'])
                entry['any_joint_dominance'][b]=witness
    # Independent scalar checks of selected counts and every event onset.
    for a in ARMS:
        for budget in (3,5):
            c=selected[a][str(budget)]['chosen'];p=scores[a]>=c['threshold']
            for mask,key in ((clear,'clear'),(~clear,'boundary'),(np.ones(288,bool),'strict')):
                ii=np.flatnonzero(mask)
                for name,value in dict(TP=sum(bool(y[i] and p[i]) for i in ii),FP=sum(bool(not y[i] and p[i]) for i in ii),FN=sum(bool(y[i] and not p[i]) for i in ii)).items():assert c[key][name]==value
            for key in ('core','events'):
                for e in c[key]['events']:
                    ii=[i for i,row in enumerate(cases) if row['episode_id']==e['episode'] and e['start_s']<=row['time_s']<=e['end_last_sample_s']]
                    hits=[i for i in ii if p[i]]
                    delay=cases[hits[0]]['time_s']-e['start_s'] if hits else None
                    assert delay==e['first_in_core_delay_s']
    write(OUT/'curves.json',curves)
    write(OUT/'summary.json',dict(authority='POSTHOC_CONSUMED_REPORT_DIAGNOSTIC_NOT_CALIBRATION_OR_NEW_GAIN',
        selection='MAX_CLEAR_TP_THEN_FEWER_FP_THEN_HIGHER_THRESHOLD; full plateaus and joint event checks retained',
        coverage=float(clear.mean()),fixed=fixed,selected=selected))
    assert all(sha(Path(p))==h for p,h in bindings.items())
    write(OUT/'completion.json',dict(status='PASS',inputs=bindings,source_sha256=sha(Path(__file__)),
        summary_sha256=sha(OUT/'summary.json'),curves_sha256=sha(OUT/'curves.json'),selected_points_checked=10,
        no_input_or_model_changes=True,training_runs=0,new_frames=0))
    for budget in (3,5):
        for a in ARMS:
            c=selected[a][str(budget)]['chosen']
            print(budget,a,c['clear']['TP'],c['clear']['FP'],c['core']['core_events_detected'],c['events']['core_events_detected'])

if __name__=='__main__':main()
