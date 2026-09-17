"""Posthoc re-evaluation of authenticated stored outputs; zero model execution."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_e1 import ROOT,HERE,CAP,read,write,sha,dataset,labels
from tolerance_eval import TOLERANCES,geometry,classify,metric,temporal

WORK=ROOT/'artifacts.local/work/corridor-tolerance-20260917'
E1=ROOT/'artifacts.local/work/corridor-depth-e1-20260917/run-v2'
S1=ROOT/'artifacts.local/work/corridor-depth-specialist-20260917'
CI=ROOT/'artifacts.local/work/corridor-depth-intermediate-20260917'
NEW=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1'
EV=ROOT/'artifacts.local/work/corridor-depth-confirmation-20260917/evaluation-v1'

def authenticated(path,expected,bindings):
    assert sha(path)==expected,str(path);bindings[str(path)]=expected
    return path

def predictions(name,ids,bindings):
    if name=='old':
        seal=read(S1/'prediction-seal.json');z=np.load(authenticated(S1/'report-predictions.npz',seal['sha256'],bindings))
        authenticated(S1/'selection-seal.json',seal['selection_sha256'],bindings)
        se=read(E1/'prediction-seal.json');scores=np.load(authenticated(E1/'report-scores.npz',se['scores_sha256'],bindings))
        authenticated(E1/'selection-seal.json',se['selection_sha256'],bindings)
        sel=read(E1/'selection-seal.json')['selection'];at=sel['A']['dev_best']['threshold'];bt=sel['B']['dev_best']['threshold']
        sc=read(CI/'prediction-seal.json');authenticated(CI/'selection-seal.json',sc['selection_sha256'],bindings)
        ct=read(CI/'selection-seal.json')['selection']['dev_best']['threshold']
        frozen=read(E1/'freeze.json');assert frozen['ids']['report']==ids
        assert np.array_equal(z['A'],scores['A'])
        return dict(A=z['A']>=at,B=scores['B']>=bt,C=z['C']>=ct,S1=z['final'].astype(bool))
    seal=read(EV/'prediction-seal.json');v=read(authenticated(EV/'predictions.json',seal['predictions_sha256'],bindings))
    authenticated(EV/'input-seal.json',seal['input_seal_sha256'],bindings)
    assert [r['id'] for r in v]==ids
    a=np.array([r['A_alert'] for r in v],bool);s=np.array([r['S1_alert'] for r in v],bool);assert np.array_equal(a,s)
    return dict(A=a,S1=s)

def distance_bin(m):
    if m is None:return 'no x/z relevant object'
    cuts=[-.10,-.05,-.03,-.01,0,.01,.03,.05,.10]
    names=['outside >10cm','outside5-10cm','outside3-5cm','outside1-3cm','outside0-1cm',
        'inside0-1cm','inside1-3cm','inside3-5cm','inside5-10cm']
    for c,n in zip(cuts,names):
        if m<c:return n
    return 'inside >=10cm'

def summarize(name,data,es,y,flags):
    gs=[geometry(e) for e in es];strict=np.array([g['strict'] for g in gs]);assert np.array_equal(strict,y)
    fs=np.array([e['family'] for e in es]);records=[]
    for r,e,g in zip(data['rows'],es,gs):
        records.append(dict(id=r['id'],episode=r['episode_id'],time_s=r['time_s'],family=e['family'],strict_truth=g['strict'],
            lateral_margin_m=g['lateral_margin_m'],signed_min_face_slack_m=g['signed_min_face_slack_m'],
            xz_relevant_objects=g['eligible_objects'],closest_object=g['closest_object'],distance_bin=distance_bin(g['lateral_margin_m'])))
    result=dict(frames=len(y),models=list(flags),no_xz_relevant_object=sum(g['lateral_margin_m'] is None for g in gs),
        strict={a:metric(y,p) for a,p in flags.items()},tolerances={})
    for t in [0.,*TOLERANCES]:
        states=np.array([classify(g,t) for g in gs]);clear=states!='boundary';boundary=~clear
        assert np.array_equal(y[clear],states[clear]=='positive')
        item=dict(coverage=float(clear.mean()),clear_frames=int(clear.sum()),core_positive=int((states=='positive').sum()),
            clear_negative=int((states=='negative').sum()),boundary_frames=int(boundary.sum()),
            boundary_original_positive=int((y&boundary).sum()),boundary_original_negative=int((~y&boundary).sum()),methods={},families={})
        for arm,p in flags.items():
            item['methods'][arm]=dict(clear=metric(y[clear],p[clear]),boundary_strict=metric(y[boundary],p[boundary]),
                boundary_alerts=int(p[boundary].sum()),complete_strict=result['strict'][arm],temporal=temporal(data['rows'],states,p))
        for fam in sorted(set(fs)):
            ix=fs==fam;ci=ix&clear;bi=ix&boundary
            item['families'][fam]=dict(frames=int(ix.sum()),clear_frames=int(ci.sum()),boundary_frames=int(bi.sum()),
                core_positive=int((ix&(states=='positive')).sum()),clear_negative=int((ix&(states=='negative')).sum()),
                methods={a:dict(clear=metric(y[ci],p[ci]),boundary_strict=metric(y[bi],p[bi])) for a,p in flags.items()})
        for i,r in enumerate(records):
            r[f'state_{round(t*100)}cm']=states[i]
            for a,p in flags.items():r[a+'_alert']=bool(p[i])
        result['tolerances'][str(round(t*100))]=item
    bins=[distance_bin(v) for v in [-.11,-.08,-.04,-.02,-.005,.005,.02,.04,.08,.11,None]]
    result['distance_bins']={}
    for b in bins:
        ix=np.array([r['distance_bin']==b for r in records]);result['distance_bins'][b]={a:metric(y[ix],p[ix]) for a,p in flags.items()}
    result['error_distance']={}
    for a,p in flags.items():
        error=p!=y;mm=np.array([g['lateral_margin_m'] if g['lateral_margin_m'] is not None else np.nan for g in gs])
        result['error_distance'][a]=dict(total_errors=int(error.sum()),
            within_cm={str(round(t*100)):int((error&np.isfinite(mm)&(np.abs(mm)<t)).sum()) for t in TOLERANCES},
            remaining_clear_errors={str(round(t*100)):int((error&np.array([classify(g,t)!='boundary' for g in gs])).sum()) for t in TOLERANCES},
            FP_margin_m=mm[error&~y].tolist(),FN_margin_m=mm[error&y].tolist())
        # JSON missingness is null, never NaN.
        for key in ['FP_margin_m','FN_margin_m']:result['error_distance'][a][key]=[float(v) if np.isfinite(v) else None for v in result['error_distance'][a][key]]
    if name=='new':
        for v in result['tolerances'].values():assert v['methods']['A']==v['methods']['S1']
    return result,records

def figures(results):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=dict(A='#2463a6',S1='#e08d24',B='#7b6d9b',C='#6b9a77')
    for col,(name,label) in enumerate([('old','Old Development'),('new','Changed backgrounds')]):
        r=results[name];ts=list(r['tolerances']);x=np.arange(len(ts));ax=axs[0,col]
        for arm in ['A','S1']:
            vals=[r['tolerances'][t]['methods'][arm]['clear']['f1']*100 for t in ts]
            ax.plot(x,vals,'o-',label=arm,color=colors[arm],linestyle='--' if arm=='S1' else '-')
        for i,t in enumerate(ts):
            v=r['tolerances'][t];ax.annotate(f"n={v['clear_frames']} ({v['coverage']:.0%})",(i,57),ha='center',fontsize=9)
        ax.set(title=label+' | clear-sample F1 + coverage',xticks=x,xticklabels=[t+'cm' for t in ts],ylim=(53,102),ylabel='F1 (%)')
        ax.grid(alpha=.2);ax.legend(loc='upper left')
        ax=axs[1,col];bins=list(r['distance_bins']);yfp=[r['distance_bins'][b]['A']['FP'] for b in bins];yfn=[r['distance_bins'][b]['A']['FN'] for b in bins]
        ax.bar(np.arange(len(bins)),yfp,color='#c55d44',label='FP');ax.bar(np.arange(len(bins)),yfn,bottom=yfp,color='#6a77ad',label='FN')
        ax.set(title='All strict A errors by native lateral position',xticks=np.arange(len(bins)),
            xticklabels=['out>10','out5-10','out3-5','out1-3','out0-1','in0-1','in1-3','in3-5','in5-10','in>=10','no x/z'],ylabel='Frame count',xlabel='Signed position bins (cm); all objects, unchanged x/z')
        ax.tick_params(axis='x',rotation=45);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Stored predictions only: tolerance changes evaluated coverage, not the model\nBoundary frames remain separately scored with original strict labels',fontsize=13)
    fig.savefig(WORK/'tolerance-overview.png',dpi=170);fig.savefig(WORK/'tolerance-overview.svg');plt.close(fig)

def main():
    assert not (WORK/'completion.json').exists();WORK.mkdir(parents=True,exist_ok=True)
    bindings={str(p):sha(p) for p in [Path(__file__),HERE/'tolerance_eval.py',HERE/'TOLERANCE_PROTOCOL_20260917.md']}
    write(WORK/'protocol-seal.json',dict(bindings=bindings,tolerances_m=TOLERANCES,model_fits=0,inference_calls=0,threshold_changes=0))
    results={};allrecords=[]
    for name,cap in [('old',CAP),('new',NEW)]:
        data=dataset(cap,'confirmation');es,y=labels(data)
        for n in ['spec.json','receipt.json','raw.jsonl','evaluator.jsonl']:bindings[str(cap/n)]=sha(cap/n)
        flags=predictions(name,[r['id'] for r in data['rows']],bindings)
        result,records=summarize(name,data,es,y,flags);results[name]=result
        allrecords.extend([dict(domain=name,**r) for r in records])
    for name,arm,expected in [('old','A',(137,28,7)),('old','B',(135,32,9)),('old','C',(103,3,41)),('old','S1',(137,24,7)),('new','A',(120,45,24))]:
        assert tuple(results[name]['strict'][arm][k] for k in ['TP','FP','FN'])==expected
    write(WORK/'summary.json',results)
    with (WORK/'frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in allrecords for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrecords)
    figures(results);write(WORK/'input-seal.json',bindings)
    assert all(sha(p)==h for p,h in bindings.items())
    write(WORK/'completion.json',dict(status='PASS',frames=576,strict_result_parity=True,new_A_S1_identical_all_tolerances=True,
        summary_sha256=sha(WORK/'summary.json'),frames_sha256=sha(WORK/'frames.csv'),figure_sha256=sha(WORK/'tolerance-overview.png'),
        model_fits=0,inference_calls=0,threshold_changes=0,new_capture_used=False,scope='POSTHOC_CONSUMED_DEVELOPMENT',backend='CPU_TASK_NOT_GPU_SUITABLE_SMALL_GEOMETRY_AGGREGATION'))
    print(json.dumps({d:{t:dict(n=v['clear_frames'],boundary=v['boundary_frames'],A=v['methods']['A']['clear']) for t,v in r['tolerances'].items()} for d,r in results.items()},indent=2))

if __name__=='__main__':main()
