"""Posthoc train/dev-only score audit; never selects using held-out outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_mz136_corridor_pair import score,retention
from run_mz107_four_sensor import readrows,truth,sha,write


def run(root,output):
    cap=root/'source/returned-v1/capture-v1';fits=root/'full-fits-v1'
    rows=readrows(cap/'raw.jsonl');es=readrows(cap/'evaluator.jsonl')
    spec=json.loads((cap/'spec.json').read_text())
    splits=[f['split'] for f in spec['frames']]
    # Do not compute held-out labels; their scores/outcomes are never inspected.
    gt=np.array([truth(e) if split in ('train','dev') else False for e,split in zip(es,splits)])
    basepath=root/'incumbent/fresh-v1/nominal/predictions.json'
    base=np.array([p['candidate'] for p in json.loads(basepath.read_text())['predictions']])
    result={}
    for arm in ('bce','pair'):
        s=np.load(fits/(arm+'-scores.npy'));result[arm]={}
        for part in ('train','dev'):
            ix=[i for i,v in enumerate(splits) if v==part];bm=score(rows,es,gt,base,ix)
            values=np.unique(s[ix]);thresholds=np.r_[values[0]-1,(values[:-1]+values[1:])/2,values[-1]+1]
            feasible=[]
            for t in thresholds:
                m=score(rows,es,gt,s>=t,ix,base);r=retention(m,bm)
                if r['pass_retention']:
                    feasible.append(dict(logit_threshold=float(t),metrics=m['metrics'],retention=r,
                                         events=m['events'],families=m['families']))
            best=min(feasible,key=lambda r:(r['metrics']['FP'],-r['metrics']['TP'],-r['logit_threshold']))
            result[arm][part]=dict(best_exact_threshold_diagnostic=best,
                positive_score_range=[float(s[ix][gt[ix]].min()),float(s[ix][gt[ix]].max())],
                negative_score_range=[float(s[ix][~gt[ix]].min()),float(s[ix][~gt[ix]].max())],
                zero_threshold=score(rows,es,gt,s>=0,ix,base))
    assert not output.exists()
    write(output,dict(authority='POSTHOC_TRAIN_DEV_ONLY_AFTER_PRIMARY_HOLDOUT_VIEW_NO_TEST_SELECTION',
        values=result,code_sha256=sha(Path(__file__)),
        scores_sha256={a:sha(fits/(a+'-scores.npy')) for a in result}))
    print(json.dumps({a:{part:v['best_exact_threshold_diagnostic']['metrics'] for part,v in parts.items()}
                      for a,parts in result.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
