"""Independent closed-AABB audit of stored tolerance results, no inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'artifacts.local/work/corridor-tolerance-20260917'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def intersects(e,halfwidth):
    origin=e['body_origin_m'];qlo=[.2,-halfwidth,.4];qhi=[3.6,halfwidth,2.05]
    for obj in e['native_bounds']:
        lo=[c-r-b for c,r,b in zip(obj['center_m'],obj['extent_m'],origin)]
        hi=[c+r-b for c,r,b in zip(obj['center_m'],obj['extent_m'],origin)]
        if all(hi[k]>=qlo[k] and lo[k]<=qhi[k] for k in range(3)):return True
    return False


def run():
    done=read(WORK/'completion.json');assert done['status']=='PASS'
    assert sha(WORK/'summary.json')==done['summary_sha256'] and sha(WORK/'frames.csv')==done['frames_sha256']
    summary=read(WORK/'summary.json');records=list(csv.DictReader((WORK/'frames.csv').open(encoding='utf-8-sig')))
    inputs=read(WORK/'input-seal.json');out={}
    for domain,folder in [('old','mz170-mean-confirmation-20260916'),('new','corridor-depth-confirmation-recovery-20260917')]:
        cap=ROOT/'artifacts.local/work'/folder/'source/returned-v1/capture-v1'
        path=cap/'evaluator.jsonl';assert sha(path)==inputs[str(path)]
        native=[json.loads(line) for line in path.read_text().splitlines()]
        rr=[r for r in records if r['domain']==domain];assert [r['id'] for r in rr]==[e['id'] for e in native]
        out[domain]={}
        for cm in (0,3,5,10):
            tol=cm/100
            states=['positive' if intersects(e,.3-tol) else 'boundary' if intersects(e,.3+tol) else 'negative' for e in native]
            assert states==[r[f'state_{cm}cm'] for r in rr],(domain,cm)
            strict=[intersects(e,.3) for e in native];clear=[s!='boundary' for s in states]
            item=dict(frames=len(rr),clear=sum(clear),boundary=len(rr)-sum(clear),methods={})
            for arm in ('A','S1'):
                pred=[r[arm+'_alert']=='True' for r in rr]
                counts=dict(TP=sum(c and y and p for c,y,p in zip(clear,strict,pred)),
                    FP=sum(c and not y and p for c,y,p in zip(clear,strict,pred)),
                    FN=sum(c and y and not p for c,y,p in zip(clear,strict,pred)),
                    TN=sum(c and not y and not p for c,y,p in zip(clear,strict,pred)))
                expected=summary[domain]['tolerances'][str(cm)]['methods'][arm]['clear']
                assert all(counts[k]==expected[k] for k in counts)
                item['methods'][arm]=counts
            if domain=='new':assert [r['A_alert'] for r in rr]==[r['S1_alert'] for r in rr]
            out[domain][str(cm)]=item
    # Exact binary-coordinate synthetic contact: all-object and thin-central checks.
    def e(objects):return dict(body_origin_m=[0,0,0],native_bounds=objects)
    def o(c,r):return dict(center_m=c,extent_m=r)
    assert intersects(e([o([2,0,1],[.1,.001,.1])]),.2)
    assert intersects(e([o([2,.375,1],[.125,.125,.125])]),.25)
    assert not intersects(e([o([4.2,0,1],[.1,1,1])]),.35)
    assert intersects(e([o([4.2,0,1],[.1,1,1]),o([2,0,1],[.1,.001,.1])]),.2)
    result=dict(status='PASS',source='DIRECT_CLOSED_AABB_INTERSECTION_NO_TOLERANCE_MODULE_IMPORT',
        checked_frames=576,tolerances_cm=[0,3,5,10],results=out,new_A_S1_identical=True,
        summary_sha256=sha(WORK/'summary.json'),frames_sha256=sha(WORK/'frames.csv'),
        model_calls=0,new_capture=False)
    (WORK/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':run()
