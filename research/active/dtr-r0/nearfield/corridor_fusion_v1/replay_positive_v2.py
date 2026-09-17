"""Public raw-row replay of both completed arms; no evaluator imports."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch
from public_positive_inference import PublicPositivePredictor

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'artifacts.local/work'
HOME=WORK/'corridor-public-positive-v2-20260917'
CAPS={'old':'mz170-mean-confirmation-20260916',
      'new':'corridor-depth-confirmation-recovery-20260917',
      'mz146':'mz146-fresh-corridor-confirmation-20260916',
      'mz158':'mz158-crossview-agreement-20260916'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    torch.set_num_threads(4)
    meta=read(HOME/'preparation/metadata.json')
    reports={}
    for arm in ['bce','peak']:
        out=HOME/arm
        if not (out/'completion.json').exists():
            continue
        assert read(out/'completion.json')['status']=='PASS'
        pred=np.load(out/'predictions.npz')
        by_id={meta[i]['id']:n for n,i in enumerate(pred['indices'])}
        expected=read(out/'cases.json')
        heads={k:PublicPositivePredictor(out/f'outer{k}-final.pt',
            read(out/f'outer{k}-selection.json')['chosen']['threshold']) for k in range(6)}
        errors,times=[],[]
        for cohort,stem in CAPS.items():
            rows=[json.loads(s) for s in (WORK/stem/'source/returned-v1/capture-v1/raw.jsonl').read_text().splitlines()]
            yaw,episode=0.,None
            for row in rows:
                if row['episode_id']!=episode:
                    yaw=0.
                if row['imu_valid']:
                    yaw+=row['delta_yaw']
                episode=row['episode_id']
                n=by_id[row['id']]
                e=expected[n]
                model=heads[int(pred['folds'][n])]
                start=time.perf_counter()
                actual=model.predict(row,yaw,e['A'])
                times.append((time.perf_counter()-start)*1000)
                errors.append(abs(actual['evidence_logit']-e['score']))
                assert actual['alert']==e['calibrated']
                assert actual['positive_evidence']==e['branch_calibrated']
                assert bool(e['A'] or (actual['usable_tof_returns']>0 and actual['evidence_logit']>=0))==e['fixed_zero']
        assert len(errors)==1152 and max(errors)<1e-5
        reports[arm]=dict(frames=len(errors),max_score_error=max(errors),all_decisions_match=True,
            additional_frontend_head_ms=dict(mean=float(np.mean(times)),p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95))))
    result=dict(status='PASS',arms=reports,scope='Public raw ToF/IMU input, host CPU, excludes frozen A/capture/transport; not phone or full-chain latency',
        inference_sha256=hashlib.sha256((Path(__file__).parent/'public_positive_inference.py').read_bytes()).hexdigest())
    (HOME/'public-replay.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
