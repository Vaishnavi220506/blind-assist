"""Unchanged MZ175 public inference on the consumed MZ146 cohort."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from run_mz175_radar_consensus import (ROOT, CODE, WORK, read, write, sha,
    predict_frame, public_observations, local_dependencies, summarize)

PARENT = ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
CAP = PARENT/'source/returned-v1/capture-v1'
DISCOVERY = WORK/'run-v2'


def run(output):
    output = output.resolve();start=time.perf_counter()
    assert output.is_relative_to(WORK.resolve()) and not output.exists()
    output.mkdir(parents=True)
    inputs={}
    def checked(p,expected=None,decode=False):
        h=sha(p);assert expected is None or h==expected,str(p)
        inputs[str(p.resolve())]=h
        return read(p) if decode else h
    try:
        d=checked(DISCOVERY/'completion.json',decode=True);assert d['status']=='PASS'
        s=checked(DISCOVERY/'summary.json',d['hashes']['summary.json'],True);assert s['passed']
        ps=checked(DISCOVERY/'prediction-seal.json',d['hashes']['prediction-seal.json'],True)
        fr=checked(DISCOVERY/'freeze.json',ps['freeze_sha256'],True)
        for name in ('mz175_radar_consensus.py','mz111_spatial_evidence.py','mz107_rgb_association.py'):
            p=(CODE/name).resolve();checked(p,fr['sources'][str(p)])
        inc=PARENT/'incumbent-v1'
        done=checked(inc/'completion.json',decode=True);assert done['status']=='PASS'
        seal=checked(inc/'prediction-seal.json',done['prediction_seal_sha256'],True)
        source=checked(inc/'input-seal.json',seal['input_seal_sha256'],True)
        for name in ('mz111_spatial_evidence.py','mz107_rgb_association.py','mz136_incumbent.py'):
            p=CODE/name;expected=[h for path,h in seal['source_hashes'].items() if Path(path).name==name]
            assert len(expected)==1;checked(p,expected[0])
        receipt=checked(CAP/'receipt.json',decode=True);assert receipt['status']=='PASS' and receipt['frames']==288
        for name in ('raw.jsonl','spec.json','receipt.json'):
            p=CAP/name;expected=[h for path,h in source['inputs'].items() if Path(path).resolve()==p.resolve()]
            assert len(expected)==1;checked(p,expected[0])
        checked(CAP/'evaluator.jsonl',receipt['hashes']['evaluator.jsonl'])
        old=checked(inc/'predictions.json',seal['predictions_sha256'],True)
        rows=public_observations([json.loads(s) for s in (CAP/'raw.jsonl').read_text().splitlines()])
        ids=[r['id'] for r in rows]
        assert len(ids)==len(set(ids))==288 and all(i.startswith('mz146_') for i in ids)
        assert [p['id'] for p in old['predictions']]==ids and len(old['corrected'])==288
        previous_done=checked(PARENT/'evaluation-v1/completion.json',decode=True)
        assert previous_done['status']=='PASS'
        checked(PARENT/'evaluation-v1/summary.json',previous_done['summary_sha256'])
        sources=local_dependencies(__file__)
        sources[str(CODE/'MZ175_TRANSFER_PROTOCOL_20260916.md')]=sha(CODE/'MZ175_TRANSFER_PROTOCOL_20260916.md')
        for path in sources:
            p=Path(path);dest=output/'source-snapshot'/p.relative_to(ROOT.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        write(output/'freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,sources=sources,
            ids=ids,authority='CONSUMED_MZ146288_UNCHANGED_RECALL_COMPONENT_TRANSFER',compute_cap_s=120))
        tick=time.perf_counter()
        predictions=[predict_frame(r,c,b) for r,c,b in zip(rows,old['corrected'],old['predictions'])]
        inference_s=time.perf_counter()-tick
        write(output/'predictions.json',predictions)
        write(output/'prediction-seal.json',dict(freeze_sha256=sha(output/'freeze.json'),predictions_sha256=sha(output/'predictions.json'),
            inference_seconds=inference_s,authority='PUBLIC_PREDICTIONS_BEFORE_NATIVE_DECODE'))
        write(output/'evaluation-start.json',dict(prediction_seal_sha256=sha(output/'prediction-seal.json')))
        native=[json.loads(line) for line in (CAP/'evaluator.jsonl').read_text().splitlines()]
        assert [e['id'] for e in native]==ids
        from run_mz143_corridor_evidence import native_account
        baseline=[p['baseline'] for p in predictions]
        native_counts=native_account(rows,native,baseline,baseline)
        annotations=[dict(n,family=e['family']) for n,e in zip(native_counts['cases'],native)]
        summary=summarize(rows,predictions,annotations)
        previous=read(PARENT/'evaluation-v1/summary.json')
        assert summary['arms']['baseline']['metrics']==previous['baseline']['metrics']
        summary.update(authority='CONSUMED_MZ146288_NOT_FRESH_CONFIRMATION',inference_seconds=inference_s,
            decision='MZ175_CONSUMED_TRANSFER_RECALL_COMPONENT' if summary['passed'] else 'MZ175_CONSENSUS_TRANSFER_GAIN_NOT_MET')
        write(output/'annotations.json',annotations)
        write(output/'row-metadata.json',[{k:r[k] for k in ('id','episode_id','time_s')} for r in rows])
        write(output/'summary.json',summary)
        write(output/'backend.json',dict(device='cpu',reason='TASK_NOT_GPU_SUITABLE',inference_seconds=inference_s))
        assert all(sha(p)==h for p,h in (sources|inputs).items()) and time.perf_counter()-start<120
        write(output/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=summary['decision'],
            hashes={n:sha(output/n) for n in ('prediction-seal.json','summary.json','annotations.json','row-metadata.json','backend.json')},
            sources_inputs_unchanged=True,resource_state='NO_PERSISTENT_PROCESS_OR_ALLOCATION'))
        print(json.dumps(dict(decision=summary['decision'],checks=summary['checks'],arms={a:r['metrics'] for a,r in summary['arms'].items()})),flush=True)
    except Exception as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
