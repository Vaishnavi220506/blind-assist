"""Frozen MZ129 inference on one new capture, no evaluator parsing or scoring."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
from mz136_incumbent import public_observations,predict_incumbent,sha,read,ROOT


def run(capture,output):
    capture=capture.resolve();output=output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    receipt=read(capture/'receipt.json')
    assert receipt['status']=='PASS' and receipt['frames']==288
    assert sha(capture/'spec.json')==receipt['spec_sha256']
    for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest,name
    rows=public_observations([json.loads(s) for s in (capture/'raw.jsonl').read_text(encoding='utf-8').splitlines()])
    assert len(rows)==len({r['id'] for r in rows})==288
    assert all(r['id'].startswith('mz146_') for r in rows)
    assert cv2.__version__=='5.0.0'
    output.mkdir(parents=True)
    def write(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    inputs={str(capture/name):sha(capture/name) for name in ('receipt.json','raw.jsonl','spec.json')}
    for row in rows:
        path=(capture/row['rgb_path']).resolve();assert path.is_relative_to(capture)
        inputs[str(path)]=sha(path);assert inputs[str(path)]==receipt['hashes'][row['rgb_path']]
    write('input-seal.json',dict(inputs=inputs,authority='FRESH_OBSERVATIONS_ONLY_NO_EVALUATOR_PARSE'))
    def loader(row):
        image=cv2.imread(str(capture/row['rgb_path']));assert image is not None
        return image
    result=predict_incumbent(rows,loader)
    write('predictions.json',result)
    sources={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'
        and Path(m.__file__).resolve().is_relative_to(Path(__file__).parent.resolve())}
    write('prediction-seal.json',dict(predictions_sha256=sha(output/'predictions.json'),
        source_hashes=sources,input_seal_sha256=sha(output/'input-seal.json'),
        authority='FROZEN_MZ129_NO_LABELS_OR_SOURCE_GEOMETRY_IN_PREDICTOR'))
    write('completion.json',dict(status='PASS',frames=288,
        prediction_seal_sha256=sha(output/'prediction-seal.json'),resources='Inference process exits'))
    print(json.dumps(result['audit'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.capture,args.output)
