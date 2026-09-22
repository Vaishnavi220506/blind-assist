"""Governed stage runner for the single-frame Development pilot."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from query_occupancy_data import read, write, sha, stage_path

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO/'artifacts.local/evidence/ba-query-occupancy-20260922'
HERE = Path(__file__).resolve().parent


def runspec(stage, attempt):
    plan = ROOT/'plan'
    if stage == 'capture':
        for src, dst in [(REPO/'artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth', plan/'mobilenet_v3_small.pth'),
                         (HERE/'QUERY_OCCUPANCY_PROTOCOL_20260922.md', plan/'learning-protocol.md')]:
            if not dst.exists():
                shutil.copyfile(src, dst)
            assert sha(src) == sha(dst)
        inputs = [dict(alias='plan', path=str(plan), role='configuration', purpose='new-development-capture-plan'),
                  dict(alias='scene', asset='unreal/BlindAssistStreetLab', relative_path='Content/StreetLab',
                       role='configuration', purpose='existing-source-map-and-materials')]
        result = stage_path(ROOT,'capture')/'launcher-terminal.json'
        command = [sys.executable, str(HERE/'launch_query_occupancy.py'), '--spec', str(plan/'spec.json'),
                   '--protocol', str(plan/'protocol.json'), '--output', str(stage_path(ROOT,'capture'))]
    else:
        inputs = [dict(alias='plan', path=str(plan), role='configuration', purpose='fixed-development-plan')]
        additions = {
            'materialize': [('rgb','capture/observations','observation'), ('native','capture/evaluator','evaluator'),
                            ('receipt','capture/receipt.json','configuration'), ('launch','capture/launch-receipt.json','configuration'),
                            ('release','capture/process-release.json','configuration')],
            'fit': [('observations','prepared/observations','observation'), ('manifest','prepared/materialization.json','configuration'), ('train','prepared/labels/train.npz','evaluator'),
                    ('dev','prepared/labels/dev.npz','evaluator')],
            'predict': [('observations','prepared/observations','observation'), ('weights','fit','configuration')],
            'evaluate': [('observations','prepared/observations','observation'), ('predictions','predictions','evaluator'),
                         ('labels','prepared/labels/evaluation.npz','evaluator'), ('visibility','prepared/labels/visibility-audit.json','evaluator'),
                         ('selection','fit/selection.json','configuration')],
        }[stage]
        inputs += [dict(alias=a,path=str(stage_path(ROOT,p.split('/')[0]).joinpath(*p.split('/')[1:])),
                       role=r,purpose='scoped-'+stage) for a,p,r in additions]
        result = stage_path(ROOT, {'materialize':'prepared','predict':'predictions','evaluate':'evaluated'}.get(stage,stage))/'result.json'
        command = [sys.executable, str(Path(__file__).resolve()), 'execute', '--stage', stage,
                   '--root', str(ROOT), '--result', '{{output:result}}']
    spec = dict(schema='blindassist-asset-run-v1', id='query-occupancy-20260922-'+stage+'-'+attempt,
        route='ue-query-occupancy', question='Does first-hit and localization supervision improve matched single-frame alerts?',
        evaluator='research/active/dtr-r0/nearfield/run_query_occupancy.py',
        evidence_boundary='Same-generator controlled simulation Development; no final, hardware or safety authority',
        reuse=dict(mode='development', query='BODY HEAD nearfield RGB native depth first occupancy distance source'),
        command=command, inputs=inputs, outputs=[dict(alias='result',path=str(result),role='result',required=True)],
        result_output='result', parameters=dict(stage=stage))
    path = ROOT/(stage+'-run-spec-'+attempt+'.json')
    write(path, spec)
    print(path, flush=True)


def execute(stage, root, result):
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or read(journal).get('state') != 'running':
        raise RuntimeError('Use tools/ba.ps1 run research-ue -RunSpec')
    if stage == 'materialize':
        from query_occupancy_data import materialize
        answer = materialize(root)
    elif stage == 'fit':
        from query_occupancy_learning import fit
        answer = fit(root)
    elif stage == 'predict':
        from query_occupancy_learning import seal_predictions
        for name,digest in read(stage_path(root,'fit')/'learning-source-seal.json')['code'].items():
            assert sha(HERE/name) == digest
        answer = seal_predictions(root)
    else:
        from query_occupancy_evaluate import evaluate
        answer = evaluate(root)
    write(result, answer)
    print(json.dumps(answer, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['runspec','execute'])
    p.add_argument('--stage', required=True, choices=['capture','materialize','fit','predict','evaluate'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--result',type=Path)
    p.add_argument('--attempt',default='v2')
    args = p.parse_args()
    if args.command == 'runspec':
        runspec(args.stage, args.attempt)
    else:
        execute(args.stage, args.root, args.result)
