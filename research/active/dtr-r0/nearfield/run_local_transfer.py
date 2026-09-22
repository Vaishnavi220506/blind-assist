"""Stage specifications for one frozen LOCAL transfer; never launches a fit."""
import argparse
import json
import os
from pathlib import Path
import sys

from query_occupancy_data import read, write, stage_path

REPO=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ROOT=REPO/'artifacts.local/evidence/ba-local-transfer-20260922'
FROZEN=REPO/'artifacts.local/evidence/ba-inherit-spatial-20260922-run'


def runspec(stage, attempt):
    inputs=[dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='frozen-paired-transfer-plan')]
    prepared=stage_path(ROOT,'prepared')
    predictions=stage_path(ROOT,'predictions')
    if stage=='capture':
        inputs.append(dict(alias='scene',asset='unreal/BlindAssistStreetLab',relative_path='Content/StreetLab',
                           role='configuration',purpose='fixed-existing-map-and-materials'))
        result=stage_path(ROOT,'capture')/'launcher-terminal.json'
        command=[sys.executable,str(HERE/'launch_local_transfer.py'),'--spec',str(ROOT/'plan/spec.json'),
                 '--protocol',str(ROOT/'plan/protocol.json'),'--output',str(result.parent)]
    elif stage=='materialize':
        for alias,part,role in [('rgb','observations','observation'),('native','evaluator','evaluator'),
                              ('receipt','receipt.json','configuration'),('launch','launch-receipt.json','configuration'),
                              ('release','process-release.json','configuration')]:
            inputs.append(dict(alias=alias,path=str(stage_path(ROOT,'capture')/part),role=role,purpose='fixed-observation-construction'))
        result=prepared/'result.json'
        command=[sys.executable,str(Path(__file__).resolve()),'execute','--stage','materialize','--result','{{output:result}}']
    elif stage=='predict':
        for alias,part,role in [('observations','observations','observation'),('materialization','materialization.json','configuration')]:
            inputs.append(dict(alias=alias,path=str(prepared/part),role=role,purpose='fixed-public-inference'))
        for i,name in enumerate(('raw.pkl','local.pkl','selection.json','model-seal.json','freeze.json','feature-seal.json','source-snapshot')):
            inputs.append(dict(alias=f'frozen_{i}',path=str(FROZEN/name),role='configuration',purpose='unchanged-inherited-model'))
        result=predictions/'result.json'
        command=[sys.executable,str(HERE/'local_transfer_inference.py'),'predict','--observations',str(prepared/'observations'),
            '--materialization',str(prepared/'materialization.json'),'--frozen-run',str(FROZEN),
            '--protocol',str(ROOT/'plan/transfer-protocol.md'),'--result','{{output:result}}']
    elif stage=='evaluate':
        inputs += [dict(alias='predictions',path=str(predictions),role='evaluator',purpose='sealed-fixed-outputs'),
                   dict(alias='labels',path=str(prepared/'labels/evaluation.npz'),role='evaluator',purpose='post-seal-full-extent-truth')]
        result=stage_path(ROOT,'evaluated')/'result.json'
        command=[sys.executable,str(HERE/'local_transfer_inference.py'),'evaluate','--predictions',str(predictions),
            '--labels',str(prepared/'labels/evaluation.npz'),'--protocol',str(ROOT/'plan/transfer-protocol.md'),
            '--result','{{output:result}}']
    else:
        raise ValueError(stage)
    spec=dict(schema='blindassist-asset-run-v1',id='local-transfer-20260922-'+stage+'-'+attempt,
        route='ue-local-transfer',question='Does frozen LOCAL retain useful geometry sensitivity under new layouts and paired backdrop appearance?',
        evaluator='research/active/dtr-r0/nearfield/run_local_transfer.py',
        evidence_boundary='New controlled same-generator Development; background-only intervention, no training or hardware claim',
        reuse=dict(mode='development',query='query occupancy local HGB same geometry paired background appearance transfer'),
        command=command,inputs=inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],
        result_output='result',parameters=dict(stage=stage,frames=576,fit=False,threshold_selection=False))
    path=ROOT/(stage+'-run-spec-'+attempt+'.json')
    write(path,spec)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['runspec','execute'])
    p.add_argument('--stage',choices=['capture','materialize','predict','evaluate'],required=True)
    p.add_argument('--attempt',default='v1');p.add_argument('--result',type=Path)
    args=p.parse_args()
    if args.command=='runspec':
        print(runspec(args.stage,args.attempt))
    else:
        journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
        assert journal and read(journal)['state']=='running','Use governed research-ue entry'
        assert args.stage=='materialize'
        from local_transfer_data import materialize
        result=materialize(ROOT)
        write(args.result,result)
        print(json.dumps(result),flush=True)
