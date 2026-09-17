"""Prepare the frozen source/bundle from authenticated existing capture helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE.parent))
import confirmation_source
import mz170_fresh_source as inherited_freeze

WORK=ROOT/'artifacts.local/work/corridor-depth-confirmation-20260917'
SOURCE=WORK/'source'
OLD=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916/source'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def helpers():
    SOURCE.mkdir(parents=True,exist_ok=False)
    names=['launch_capture.py','prepare_bundle.py','pack_capture.py','verify_returned.py',
        'upload_files.ps1','return_capture.ps1','worker_prepare_directory.ps1',
        'worker_prepare_bundle.ps1','worker_inspect_ready.ps1','worker_dispatch_capture.ps1',
        'worker_capture_status.ps1','worker_verify_release.ps1','worker_pack_capture.ps1']
    bindings={}
    for name in names:
        p=OLD/name
        value=p.read_text(encoding='utf-8-sig')
        value=value.replace('mz170-mean-confirmation-20260916','corridor-depth-confirmation-20260917')
        value=value.replace('mz170-source-capture-20260916','s1-confirmation-capture-20260917')
        value=value.replace('mz170_fresh_source','confirmation_source')
        value=value.replace('MZ170','S1_CONFIRMATION').replace('mz170_capture_capsule','s1_capture_capsule')
        if name=='prepare_bundle.py':
            value=value.replace("shutil.copyfile(near/name, bundle/name)","shutil.copyfile((near/'corridor_fusion_v1'/name) if name=='confirmation_source.py' else (near/name), bundle/name)")
        if name=='verify_returned.py':
            value=value.replace("from confirmation_source import", "sys.path.insert(0,str(repo/'research/active/dtr-r0/nearfield/corridor_fusion_v1'))\nfrom confirmation_source import")
        (SOURCE/name).write_text(value,encoding='utf-8')
        bindings[str(p)]=sha(p)
    write(SOURCE/'helper-provenance.json',dict(originals=bindings,
        adaptation='TASK_IDENTITIES_PATHS_SOURCE_MODULE_ONLY; inherited Zen port adapter retained',
        prepared={n:sha(SOURCE/n) for n in names}))

def freeze(recipe):
    # Reuse the existing no-capture freeze function with explicit new task bindings.
    inherited_freeze.WORK=WORK
    inherited_freeze.SOURCE=SOURCE
    inherited_freeze.SEED=confirmation_source.SEED
    inherited_freeze.AUTHORITY=confirmation_source.AUTHORITY
    inherited_freeze.source=confirmation_source.source
    inherited_freeze.check_source=confirmation_source.check_source
    result=inherited_freeze.freeze(SOURCE/'design-v1',recipe,sha(recipe))
    p=SOURCE/'design-v1/freeze.json';value=json.loads(p.read_text())
    extra=[HERE/'confirmation_source.py',Path(__file__),HERE/'CONFIRMATION_PROTOCOL_20260917.md']
    for index,item in enumerate(extra):
        value['frozen_inputs'][str(item.resolve())]=sha(item)
        dest=SOURCE/'design-v1/frozen-inputs'/('extra-'+str(index)+'-'+item.name)
        shutil.copyfile(item,dest)
        value['source_snapshots'][str(dest.relative_to(p.parent)).replace('\\','/')]=sha(dest)
    value['source_sha256']=sha(HERE/'confirmation_source.py')
    value['source_seed']=confirmation_source.SEED
    write(p,value)
    print(json.dumps(dict(result,freeze_sha256=sha(p)),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--helpers',action='store_true');parser.add_argument('--recipe',type=Path)
    args=parser.parse_args()
    if args.helpers:helpers()
    else:freeze(args.recipe.resolve(strict=True))
