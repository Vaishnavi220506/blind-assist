"""Hash-bound source capture reusing the successful no-total-timeout capsule."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from run_e1 import ROOT,HERE,sha,read,write
from intrusion_source import source,check_source

OLD=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source'
WORK=ROOT/'artifacts.local/work/corridor-intrusion-20260917'

def prepare(mode,recipe):
    name=f'corridor-intrusion-{mode}-20260917'
    task=ROOT/'artifacts.local/work'/name
    dest=task/'source';assert not dest.exists();dest.mkdir(parents=True)
    helpers=['launch_capture.py','pack_capture.py','verify_returned.py','upload_files.ps1',
        'return_capture.ps1','worker_prepare_directory.ps1','worker_prepare_bundle.ps1',
        'worker_inspect_ready.ps1','worker_dispatch_capture.ps1','worker_capture_status.ps1',
        'worker_verify_release.ps1','worker_pack_capture.ps1','worker_progress_compact.ps1','worker_cleanup_ddc.ps1']
    for n in helpers:
        text=(OLD/n).read_text(encoding='utf-8-sig').replace('corridor-depth-confirmation-recovery-20260917',name)
        text=text.replace('s1-confirmation-recovery-20260917',f'intrusion-{mode}-20260917')
        text=text.replace('confirmation_source','intrusion_source').replace('S1_CONFIRMATION','INTRUSION')
        text=text.replace("freeze['capture_timeout_seconds'] == 1200","freeze['capture_timeout_seconds'] is None")
        (dest/n).write_text(text,encoding='utf-8')
    design=dest/'design-v1';design.mkdir()
    recipe=recipe.resolve(strict=True);spec=source(mode)
    spec['recipe_binding']=dict(path=str(recipe),sha256=sha(recipe))
    write(design/'spec.json',spec);write(design/'source-audit.json',check_source(spec))
    shutil.copyfile(recipe,design/'recipe-freeze.json')
    paths=[HERE/'intrusion_source.py',HERE/'INTRUSION_PROTOCOL_20260917.md',Path(__file__),recipe]
    write(design/'freeze.json',dict(files={n:sha(design/n) for n in ['spec.json','source-audit.json','recipe-freeze.json']},
        recipe_freeze=dict(path=str(recipe),sha256=sha(recipe)),capture_attempt_budget=1,capture_timeout_seconds=None,
        frozen_inputs={str(p):sha(p) for p in paths},source_seed=spec['seed'],mode=mode,
        authority=spec['authority'],report_model_selection_complete=mode=='report'))
    bundle=dest/'capture-bundle-v1';bundle.mkdir()
    for p in (OLD/'capture-bundle-v1').iterdir():
        if p.name not in ['bundle-manifest.json','confirmation_source.py','recovery.json']:
            shutil.copyfile(p,bundle/p.name)
    shutil.copyfile(HERE/'intrusion_source.py',bundle/'intrusion_source.py')
    shutil.copyfile(dest/'launch_capture.py',bundle/'launch_capture.py')
    for n in ['spec.json','source-audit.json','recipe-freeze.json','freeze.json']:shutil.copyfile(design/n,bundle/n)
    write(bundle/'bundle-manifest.json',{p.name:sha(p) for p in bundle.iterdir()})
    archive=dest/'capture-bundle-v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():z.write(p,p.name)
    receipt=dict(status='BUNDLE_READY',zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,
        bundle_manifest_sha256=sha(bundle/'bundle-manifest.json'),design_freeze_sha256=sha(design/'freeze.json'),
        recipe_freeze_sha256=sha(recipe))
    write(dest/'bundle-receipt.json',receipt)
    write(dest/'capture-go.json',dict(receipt,status='ROOT_AUTHORIZED_SINGLE_CAPTURE',
        authorization='User authorized counterfactual training and third-source evaluation',total_timeout_seconds=None))
    print(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','report'],required=True);p.add_argument('--recipe',type=Path,required=True)
    args=p.parse_args();prepare(args.mode,args.recipe)
