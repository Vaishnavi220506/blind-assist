"""Retry identical source with no total wall-clock cutoff; preserve failed attempt."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[5]
OLD=ROOT/'artifacts.local/work/corridor-depth-confirmation-20260917/source'
WORK=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917'
SOURCE=WORK/'source'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    assert not SOURCE.exists();SOURCE.mkdir(parents=True)
    names=['launch_capture.py','pack_capture.py','verify_returned.py','upload_files.ps1',
        'return_capture.ps1','worker_prepare_directory.ps1','worker_prepare_bundle.ps1',
        'worker_inspect_ready.ps1','worker_dispatch_capture.ps1','worker_capture_status.ps1',
        'worker_verify_release.ps1','worker_pack_capture.ps1','worker_progress_compact.ps1',
        'worker_cleanup_ddc.ps1']
    for name in names:
        text=(OLD/name).read_text(encoding='utf-8-sig')
        text=text.replace('corridor-depth-confirmation-20260917','corridor-depth-confirmation-recovery-20260917')
        text=text.replace('s1-confirmation-capture-20260917','s1-confirmation-recovery-20260917')
        (SOURCE/name).write_text(text,encoding='utf-8')
    launcher=SOURCE/'launch_capture.py';text=launcher.read_text(encoding='utf-8')
    marker="    ast.fix_missing_locations(adapted)"
    replacement="""    timeout_nodes=[n for n in ast.walk(adapted) if isinstance(n,ast.Constant) and n.value==1200]
    assert len(timeout_nodes)==2  # launch metadata and run_owned argument only
    for n in timeout_nodes:n.value=None
    ast.fix_missing_locations(adapted)"""
    assert text.count(marker)==1;text=text.replace(marker,replacement)
    text=text.replace("only_change='REPLACE_EPHEMERAL_BIND_ZERO_WITH_EXISTING_EXCLUSIVE_PROBE_20000_TO_29999',",
        "only_change='INHERITED_ZEN_PORT_ADAPTER_PLUS_NO_TOTAL_WALL_CLOCK_CUTOFF',\n        recovery_authorization='User corrected self-imposed1200s cutoff; complete identical source',")
    text=text.replace('renderer_sensor_capture_and_owned_process_cleanup_unchanged=True',
        'renderer_sensor_capture_and_owned_process_cleanup_unchanged=True,\n        total_timeout_seconds=None, original_attempt_timeout_seconds=1200')
    launcher.write_text(text,encoding='utf-8')
    shutil.copytree(OLD/'design-v1',SOURCE/'design-v1')
    bundle=SOURCE/'capture-bundle-v1';shutil.copytree(OLD/'capture-bundle-v1',bundle)
    shutil.copyfile(launcher,bundle/'launch_capture.py')
    lifecycle=bundle/'street_process_lifecycle.py';text=lifecycle.read_text(encoding='utf-8')
    replacements={
        'deadline = time.monotonic() + self._duration(timeout, "timeout")':'deadline = None if timeout is None else time.monotonic() + self._duration(timeout, "timeout")',
        'remaining = deadline - time.monotonic()':'remaining = interval if deadline is None else deadline - time.monotonic()'}
    start=text.index('    def wait(');end=text.index('    def _states(',start)
    section=text[start:end]
    for a,b in replacements.items():assert section.count(a)==1;section=section.replace(a,b)
    text=text[:start]+section+text[end:]
    ast.parse(text);lifecycle.write_text(text,encoding='utf-8')
    oldmanifest=json.loads((OLD/'capture-bundle-v1/bundle-manifest.json').read_text())
    changed=[n for n,h in oldmanifest.items() if sha(bundle/n)!=h]
    assert sorted(changed)==['launch_capture.py','street_process_lifecycle.py']
    write(bundle/'recovery.json',dict(previous_attempt='corridor-depth-confirmation-20260917/capture-v1',
        reason='Engineering timeout only; raw/evaluator buffers were lost and cannot resume from RGB',
        user_authorized_completion=True,changed_files=changed,total_timeout_seconds=None,
        spec_sha256=sha(bundle/'spec.json'),recipe_sha256=sha(bundle/'recipe-freeze.json'),
        evaluation_unread=True,progress_monitor='Observe same job and frame growth; diagnose actual stalls rather than elapsed total time'))
    manifest={p.name:sha(p) for p in bundle.iterdir() if p.name!='bundle-manifest.json'}
    write(bundle/'bundle-manifest.json',manifest)
    archive=SOURCE/'capture-bundle-v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():z.write(p,p.name)
    receipt=dict(status='RECOVERY_READY',zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,
        bundle_manifest_sha256=sha(bundle/'bundle-manifest.json'),design_freeze_sha256=sha(bundle/'freeze.json'),
        recipe_freeze_sha256=sha(bundle/'recipe-freeze.json'))
    write(SOURCE/'bundle-receipt.json',receipt)
    write(SOURCE/'capture-go.json',dict(receipt,status='ROOT_AUTHORIZED_SINGLE_CAPTURE',
        authorization='User explicitly corrected arbitrary timeout and requested completion'))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
