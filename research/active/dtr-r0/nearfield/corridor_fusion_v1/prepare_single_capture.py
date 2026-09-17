"""Build one frozen source with the already tested no-timeout capture adapter."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from single_confirmation_source import source,check_source
ROOT=Path(__file__).resolve().parents[5];HOME=ROOT/'artifacts.local/work/corridor-public-single-20260917'
OLD=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source';OUT=HOME/'source'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    recipe=HOME/'recipe-freeze.json';assert sha(recipe)=='786d19de6a599c4d562275503d7edbf6ba6fda8bab49951b6ab9ca594cb5d53b'
    spec=source();spec['recipe_binding']=dict(sha256=sha(recipe))
    audit=check_source(spec)
    from test_single_confirmation_source import admit_geometry
    audit.update(admit_geometry(spec))
    design=OUT/'design-v1';design.mkdir();bundle=OUT/'capture-bundle-v1'
    shutil.copytree(OLD/'capture-bundle-v1',bundle)
    for p in OLD.iterdir():
        if p.suffix in ('.py','.ps1'):
            text=p.read_text(encoding='utf-8-sig').replace('corridor-depth-confirmation-recovery-20260917','corridor-public-single-20260917').replace('s1-confirmation-recovery-20260917','single-confirmation-20260917').replace('S1_CONFIRMATION','SINGLE_CONFIRMATION')
            text=text.replace('from confirmation_source import check_source','from single_confirmation_source import check_source')
            text=text.replace("freeze['capture_timeout_seconds'] == 1200","freeze['capture_timeout_seconds'] is None")
            (OUT/p.name).write_text(text,encoding='utf-8')
    for folder in [design,bundle]:
        write(folder/'spec.json',spec);write(folder/'source-audit.json',audit);shutil.copyfile(recipe,folder/'recipe-freeze.json')
    shutil.copyfile(OUT/'launch_capture.py',bundle/'launch_capture.py')
    shutil.copyfile(Path(__file__).with_name('single_confirmation_source.py'),bundle/'single_confirmation_source.py')
    freeze=dict(source_seed=186017,files={n:sha(bundle/n) for n in ['spec.json','source-audit.json','recipe-freeze.json']},
        recipe_freeze=dict(sha256=sha(recipe)),capture_attempt_budget=1,capture_timeout_seconds=None,
        inherited_no_timeout_bundle=str(OLD/'capture-bundle-v1'),source_sha256=sha(bundle/'single_confirmation_source.py'))
    for folder in [design,bundle]:write(folder/'freeze.json',freeze)
    write(bundle/'bundle-manifest.json',{p.name:sha(p) for p in bundle.iterdir() if p.is_file() and p.name!='bundle-manifest.json'})
    archive=OUT/'capture-bundle-v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():
            if p.is_file():z.write(p,p.name)
    receipt=dict(status='READY',zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,bundle_manifest_sha256=sha(bundle/'bundle-manifest.json'),
        design_freeze_sha256=sha(bundle/'freeze.json'),recipe_freeze_sha256=sha(recipe))
    write(OUT/'bundle-receipt.json',receipt);write(OUT/'capture-go.json',dict(receipt,status='ROOT_AUTHORIZED_SINGLE_CAPTURE',authorization='User single complete prospective source; root authorized dispatch'))
    print(json.dumps(dict(receipt=receipt,audit=audit),indent=2))
if __name__=='__main__':main()
