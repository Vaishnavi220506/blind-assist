"""Freeze a texture-factorial source using the prior authenticated UE adapter."""
from pathlib import Path
import shutil
import zipfile
from bg_invariance_source import source, check_source
from prepare_single_a import ROOT, read, write, sha

HOME = ROOT/'artifacts.local/work/corridor-bg-invariance-20260918'
OUT = HOME/'source'
OLD = ROOT/'artifacts.local/work/corridor-public-single-20260917/source'
HERE = Path(__file__).resolve().parent


def main():
    assert not HOME.exists()
    OUT.mkdir(parents=True)
    recipe = dict(protocol_sha256=sha(HERE/'BG_INVARIANCE_PROTOCOL_20260918.md'),
        source_sha256=sha(HERE/'bg_invariance_source.py'), seed=189018,
        old_raw_model_sha256=sha(ROOT/'artifacts.local/work/corridor-representation-20260918/raw-final.pkl'),
        arm_losses=['BCE', 'BCE+.25*hinge', 'BCE+.25*hinge+.1*absolute_logit_consistency'],
        steps=120, lr=.001, weight_decay=.01, margin=1., residual='2224->16->1',
        source_revision='2e46bfba2e58da6f16fcdcb2be7eb225ff1b7f75',
        authorized_task_owned_WIP=True, train_new=192, held_new=96,
        scope='ONE_TEXTURE_FACTORIAL_SOURCE_AND_THREE_MATCHED_LOSSES')
    write(HOME/'recipe-freeze.json', recipe)
    spec = source()
    spec['recipe_binding'] = dict(sha256=sha(HOME/'recipe-freeze.json'))
    audit = check_source(spec)
    bundle = OUT/'capture-bundle-v1'
    shutil.copytree(OLD/'capture-bundle-v1', bundle, ignore=shutil.ignore_patterns('__pycache__'))
    for p in OLD.iterdir():
        if p.suffix not in ('.py', '.ps1'):
            continue
        text = p.read_text(encoding='utf-8-sig')
        text = text.replace('corridor-public-single-20260917', 'corridor-bg-invariance-20260918')
        text = text.replace('single-confirmation-20260917', 'bg-invariance-20260918')
        text = text.replace('SINGLE_CONFIRMATION', 'BG_INVARIANCE')
        text = text.replace('single_confirmation_source', 'bg_invariance_source')
        (OUT/p.name).write_text(text, encoding='utf-8')
    # Launch checks only the source contract; existing renderer/sensors unchanged.
    shutil.copyfile(OUT/'launch_capture.py', bundle/'launch_capture.py')
    shutil.copyfile(HERE/'bg_invariance_source.py', bundle/'bg_invariance_source.py')
    shutil.copyfile(HOME/'recipe-freeze.json', bundle/'recipe-freeze.json')
    write(bundle/'spec.json', spec)
    write(bundle/'source-audit.json', audit)
    freeze = dict(source_seed=189018, files={n: sha(bundle/n) for n in ('spec.json', 'source-audit.json', 'recipe-freeze.json')},
        recipe_freeze=dict(sha256=sha(HOME/'recipe-freeze.json')), capture_attempt_budget=1,
        capture_timeout_seconds=None, source_sha256=sha(bundle/'bg_invariance_source.py'),
        inherited_bundle=str(OLD/'capture-bundle-v1'))
    write(bundle/'freeze.json', freeze)
    design = OUT/'design-v1'
    design.mkdir()
    for name in ('spec.json', 'source-audit.json', 'recipe-freeze.json', 'freeze.json'):
        shutil.copyfile(bundle/name, design/name)
    write(bundle/'bundle-manifest.json', {p.name: sha(p) for p in bundle.iterdir() if p.is_file() and p.name != 'bundle-manifest.json'})
    with zipfile.ZipFile(OUT/'capture-bundle-v1.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in bundle.iterdir():
            if p.is_file():
                z.write(p, p.name)
    receipt = dict(status='READY', zip_sha256=sha(OUT/'capture-bundle-v1.zip'), zip_bytes=(OUT/'capture-bundle-v1.zip').stat().st_size,
        bundle_manifest_sha256=sha(bundle/'bundle-manifest.json'), design_freeze_sha256=sha(bundle/'freeze.json'),
        recipe_freeze_sha256=sha(HOME/'recipe-freeze.json'))
    write(OUT/'bundle-receipt.json', receipt)
    write(OUT/'capture-go.json', dict(receipt, status='ROOT_AUTHORIZED_SINGLE_CAPTURE',
        authorization='User requested new background pairs and matched three-loss experiment'))
    print(audit)


if __name__ == '__main__':
    main()
