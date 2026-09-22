"""Owned HD City Sample illustration snapshot, launch, and source audit.

The shared collector and source maps are never edited. Native inference pairs
remain 640x360; a separately rendered, pose-matched beauty target is 1920x1080.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = ROOT / 'artifacts.local/work/ba-city-showcase-20260920'
PROJECT = ROOT / 'artifacts.local/unreal/CitySample/CitySample.uproject'
PLUGIN = ROOT / 'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def freeze():
    source = HERE.parent / 'city_pcg_capture.py'
    text = source.read_text(encoding='utf-8')
    patches = [
        ('u.TextureRenderTargetFormat.RTF_RGBA8_SRGB, 1280, 720)',
         'u.TextureRenderTargetFormat.RTF_RGBA8_SRGB, 1920, 1080)'),
        ("for kind, actor in (('rgb', rgb), ('depth', depth)):",
         "for kind, actor in (('rgb', rgb), ('depth', depth), ('beauty', beauty)):")]
    for old, new in patches:
        assert text.count(old) == 1, 'Snapshot patch must match exactly once'
        text = text.replace(old, new)
    anchor = "            report.setdefault('capture_poses', []).append(dict(sample_index=index,"
    addition = """            measured['dimensions'] = dict(rgb=[640,360], depth=[640,360], beauty=[1920,1080])
            measured['hfov_degrees'] = {kind: float(actor.capture_component2d.fov_angle)
                for kind, actor in (('rgb', rgb), ('depth', depth), ('beauty', beauty))}
            assert measured['rgb'] == measured['depth'] == measured['beauty'], 'Capture pose mismatch'
            assert all(abs(v - 100.) < 1e-6 for v in measured['hfov_degrees'].values()), 'Capture FOV mismatch'
"""
    assert text.count(anchor) == 1
    text = text.replace(anchor, addition + anchor)
    compile(text, 'city_pcg_capture.py', 'exec')
    target = OUT / 'source/city_pcg_capture.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    dependencies = [HERE.parent / n for n in ('ue_pair_export.py', 'ue_capture_readiness.py',
        'city_native_inspect.py')]
    dependencies += [ROOT / 'tools' / n for n in ('run_city_pcg_capture.py', 'ue_native_capture.py', 'city_render_health.py')]
    write(OUT / 'snapshot-seal.json', dict(authority='ENGINEERING_ILLUSTRATION_NO_EVALUATION_COHORT',
        source=str(source), source_sha256=sha(source), snapshot_sha256=sha(target),
        changes=['Beauty resolution 1280x720 to 1920x1080',
                 'Record/assert RGB, depth, beauty actual poses and common HFOV'],
        dependencies={str(p): sha(p) for p in dependencies},
        builder_sha256=sha(__file__), project_sha256=sha(PROJECT), plugin_sha256=sha(PLUGIN),
        plugin_binary_sha256=sha(PLUGIN.parent / 'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll')))
    print(json.dumps(dict(status='SNAPSHOT_FROZEN', path=str(target))))


def capture(region):
    assert region in ('smoke', 'plaza', 'boulevard', 'residential', 'big01', 'big04', 'smallcity')
    seal = json.loads((OUT / 'snapshot-seal.json').read_text(encoding='utf-8'))
    assert sha(OUT / 'source/city_pcg_capture.py') == seal['snapshot_sha256']
    for p, expected in seal['dependencies'].items():
        assert sha(p) == expected, 'Dependency changed since freeze: ' + p
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    spec = OUT / 'specs' / (region + '.json')
    assert sha(spec) == protocol['spec_hashes'][region]
    sys.path.insert(0, str(ROOT / 'tools'))
    from run_city_pcg_capture import capture as native_capture
    native_capture(SimpleNamespace(project=PROJECT, spec=spec, plugin=PLUGIN,
        output=OUT / region, timeout=2400., engine=Path('F:/epic/UE_5.8'),
        ddc_path=PROJECT.parent / 'DerivedDataCache', capture_source=OUT / 'source/city_pcg_capture.py'))
    audit(region)


def audit(region):
    from PIL import Image
    path = OUT / region
    spec = json.loads((path / 'source/spec.json').read_text(encoding='utf-8'))
    receipt = json.loads((path / 'receipt.json').read_text(encoding='utf-8'))
    release = json.loads((path / 'process-release.json').read_text(encoding='utf-8'))
    health = json.loads((path / 'render-resource-health.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'PASS' and receipt['source_unchanged'] and release['released']
    assert health['ready_data_eligible'], 'Missing native rendering resources'
    assert len(receipt['capture_poses']) == len(spec['cases'])
    rows = []
    for i, case in enumerate(spec['cases']):
        pose = receipt['capture_poses'][i]
        assert pose['sample_index'] == i and pose['rgb'] == pose['depth'] == pose['beauty']
        for key, value in case['camera'].items():
            assert abs(pose['rgb'][key] - value) < 1e-4
        image_path = path / f'appearance/{i:04d}.png'
        with Image.open(image_path) as image:
            image.load()
            assert image.size == (1920, 1080)
        rows.append(dict(id=case['name'], clip_id=case.get('clip_id'),
            frame_in_clip=case.get('frame_in_clip'), time_s=case.get('time_s'),
            rgb_path=str(image_path.relative_to(OUT)), rgb_sha256=sha(image_path),
            sensor_rgb_path=str((path / f'model/sample/{i:04d}.png').relative_to(OUT)),
            native_path=str((path / f'evaluator/native/{i:04d}.npy').relative_to(OUT)),
            native_sha256=sha(path / f'evaluator/native/{i:04d}.npy'), camera=pose['rgb']))
    write(path / 'hd-source-receipt.json', dict(status='PASS', frame_count=len(rows),
        illustration_only=True, source_unchanged=True, task_processes_released=True,
        receipt_sha256=sha(path / 'receipt.json'), release_sha256=sha(path / 'process-release.json'),
        health_sha256=sha(path / 'render-resource-health.json'), frames=rows))
    print(json.dumps(dict(status='HD_SOURCE_PASS', region=region, frame_count=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('freeze', 'capture', 'audit'))
    parser.add_argument('--region')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        globals()[args.stage](args.region)
