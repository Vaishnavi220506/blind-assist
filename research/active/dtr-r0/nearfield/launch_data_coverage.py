"""Launch the one frozen 144-clip/3456-frame complete-event UE capture.

The protocol seals the spec plus code_hashes (nearfield filenames) and
input_hashes (repository-relative paths). No source, budget, cohort or
output-overwrite override.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

EXPECTED_FRAMES = 3456
EXPECTED_CLIPS = 144
FRAMES_PER_CLIP = 24
CAPTURE_BUDGET_SECONDS = 3600
EXPECTED_MAP_SHA = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'


from launch_spatial_bce import sha, write


def verify_protocol(protocol, spec, repo):
    data = json.loads(protocol.read_text(encoding='utf-8-sig'))
    assert data['spec_sha256'] == sha(spec), 'Frozen spec mismatch'
    assert data['frames'] == EXPECTED_FRAMES and data['clips'] == EXPECTED_CLIPS
    assert data['capture_timeout_s'] == CAPTURE_BUDGET_SECONDS
    codes, inputs = data['code_hashes'], data['input_hashes']
    assert isinstance(codes, dict) and isinstance(inputs, dict)
    mandatory = {'data_coverage_capture.py', 'launch_data_coverage.py', 'ue_capture_readiness.py', 'data_coverage_spec.py', 'core_transfer_spec.py', 'launch_spatial_bce.py', 'spatial_bce_spec.py'}
    assert mandatory <= set(codes), 'Capture/launcher/readiness code must be frozen'
    nearfield = Path(__file__).resolve().parent
    resolved = {}
    for base, files in ((nearfield, codes), (repo, inputs)):
        for name, digest in files.items():
            assert not Path(name).is_absolute(), 'Protocol paths must be relative'
            path = (base/name).resolve()
            assert path.is_relative_to(base.resolve()) or base == repo, 'Code paths must remain under nearfield'
            assert path.is_file() and sha(path) == digest, 'Frozen input mismatch: '+str(path)
            assert path not in resolved or resolved[path] == digest, 'Conflicting frozen path'
            resolved[path] = digest
    return resolved


def validate_cases(data):
    cases = data['cases']
    assert data['expected_map_sha256'] == EXPECTED_MAP_SHA
    assert len(cases) == EXPECTED_FRAMES
    assert len({c['name'] for c in cases}) == EXPECTED_FRAMES
    assert len({c['clip_id'] for c in cases}) == EXPECTED_CLIPS
    for offset in range(0, EXPECTED_FRAMES, FRAMES_PER_CLIP):
        clip = cases[offset:offset+FRAMES_PER_CLIP]
        assert len({c['clip_id'] for c in clip}) == 1
        assert [c['frame_in_clip'] for c in clip] == list(range(FRAMES_PER_CLIP))
        assert all(clip[i]['time_s'] < clip[i+1]['time_s'] for i in range(FRAMES_PER_CLIP-1))
        assert all(c['objects'] == clip[0]['objects'] for c in clip)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for option in ('spec', 'protocol', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--plugin', type=Path, default=os.environ.get('BLINDASSIST_UE_CAPTURE_PLUGIN',
        str(Path(__file__).resolve().parents[4]/'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin')))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(repo/'tools'), str(repo/'research/active/dtr-r0/unreal')]
    from run_obstacle_research import engine_root
    from street_process_lifecycle import TaskProcessTree
    root = (repo/'artifacts.local').resolve()
    spec, protocol, out = (p.resolve() for p in (args.spec, args.protocol, args.output))
    assert all(p.is_relative_to(root) and p.is_file() for p in (spec, protocol))
    assert out.is_relative_to(root) and not out.exists(), 'New canonical capture output required'
    capture_script = Path(__file__).with_name('data_coverage_capture.py').resolve()
    helper = capture_script.with_name('ue_capture_readiness.py')
    frozen = verify_protocol(protocol, spec, repo)
    data = json.loads(spec.read_text(encoding='utf-8-sig'))
    validate_cases(data)
    project = repo/'artifacts.local/unreal/BlindAssistStreetLab'
    engine = engine_root()
    map_file = project/'Content/StreetLab/WillowSampleV1.umap'
    assert sha(map_file) == EXPECTED_MAP_SHA
    plugin = args.plugin.resolve()
    assert plugin.name == 'BlindAssistCapture.uplugin' and plugin.is_file()
    plugin_binary = plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'
    assert plugin_binary.is_file()
    assets = {}
    for case in data['cases']:
        for obj in case['objects']:
            mesh = {'cube': 'Cube', 'cylinder': 'Cylinder'}[obj['kind']]
            paths = [engine/f'Engine/Content/BasicShapes/{mesh}.uasset',
                     project/('Content/'+obj['material'].removeprefix('/Game/')+'.uasset')]
            for path in paths:
                assets[str(path)] = sha(path)
    source = dict(spec_sha256=sha(spec), protocol_sha256=sha(protocol),
        frozen_files={str(p): h for p, h in frozen.items()}, capture_script_sha256=sha(capture_script),
        launcher_sha256=sha(__file__), map_sha256=sha(map_file),
        uproject_sha256=sha(project/'BlindAssistStreetLab.uproject'), readiness_helper_sha256=sha(helper),
        plugin_path=str(plugin), plugin_sha256=sha(plugin), plugin_binary_sha256=sha(plugin_binary),
        assets=assets, engine=str(engine), project=str(project), sampling=data['sampling'],
        frame_count=EXPECTED_FRAMES, clip_count=EXPECTED_CLIPS, budget_seconds=CAPTURE_BUDGET_SECONDS)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'launch-receipt.json', source)
    env = dict(os.environ, BA_CAMERA_CORRIDOR_SPEC=str(spec), BA_CAMERA_CORRIDOR_OUTPUT=str(out),
        BA_CAMERA_CORRIDOR_SCRIPT=str(capture_script), BA_CAMERA_CORRIDOR_PROTOCOL_SHA=source['protocol_sha256'],
        BA_CAMERA_CORRIDOR_SPEC_SHA=source['spec_sha256'], BA_CAMERA_CORRIDOR_SCRIPT_SHA=source['capture_script_sha256'])
    env['UE-LocalDataCachePath'] = str(project/'DerivedDataCache')
    startup = subprocess.STARTUPINFO() if os.name == 'nt' else None
    if startup is not None:
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    tree = None
    terminal = dict(status='FAIL', budget_seconds=CAPTURE_BUDGET_SECONDS)
    try:
        verify_protocol(protocol, spec, repo)
        assert sha(protocol) == source['protocol_sha256']
        proc = subprocess.Popen([str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'), str(project/'BlindAssistStreetLab.uproject'),
            '-ExecCmds=py '+capture_script.as_posix(), '-RenderOffscreen', '-unattended', '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared',
            '-abslog='+str(out/'editor.log'), '-PLUGIN='+str(plugin)], env=env, startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        tree = TaskProcessTree(proc, owner=str(out))
        code = tree.wait(timeout=CAPTURE_BUDGET_SECONDS)
        receipt = json.loads((out/'receipt.json').read_text())
        assert code == 0 and receipt['status'] == 'PASS' and receipt['frame_count'] == EXPECTED_FRAMES
        assert receipt['source_unchanged'] and receipt['task_actors_released']
        assert receipt['readiness_helper_sha256'] == source['readiness_helper_sha256'] == sha(helper)
        assert len(receipt['view_readiness']) == EXPECTED_FRAMES and all(r['status'] == 'READY' for r in receipt['view_readiness'])
        assert sha(plugin) == source['plugin_sha256'] and sha(plugin_binary) == source['plugin_binary_sha256']
        assert receipt['protocol_sha256'] == source['protocol_sha256'] and receipt['spec_sha256'] == source['spec_sha256']
        assert receipt['script_sha256'] == source['capture_script_sha256']
        assert sha(map_file) == source['map_sha256']
        assert sha(project/'BlindAssistStreetLab.uproject') == source['uproject_sha256']
        assert sha(protocol) == source['protocol_sha256']
        verify_protocol(protocol, spec, repo)
        for path, digest in assets.items():
            assert sha(path) == digest
        terminal.update(status='PASS', returncode=code, frame_count=EXPECTED_FRAMES)
        print(json.dumps(terminal), flush=True)
    except BaseException:
        terminal['error'] = traceback.format_exc()
        raise
    finally:
        release = tree.cleanup() if tree is not None else dict(released=True, process_started=False)
        write(out/'process-release.json', release)
        terminal['processes_released'] = release['released']
        if not release['released']:
            terminal['status'] = 'FAIL'
        write(out/'launcher-terminal.json', terminal)
        if not release['released']:
            raise RuntimeError('Owned process release incomplete')


if __name__ == '__main__':
    main()

