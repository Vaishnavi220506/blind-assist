"""Launch one owned offscreen UE capture only after a root-owned protocol exists."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plugin',type=Path,default=os.environ.get('BLINDASSIST_UE_CAPTURE_PLUGIN',
        str(Path(__file__).resolve().parents[4]/'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin')))
    args=parser.parse_args();repo=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(repo/'tools'),str(repo/'research/active/dtr-r0/unreal')]
    from run_obstacle_research import engine_root
    from street_process_lifecycle import TaskProcessTree
    root=(repo/'artifacts.local').resolve()
    spec,protocol,out=args.spec.resolve(),args.protocol.resolve(),args.output.resolve()
    assert spec.is_relative_to(root) and spec.is_file()
    assert protocol.is_relative_to(root) and protocol.is_file()
    assert out.is_relative_to(root) and not out.exists(), 'New canonical capture output required'
    data=json.loads(spec.read_text());assert len(data['cases'])==96
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab';engine=engine_root()
    map_file=project/'Content/StreetLab/WillowSampleV1.umap'
    assert sha(map_file)==data['expected_map_sha256']
    capture_script=Path(__file__).with_name('ba_camera_corridor_capture.py')
    helper=capture_script.with_name('ue_capture_readiness.py')
    assert args.plugin is not None, 'Readiness requires --plugin or BLINDASSIST_UE_CAPTURE_PLUGIN'
    plugin=args.plugin.resolve()
    assert plugin.name=='BlindAssistCapture.uplugin' and plugin.is_file()
    plugin_binary=plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'
    assert plugin_binary.is_file()
    assets={}
    for case in data['cases']:
        for obj in case['objects']:
            mesh={'cube':'Cube','cylinder':'Cylinder'}[obj['kind']]
            paths=[engine/f'Engine/Content/BasicShapes/{mesh}.uasset',
                   project/('Content/'+obj['material'].removeprefix('/Game/')+'.uasset')]
            for path in paths:assets[str(path)]=sha(path)
    out.mkdir(parents=True)
    source=dict(spec_sha256=sha(spec),protocol_sha256=sha(protocol),capture_script_sha256=sha(capture_script),
        launcher_sha256=sha(__file__),map_sha256=sha(map_file),uproject_sha256=sha(project/'BlindAssistStreetLab.uproject'),
        readiness_helper_sha256=sha(helper),plugin_path=str(plugin),plugin_sha256=sha(plugin),
        plugin_binary_sha256=sha(plugin_binary),
        assets=assets,engine=str(engine),project=str(project),sampling=data['sampling'])
    (out/'launch-receipt.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    env=dict(os.environ,BA_CAMERA_CORRIDOR_SPEC=str(spec),BA_CAMERA_CORRIDOR_OUTPUT=str(out),
             BA_CAMERA_CORRIDOR_SCRIPT=str(capture_script),BA_CAMERA_CORRIDOR_PROTOCOL_SHA=sha(protocol))
    env['UE-LocalDataCachePath']=str(project/'DerivedDataCache')
    startup=subprocess.STARTUPINFO() if os.name=='nt' else None
    if startup is not None:startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen([str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project/'BlindAssistStreetLab.uproject'),
        '-ExecCmds=py '+capture_script.as_posix(),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-abslog='+str(out/'editor.log'),'-PLUGIN='+str(plugin)],env=env,startupinfo=startup,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    tree=TaskProcessTree(proc,owner=str(out))
    try:
        code=tree.wait(timeout=900)
        receipt=json.loads((out/'receipt.json').read_text())
        assert code==0 and receipt['status']=='PASS' and receipt['frame_count']==96
        assert receipt['source_unchanged'] and receipt['task_actors_released']
        assert receipt['readiness_helper_sha256']==source['readiness_helper_sha256']==sha(helper)
        assert len(receipt['view_readiness'])==96 and all(r['status']=='READY' for r in receipt['view_readiness'])
        assert sha(plugin)==source['plugin_sha256'] and sha(plugin_binary)==source['plugin_binary_sha256']
        assert receipt['protocol_sha256']==source['protocol_sha256'] and receipt['spec_sha256']==source['spec_sha256']
        assert sha(map_file)==source['map_sha256']
        for path,digest in assets.items():assert sha(path)==digest
        print(json.dumps(receipt,indent=2),flush=True)
    finally:
        release=tree.cleanup()
        (out/'process-release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
        if not release['released']:raise RuntimeError('Owned process release incomplete')


if __name__=='__main__':main()
