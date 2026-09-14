"""Mask-only engineering rerender of frozen MZ123 scene; never calls sensors.

The frozen RGB renderer is instrumented with a categorical unlit material pass.
All logical scene instances, including floor/background and texture child cubes,
remain present for occlusion. Exact palette decoding rejects blended pixels.
Native geometry readbacks determine source admission; RGB differences are
reported separately without an image-error threshold.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import sys

ROOT = Path(os.environ.get('BA_MZ132_REPO_ROOT', str(Path(__file__).resolve().parents[4])))
PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
           [255, 0, 255], [0, 255, 255], [255, 255, 255]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def instrument(source):
    """Fail closed if frozen renderer text does not match the expected surface."""
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old[:80]
        source = source.replace(old, new)
    replace('    from mz115_zonal_sensors import sensors\n', '')
    replace('actors=[];targets=[];captures=[];manifest=[];hashes={}',
            'actors=[];targets=[];captures=[];manifest=[];hashes={};groups=[];original_materials={}\n    palette=' + repr(PALETTE))
    replace('    def textured(obj,owned):', '    def textured_original(obj,owned):')
    replace('    # Controlled native scene boundaries; source geometry is explicit in receipt.', '''    def textured(obj,owned):
        start=len(actors)
        base=textured_original(obj,owned)
        groups.append(list(actors[start:]))
        return base

    # Controlled native scene boundaries; source geometry is explicit in receipt.''')
    replace("        floor_actor=cube(floor['center_m'],floor['size_m'],.22)",
            "        floor_actor=cube(floor['center_m'],floor['size_m'],.22)\n        groups.append([floor_actor])")
    replace("state=dict(i=0,warm=0,prepared=False,after=time.monotonic()+2)",
            "state=dict(i=0,warm=0,prepared=False,phase='rgb',after=time.monotonic()+2)")
    replace('                for a in targets:api.destroy_actor(a);actors.remove(a)', '''                groups[:]=[g for g in groups if not any(a in targets for a in g)]
                for a,mat in original_materials.items():
                    if a not in targets:a.static_mesh_component.set_material(0,mat)
                original_materials.clear()
                left.capture_component2d.set_editor_property('show_flag_settings', [])
                left.capture_component2d.texture_target.target_gamma=2.2
                for a in targets:api.destroy_actor(a);actors.remove(a)''')
    start=source.index("            folder=out/'frame'/frame['id'];folder.mkdir(parents=True,exist_ok=False)")
    end=source.index('            state.update(i=state[\'i\']+1,warm=0,prepared=False)', start)
    source=source[:start]+'''            folder=out/'frame'/frame['id']
            if state['phase']=='rgb':
                folder.mkdir(parents=True,exist_ok=False)
                u.RenderingLibrary.export_render_target(world,left.capture_component2d.texture_target,str(folder),'rgb.png')
                assert len(groups)<=len(palette)
                for group_index,group in enumerate(groups):
                    color=palette[group_index]
                    for actor in group:
                        comp=actor.static_mesh_component
                        original_materials[actor]=comp.get_material(0)
                        material=comp.create_dynamic_material_instance(0,parent)
                        material.set_vector_parameter_value('Color',u.LinearColor(*(v/255 for v in color),1.))
                left.capture_component2d.set_editor_property('show_flag_settings', [
                    u.EngineShowFlagsSetting(show_flag_name=name, enabled=False)
                    for name in ('AntiAliasing','TemporalAA','Tonemapper','PostProcessing')])
                left.capture_component2d.texture_target.target_gamma=1.
                state.update(phase='mask',warm=0)
                return
            u.RenderingLibrary.export_render_target(world,left.capture_component2d.texture_target,str(folder),'instance-rgb.png')
            paths=dict(rgb='frame/'+frame['id']+'/rgb.png',instance_rgb='frame/'+frame['id']+'/instance-rgb.png')
            for rel in paths.values():hashes[rel]=sha(out/rel)
            location=left.get_actor_location();rotation=left.get_actor_rotation()
            native_instances=[]
            for group in groups:
                center,extent=group[0].get_actor_bounds(False)
                native_instances.append(dict(center_m=[center.x/100,center.y/100,center.z/100],
                    extent_m=[extent.x/100,extent.y/100,extent.z/100]))
            manifest.append(dict(id=frame['id'],paths=paths,instance_count=len(groups),palette=palette[:len(groups)],
                renderer_group_sizes=[len(g) for g in groups],evaluator_geometry=dict(
                    camera_location_m=[location.x/100,location.y/100,location.z/100],
                    camera_rotation_deg=[rotation.pitch,rotation.yaw,rotation.roll],
                    capture_hfov_deg=left.capture_component2d.fov_angle,
                    render_target_size=[left.capture_component2d.texture_target.size_x,left.capture_component2d.texture_target.size_y],
                    native_base_instances=native_instances)))
            state['phase']='rgb'
''' + source[end:]
    replace("authority='SINGLE_RGB_NATIVE_COLLISION_TOF_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF'",
            "authority='UE_RENDERED_INSTANCE_COLOR_PASS_ENGINEERING_NOT_YET_ADMITTED' ")
    replace("rgb_camera_count=1,depth_images_produced=False,frames=manifest",
            "rgb_camera_count=1,depth_images_produced=False,sensor_calls=0,frames=manifest")
    # Frozen imports are retained only as file provenance strings; engine does not
    # load sensor modules, and original main() is not invoked in engine mode.
    assert 'row,evaluation,provenance=sensors(' not in source
    return source


def verify_geometry(frame, scene, spec, original_evaluation, require_capture_readback=True):
    geometry=frame['evaluator_geometry'];camera=scene['camera']
    def equal(a,b):
        return len(a)==len(b) and all(math.isclose(x,y,rel_tol=0.,abs_tol=1e-9) for x,y in zip(a,b))
    # Python's UE Rotator constructor narrows each input to float32 before the
    # engine quaternion roundtrip. Compare to that exact input representation;
    # never enlarge a geometry tolerance to accommodate raw decimal mismatch.
    float32=lambda value:struct.unpack('<f',struct.pack('<f',value))[0]
    checks=dict(camera_location=equal(geometry['camera_location_m'],[camera[k] for k in ('x','y','z')]),
        camera_rotation=equal(geometry['camera_rotation_deg'],[float32(camera.get(k,0.)) for k in ('pitch','yaw','roll')]))
    if require_capture_readback:
        checks['capture_hfov']=geometry['capture_hfov_deg']==spec['rig']['hfov_deg']
        checks['render_target_size']=geometry['render_target_size']==[spec['rig']['width'],spec['rig']['height']]
    expected=[];group_sizes=[]
    for name in ('background','floor'):
        obj=spec.get(name)
        if obj:
            expected.append(dict(center_m=obj['center_m'],extent_m=[v/2 for v in obj['size_m']]))
            grid=obj.get('texture_grid',[8,8])
            group_sizes.append(1+grid[0]*grid[1] if name=='background' and not obj.get('material') and obj.get('texture',True) else 1)
    native={obj['name']:obj for obj in original_evaluation['native_bounds']}
    for obj in scene['objects']:
        expected.append(native[obj['name']]);grid=obj.get('texture_grid',[8,8])
        textured=scene.get('appearance','textured')=='textured' and obj.get('texture',True) and not obj.get('material')
        group_sizes.append(1+grid[0]*grid[1] if textured else 1)
    actual=geometry['native_base_instances']
    checks['all_instances_retained']=len(actual)==len(expected)==frame['instance_count']
    checks['base_bounds']=len(actual)==len(expected) and all(equal(a['center_m'],b['center_m']) and equal(a['extent_m'],b['extent_m']) for a,b in zip(actual,expected))
    checks['texture_children_grouped']=frame['renderer_group_sizes']==group_sizes
    return dict(passed=all(checks.values()),checks=checks,
        numeric_roundoff_absolute_tolerance=1e-9,
        camera_rotation_input_representation='IEEE754_FLOAT32_UE_PYTHON_ROTATOR',
        tolerance_reason='UE transform metre/centimetre and quaternion roundtrip arithmetic; Rotator inputs first converted exactly to float32; not image or outcome tolerance')


def verify(out, original_rgb_root, original_receipt):
    import numpy as np
    from PIL import Image
    receipt=json.loads((out/'receipt.json').read_text())
    assert receipt['status']=='PASS'
    manifest=json.loads((out/'manifest.json').read_text())
    frozen=json.loads(original_receipt.read_text())
    evaluatorpath=original_receipt.with_name('evaluator.jsonl')
    assert sha(evaluatorpath)==frozen['hashes']['evaluator.jsonl']
    evaluations={r['id']:r for r in map(json.loads,evaluatorpath.read_text().splitlines())}
    spec=json.loads((out/'spec.json').read_text())
    specs={f['id']:f for f in spec['frames']}
    rows=[];legends=[]
    for frame in manifest['frames']:
        rgb=out/frame['paths']['rgb']; reference=original_rgb_root/frame['paths']['rgb']
        assert sha(reference)==frozen['hashes'][frame['paths']['rgb']]
        a=np.asarray(Image.open(rgb).convert('RGB'))
        b=np.asarray(Image.open(reference).convert('RGB'))
        codes=np.asarray(Image.open(out/frame['paths']['instance_rgb']).convert('RGB'))
        labels=np.zeros(codes.shape[:2],dtype=np.uint8)
        recognized=np.all(codes==0,axis=-1)
        for i,color in enumerate(frame['palette'],1):
            selected=np.all(codes==color,axis=-1);labels[selected]=i;recognized|=selected
        delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
        valid=bool(recognized.all() and np.any(labels>0))
        if valid:Image.fromarray(labels).save(rgb.with_name('instance-id.png'))
        private_names=[]
        if spec.get('background',True):private_names.append('CONTEXT/background')
        if spec.get('floor',True):private_names.append('CONTEXT/floor')
        private_names.extend(obj['name'] for obj in specs[frame['id']]['objects'])
        assert len(private_names)==frame['instance_count']
        legends.append(dict(id=frame['id'],instances=[dict(render_id=i+1,source_name=name,
            palette=frame['palette'][i],rendered_actor_count=frame['renderer_group_sizes'][i])
            for i,name in enumerate(private_names)]))
        geometry=verify_geometry(frame,specs[frame['id']],spec,evaluations[frame['id']]) if 'capture_hfov_deg' in frame.get('evaluator_geometry',{}) else None
        rows.append(dict(id=frame['id'],rgb_equal=bool(np.array_equal(a,b)),
            rgb_mae=float(delta.mean()),rgb_max_error=int(delta.max()),rgb_changed_pixels=int(np.any(delta,axis=-1).sum()),
            exact_palette_decode=valid,nonzero_rendered_pixels=int((labels>0).sum()),unrecognized_pixels=int((~recognized).sum()),
            geometry=geometry,instance_id_sha256=sha(rgb.with_name('instance-id.png')) if valid else None,
            unrecognized_colors=np.unique(codes[~recognized],axis=0).tolist()[:20],
            visible_pixels=[int((labels==i).sum()) for i in range(1,len(frame['palette'])+1)]))
    structural=all(r['exact_palette_decode'] and r['geometry'] and r['geometry']['passed'] for r in rows)
    identical=all(r['rgb_equal'] for r in rows)
    result=dict(status=('GEOMETRY_PASS_RGB_IDENTICAL' if identical else 'GEOMETRY_PASS_RGB_NOT_IDENTICAL') if structural else 'NOT_ADMITTED',
        source='MASK_ONLY_RERENDER_OF_CONSUMED_MZ123',frames=rows,sensor_calls=0,
        original_evaluator_sha256=sha(evaluatorpath),
        limitation='Pixel-center categorical raster with AA disabled; not fractional edge coverage. Geometry readbacks and exact palette decode required. RGB equality disclosed separately; no image-error threshold. No evaluator identity/range/role may enter associator.')
    write(out/'verification.json',result)
    write(out/'evaluator-instance-legend.json',dict(authority='EVALUATOR_ONLY_DO_NOT_PASS_TO_ASSOCIATOR',frames=legends))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rgb-root',type=Path,required=True)
    parser.add_argument('--engine',type=Path,default=Path('F:/epic/UE_5.8'))
    parser.add_argument('--project',type=Path,default=ROOT/'artifacts.local/unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject')
    parser.add_argument('--plugin',type=Path,default=ROOT/'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin')
    parser.add_argument('--ddc',type=Path,default=ROOT/'artifacts.local/work/mz113-dynamic-four-sensor-20260913/ddc')
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--ids',nargs='+')
    selection.add_argument('--all-frames',action='store_true')
    parser.add_argument('--smoke-admission',type=Path)
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    out=args.output.resolve();art=(ROOT/'artifacts.local').resolve()
    assert out.is_relative_to(art) and out!=art and not out.exists()
    import psutil
    assert not any((p.info['name'] or '').startswith('UnrealEditor') for p in psutil.process_iter(['name']))
    source=args.source.resolve();frozen=source/'bundle/research/active/dtr-r0/nearfield'
    original=(frozen/'mz115_zonal_capture.py').read_text()
    receipt=args.receipt or source/'returned-v1/capture-v1/receipt.json'
    specpath=args.spec or source/'spec.json'
    original_receipt=json.loads(receipt.read_text())
    assert sha(frozen/'mz115_zonal_capture.py')==original_receipt['source_sha256']
    spec=json.loads(specpath.read_text())
    assert sha(specpath)==original_receipt['spec_sha256']
    if args.all_frames:
        assert args.smoke_admission and args.smoke_admission.is_file()
        admission=json.loads(args.smoke_admission.read_text())
        assert admission['status']=='GEOMETRY_SMOKE_PASS' and admission['cross_host_masks_exact'] and admission['all_geometry_readbacks_pass']
        selected=spec['frames'];assert len(selected)==288
    else:
        wanted=set(args.ids);selected=[f for f in spec['frames'] if f['id'] in wanted]
        assert len(selected)==len(wanted)==len(args.ids) and len(selected)<=8, 'Engineering smoke only'
    out.mkdir(parents=True)
    if args.smoke_admission:
        shutil.copyfile(args.smoke_admission,out/'smoke-geometry-admission.json')
    script=out/'mz115_zonal_capture.py';script.write_text(instrument(original),encoding='utf-8')
    compile(script.read_text(),str(script),'exec')
    for name in ('ue_capture_readiness.py','mz115_zonal_sensors.py','mz115_zonal_tof.py','mz113_dynamic_sensors.py','mz99_angle_information_capture.py'):
        assert sha(frozen/name)==original_receipt['source_hashes'][name]
        shutil.copyfile(frozen/name,out/name)
    spec['frames']=selected;write(out/'spec.json',spec)
    # This UE Zen build interprets the desired port as signed int16.
    port=None
    for candidate_port in range(24000,25000):
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',candidate_port))
            except OSError:continue
            port=candidate_port;break
    assert port is not None
    cache=args.ddc.resolve()
    assert cache.is_relative_to(art) and cache.is_dir(), 'Reuse an existing canonical UE DDC'
    env=dict(os.environ,BA_MZ115_OUT=str(out),BA_MZ115_SPEC=str(out/'spec.json'))
    env['UE-LocalDataCachePath']=str(cache)
    command=[str(args.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(args.project),
        '-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port),'-PLUGIN='+str(args.plugin),
        '-EnablePlugins=PythonScriptPlugin,BlindAssistCapture','-ExecCmds=py '+script.as_posix(),
        '-abslog='+str(out/'editor.log'),'-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
        '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(out/'launch.json',dict(command=command,source_sha256=sha(frozen/'mz115_zonal_capture.py'),
        instrumented_sha256=sha(script),frozen_spec_sha256=sha(specpath),
        spec_sha256=sha(out/'spec.json'),project_sha256=sha(args.project),
        plugin_sha256=sha(args.plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'),
        owner='MZ132 mask-only contour diagnostic',sensor_calls=0,ids=[f['id'] for f in selected],ddc=str(cache),
        smoke_admission_sha256=sha(args.smoke_admission) if args.smoke_admission else None))
    sys.path.insert(0,str(ROOT/'tools'))
    from ue_native_capture import run_owned
    run_owned(command,env,out,args.timeout)
    result=verify(out,args.rgb_root,receipt)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
