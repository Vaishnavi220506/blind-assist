"""One fixed-geometry appearance intervention; commands separate truth from inference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONDITIONS = ('reference', 'repeat', 'target', 'background')
FRAMES = 1152
T = .4071309640537889
CUTS = dict(B=6.888704776763916, N=7.03014612197876)
DEPS = ('appearance_probe.py', 'appearance_capture.py', 'ue_capture_readiness.py',
        'spatial_bce_model.py', 'tof_fov45_core.py', 'tof_corridor_calibration.py',
        'ba_camera_corridor.py', 'ba_camera_corridor_spec.py', 'core_transfer_spec.py', 'full_event_metrics_20260920.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def seal(out, name, files):
    write(out/name, dict(time_ns=time.time_ns(), hashes={f: sha(out/f) for f in files}))


def check_seal(out, name):
    for f, h in read(out/name)['hashes'].items():
        assert sha(out/f) == h, f


def make_spec(source):
    families = sorted({g['type_id'] for g in source['groups']})
    groups = [sorted(g['base_group_id'] for g in source['groups']
                     if g['type_id'] == family and g['split'] == 'train')[0] for family in families]
    cases = []
    selected = [c for c in source['cases'] if c['base_group_id'] in groups]
    assert len(selected) == 288
    for clip_id in sorted({c['clip_id'] for c in selected}):
        clip = sorted((c for c in selected if c['clip_id'] == clip_id), key=lambda c:c['frame_in_clip'])
        for condition in CONDITIONS:
            for c in clip:
                row = copy.deepcopy(c)
                row.update(source_id=c['name'], source_clip_id=clip_id, condition=condition,
                           clip_id=clip_id+'__'+condition, name=c['name']+'__'+condition)
                if condition in ('target', 'background'):
                    obj = next(o for o in row['objects'] if o['name'] == condition)
                    old = obj['material'].rsplit('/', 1)[-1]
                    new = ('Cream' if old == 'Sage' else 'Sage') if condition == 'target' else ('Wood' if old == 'Brick' else 'Brick')
                    obj['material'] = '/Game/StreetLab/Materials/'+new
                cases.append(row)
    return dict(schema='appearance-intervention-v1', cases=cases, frames=len(cases),
                groups=groups, clips=48, expected_map_sha256=source['expected_map_sha256'],
                profile=source['profile'], sampling=source['sampling'])


def prepare(out):
    assert not out.exists()
    original = ROOT/'artifacts.local/work/ba-data-coverage-20260921'
    source = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
    inputs = dict(source_spec=original/'spec.json', B=source/'fit/head_last.pt',
                  N=original/'fit/head_last.pt', selection=original/'selection.json',
                  encoder=ROOT/'artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth')
    selection = read(inputs['selection'])['arms']
    assert selection['B_control']['threshold'] == CUTS['B'] and selection['N']['threshold'] == CUTS['N']
    spec = make_spec(read(inputs['source_spec']))
    assert spec['frames'] == FRAMES
    out.mkdir(parents=True)
    for name, src in inputs.items():
        dest = out/'inputs'/(name+src.suffix)
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dest)
    write(out/'spec.json', spec)
    shutil.copyfile(HERE/'APPEARANCE_PROTOCOL_20260921.md', out/'protocol-before-run.md')
    codes = {str((HERE/f).relative_to(ROOT)).replace('\\','/'): sha(HERE/f) for f in DEPS}
    for rel in ('tools/research_backend.py', 'research/active/dtr-r0/unreal/street_process_lifecycle.py'):
        codes[rel] = sha(ROOT/rel)
    write(out/'protocol.json', dict(frames=FRAMES, groups=spec['groups'], cuts=CUTS,
          spec_sha256=sha(out/'spec.json'), text_sha256=sha(out/'protocol-before-run.md'),
          code_hashes=codes, input_hashes={p.name:sha(p) for p in (out/'inputs').iterdir()},
          capture_budget_seconds=1800, original_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print('FROZEN', len(spec['cases']), spec['groups'], flush=True)


def verify(out):
    p = read(out/'protocol.json')
    assert p['frames'] == FRAMES and p['cuts'] == CUTS
    assert p['spec_sha256'] == sha(out/'spec.json') and p['text_sha256'] == sha(out/'protocol-before-run.md')
    for rel,h in p['code_hashes'].items():
        assert sha(ROOT/rel) == h, rel
    for rel,h in p['input_hashes'].items():
        assert sha(out/'inputs'/rel) == h, rel
    return p


def capture(out):
    p = verify(out)
    dest = out/'capture'
    assert not dest.exists()
    artifact_root = Path(os.environ.get('BLINDASSIST_ARTIFACTS', str(ROOT/'artifacts.local'))).resolve()
    assert out.resolve().is_relative_to(artifact_root)
    project = artifact_root/'unreal/BlindAssistStreetLab'
    engine = Path(os.environ['UE_ENGINE_ROOT'])
    plugin = Path(os.environ['BLINDASSIST_UE_CAPTURE_PLUGIN'])
    spec = read(out/'spec.json')
    map_file = project/'Content/StreetLab/WillowSampleV1.umap'
    assert sha(map_file) == spec['expected_map_sha256']
    source_files = [map_file, project/'BlindAssistStreetLab.uproject', plugin,
                    plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll']
    source_files += sorted({project/('Content/'+o['material'].removeprefix('/Game/')+'.uasset') for c in spec['cases'] for o in c['objects']})
    assets = {str(f):sha(f) for f in source_files}
    script = HERE/'appearance_capture.py'
    dest.mkdir()
    write(dest/'launch-receipt.json', dict(assets=assets, protocol_sha256=sha(out/'protocol.json'),
          script_sha256=sha(script), started_ns=time.time_ns(), engine=str(engine), project=str(project),
          host=os.environ.get('COMPUTERNAME'), budget_seconds=1800))
    env = dict(os.environ, BA_CAMERA_CORRIDOR_SPEC=str(out/'spec.json'), BA_CAMERA_CORRIDOR_OUTPUT=str(dest),
        BA_CAMERA_CORRIDOR_SCRIPT=str(script), BA_CAMERA_CORRIDOR_PROTOCOL_SHA=sha(out/'protocol.json'),
        BA_CAMERA_CORRIDOR_SPEC_SHA=sha(out/'spec.json'), BA_CAMERA_CORRIDOR_SCRIPT_SHA=sha(script))
    env['UE-LocalDataCachePath'] = str(project/'DerivedDataCache')
    sys.path.insert(0,str(ROOT/'research/active/dtr-r0/unreal'))
    from street_process_lifecycle import TaskProcessTree
    tree = None
    terminal = dict(status='FAIL')
    try:
        si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        proc=subprocess.Popen([str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project/'BlindAssistStreetLab.uproject'),
            '-ExecCmds=py '+script.as_posix(),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
            '-abslog='+str(dest/'editor.log'),'-PLUGIN='+str(plugin)],env=env,startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW)
        tree=TaskProcessTree(proc,owner=str(dest))
        code=tree.wait(timeout=p['capture_budget_seconds'])
        receipt=read(dest/'receipt.json')
        assert code==0 and receipt['status']=='PASS' and receipt['frame_count']==FRAMES
        assert receipt['source_unchanged'] and receipt['task_actors_released']
        assert receipt['protocol_sha256']==sha(out/'protocol.json') and receipt['spec_sha256']==sha(out/'spec.json')
        for path,h in assets.items():assert sha(path)==h,path
        verify(out)
        terminal=dict(status='PASS',returncode=code)
    except BaseException:
        terminal['error']=traceback.format_exc();raise
    finally:
        release=tree.cleanup() if tree else dict(released=True,process_started=False)
        write(dest/'process-release.json',release)
        terminal['processes_released']=release['released'];write(dest/'launcher-terminal.json',terminal)
        assert release['released']


def held(flags, ids):
    flags=list(map(bool,flags));result=flags.copy()
    for i in range(1,len(flags)):
        if ids[i]['clip_id']==ids[i-1]['clip_id'] and ids[i]['frame_in_clip']==ids[i-1]['frame_in_clip']+1:
            result[i]=flags[i] or flags[i-1]
    return result


def materialize(out):
    import numpy as np
    from tof_fov45_core import boxes45, simulate
    from ba_camera_corridor import sample_native
    from tof_corridor_calibration import score_frame, decide
    verify(out)
    assert read(out/'capture/launcher-terminal.json')['status']=='PASS'
    assert read(out/'capture/process-release.json')['released']
    spec=read(out/'spec.json');manifest=read(out/'capture/observations/manifest.json')['frames']
    geometry=read(out/'capture/evaluator/geometry.json')
    assert len(manifest)==len(geometry)==len(spec['cases'])==FRAMES
    ranges=[];ids=[];baseline=[];refs={};audit=[]
    boxes=boxes45()
    for i,(c,m,g) in enumerate(zip(spec['cases'],manifest,geometry)):
        assert m['id']==g['id']==c['name'] and m['sample_index']==g['sample_index']==i
        npth=out/'capture/evaluator'/g['native_path'];rgb=out/'capture/observations'/m['rgb_path']
        assert sha(npth)==g['native_sha256'] and sha(rgb)==m['rgb_sha256']==g['rgb_sha256']
        depth=np.load(npth,allow_pickle=False)
        vector,_=simulate(sample_native(depth),'spatial-bce-v1/'+c['sensor_noise_key'],boxes)
        sc=score_frame(boxes,vector);a=decide(sc,T)
        if c['condition']=='reference':refs[c['source_id']]=(depth,vector)
        refd,refr=refs[c['source_id']]
        audit.append(dict(index=i,source_id=c['source_id'],condition=c['condition'],
            depth_equal=bool(np.array_equal(depth,refd,equal_nan=True)),
            tof_equal=bool(np.array_equal(vector,refr,equal_nan=True)),
            max_depth_difference=float(np.nanmax(np.abs(depth-refd)))))
        ranges.append(vector)
        ids.append(dict(index=i,id=m['id'],clip_id=m['clip_id'],frame_in_clip=m['frame_in_clip'],time_s=m['time_s'],
                   rgb_path=rgb.relative_to(out).as_posix(),rgb_sha256=m['rgb_sha256']))
        baseline.append(dict(current=bool(a['alert']),unknown=bool(a['unknown'])))
    write(out/'pair-admission.json',dict(status='PASS' if all(r['depth_equal'] and r['tof_equal'] for r in audit) else 'NOT_EVALUABLE',
                                     rows=audit,backend_reason='TASK_NOT_GPU_SUITABLE',pairs=864))
    np.save(out/'ranges.npy',np.asarray(ranges))
    write(out/'identities.json',ids);write(out/'baseline.json',baseline)
    seal(out,'observation-seal.json',['ranges.npy','identities.json','baseline.json','pair-admission.json',
         'capture/observations/manifest.json','capture/evaluator/geometry.json','capture/receipt.json'])
    print('OBSERVATIONS_SEALED',FRAMES,flush=True)


def predict(out):
    verify(out)
    if read(out/'pair-admission.json')['status'] != 'PASS':
        raise ValueError('Paired source NOT_EVALUABLE; frozen model inference is not admitted')
    check_seal(out,'observation-seal.json')
    import numpy as np
    import torch
    import spatial_bce_model as model
    ids=read(out/'identities.json');base=read(out/'baseline.json')
    features=model.encode_rgb([out/r['rgb_path'] for r in ids],out/'inputs/encoder.pth',out/'features')
    inputs=model.build_inputs(features,np.load(out/'ranges.npy'))
    flags=dict(A_current=[r['current'] for r in base]);logits={}
    model._precision()
    for name in CUTS:
        head=model.load_head(out/'inputs'/(name+'.pt'))
        device,backend=model._select(head,torch.from_numpy(inputs[:64]),out/(name+'-backend.json'))
        head.to(device)
        logits[name]=model.predict(head,inputs).astype(float)
        flags[name+'_current']=(np.array(flags['A_current'])|(logits[name]>=CUTS[name])).tolist()
    for name in tuple(flags):flags[name.replace('_current','_hold')]=held(flags[name],ids)
    predictions=[dict(index=i,id=r['id'],logits={k:float(v[i]) for k,v in logits.items()},
        supplements={k:bool(v[i]>=CUTS[k]) for k,v in logits.items()},flags={k:bool(v[i]) for k,v in flags.items()},
        unknown=base[i]['unknown']) for i,r in enumerate(ids)]
    write(out/'predictions.json',predictions)
    seal(out,'prediction-seal.json',['predictions.json','B-backend.json','N-backend.json','features/feature_receipt.json'])
    print('PREDICTIONS_SEALED',FRAMES,flush=True)


def evaluate(out):
    import numpy as np
    import full_event_metrics_20260920 as metrics
    from core_transfer_spec import bounds, classify
    verify(out);check_seal(out,'prediction-seal.json');check_seal(out,'observation-seal.json')
    spec=read(out/'spec.json');pred=read(out/'predictions.json')
    conditions={c:[] for c in CONDITIONS};paired={}
    metrics.ARMS=('A_current','A_hold','B_current','B_hold','N_current','N_hold')
    for c,p in zip(spec['cases'],pred):
        assert p['id']==c['name']
        row={k:c[k] for k in ('base_group_id','clip_id','frame_in_clip','time_s','layer','background','layout_relation','type_id','phase')}
        row.update(index=p['index'],id=p['id'],truth=bool(classify(*bounds(c))['truth']),flags=p['flags'],
                   current_unknown={k:p['unknown'] for k in metrics.ARMS})
        conditions[c['condition']].append(row)
        paired.setdefault(c['source_id'],{})[c['condition']]=(c,p)
    summaries={name:metrics.evaluate(rows) for name,rows in conditions.items()}
    differences={}
    for name in CUTS:
        by_condition={}
        for condition in CONDITIONS[1:]:
            changes=[]
            for sid,rows in paired.items():
                c,p=rows[condition];_,r=rows['reference']
                changes.append(dict(source_id=sid,group=c['base_group_id'],family=c['type_id'],relation=c['layout_relation'],
                    logit_delta=p['logits'][name]-r['logits'][name],
                    supplement_flip=p['supplements'][name]!=r['supplements'][name],
                    current_flip=p['flags'][name+'_current']!=r['flags'][name+'_current'],
                    hold_flip=p['flags'][name+'_hold']!=r['flags'][name+'_hold']))
            groups={g:sum(r['current_flip'] for r in changes if r['group']==g) for g in spec['groups']}
            by_condition[condition]=dict(current_flips=sum(r['current_flip'] for r in changes),
                 supplement_flips=sum(r['supplement_flip'] for r in changes),held_flips=sum(r['hold_flip'] for r in changes),
                 absolute_logit_median=float(np.median([abs(r['logit_delta']) for r in changes])),
                 absolute_logit_max=float(max(abs(r['logit_delta']) for r in changes)),groups=groups,rows=changes)
        for condition in ('target','background'):
            result=by_condition[condition];repeat=by_condition['repeat']
            excess=result['current_flips']-repeat['current_flips']
            groups=sum(result['groups'][g]>repeat['groups'][g] for g in spec['groups'])
            result['decision_sensitive']=excess>=3 and groups>=2
        differences[name]=by_condition
    admitted=read(out/'pair-admission.json')['status']=='PASS'
    signal=any(differences[n][c]['decision_sensitive'] for n in CUTS for c in ('target','background'))
    status='NOT_EVALUABLE' if not admitted else 'APPEARANCE_SENSITIVITY_SUPPORTED' if signal else 'NOT_SUPPORTED_FOR_THIS_INTERVENTION'
    write(out/'metrics.json',summaries);write(out/'paired-differences.json',differences)
    write(out/'result.json',dict(status=status,frames=FRAMES,paired_frames=288,groups=spec['groups'],
        source_admitted=admitted,scope='CONSUMED_TRAINING_GEOMETRY_FROZEN_MODEL_MECHANISM_DIAGNOSTIC',
        cutoffs=CUTS,conditions={n:{c:{k:v for k,v in r.items() if k!='rows'} for c,r in d.items()} for n,d in differences.items()}))
    write(out/'local-inheritance.json',dict(role='COMPONENT_OR_CHALLENGER',mode='COMPONENT',
        scope='Paired appearance sensitivity diagnostic, no alert policy promotion',status=status,
        preserved_baselines=True,global_registration='PENDING_SUPPORTED_COMMAND'))
    seal(out,'evaluation-seal.json',['metrics.json','paired-differences.json','result.json','local-inheritance.json'])
    print('RESULT',status,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['prepare','capture','materialize','predict','evaluate','run'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if args.command=='run':
        for stage in (capture,materialize,predict,evaluate):stage(out)
    else:globals()[args.command](out)


if __name__=='__main__':main()
