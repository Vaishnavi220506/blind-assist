"""One matched regional/center measurement pilot, with sealed stage boundaries."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

import regional_depth_data as data
import regional_depth_model as model
import run_corridor_relative as base
from spatial_bce_model import _precision

HERE, ROOT = base.HERE, base.ROOT
NAME = 'ba-regional-depth-20260921'
OUT = ROOT/'artifacts.local/work'/NAME
PROTO = HERE/'REGIONAL_DEPTH_PROTOCOL_20260921.md'
SOURCES = base.SOURCES
PRIOR = ROOT/'artifacts.local/work/ba-spatial-structure-20260921'
ARMS = tuple(a+'_'+s for s in ('current','hold') for a in ('A','B','R','U','G',*model.ARMS))
read, write = base.read, base.write
CODE = ['run_regional_depth.py','regional_depth_data.py','regional_depth_model.py',
        'regional_depth_inference.py','test_regional_depth_data.py','test_regional_depth_model.py',
        'run_corridor_relative.py','corridor_relative_model.py','spatial_bce_model.py',
        'run_spatial_bce.py','full_event_metrics_20260920.py','tof_fov45_core.py',
        'ba_camera_corridor.py','ba_camera_corridor_metrics.py']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def seal(name, files):
    write(OUT/name,dict(protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(OUT/n) for n in files}))


def check(directory, name):
    record = read(directory/name)
    assert record['protocol_sha256'] == sha(directory/'protocol.json')
    for n, digest in record['hashes'].items():
        assert sha(directory/n) == digest, n


def verify():
    protocol = read(OUT/'protocol.json')
    assert sha(PROTO) == protocol['protocol_text_sha256'] == sha(OUT/'protocol-before-run.md')
    for section in ('source_hashes','code_hashes'):
        for n, digest in protocol[section].items():
            assert sha(ROOT/n) == digest, n
    return protocol


def indices(source, split):
    return [r['index'] for r in read(source/'evaluator/metadata.json') if r['split'] == split]


def freeze():
    assert not OUT.exists(), 'One immutable pilot only'
    files = []; groups = {}; skipped = {}
    for key, source in SOURCES.items():
        observation_seal=read(source/'observation-seal.json')
        parent_seal=read(source/('fit-seal.json' if key == 'original' else 'prediction-seal.json'))
        assert observation_seal['protocol_sha256']==parent_seal['protocol_sha256']==sha(source/'protocol.json')
        names = ['protocol.json','observation-seal.json','observations.npz','identities.json',
                 'baseline.json','evaluator/metadata.json','capture/evaluator/geometry.json']
        names += ['fit-seal.json','evaluator/train-labels.json','evaluator/dev-labels.json'] if key == 'original' else [
            'prediction-seal.json','source-admission.json','evaluator/transfer-labels.json']
        # Pin the source seals, but never recursively open protected test labels
        # or unused all-split feature caches merely to verify unrelated members.
        for name in names:
            digest=sha(source/name)
            for source_seal in (observation_seal,parent_seal):
                if name in source_seal['hashes']:
                    assert digest==source_seal['hashes'][name],name
        skipped[key]=sorted((set(observation_seal['hashes'])|set(parent_seal['hashes']))-set(names))
        metadata = read(source/'evaluator/metadata.json')
        for split, count, ng in ([('train',1728,24),('dev',576,8)] if key == 'original' else [('transfer',1152,16)]):
            rows = [r for r in metadata if r['split'] == split]
            groups[split] = {r['base_group_id'] for r in rows}
            assert len(rows) == count and len(groups[split]) == ng
            assert all(sum(r['base_group_id']==g for r in rows)==72 for g in groups[split])
        files.extend(source/n for n in names)
    assert all(not groups[a]&groups[b] for a,b in [('train','dev'),('train','transfer'),('dev','transfer')])
    for marker in ('test-start.json','test-logits.npy','test-prediction-seal.json','test-metrics.json'):
        assert not (SOURCES['original']/marker).exists(), marker
    check(PRIOR,'prediction-seal.json'); check(PRIOR,'evaluation-seal.json')
    files.extend(PRIOR/n for n in ['protocol.json','prediction-seal.json','predictions.json','evaluation-seal.json','result.json'])
    OUT.mkdir()
    (OUT/'protocol-before-run.md').write_bytes(PROTO.read_bytes())
    write(OUT/'protocol.json',dict(id=NAME,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        recipe=model.RECIPE,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_text_sha256=sha(PROTO),code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [*(HERE/n for n in CODE), ROOT/'tools/research_backend.py']},
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        source_seal_members_not_opened=skipped,
        groups={k:sorted(v) for k,v in groups.items()},scope='CONSUMED_SIMULATION_DEVELOPMENT',
        protected_test_activated=False,automatic_successor=False))
    print('FROZEN', NAME, flush=True)


def prepare():
    verify(); write(OUT/'prepare-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat()))
    start = time.perf_counter(); manifests = {}; files = []
    for split, key in [('train','original'),('dev','original'),('transfer','transfer')]:
        source = SOURCES[key]; idx = indices(source,split); ids = read(source/'identities.json')
        with np.load(source/'observations.npz',allow_pickle=False) as observations:
            boxes = observations['boxes']; ranges = observations['ranges'][idx]
        values = np.empty((len(idx),7,64,64),np.float32); manifest=[]
        for j,i in enumerate(idx):
            identity = ids[i]; path = source/identity['rgb_path']
            assert sha(path) == identity['rgb_sha256']
            with Image.open(path) as image:
                values[j] = data.build_observation(np.asarray(image.convert('RGB')),ranges[j],boxes)
            manifest.append(dict(index=i,id=identity['id'],rgb_sha256=identity['rgb_sha256']))
        np.save(OUT/f'{split}-inputs.npy',values); write(OUT/f'{split}-indices.json',idx)
        manifests[split] = manifest; files.extend([f'{split}-inputs.npy',f'{split}-indices.json'])
        print('OBSERVATIONS',split,values.shape,flush=True)
    write(OUT/'observation-manifest.json',manifests)
    write(OUT/'observation-receipt.json',dict(elapsed_s=time.perf_counter()-start,backend='CPU',
        reason='TASK_NOT_GPU_SUITABLE',operation='PNG decode, exact point samples and metadata',
        native_depth_read=False,protected_test_payloads_read=0))
    seal('inputs-seal.json',files+['observation-manifest.json','observation-receipt.json'])
    # This separate evaluator interface only receives selected TRAIN indices.
    source = SOURCES['original']; idx = read(OUT/'train-indices.json')
    geometries = read(source/'capture/evaluator/geometry.json')
    with np.load(source/'observations.npz',allow_pickle=False) as observations:
        boxes = observations['boxes']
    targets = np.empty((len(idx),64,64),np.int64); receipts=[]
    for j,i in enumerate(idx):
        geo=geometries[i]; path=source/'capture/evaluator'/geo['native_path']
        assert geo['sample_index']==i and sha(path)==geo['native_sha256']
        targets[j]=data.depth_labels(np.load(path,allow_pickle=False),boxes)['class_index']
        receipts.append(dict(index=i,sha256=geo['native_sha256']))
    np.save(OUT/'train-depth-labels.npy',targets)
    write(OUT/'train-depth-receipt.json',dict(native_payloads=receipts,frames=len(idx),
        valid_pixels=int((targets!=-100).sum()),missing_pixels=int((targets==-100).sum()),
        protected_test_payloads_read=0,dev_transfer_native_payloads_read=0,
        native_geometry_role='TRAIN_SUPERVISION_ONLY'))
    seal('labels-seal.json',['train-depth-labels.npy','train-depth-receipt.json'])
    print('TRAIN_LABELS_SEALED',targets.shape,flush=True)


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def infer(head, values, maps=False):
    device=next(head.parameters()).device; scores=[]; probabilities=[]; expected=[]; classes=[]
    for begin in range(0,len(values),model.RECIPE['batch_size']):
        x=torch.from_numpy(np.array(values[begin:begin+model.RECIPE['batch_size']],copy=True)).to(device)
        logits=head.distribution_logits(x); score,local=model.alert_score(logits,x)
        scores.append(score.cpu().numpy())
        if maps:
            probabilities.append(local.cpu().numpy())
            expected.append((logits.softmax(1)*model.depths(device)[None,:,None,None]).sum(1).cpu().numpy())
            classes.append(logits.argmax(1).to(torch.uint8).cpu().numpy())
    return dict(scores=np.concatenate(scores),**(dict(probability=np.concatenate(probabilities),
        expected_depth=np.concatenate(expected),classes=np.concatenate(classes)) if maps else {}))


def fit():
    verify(); check(OUT,'inputs-seal.json'); check(OUT,'labels-seal.json'); _precision()
    write(OUT/'fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),fits=2))
    label_rows=read(SOURCES['original']/'evaluator/train-labels.json')
    assert [r['index'] for r in label_rows]==read(OUT/'train-indices.json')
    values=torch.from_numpy(np.load(OUT/'train-inputs.npy'))
    classes=torch.from_numpy(np.load(OUT/'train-depth-labels.npy'))
    truth=torch.tensor([r['truth'] for r in label_rows],dtype=torch.float32)
    torch.manual_seed(model.RECIPE['seed']); prototype=model.Head()
    initial={k:v.clone() for k,v in prototype.state_dict().items()}
    bs=model.RECIPE['batch_size']
    device,backend=model.select_device(prototype,values[:bs],OUT/'fit-backend.json',classes[:bs],truth[:bs])
    batches=np.random.default_rng(model.RECIPE['seed']).integers(0,len(values),(model.RECIPE['steps'],bs))
    np.save(OUT/'batch-indices.npy',batches)
    values,classes,truth=values.to(device),classes.to(device),truth.to(device)
    receipts={}; files=['fit-backend.json','batch-indices.npy']
    for arm in model.ARMS:
        write(OUT/f'{arm}-fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat()))
        head=model.Head(); head.load_state_dict(initial)
        assert state_hash(head.state_dict())==state_hash(initial)
        head.to(device).train()
        optimizer=torch.optim.AdamW(head.parameters(),lr=model.RECIPE['learning_rate'],weight_decay=model.RECIPE['weight_decay'])
        start=time.perf_counter()
        with (OUT/f'{arm}-training.jsonl').open('x',encoding='utf-8') as stream:
            for step,batch in enumerate(batches,1):
                idx=torch.as_tensor(batch,device=device); optimizer.zero_grad(set_to_none=True)
                loss,parts=model.loss(head,values[idx],classes[idx],truth[idx],arm)
                assert bool(torch.isfinite(loss)), (arm,step)
                loss.backward(); optimizer.step()
                if step==1 or step%200==0:
                    row=dict(step=step,loss=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-start,
                             **{k:float(v.detach().cpu()) for k,v in parts.items()})
                    stream.write(json.dumps(row)+'\n'); stream.flush(); print('FIT',arm,row,flush=True)
        receipts[arm]=dict(initial_state_sha256=state_hash(initial),batch_indices_sha256=sha(OUT/'batch-indices.npy'),
            steps=model.RECIPE['steps'],parameters=sum(p.numel() for p in head.parameters()),elapsed_s=time.perf_counter()-start,
            device=str(next(head.parameters()).device),final_state_sha256=state_hash(head.state_dict()))
        torch.save(dict(recipe=model.RECIPE,arm=arm,state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()}),OUT/f'{arm}-head_last.pt')
        head.eval()
        for split in ('train','dev'):
            scores=infer(head,np.load(OUT/f'{split}-inputs.npy',mmap_mode='r'))['scores']
            assert np.isfinite(scores).all(); np.save(OUT/f'{arm}-{split}-scores.npy',scores)
        files.extend(f'{arm}-{f}' for f in ('head_last.pt','training.jsonl','train-scores.npy','dev-scores.npy'))
        del head,optimizer
    write(OUT/'fit-receipt.json',dict(backend=backend,arms=receipts))
    seal('fit-seal.json',files+['fit-receipt.json'])


def select():
    verify(); check(OUT,'fit-seal.json')
    source=SOURCES['original']; idx=read(OUT/'dev-indices.json')
    metadata=read(source/'evaluator/metadata.json'); metadata=[metadata[i] for i in idx]
    labels=read(source/'evaluator/dev-labels.json'); assert idx==[r['index'] for r in labels]
    baseline=read(source/'baseline.json'); a=[baseline[i]['current'] for i in idx]
    truth=[r['truth'] for r in labels]; selections={}
    for arm in model.ARMS:
        scores=np.load(OUT/f'{arm}-dev-scores.npy').astype(float)
        candidates=sorted(set(scores.tolist())|{float(np.nextafter(scores.max(),np.inf))})
        records=[]; chosen=None
        for threshold in candidates:
            cost=base.costs(scores,threshold,truth,metadata,a)
            admissible=all(c['pass_cost'] for c in cost.values())
            records.append(dict(threshold=threshold,cost=cost,admissible=admissible))
            if admissible and chosen is None:
                chosen=threshold
        assert chosen is not None
        selections[arm]=dict(threshold=chosen,cost=base.costs(scores,chosen,truth,metadata,a),candidates=len(records))
        write(OUT/f'{arm}-selection-candidates.json',records)
    write(OUT/'selection.json',selections)
    seal('selection-seal.json',['selection.json',*[f'{a}-selection-candidates.json' for a in model.ARMS]])
    print('SELECTED',selections,flush=True)


def predict():
    verify(); check(OUT,'fit-seal.json'); check(OUT,'inputs-seal.json'); check(OUT,'selection-seal.json'); _precision()
    write(OUT/'prediction-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),truth_read=False))
    values=np.load(OUT/'transfer-inputs.npy',mmap_mode='r'); prior=read(PRIOR/'predictions.json')
    ids=read(SOURCES['transfer']/'identities.json'); selection=read(OUT/'selection.json')
    assert read(OUT/'transfer-indices.json')==list(range(1152))
    prototype=model.Head()
    device,backend=model.select_device(prototype,torch.from_numpy(np.array(values[:16],copy=True)),OUT/'inference-backend.json')
    a=np.array([r['flags']['A_current'] for r in prior],bool); flags={}; scores={}; receipts={}; files=[]
    for arm in model.ARMS:
        payload=torch.load(OUT/f'{arm}-head_last.pt',map_location='cpu',weights_only=True)
        assert payload['arm']==arm and payload['recipe']==model.RECIPE
        head=model.Head(); head.load_state_dict(payload['state_dict']); head.to(device).eval()
        start=time.perf_counter(); outputs=infer(head,values,maps=True)
        receipts[arm]=dict(elapsed_s=time.perf_counter()-start,frames=len(values),device=device,
                           includes='batched model, CPU transfer and map extraction; excludes RGB decoding')
        for key,value in outputs.items():
            assert np.isfinite(value).all(); name=f'{arm}-transfer-{key}.npy'
            np.save(OUT/name,value); files.append(name)
        scores[arm]=outputs['scores']
        flags[arm+'_current']=a|(scores[arm].astype(float)>=selection[arm]['threshold'])
        flags[arm+'_hold']=base.held(flags[arm+'_current'],ids)
        del head,outputs
    rows=[]
    for i,identity in enumerate(ids):
        assert identity['id']==prior[i]['id'] and i==identity['index']==prior[i]['index']
        rows.append(dict(**{k:identity[k] for k in ('id','index','clip_id','frame_in_clip','time_s')},
            scores={k:float(v[i]) for k,v in scores.items()},
            flags={**prior[i]['flags'],**{k:bool(v[i]) for k,v in flags.items()}},
            current_unknown={k:prior[i]['current_unknown']['A_current'] for k in ARMS}))
    write(OUT/'predictions.json',rows); write(OUT/'inference-receipt.json',dict(backend=backend,arms=receipts))
    seal('prediction-seal.json',files+['predictions.json','inference-backend.json','inference-receipt.json'])
    print('PREDICTIONS_SEALED',flush=True)


def metrics(rows):
    spec=importlib.util.spec_from_file_location('regional_metrics',HERE/'full_event_metrics_20260920.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.ARMS=ARMS
    return module.evaluate(rows)


def geometry_metrics():
    """Call only after transfer prediction sealing; no native masks required."""
    check(OUT,'prediction-seal.json')
    source=SOURCES['transfer']; geometries=read(source/'capture/evaluator/geometry.json')
    with np.load(source/'observations.npz',allow_pickle=False) as observations:
        boxes=observations['boxes']
    labels=[]; receipts=[]
    for i,geo in enumerate(geometries):
        path=source/'capture/evaluator'/geo['native_path']
        assert geo['sample_index']==i and sha(path)==geo['native_sha256']
        labels.append(data.depth_labels(np.load(path,allow_pickle=False),boxes))
        receipts.append(dict(index=i,sha256=geo['native_sha256']))
    assert len(labels)==1152
    classes=np.stack([r['class_index'] for r in labels]); truth=np.stack([r['occupancy'] for r in labels])
    valid=classes!=-100; depth=np.stack([r['depth'] for r in labels]); near=valid&(depth<8)
    results={}
    for arm in model.ARMS:
        probability=np.load(OUT/f'{arm}-transfer-probability.npy'); predicted=probability>=.5
        expected=np.load(OUT/f'{arm}-transfer-expected_depth.npy'); bins=np.load(OUT/f'{arm}-transfer-classes.npy')
        tp=int((valid&truth&predicted).sum()); fp=int((valid&~truth&predicted).sum()); fn=int((valid&truth&~predicted).sum())
        results[arm]=dict(TP=tp,FP=fp,FN=fn,precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None,
            known_pixels=int(valid.sum()),missing_pixels=int((~valid).sum()),positive_pixels=int(truth.sum()),
            class_accuracy=float((bins[valid]==classes[valid]).mean()),finite_under8_MAE=float(np.abs(expected[near]-depth[near]).mean()),
            MAE_pixels=int(near.sum()))
    write(OUT/'geometry-metrics.json',results); write(OUT/'geometry-native-receipt.json',receipts)
    return results


def evaluate():
    verify(); check(OUT,'prediction-seal.json'); check(OUT,'selection-seal.json')
    source=SOURCES['transfer']; predictions=read(OUT/'predictions.json')
    labels=read(source/'evaluator/transfer-labels.json'); metadata=read(source/'evaluator/metadata.json')
    admission=read(source/'source-admission.json')['frames']; rows=[]
    for p,y,m,n in zip(predictions,labels,metadata,admission):
        assert p['index']==y['index']==m['index'] and p['id']==m['id']==n['id']
        rows.append(dict(**m,truth=y['truth'],flags=p['flags'],current_unknown=p['current_unknown'],
            native_target_corridor_samples=n['returned_target_corridor_samples']))
    assert len(rows)==1152
    result=metrics(rows)
    groups={g:metrics([r for r in rows if r['base_group_id']==g]) for g in sorted({r['base_group_id'] for r in rows})}
    horizontal=metrics([r for r in rows if 'head_horizontal' in r['base_group_id'] and r['layout_relation']=='BOUNDARY'])
    selection=read(OUT/'selection.json'); findings={}
    for arm in model.ARMS:
        cost=base.costs(np.load(OUT/f'{arm}-transfer-scores.npy'),selection[arm]['threshold'],
            [r['truth'] for r in rows],metadata,[r['flags']['A_current'] for r in rows])
        gains=[g for g,v in groups.items() if v['Boundary']['arms'][arm+'_hold']['frames']['TP']>v['Boundary']['arms']['A_hold']['frames']['TP']]
        events=result['Core']['arms'][arm+'_hold']['events']; old=result['Core']['arms']['A_hold']['events']
        event_ok=all(n['clip_id']==o['clip_id'] and (not o['detected'] or (n['detected'] and
            n['first_in_event_alert_delay_s']<=o['first_in_event_alert_delay_s']+1e-8)) for n,o in zip(events,old))
        gates=dict(cost=all(c['pass_cost'] for c in cost.values()),
            retain_A=all(not r['flags']['A_'+s] or r['flags'][arm+'_'+s] for r in rows for s in ('current','hold')),
            core_events_timing=len(events)==len(old) and event_ok,
            boundary_recall=result['Boundary']['arms'][arm+'_hold']['frames']['recall']>=.5,
            broad_gain=len(gains)>=8)
        new=[r for r in rows if r['truth'] and r['flags'][arm+'_current'] and not r['flags']['A_current']]
        findings[arm]=dict(usable=all(gates.values()),gates=gates,costs=cost,gain_groups=gains,
            new_current_TP=len(new),new_current_TP_without_native_corridor_contributor=sum(r['native_target_corridor_samples']==0 for r in new))
    extra=result['Boundary']['arms']['REGION_hold']['frames']['TP']-result['Boundary']['arms']['POINT_hold']['frames']['TP']
    gains=[g for g,v in groups.items() if v['Boundary']['arms']['REGION_hold']['frames']['TP']>v['Boundary']['arms']['POINT_hold']['frames']['TP']]
    supported=findings['REGION']['usable'] and extra>=5 and len(gains)>=2
    geometry_metrics()
    terminal=dict(regional_hypothesis='SUPPORTED_DEVELOPMENT' if supported else 'NOT_SUPPORTED',arms=findings,
        contribution=dict(extra_Boundary_TP=extra,gain_groups=gains),scope='CONSUMED_SIMULATION_DEVELOPMENT',
        protected_test_activated=False,automatic_successor=False)
    for name,value in [('metrics.json',result),('group-metrics.json',groups),('horizontal-metrics.json',horizontal),
                       ('frame-results.json',rows),('result.json',terminal)]:
        write(OUT/name,value)
    write(OUT/'local-inheritance.json',dict(terminal_id=NAME,
        inheritance_role='COMPONENT_OR_CHALLENGER' if any(r['usable'] for r in findings.values()) else 'NEGATIVE_CONTROL',
        arm_roles={a:'COMPONENT_OR_CHALLENGER' if r['usable'] else 'NEGATIVE_CONTROL' for a,r in findings.items()},
        role_scope='Exact two-arm 1200-step regional/central depth-learning recipe and fixed alert decoder',
        retained_surface='A, UNKNOWN and earlier B/R/U/G dispositions unchanged',
        regional_hypothesis=terminal['regional_hypothesis'],revisit_trigger='Separately justified mechanism and authorization'))
    seal('evaluation-seal.json',['metrics.json','group-metrics.json','horizontal-metrics.json','frame-results.json',
        'result.json','local-inheritance.json','geometry-metrics.json','geometry-native-receipt.json'])
    print(json.dumps(terminal),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','prepare','fit','select','predict','evaluate'])
    globals()[parser.parse_args().stage]()
