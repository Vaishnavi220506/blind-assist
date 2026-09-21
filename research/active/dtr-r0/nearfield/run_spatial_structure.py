"""Matched U/G spatial representation contrast on disclosed consumed simulation."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

import run_corridor_relative as base
import spatial_structure_model as model
from spatial_bce_model import _precision,_select

HERE,ROOT=base.HERE,base.ROOT
NAME='ba-spatial-structure-20260921'
OUT=ROOT/'artifacts.local/work'/NAME
PROTO=HERE/'SPATIAL_STRUCTURE_PROTOCOL_20260921.md'
SOURCES=base.SOURCES
PRIOR=base.OUT
ARMS=tuple(n+'_'+s for s in ('current','hold') for n in ('A','B','R','U','G'))
read,write,sha=base.read,base.write,base.sha
CODE=['run_spatial_structure.py','spatial_structure_model.py','test_spatial_structure.py',
      'run_corridor_relative.py','corridor_relative_model.py','spatial_bce_model.py',
      'run_spatial_bce.py','full_event_metrics_20260920.py','tof_fov45_core.py',
      'ba_camera_corridor.py','ba_camera_corridor_metrics.py']


def seal(name,files):
    write(OUT/name,dict(protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(OUT/n) for n in files}))


def check(directory,name):
    record=read(directory/name)
    assert record['protocol_sha256']==sha(directory/'protocol.json')
    for n,d in record['hashes'].items(): assert sha(directory/n)==d,n


def verify():
    p=read(OUT/'protocol.json')
    assert sha(PROTO)==p['protocol_text_sha256']==sha(OUT/'protocol-before-run.md')
    for key in ('code_hashes','source_hashes'):
        for n,d in p[key].items(): assert sha(ROOT/n)==d,n
    return p


def freeze():
    assert not OUT.exists(),'One new contrast only'
    files=[]; group_ids={}
    for key,source in SOURCES.items():
        observation=read(source/'observation-seal.json')
        parent=read(source/('fit-seal.json' if key=='original' else 'prediction-seal.json'))
        assert observation['protocol_sha256']==parent['protocol_sha256']==sha(source/'protocol.json')
        names=['protocol.json','observation-seal.json','observations.npz','identities.json','baseline.json',
               'evaluator/metadata.json','features/rgb_features.npy','features/feature_receipt.json']
        names+=['fit-seal.json','evaluator/train-labels.json','evaluator/dev-labels.json'] if key=='original' else [
            'prediction-seal.json','predictions.json','evaluator/transfer-labels.json','source-admission.json']
        for n in names:
            d=sha(source/n)
            if n in observation['hashes']: assert observation['hashes'][n]==d,n
            if n in parent['hashes']: assert parent['hashes'][n]==d,n
        receipt=read(source/'features/feature_receipt.json')
        assert receipt['cache_sha256']==sha(source/'features/rgb_features.npy')
        metadata=read(source/'evaluator/metadata.json')
        for split,frames,groups in ([('train',1728,24),('dev',576,8)] if key=='original' else [('transfer',1152,16)]):
            subset=[r for r in metadata if r['split']==split]
            group_ids[split]={r['base_group_id'] for r in subset}
            assert len(subset)==frames and len(group_ids[split])==groups
        files.extend(source/n for n in names)
    original=read(SOURCES['original']/'features/feature_receipt.json')
    transfer=read(SOURCES['transfer']/'features/feature_receipt.json')
    assert original['recipe']==transfer['recipe'] and original['checkpoint_sha256']==transfer['checkpoint_sha256']
    for marker in ('test-start.json','test-logits.npy','test-prediction-seal.json','test-metrics.json'):
        assert not (SOURCES['original']/marker).exists(),'Protected test already activated: '+marker
    assert all(not group_ids[a]&group_ids[b] for a,b in [('train','dev'),('train','transfer'),('dev','transfer')])
    check(PRIOR,'prediction-seal.json'); check(PRIOR,'evaluation-seal.json')
    files.extend(PRIOR/n for n in ['protocol.json','prediction-seal.json','predictions.json','evaluation-seal.json','result.json'])
    OUT.mkdir(); (OUT/'protocol-before-run.md').write_bytes(PROTO.read_bytes())
    write(OUT/'protocol.json',dict(id=NAME,frozen_at_utc=datetime.now(timezone.utc).isoformat(),recipe=model.RECIPE,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_text_sha256=sha(PROTO),code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [*(HERE/n for n in CODE),ROOT/'tools/research_backend.py']},
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files},arms=ARMS,
        protected_test_activated=False,automatic_successor=False,scope='CONSUMED_SIMULATION_MATCHED_INPUT_ABLATION'))
    print('FROZEN',NAME,flush=True)


def prepare():
    verify(); write(OUT/'prepare-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat()))
    start=time.perf_counter()
    for split,key in [('train','original'),('dev','original'),('transfer','transfer')]:
        source=SOURCES[key]
        idx=[r['index'] for r in read(source/'evaluator/metadata.json') if r['split']==split]
        features=np.load(source/'features/rgb_features.npy',mmap_mode='r',allow_pickle=False)[idx]
        with np.load(source/'observations.npz',allow_pickle=False) as data:
            x=model.build_inputs(features,data['ranges'][idx],data['boxes'])
        inverse=x[:,:24].reshape(-1,24,8,3,8,3).transpose(0,1,3,5,2,4).reshape(-1,216,8,8)
        assert np.array_equal(inverse,features),'Ordered image representation must be lossless'
        np.save(OUT/f'{split}-inputs.npy',x); write(OUT/f'{split}-indices.json',idx)
        print('PREPARED',split,x.shape,flush=True)
    write(OUT/'prepare-receipt.json',dict(elapsed_s=time.perf_counter()-start,backend='CPU',
        reason='TASK_NOT_GPU_SUITABLE',operation='Array reindexing and small deterministic geometry; no encoder execution',
        labels_read=False,protected_test_rows_indexed=0,native_depth_read=False,inverse_order_exact=True))
    seal('inputs-seal.json',[f'{s}-{f}' for s in ('train','dev','transfer') for f in ('inputs.npy','indices.json')]+['prepare-receipt.json'])


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()): h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def fit():
    verify(); check(OUT,'inputs-seal.json'); _precision()
    write(OUT/'fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),fits=2))
    labels=read(SOURCES['original']/'evaluator/train-labels.json')
    assert [r['index'] for r in labels]==read(OUT/'train-indices.json')
    values=np.load(OUT/'train-inputs.npy'); y=torch.tensor([r['truth'] for r in labels],dtype=torch.float32)
    torch.manual_seed(model.RECIPE['seed']); prototype=model.Head()
    initial={k:v.clone() for k,v in prototype.state_dict().items()}
    device,backend=_select(prototype,torch.from_numpy(values[:64]),OUT/'fit-backend.json',y[:64])
    batches=np.random.default_rng(model.RECIPE['seed']).integers(0,len(values),(1200,64))
    np.save(OUT/'batch-indices.npy',batches)
    receipts={}; y=y.to(device)
    for arm in ('U','G'):
        write(OUT/f'{arm}-fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat()))
        head=model.Head(); head.load_state_dict(initial); assert state_hash(head.state_dict())==state_hash(initial)
        head.to(device).train(); x=torch.from_numpy(model.arm_inputs(values,arm)).to(device)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        start=time.perf_counter()
        with (OUT/f'{arm}-training.jsonl').open('x',encoding='utf-8') as stream:
            for step,indices in enumerate(batches,1):
                idx=torch.as_tensor(indices,device=device); optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(head(x[idx]),y[idx]); loss.backward(); optimizer.step()
                if step==1 or step%200==0:
                    row=dict(step=step,bce=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-start)
                    import json
                    stream.write(json.dumps(row)+'\n'); stream.flush(); print('FIT',arm,row,flush=True)
        receipts[arm]=dict(initial_state_sha256=state_hash(initial),batch_indices_sha256=sha(OUT/'batch-indices.npy'),
            train_labels_sha256=sha(SOURCES['original']/'evaluator/train-labels.json'),parameters=sum(p.numel() for p in head.parameters()),
            elapsed_s=time.perf_counter()-start,device=str(next(head.parameters()).device),steps=1200)
        torch.save(dict(recipe=model.RECIPE,arm=arm,state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()}),OUT/f'{arm}-head_last.pt')
        head.eval()
        for split in ('train','dev'):
            np.save(OUT/f'{arm}-{split}-logits.npy',base.infer(head,model.arm_inputs(np.load(OUT/f'{split}-inputs.npy'),arm)))
        del head,x,optimizer
    write(OUT/'fit-receipt.json',dict(backend=backend,arms=receipts))
    seal('fit-seal.json',['fit-backend.json','fit-receipt.json','batch-indices.npy']+
         [f'{a}-{f}' for a in ('U','G') for f in ('head_last.pt','training.jsonl','train-logits.npy','dev-logits.npy')])


def select():
    verify(); check(OUT,'fit-seal.json')
    source=SOURCES['original']; idx=read(OUT/'dev-indices.json')
    metadata=read(source/'evaluator/metadata.json'); metadata=[metadata[i] for i in idx]
    labels=read(source/'evaluator/dev-labels.json'); assert idx==[r['index'] for r in labels]
    truth=[r['truth'] for r in labels]; baseline=read(source/'baseline.json'); a=[baseline[i]['current'] for i in idx]
    selections={}
    for arm in ('U','G'):
        logits=np.load(OUT/f'{arm}-dev-logits.npy').astype(float); assert np.isfinite(logits).all()
        thresholds=sorted(set(logits.tolist())|{float(np.nextafter(logits.max(),np.inf))})
        records=[]; chosen=None
        for t in thresholds:
            cost=base.costs(logits,t,truth,metadata,a); okay=all(c['pass_cost'] for c in cost.values())
            records.append(dict(threshold=t,cost=cost,admissible=okay))
            if okay and chosen is None: chosen=t
        assert chosen is not None
        selections[arm]=dict(threshold=chosen,cost=base.costs(logits,chosen,truth,metadata,a),candidates=len(records))
        write(OUT/f'{arm}-selection-candidates.json',records)
    write(OUT/'selection.json',selections); seal('selection-seal.json',['selection.json','U-selection-candidates.json','G-selection-candidates.json'])
    print('SELECTED',selections,flush=True)


def predict():
    verify(); check(OUT,'fit-seal.json'); check(OUT,'inputs-seal.json'); check(OUT,'selection-seal.json'); _precision()
    write(OUT/'prediction-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),truth_read=False))
    values=np.load(OUT/'transfer-inputs.npy'); source=SOURCES['transfer']
    ids=read(source/'identities.json'); prior=read(PRIOR/'predictions.json'); selection=read(OUT/'selection.json')
    assert read(OUT/'transfer-indices.json')==list(range(1152))
    prototype=model.Head(); device,backend=_select(prototype,torch.from_numpy(values[:64]),OUT/'inference-backend.json')
    a=np.array([r['flags']['A_current'] for r in prior],bool); flags={}; scores={}
    start=time.perf_counter()
    for arm in ('U','G'):
        payload=torch.load(OUT/f'{arm}-head_last.pt',map_location='cpu',weights_only=True)
        assert payload['arm']==arm and payload['recipe']==model.RECIPE
        head=model.Head(); head.load_state_dict(payload['state_dict']); head.to(device).eval()
        scores[arm]=base.infer(head,model.arm_inputs(values,arm)); assert np.isfinite(scores[arm]).all()
        np.save(OUT/f'{arm}-transfer-logits.npy',scores[arm])
        flags[arm+'_current']=a|(scores[arm].astype(float)>=selection[arm]['threshold'])
        flags[arm+'_hold']=base.held(flags[arm+'_current'],ids)
    rows=[]
    for i,identity in enumerate(ids):
        assert identity['id']==prior[i]['id'] and i==identity['index']==prior[i]['index']
        combined={**prior[i]['flags'],**{k:bool(v[i]) for k,v in flags.items()}}
        rows.append(dict(**{k:identity[k] for k in ('id','index','clip_id','frame_in_clip','time_s')},
            logits={k:float(v[i]) for k,v in scores.items()},flags=combined,
            current_unknown={k:prior[i]['current_unknown']['A_current'] for k in ARMS}))
    write(OUT/'predictions.json',rows); write(OUT/'inference-receipt.json',dict(backend=backend,elapsed_s=time.perf_counter()-start,frames=1152,arms=2))
    seal('prediction-seal.json',['predictions.json','U-transfer-logits.npy','G-transfer-logits.npy','inference-backend.json','inference-receipt.json'])
    print('PREDICTIONS_SEALED',flush=True)


def metrics(rows):
    spec=importlib.util.spec_from_file_location('structure_metrics',HERE/'full_event_metrics_20260920.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.ARMS=ARMS
    return module.evaluate(rows)


def evaluate():
    verify(); check(OUT,'prediction-seal.json'); check(OUT,'selection-seal.json')
    source=SOURCES['transfer']; predictions=read(OUT/'predictions.json'); labels=read(source/'evaluator/transfer-labels.json')
    metadata=read(source/'evaluator/metadata.json'); admission=read(source/'source-admission.json')['frames']
    rows=[]
    for p,y,m,n in zip(predictions,labels,metadata,admission):
        assert p['index']==y['index']==m['index'] and p['id']==m['id']==n['id']
        rows.append(dict(**m,truth=y['truth'],flags=p['flags'],current_unknown=p['current_unknown'],native_target_corridor_samples=n['returned_target_corridor_samples']))
    assert len(rows)==1152
    result=metrics(rows); groups={g:metrics([r for r in rows if r['base_group_id']==g]) for g in sorted({r['base_group_id'] for r in rows})}
    horizontal=[r for r in rows if 'head_horizontal' in r['base_group_id'] and r['layout_relation']=='BOUNDARY']
    assert len(horizontal)==96 and sum(r['truth'] for r in horizontal)==43
    hmetrics=metrics(horizontal); selection=read(OUT/'selection.json'); findings={}
    for arm in ('U','G'):
        costs=base.costs(np.load(OUT/f'{arm}-transfer-logits.npy'),selection[arm]['threshold'],[r['truth'] for r in rows],metadata,[r['flags']['A_current'] for r in rows])
        gains=[g for g,m in groups.items() if m['Boundary']['arms'][arm+'_hold']['frames']['TP']>m['Boundary']['arms']['A_hold']['frames']['TP']]
        h=hmetrics['Boundary']['arms'][arm+'_hold']; b=result['Boundary']['arms'][arm+'_hold']
        timely=sum(e['detected'] and e['first_in_event_alert_delay_s']<=.4+1e-8 for e in h['events'])
        gates=dict(cost=all(c['pass_cost'] for c in costs.values()),
            retain_A=all(not r['flags']['A_'+s] or r['flags'][arm+'_'+s] for r in rows for s in ('current','hold')),
            boundary_recall=b['frames']['recall']>=.5,broad_gain=len(gains)>=8,
            horizontal_recall=h['frames']['recall']>=.5,horizontal_events=h['detected_events']>=3,
            horizontal_timing=timely>=3 and all(not e['detected'] or e['first_in_event_alert_delay_s']<=.4+1e-8 for e in h['events']))
        new=[r for r in rows if r['truth'] and r['flags'][arm+'_current'] and not r['flags']['A_current']]
        findings[arm]=dict(usable=all(gates.values()),gates=gates,costs=costs,gain_groups=gains,horizontal_timely_events=timely,
            new_current_TP=len(new),new_current_TP_without_native_corridor_contributor=sum(r['native_target_corridor_samples']==0 for r in new))
    hg=[g for g in groups if 'head_horizontal' in g]
    hgain=[g for g in hg if groups[g]['Boundary']['arms']['G_hold']['frames']['TP']>groups[g]['Boundary']['arms']['U_hold']['frames']['TP']]
    contribution=dict(G_usable=findings['G']['usable'],horizontal_extra_TP=hmetrics['Boundary']['arms']['G_hold']['frames']['TP']-hmetrics['Boundary']['arms']['U_hold']['frames']['TP'],
        horizontal_gain_groups=hgain,boundary_TP_not_lower=result['Boundary']['arms']['G_hold']['frames']['TP']>=result['Boundary']['arms']['U_hold']['frames']['TP'])
    supported=contribution['G_usable'] and contribution['horizontal_extra_TP']>=5 and len(hgain)>=2 and contribution['boundary_TP_not_lower']
    terminal=dict(geometry_hypothesis='SUPPORTED_DEVELOPMENT' if supported else 'NOT_SUPPORTED',arms=findings,contribution=contribution,
        protected_test_activated=False,automatic_successor=False,scope='CONSUMED_SIMULATION_MATCHED_INPUT_ABLATION')
    for name,data in [('metrics.json',result),('group-metrics.json',groups),('horizontal-metrics.json',hmetrics),('frame-results.json',rows),('result.json',terminal)]: write(OUT/name,data)
    write(OUT/'local-inheritance.json',dict(terminal_id=NAME,inheritance_role='COMPONENT_OR_CHALLENGER' if any(v['usable'] for v in findings.values()) else 'NEGATIVE_CONTROL',
        role_scope='Exact matched ordered-spatial U/G fit and cost/timing question',arm_roles={a:'COMPONENT_OR_CHALLENGER' if f['usable'] else 'NEGATIVE_CONTROL' for a,f in findings.items()},
        geometry_hypothesis=terminal['geometry_hypothesis'],retained_surface='A, UNKNOWN and all older frozen controls unchanged',revisit_trigger='Separate authorization and materially different hypothesis'))
    seal('evaluation-seal.json',['metrics.json','group-metrics.json','horizontal-metrics.json','frame-results.json','result.json','local-inheritance.json'])
    print(terminal,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['freeze','prepare','fit','select','predict','evaluate'])
    globals()[parser.parse_args().stage]()
