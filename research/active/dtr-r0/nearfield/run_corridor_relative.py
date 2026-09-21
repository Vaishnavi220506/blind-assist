"""One consumed-Development fit; predictions sealed before transfer truth join."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

import corridor_relative_model as model
from run_spatial_bce import held, fp_segments
from spatial_bce_model import _select, _precision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = dict(original=ROOT/'artifacts.local/work/ba-spatial-bce-20260920',
               transfer=ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921')
NAME = 'ba-corridor-relative-20260921'
OUT = ROOT/'artifacts.local/work'/NAME
PROTO = HERE/'CORRIDOR_RELATIVE_PROTOCOL_20260921.md'
ARMS = ('A_current','B_current','R_current','A_hold','B_hold','R_hold')
OLD_THRESHOLD = 7.6612162590026855
CODE = ('corridor_relative_model.py','run_corridor_relative.py','spatial_bce_model.py',
        'run_spatial_bce.py','ba_camera_corridor.py','ba_camera_corridor_metrics.py',
        'full_event_metrics_20260920.py','test_corridor_relative.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def seal(name, files):
    write(OUT/name, dict(protocol_sha256=sha(OUT/'protocol.json'), hashes={f:sha(OUT/f) for f in files}))


def check(name):
    s = read(OUT/name)
    assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    for f, digest in s['hashes'].items():
        assert sha(OUT/f) == digest, f


def verify():
    p = read(OUT/'protocol.json')
    assert sha(OUT/'protocol-before-run.md') == p['protocol_text_sha256']
    for f, digest in p['code_hashes'].items():
        assert sha(ROOT/f) == digest, f
    for f, digest in p['source_hashes'].items():
        assert sha(ROOT/f) == digest, f
    return p


def freeze():
    assert not OUT.exists(), 'New output only; no overwrite/retry'
    files = []
    split_groups = {}
    for key, source in SOURCES.items():
        names = ['observations.npz','identities.json','baseline.json','evaluator/metadata.json',
                 'observation-seal.json','protocol.json']
        names += ['evaluator/train-labels.json','evaluator/dev-labels.json','dev-logits.npy'] if key == 'original' else [
            'evaluator/transfer-labels.json','predictions.json','prediction-seal.json','source-admission.json']
        observation = read(source/'observation-seal.json')
        assert observation['protocol_sha256'] == sha(source/'protocol.json')
        for name in names:
            if name in observation['hashes']:
                assert sha(source/name) == observation['hashes'][name], name
        prior_seal = read(source/('fit-seal.json' if key=='original' else 'prediction-seal.json'))
        prior_output = 'dev-logits.npy' if key=='original' else 'predictions.json'
        assert prior_seal['protocol_sha256']==sha(source/'protocol.json')
        assert prior_seal['hashes'][prior_output]==sha(source/prior_output)
        files.append(source/('fit-seal.json' if key=='original' else 'prediction-seal.json'))
        metadata=read(source/'evaluator/metadata.json')
        for split,frames,groups in ([('train',1728,24),('dev',576,8)] if key=='original' else [('transfer',1152,16)]):
            subset=[r for r in metadata if r['split']==split]
            split_groups[split]={r['base_group_id'] for r in subset}
            assert len(subset)==frames and len(split_groups[split])==groups
            assert all(sum(r['base_group_id']==g for r in subset)==72 for g in split_groups[split])
        files.extend(source/n for n in names)
    assert all(not split_groups[a]&split_groups[b] for a,b in [('train','dev'),('train','transfer'),('dev','transfer')])
    OUT.mkdir()
    (OUT/'protocol-before-run.md').write_bytes(PROTO.read_bytes())
    write(OUT/'protocol.json',dict(id=NAME, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_text_sha256=sha(PROTO), recipe=model.RECIPE,
        code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [*(HERE/f for f in CODE), ROOT/'tools/research_backend.py']},
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        protected_test_activated=False, scope='CONSUMED_SIMULATION_DEVELOPMENT', automatic_successor=False))
    print('FROZEN', NAME, flush=True)


def prepare():
    verify()
    write(OUT/'features-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        backend='CPU', placement='TASK_NOT_GPU_SUITABLE', reason='Small per-zone ragged statistics and image IO; tensor fit probed separately'))
    started = time.perf_counter()
    for name, key, split in [('train','original','train'),('dev','original','dev'),('transfer','transfer','transfer')]:
        source = SOURCES[key]
        # Metadata routes complete splits only; no truth/geometric fields enter extract.
        indices = [m['index'] for m in read(source/'evaluator/metadata.json') if m['split']==split]
        ids = read(source/'identities.json')
        with np.load(source/'observations.npz',allow_pickle=False) as data:
            boxes, ranges = data['boxes'], data['ranges'][indices]
        features = np.empty((len(indices),64,60),np.float32)
        for j, i in enumerate(indices):
            identity = ids[i]
            path = source/identity['rgb_path']
            assert sha(path) == identity['rgb_sha256']
            features[j], valid = model.extract(path,ranges[j],boxes)
            assert np.array_equal(valid,features[j,:,-1].astype(bool))
            if j % 288 == 0:
                print('FEATURES',name,j,'/',len(indices),flush=True)
        np.save(OUT/f'{name}-features.npy', features)
        write(OUT/f'{name}-indices.json',indices)
    write(OUT/'features-receipt.json',dict(elapsed_s=time.perf_counter()-started, protected_test_images_read=0,
        native_depth_read=False, labels_read=False, features=60, sources=['native RGB','original ranges','original zone boxes']))
    seal('features-seal.json',[f'{s}-{f}' for s in ('train','dev','transfer') for f in ('features.npy','indices.json')]+['features-receipt.json'])


def fit():
    verify(); check('features-seal.json'); _precision()
    write(OUT/'fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),recipe=model.RECIPE))
    labels = read(SOURCES['original']/'evaluator/train-labels.json')
    assert [r['index'] for r in labels] == read(OUT/'train-indices.json')
    x = torch.from_numpy(np.load(OUT/'train-features.npy'))
    y = torch.tensor([r['truth'] for r in labels],dtype=torch.float32)
    torch.manual_seed(model.RECIPE['seed'])
    head = model.Head()
    device, backend = _select(head,x[:64],OUT/'fit-backend.json',y[:64])
    head.to(device).train(); x,y=x.to(device),y.to(device)
    rng = np.random.default_rng(model.RECIPE['seed'])
    opt = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    started=time.perf_counter()
    with (OUT/'training.jsonl').open('x',encoding='utf-8') as stream:
        for step in range(1,1201):
            idx=torch.as_tensor(rng.integers(0,len(x),64),device=device)
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(head(x[idx]),y[idx])
            loss.backward(); opt.step()
            if step==1 or step%100==0:
                row=dict(step=step,bce=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-started)
                stream.write(json.dumps(row)+'\n'); stream.flush(); print('FIT',json.dumps(row),flush=True)
    torch.save(dict(recipe=model.RECIPE,state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()}),OUT/'head_last.pt')
    write(OUT/'fit-receipt.json',dict(parameters=sum(p.numel() for p in head.parameters()),backend=backend,
        elapsed_s=time.perf_counter()-started,train_rows=len(x),actual_device=str(next(head.parameters()).device)))
    head.eval()
    for name in ('train','dev'):
        np.save(OUT/f'{name}-logits.npy', infer(head,np.load(OUT/f'{name}-features.npy')))
    seal('fit-seal.json',['head_last.pt','fit-receipt.json','fit-backend.json','training.jsonl','train-logits.npy','dev-logits.npy'])


@torch.inference_mode()
def infer(head,x):
    device=next(head.parameters()).device
    return np.concatenate([head(torch.from_numpy(x[s:s+64]).to(device)).cpu().numpy() for s in range(0,len(x),64)])


def costs(logits,threshold,truth,metadata,a):
    y,a=np.asarray(truth,bool),np.asarray(a,bool)
    r=a|(np.asarray(logits,float)>=threshold)
    result={}
    for region in ('Core','Boundary'):
        mask=np.array([(m['layout_relation']=='BOUNDARY')==(region=='Boundary') for m in metadata])
        negatives=int((mask&~y).sum())
        clips=len({m['clip_id'] for m,k in zip(metadata,mask) if k})
        for suffix,aa,rr,fraction in [('current',a,r,.01),('hold',held(a,metadata),held(r,metadata),.02)]:
            afp,rfp=int((mask&~y&aa).sum()),int((mask&~y&rr).sum())
            aseg,rseg=fp_segments(aa,y,metadata,mask),fp_segments(rr,y,metadata,mask)
            fp_cap=int(np.floor(fraction*negatives)); segment_cap=int(np.floor(.125*clips))
            result[region+'_'+suffix]=dict(negative_frames=negatives,clips=clips,added_FP=rfp-afp,FP_cap=fp_cap,
                added_segments=rseg-aseg,segment_cap=segment_cap,
                pass_cost=rfp-afp<=fp_cap and rseg-aseg<=segment_cap)
    return result


def select():
    verify(); check('fit-seal.json')
    idx=read(OUT/'dev-indices.json'); source=SOURCES['original']
    metadata=read(source/'evaluator/metadata.json'); metadata=[metadata[i] for i in idx]
    labels=read(source/'evaluator/dev-labels.json')
    assert [r['index'] for r in labels]==idx
    truth=[r['truth'] for r in labels]; baseline=read(source/'baseline.json')
    a=[baseline[i]['current'] for i in idx]
    logits=np.load(OUT/'dev-logits.npy').astype(float); assert np.isfinite(logits).all()
    thresholds=sorted(set(logits.tolist())|{float(np.nextafter(logits.max(),np.inf))})
    chosen=None; records=[]
    for threshold in thresholds:
        cost=costs(logits,threshold,truth,metadata,a)
        okay=all(r['pass_cost'] for r in cost.values())
        records.append(dict(threshold=threshold,cost=cost,admissible=okay))
        if okay and chosen is None:
            chosen=threshold
    assert chosen is not None
    write(OUT/'selection.json',dict(threshold=chosen,rule='Lowest dev score satisfying fixed added-cost caps',
        cost=costs(logits,chosen,truth,metadata,a),thresholds=len(thresholds)))
    write(OUT/'selection-candidates.json',records)
    seal('selection-seal.json',['selection.json','selection-candidates.json'])
    print('SELECTED',chosen,flush=True)


def predict():
    verify(); check('features-seal.json'); check('fit-seal.json'); check('selection-seal.json'); _precision()
    write(OUT/'prediction-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),evaluator_truth_read=False))
    payload=torch.load(OUT/'head_last.pt',map_location='cpu',weights_only=True)
    assert payload['recipe']==model.RECIPE
    head=model.Head(); head.load_state_dict(payload['state_dict']); head.eval()
    x=np.load(OUT/'transfer-features.npy')
    device,backend=_select(head,torch.from_numpy(x[:64]),OUT/'inference-backend.json')
    head.to(device); start=time.perf_counter(); logits=infer(head,x)
    assert np.isfinite(logits).all()
    np.save(OUT/'transfer-logits.npy',logits)
    threshold=read(OUT/'selection.json')['threshold']
    source=SOURCES['transfer']; ids=read(source/'identities.json'); baseline=read(source/'baseline.json')
    old=read(source/'predictions.json')
    assert read(OUT/'transfer-indices.json')==list(range(len(ids)))
    a=np.array([r['current'] for r in baseline],bool); r=a|(logits.astype(float)>=threshold)
    ah,rh=held(a,ids),held(r,ids)
    rows=[]
    for i,identity in enumerate(ids):
        assert identity['index']==old[i]['index']==baseline[i]['index']==i
        assert bool(a[i])==old[i]['flags']['A_current'] and bool(ah[i])==old[i]['flags']['A_hold']
        assert old[i]['flags']['C_current']==bool(a[i] or old[i]['logit']>=OLD_THRESHOLD)
        flags=dict(A_current=bool(a[i]),B_current=old[i]['flags']['C_current'],R_current=bool(r[i]),
                   A_hold=bool(ah[i]),B_hold=old[i]['flags']['C_hold'],R_hold=bool(rh[i]))
        rows.append(dict(**{k:identity[k] for k in ('id','index','clip_id','time_s','frame_in_clip')},
            logit=float(logits[i]), flags=flags,current_unknown={k:baseline[i]['unknown'] for k in ARMS}))
    write(OUT/'predictions.json',rows)
    write(OUT/'inference-receipt.json',dict(backend=backend,elapsed_s=time.perf_counter()-start,frames=len(rows),
        actual_device=str(next(head.parameters()).device),protected_test_activated=False))
    seal('prediction-seal.json',['predictions.json','transfer-logits.npy','inference-backend.json','inference-receipt.json'])
    print('PREDICTIONS_SEALED',len(rows),flush=True)


def metrics(rows):
    spec=importlib.util.spec_from_file_location('relative_metrics',HERE/'full_event_metrics_20260920.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.ARMS=ARMS
    return module.evaluate(rows)


def evaluate():
    verify(); check('prediction-seal.json'); check('selection-seal.json')
    source=SOURCES['transfer']; predictions=read(OUT/'predictions.json')
    metadata=read(source/'evaluator/metadata.json'); labels=read(source/'evaluator/transfer-labels.json')
    native=read(source/'source-admission.json')['frames']
    rows=[]
    for p,m,y,n in zip(predictions,metadata,labels,native):
        assert p['index']==m['index']==y['index'] and p['id']==m['id']==n['id']
        rows.append(dict(**m,truth=y['truth'],flags=p['flags'],current_unknown=p['current_unknown'],
                         native_target_corridor_samples=n['returned_target_corridor_samples']))
    assert len(rows)==1152 and len({r['base_group_id'] for r in rows})==16
    result=metrics(rows); write(OUT/'metrics.json',result); write(OUT/'frame-results.json',rows)
    groups={g:metrics([r for r in rows if r['base_group_id']==g]) for g in sorted({r['base_group_id'] for r in rows})}
    write(OUT/'group-metrics.json',groups)
    logits=np.load(OUT/'transfer-logits.npy'); threshold=read(OUT/'selection.json')['threshold']
    cost=costs(logits,threshold,[r['truth'] for r in rows],metadata,[r['flags']['A_current'] for r in rows])
    gains=[g for g,m in groups.items() if m['Boundary']['arms']['R_hold']['frames']['TP']>m['Boundary']['arms']['A_hold']['frames']['TP']]
    retention=all(not r['truth'] or all(not r['flags']['A_'+s] or r['flags']['R_'+s] for s in ('current','hold')) for r in rows)
    gates=dict(cost=all(v['pass_cost'] for v in cost.values()),retain_A=retention,
        boundary_recall=result['Boundary']['arms']['R_hold']['frames']['recall']>=.5,broad_gain=len(gains)>=8)
    increments=[r for r in rows if r['truth'] and r['flags']['R_current'] and not r['flags']['A_current']]
    terminal=dict(status='PASS' if all(gates.values()) else 'NO_GO',gates=gates,cost=cost,
        boundary_gain_groups=gains,new_current_TP=len(increments),
        new_current_TP_without_native_corridor_contributor=sum(r['native_target_corridor_samples']==0 for r in increments),
        protected_test_activated=False,automatic_successor=False,scope='CONSUMED_SIMULATION_DEVELOPMENT')
    write(OUT/'result.json',terminal)
    role='COMPONENT_OR_CHALLENGER' if all(gates.values()) else 'NEGATIVE_CONTROL'
    write(OUT/'local-inheritance.json',dict(terminal_id=NAME,inheritance_role=role,
        inheritance_mode='CHALLENGER' if role=='COMPONENT_OR_CHALLENGER' else None,
        role_scope='This exact 60-feature interval-band / shared max MLP / fit / fixed-cost selection recipe',
        retained_surface='A and old negative results unchanged; original ranges and UNKNOWN unchanged',
        failure_signature=terminal['gates'],revisit_trigger='Separate authorization and a materially different hypothesis',
        assignment_basis='Frozen protocol and sealed consumed-Development predictions; global registration attempted separately'))
    seal('evaluation-seal.json',['metrics.json','frame-results.json','group-metrics.json','result.json','local-inheritance.json'])
    print(json.dumps(terminal),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['freeze','prepare','fit','select','predict','evaluate'])
    args=parser.parse_args(); globals()[args.stage]()
