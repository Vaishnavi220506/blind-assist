"""One two-arm 33-parameter consumed-Development pilot; never original test."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT / 'artifacts.local/work/ba-last-layer-20260921'
PROTOCOL = HERE / 'LAST_LAYER_PROTOCOL_20260921.md'
HIGH = 7.6612162590026855
ARMS = ('A', 'C', 'uniform', 'balanced')
ROLES = ('fit', 'selection', 'evaluation')
RECIPE = dict(parameters=33, features=32, dtype='float64', optimizer='LBFGS',
              lr=1., max_iter=500, max_eval=1000, tolerance_grad=1e-9,
              tolerance_change=1e-12, line_search_fn='strong_wolfe', l2=.001,
              standard_deviation_floor=1e-6, initialization='zeros')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def original_test_closed():
    for name in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'):
        assert not (OLD/name).exists(), name


def seal(name, paths):
    write(OUT/name, dict(protocol_sha256=sha(OUT/'protocol.json'),
                        hashes={p: sha(OUT/p) for p in paths}))


def check_seal(name):
    s = read(OUT/name)
    assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    for p, h in s['hashes'].items():
        assert sha(OUT/p) == h, p


def verify():
    original_test_closed()
    p = read(OUT/'protocol.json')
    for path, h in p['sources'].items():
        assert sha(ROOT/path) == h, path
    for path, h in p['code'].items():
        assert sha(ROOT/path) == h, path
    assert p['recipe'] == RECIPE
    return p


def hold(flags, rows):
    result = np.asarray(flags, bool).copy()
    for i in range(1, len(rows)):
        if rows[i]['clip_id'] == rows[i-1]['clip_id']:
            assert abs(rows[i]['time_s'] - rows[i-1]['time_s'] - .2) < 1e-8
            result[i] = bool(flags[i] or flags[i-1])
    return result


def role_for(row, source):
    if source == 'dev':
        assert row['split'] == 'dev'
        return 'fit'
    suffix = row['base_group_id'].rsplit('_', 1)[1]
    assert suffix in ('g00', 'g01', 'g02', 'g03')
    return 'selection' if suffix in ('g00', 'g01') else 'evaluation'


def row_weights(rows, y, balanced):
    if not balanced:
        return np.full(len(rows), 1/len(rows), np.float64)
    cells = [(r['base_group_id'], bool(v)) for r, v in zip(rows, y)]
    counts = Counter(cells)
    assert len(counts) == 2*len({r['base_group_id'] for r in rows})
    return np.array([1/(len(counts)*counts[c]) for c in cells], np.float64)


def cutoff(scores, a, y, rows):
    """Exact lowest cutoff without added current/one-frame-held false alerts."""
    scores, a, y = np.asarray(scores), np.asarray(a, bool), np.asarray(y, bool)
    forbidden = (~y) & (~a)
    ah = hold(a, rows)
    for i in range(1, len(rows)):
        if rows[i]['clip_id'] == rows[i-1]['clip_id'] and not y[i] and not ah[i]:
            forbidden[i-1] = True
    assert forbidden.any(), 'Expected negative calibration opportunities'
    value = float(np.nextafter(scores[forbidden].max(), np.inf))
    c = a | (scores >= value)
    assert not np.any(c & ~a & ~y)
    assert not np.any(hold(c, rows) & ~ah & ~y)
    return dict(threshold=value, forbidden_ids=[r['id'] for r, b in zip(rows, forbidden) if b],
                forbidden_max=float(scores[forbidden].max()))


def prepare():
    assert not OUT.exists(), 'One pilot, no overwrite'
    original_test_closed()
    assert sha(OLD/'fit/head_last.pt') == read(OLD/'fit/train_receipt.json')['checkpoint_sha256']
    sources = []
    for source in (OLD, TRANSFER):
        sources += [source/n for n in ('protocol.json','observations.npz','identities.json',
            'baseline.json','evaluator/metadata.json','source-admission.json',
            'features/feature_receipt.json','features/rgb_features.npy')]
    sources += [OLD/'evaluator/dev-labels.json', OLD/'dev-logits.npy', OLD/'fit/head_last.pt',
                OLD/'fit/train_receipt.json', TRANSFER/'evaluator/transfer-labels.json',
                TRANSFER/'logits.npy', TRANSFER/'prediction-seal.json']
    code = [Path(__file__), PROTOCOL, HERE/'spatial_bce_model.py', HERE/'run_spatial_bce.py',
            HERE/'full_event_metrics_20260920.py', ROOT/'tools/research_backend.py',
            HERE/'tof_fov45_core.py']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', dict(id='ba-last-layer-20260921',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), recipe=RECIPE,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        code={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        scope='CONSUMED_DEVELOPMENT', roles=dict(fit='original dev8',selection='transfer g00/g01',
        evaluation='transfer g02/g03'), challenger='balanced', control='uniform',
        original_test_access=False, automatic_successor=False))
    (OUT/'protocol-before-run.md').write_bytes(PROTOCOL.read_bytes())
    roles = {role:[] for role in ROLES}
    labels = {role:[] for role in ROLES}
    for tag, source, label_file in (('dev',OLD,'dev-labels.json'),('transfer',TRANSFER,'transfer-labels.json')):
        metadata = read(source/'evaluator/metadata.json')
        if tag == 'dev':
            metadata = [r for r in metadata if r['split']=='dev']
        ys = {r['index']:r['truth'] for r in read(source/'evaluator'/label_file)}
        baseline = read(source/'baseline.json')
        ids = read(source/'identities.json')
        for r in metadata:
            i = r['index']
            assert r['id'] == ids[i]['id'] and r['clip_id'] == ids[i]['clip_id']
            assert type(ys[i]) is bool
            role = role_for(r,tag)
            rid = tag+':'+r['id']
            roles[role].append(dict(**{k:v for k,v in r.items() if k!='id'},
                id=rid, source=tag, source_id=r['id'], a=bool(baseline[i]['current']),
                unknown=bool(baseline[i]['unknown'])))
            labels[role].append(dict(id=rid, truth=ys[i]))
    group_sets = []
    for role in ROLES:
        rows = roles[role]
        groups = {r['base_group_id'] for r in rows}
        group_sets.append(groups)
        assert len(rows)==576 and len(groups)==8
        assert set(Counter(r['clip_id'] for r in rows).values()) == {24}
        assert set(Counter(r['type_id'] for r in rows).values()) == {144}
        write(OUT/(role+'-rows.json'),rows)
        write(OUT/'labels'/(role+'.json'),labels[role])
    assert all(not a&b for i,a in enumerate(group_sets) for b in group_sets[i+1:])
    seal('role-seal.json',[r+'-rows.json' for r in ROLES]+['labels/'+r+'.json' for r in ROLES])
    write(OUT/'role-summary.json',{role:dict(frames=len(roles[role]),
        groups=sorted({r['base_group_id'] for r in roles[role]})) for role in ROLES})
    verify()
    print('PREPARED 8 fit / 8 selection / 8 evaluation layouts',flush=True)


def extract():
    import torch
    from spatial_bce_model import build_inputs, load_head, _select, _precision
    verify(); check_seal('role-seal.json'); _precision()
    head = load_head(OLD/'fit/head_last.pt').eval().requires_grad_(False)
    trunk = head.net[:-1].eval().requires_grad_(False)
    write(OUT/'extraction-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        labels_read=False, original_test_rows_selected=0))
    records = {}
    for tag, source in (('dev',OLD),('transfer',TRANSFER)):
        rows = [r for role in ROLES for r in read(OUT/(role+'-rows.json')) if r['source']==tag]
        indices = np.array([r['index'] for r in rows])
        if tag=='dev':
            assert len(rows)==576 and all(r['split']=='dev' for r in rows)
        cache = np.load(source/'features/rgb_features.npy',mmap_mode='r')
        receipt = read(source/'features/feature_receipt.json')
        assert sha(source/'features/rgb_features.npy') == receipt['cache_sha256']
        with np.load(source/'observations.npz') as observed:
            inputs = build_inputs(cache[indices],observed['ranges'][indices])
        if tag=='dev':
            device, backend = _select(trunk,torch.from_numpy(inputs[:64].copy()),OUT/'extract-backend.json')
            trunk.to(device); head.to(device)
        started=time.perf_counter(); hidden=[]; logits=[]
        with torch.inference_mode():
            for start in range(0,len(rows),64):
                z=trunk(torch.from_numpy(inputs[start:start+64]).to(device))
                hidden.append(z.cpu().numpy())
                logits.append(head.net[-1](z).squeeze(-1).cpu().numpy())
        hidden=np.concatenate(hidden); logits=np.concatenate(logits)
        saved=np.load(source/('dev-logits.npy' if tag=='dev' else 'logits.npy'))
        if tag=='transfer': saved=saved[indices]
        error=float(np.max(np.abs(saved-logits)))
        assert hidden.shape==(len(rows),32) and np.isfinite(hidden).all()
        assert error<=1e-4, ('Original B did not reproduce',error)
        for row,z,s in zip(rows,hidden,saved): records[row['id']]=(z,float(s))
        print('EXTRACTED',tag,len(rows),'max_original_logit_error',error,flush=True)
        write(OUT/(tag+'-extraction.json'),dict(rows=len(rows),elapsed_s=time.perf_counter()-started,
            max_original_logit_error=error,device=device,original_test_rows=0))
    files=[]
    for role in ROLES:
        rows=read(OUT/(role+'-rows.json'))
        np.save(OUT/(role+'-hidden.npy'),np.stack([records[r['id']][0] for r in rows]))
        np.save(OUT/(role+'-original.npy'),np.array([records[r['id']][1] for r in rows]))
        files += [role+'-hidden.npy',role+'-original.npy']
    seal('hidden-seal.json',files+['extract-backend.json','dev-extraction.json','transfer-extraction.json'])
    verify()


def fit_one(x,y,weights,device,max_iter=None):
    import torch
    import torch.nn.functional as F
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float64,device=device)
    xx,yy,ww=map(tensor,(x,y,weights))
    model=torch.nn.Linear(32,1,dtype=torch.float64,device=device)
    with torch.no_grad(): model.weight.zero_(); model.bias.zero_()
    options={k:RECIPE[k] for k in ('lr','max_iter','max_eval','tolerance_grad','tolerance_change','line_search_fn')}
    if max_iter is not None: options.update(max_iter=max_iter,max_eval=max_iter*2)
    optimizer=torch.optim.LBFGS(model.parameters(),**options)
    trace=[]
    def closure():
        optimizer.zero_grad()
        bce=(F.binary_cross_entropy_with_logits(model(xx).squeeze(-1),yy,reduction='none')*ww).sum()
        loss=bce+RECIPE['l2']/2*model.weight.square().sum()
        loss.backward()
        trace.append(float(loss.detach().cpu()))
        return loss
    started=time.perf_counter(); optimizer.step(closure)
    if device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    loss=closure()
    state=optimizer.state[model.weight]
    receipt=dict(parameters=sum(p.numel() for p in model.parameters()),objective=float(loss.detach()),
        gradient_max=max(float(p.grad.abs().max()) for p in model.parameters()),
        iterations=int(state['n_iter']),function_evaluations=int(state['func_evals']),
        elapsed_s=elapsed,objective_trace=trace,device=device)
    assert receipt['parameters']==33 and np.isfinite(receipt['objective'])
    return model,receipt


def fit():
    import torch
    from tools.research_backend import BackendCandidate,Workload,select_backend,torch_observation
    from spatial_bce_model import _precision
    verify();check_seal('hidden-seal.json');_precision()
    rows=read(OUT/'fit-rows.json');ys=read(OUT/'labels/fit.json')
    assert [r['id'] for r in rows]==[r['id'] for r in ys]
    y=np.array([r['truth'] for r in ys],np.float64)
    z=np.load(OUT/'fit-hidden.npy').astype(np.float64)
    mean=z.mean(0);scale=np.maximum(z.std(0),RECIPE['standard_deviation_floor'])
    x=(z-mean)/scale
    write(OUT/'fit-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        label_role='fit_only',fit_rows=len(x),selection_or_evaluation_labels_read=False))
    weights={arm:row_weights(rows,y,arm=='balanced') for arm in ('uniform','balanced')}
    candidates={}
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        def probe(d=device): return fit_one(x,y,weights['balanced'],d,max_iter=8)[0]
        candidates[device]=BackendCandidate(device,device,probe,
            lambda model:torch_observation(model=model),torch.cuda.synchronize if device=='cuda' else lambda:None)
    backend=select_backend(Workload.BATCH_TENSOR,cpu=candidates['cpu'],gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=OUT/'fit-backend.json',warmups=1,repeats=3)
    device=backend['selected_device_type'];files=['fit-backend.json']
    for arm in ('uniform','balanced'):
        model,receipt=fit_one(x,y,weights[arm],device)
        w=model.weight.detach().cpu().numpy().ravel();b=float(model.bias.detach().cpu()[0])
        np.savez(OUT/(arm+'-readout.npz'),weight=w,bias=b,mean=mean,scale=scale)
        logits=x@w+b
        receipt.update(arm=arm,correct_at_zero=int(np.sum((logits>=0)==y)),rows=len(y),
            weight_sum=float(weights[arm].sum()),weight_min=float(weights[arm].min()),
            weight_max=float(weights[arm].max()),classes=Counter(map(int,y)),
            convergence_note='Final prescribed LBFGS iterate; no tuned retry')
        write(OUT/(arm+'-fit.json'),receipt)
        files += [arm+'-readout.npz',arm+'-fit.json']
        print('FIT',arm,{k:receipt[k] for k in ('correct_at_zero','objective','gradient_max','iterations','elapsed_s')},flush=True)
    seal('fit-seal.json',files);verify()


def predict():
    verify();check_seal('fit-seal.json');check_seal('hidden-seal.json')
    selections={};scores={}
    # Prediction for all roles depends only on saved observed features.
    for role in ROLES:
        z=np.load(OUT/(role+'-hidden.npy')).astype(np.float64)
        scores[role]={}
        for arm in ('uniform','balanced'):
            with np.load(OUT/(arm+'-readout.npz')) as m:
                scores[role][arm]=((z-m['mean'])/m['scale'])@m['weight']+float(m['bias'])
    rows=read(OUT/'selection-rows.json');ys=read(OUT/'labels/selection.json')
    assert [r['id'] for r in rows]==[r['id'] for r in ys]
    y=np.array([r['truth'] for r in ys],bool);a=np.array([r['a'] for r in rows],bool)
    for arm in ('uniform','balanced'):
        selections[arm]=cutoff(scores['selection'][arm],a,y,rows)
    write(OUT/'selection.json',selections)
    files=['selection.json']
    for role in ROLES:
        rows=read(OUT/(role+'-rows.json'));a=np.array([r['a'] for r in rows],bool)
        original=np.load(OUT/(role+'-original.npy'))
        currents={'A':a,'C':a|(original>=HIGH)}
        currents.update({arm:a|(scores[role][arm]>=selections[arm]['threshold']) for arm in selections})
        flags={}
        for arm,cur in currents.items():
            flags[arm+'_current']=cur;flags[arm+'_hold']=hold(cur,rows)
        predictions=[dict(id=r['id'],scores=dict(original=float(original[i]),
            **{arm:float(scores[role][arm][i]) for arm in selections}),
            flags={k:bool(v[i]) for k,v in flags.items()},
            current_unknown={k:r['unknown'] for k in flags}) for i,r in enumerate(rows)]
        write(OUT/(role+'-predictions.json'),predictions);files.append(role+'-predictions.json')
    write(OUT/'prediction-receipt.json',dict(evaluation_labels_read=False,role_materialization_disclosed=True,
        cutoffs_fixed_before_evaluation=True,original_test_activated=False))
    seal('prediction-seal.json',files+['prediction-receipt.json']);verify()
    print('PREDICTIONS_SEALED', {a:v['threshold'] for a,v in selections.items()},flush=True)


def metric_module():
    spec=importlib.util.spec_from_file_location('last_layer_metrics',HERE/'full_event_metrics_20260920.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ARMS=tuple(a+'_'+s for a in ARMS for s in ('current','hold'))
    return module


def xauc(rows):
    groups=sorted({r['base_group_id'] for r in rows})
    paired={}
    for r in rows: paired.setdefault((r['base_group_id'],r['frame_in_clip']),{})[r['layout_relation']]=r
    primary=[p for p in paired.values() if p['BOUNDARY']['truth'] and p['INSIDE']['truth'] and not p['OUTSIDE']['truth']]
    result={}
    for arm in ('original','uniform','balanced'):
        positive={g:[p['BOUNDARY']['scores'][arm] for p in primary if p['BOUNDARY']['base_group_id']==g] for g in groups}
        negative={g:[p['OUTSIDE']['scores'][arm] for p in primary if p['OUTSIDE']['base_group_id']==g] for g in groups}
        matrix=[];counts=[];off=[]
        for g in groups:
            line=[];n=[]
            for h in groups:
                d=np.subtract.outer(positive[g],negative[h])
                value=float(np.mean((d>0)+.5*(d==0))) if d.size else None
                line.append(value);n.append(int(d.size))
                if g!=h and value is not None: off.append(value)
            matrix.append(line);counts.append(n)
        result[arm]=dict(groups=groups,matrix=matrix,pair_counts=counts,
            off_diagonal_equal_layout_mean=float(np.mean(off)),
            off_diagonal_min=float(min(off)),primary_matched_pairs=len(primary),
            primary_matched_ordered=sum(p['BOUNDARY']['scores'][arm]>p['OUTSIDE']['scores'][arm] for p in primary))
    return result


def assess(rows,report,arm):
    boundary=[r for r in rows if r['layout_relation']=='BOUNDARY']
    rescue=[r for r in boundary if r['truth'] and r['flags']['C_current'] and not r['flags']['A_current']]
    retained=sum(r['flags'][arm+'_current'] for r in rescue)
    flags_retained=all(not r['flags']['A_'+s] or r['flags'][arm+'_'+s] for r in rows for s in ('current','hold'))
    no_extra=all(report[g]['arms'][arm+'_'+s]['frames']['FP']==report[g]['arms']['A_'+s]['frames']['FP']
        and report[g]['arms'][arm+'_'+s]['false_alert_segment_count']<=report[g]['arms']['A_'+s]['false_alert_segment_count']
        for g in ('Core','Boundary') for s in ('current','hold'))
    gaining=len({r['base_group_id'] for r in boundary if r['truth'] and r['flags'][arm+'_current'] and not r['flags']['A_current']})
    event_ids=lambda a:{e['clip_id'] for e in report['Boundary']['arms'][a+'_current']['events'] if e['detected']}
    events_retained=event_ids('C')<=event_ids(arm)
    gain=report['Boundary']['arms'][arm+'_current']['frames']['recall']-report['Boundary']['arms']['A_current']['frames']['recall']
    rate=retained/len(rescue) if rescue else None
    return dict(all_A_flags_retained=flags_retained,no_added_FP_current_and_hold=no_extra,
        C_boundary_rescues=len(rescue),C_boundary_rescues_retained=retained,C_rescue_retention=rate,
        C_boundary_detected_events_retained=events_retained,boundary_recall_gain=gain,gaining_groups=gaining,
        useful_repair=bool(flags_retained and no_extra and gain>=.10 and gaining>=4 and rate is not None and rate>=.90 and events_retained))


def evaluate():
    verify();check_seal('prediction-seal.json');check_seal('role-seal.json')
    write(OUT/'evaluation-start.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json')))
    module=metric_module();summaries={};gates={};files=[]
    admissions={tag:{r['id']:r for r in read(source/'source-admission.json')['frames']}
                for tag,source in (('dev',OLD),('transfer',TRANSFER))}
    all_transfer=[]
    for role in ROLES:
        meta=read(OUT/(role+'-rows.json'));pred=read(OUT/(role+'-predictions.json'));ys=read(OUT/'labels'/(role+'.json'))
        assert [r['id'] for r in meta]==[r['id'] for r in pred]==[r['id'] for r in ys]
        rows=[dict(**r,truth=y['truth'],scores=p['scores'],flags=p['flags'],current_unknown=p['current_unknown'],
            native_target_corridor_samples=admissions[r['source']][r['source_id']]['returned_target_corridor_samples'])
            for r,p,y in zip(meta,pred,ys)]
        report=module.evaluate(rows)
        groups={g:module.evaluate([r for r in rows if r['base_group_id']==g]) for g in sorted({r['base_group_id'] for r in rows})}
        gates[role]={a:assess(rows,report,a) for a in ('uniform','balanced')}
        summaries[role]={g:{a:dict(counts=[v['frames'][k] for k in ('TP','FP','FN')],
            precision=v['frames']['precision'],FPR=v['frames']['FPR'],events=[v['detected_events'],v['event_count']],
            false_segments=v['false_alert_segment_count'],unknown=v['current_unknown']) for a,v in report[g]['arms'].items()}
            for g in ('Core','Boundary')}
        changes={}
        for arm in ('uniform','balanced'):
            changes[arm]={s:{name:[r['id'] for r in rows if condition(r)] for name,condition in (
                ('rescued_A_TP',lambda r:r['truth'] and r['flags'][arm+'_'+s] and not r['flags']['A_'+s]),
                ('added_A_FP',lambda r:not r['truth'] and r['flags'][arm+'_'+s] and not r['flags']['A_'+s]),
                ('lost_C_TP',lambda r:r['truth'] and r['flags']['C_'+s] and not r['flags'][arm+'_'+s]),
                ('removed_C_FP',lambda r:not r['truth'] and r['flags']['C_'+s] and not r['flags'][arm+'_'+s]))}
                for s in ('current','hold')}
        for suffix,value in [('frame-results',rows),('metrics',report),('group-metrics',groups),('changes',changes),('xauc',xauc(rows))]:
            name=role+'-'+suffix+'.json';write(OUT/name,value);files.append(name)
        if role!='fit': all_transfer+=rows
    write(OUT/'transfer-xauc.json',xauc(all_transfer));files.append('transfer-xauc.json')
    result=dict(status='COMPLETE',scope='CONSUMED_DEVELOPMENT',recipe=RECIPE,metrics=summaries,gates=gates,
        balanced_useful_repair=all(gates[r]['balanced']['useful_repair'] for r in ('selection','evaluation')),
        uniform_useful_repair=all(gates[r]['uniform']['useful_repair'] for r in ('selection','evaluation')),
        original_test_activated=False,automatic_successor=False)
    write(OUT/'result.json',result);files.append('result.json');seal('evaluation-seal.json',files);verify()
    print(json.dumps(dict(status=result['status'],metrics=summaries['evaluation'],gates=gates['evaluation'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','extract','fit','predict','evaluate'))
    globals()[parser.parse_args().stage]()
