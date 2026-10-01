"""Bounded response/calibration diagnostic. Frozen models, no training or test access.

Dependencies for inference are imported from the v5 source snapshot, not live WIP.
See the pre-result DECISIONS.md beside the output for assumptions and thresholds.
"""
import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'artifacts.local/work/cnh-track-a-v5-20260928/data'
OUT = ROOT / 'artifacts.local/work/cnh-proposal-diagnostics-20260929/observation'
CONDITIONS = [('nominal', 1., 0., 1.), ('gain0.5', .5, 0., 1.),
              ('gain2', 2., 0., 1.), ('offset8', 1., 8., 1.),
              ('offset24', 1., 24., 1.), ('ambient2', 1., 0., 2.),
              ('ambient4', 1., 0., 4.)]
FEATURE_ARMS = ('NONE', 'SIMPLE_SAFE', 'SIMPLE_MIXED', 'PROPOSED_SAFE', 'PROPOSED_MIXED')
CALIB, AUDIT = list(range(8)), list(range(32, 48))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_hash(source):
    """Evaluation-only edits may resume; response/inference edits may not."""
    tree=ast.parse(source)
    names={'perturb','fit_response','setup','collect_calibration','infer_unit'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    payload=ast.dump(ast.Module(body=nodes,type_ignores=[]))+repr((CONDITIONS,FEATURE_ARMS,CALIB,AUDIT))
    return hashlib.sha256(payload.encode()).hexdigest()


def perturb(hist, ambient, condition, unit, config):
    name, gain, offset, noise = condition
    h, a = hist * gain + offset, ambient * gain
    if noise > 1:
        seed = np.random.SeedSequence([20260929, unit, config, int(noise)])
        rng = np.random.default_rng(seed)
        lam = np.broadcast_to(8 * (noise - 1) * ambient[..., None], hist.shape)
        h = h + rng.poisson(lam) - rng.poisson(lam)
        a = ambient * noise
    return h.astype(np.float32), a.astype(np.float32)


def fit_response(reference, target):
    r = np.quantile(reference, [.25, .5, .75], axis=0)
    t = np.quantile(target, [.25, .5, .75], axis=0)
    ri, ti = r[2]-r[0], t[2]-t[0]
    good = ri > 1e-6
    gain = float(np.median(ti[good]/ri[good]))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('nonpositive response estimate')
    return dict(gain=gain, offset=t[1]-gain*r[1], delta=t[1]-r[1])


def setup():
    sys.path.insert(0, str(DATA/'source'))
    import torch
    import cnh_track_a_scale_evaluate as se
    import cnh_track_a_gpu_readout as g
    import cnh_learned_readout as L
    import cnh_track_a_readout as R
    from cnh_learned_memory_fusion import causal_ewma
    se.sensor_module.FAMILY = json.loads((DATA/'request.json').read_text())['family']
    torch.set_num_threads(2)
    return torch, se, g, L, R, causal_ewma


def collect_calibration(se):
    conditions = {c[0]: [] for c in CONDITIONS}
    labels, unit_ids = [], []
    for u in CALIB:
        _, records, _ = se.unit_records(DATA/'geometry', DATA/'sensor', u, -10, 1)
        for rec in records:
            labels.append(rec['labels'][3:])
            unit_ids.extend([u]*9)
            for c in CONDITIONS:
                h, _ = perturb(rec['hist'], rec['ambient'], c, u, rec['config'])
                conditions[c[0]].append(h[3:])
    labels = np.concatenate(labels)
    unit_ids = np.array(unit_ids)
    safe = np.flatnonzero((labels == 0).all(axis=1))
    mixed = np.arange(len(labels))
    n = min(128, len(safe), len(mixed))
    if n < 32:
        raise ValueError('insufficient predeclared calibration pool')
    indices = {k: x[np.linspace(0, len(x)-1, n).astype(int)] for k, x in [('SAFE',safe),('MIXED',mixed)]}
    conditions = {k: np.concatenate(v) for k, v in conditions.items()}
    reference = conditions['nominal'][indices['SAFE']]
    fit = {c: {pool: fit_response(reference, h[idx]) for pool, idx in indices.items()}
           for c, h in conditions.items()}
    np.savez_compressed(OUT/'response_fit.npz', **{f'{c}__{p}__{k}': v for c,d in fit.items() for p,e in d.items() for k,v in e.items()})
    write(OUT/'calibration.json', dict(frames_per_pool=n, available_safe_frames=len(safe),
          threshold_calibration_units=CALIB, threshold_calibration_sequences=8*32,
          pools={p:dict(indices=idx.tolist(), units=np.unique(unit_ids[idx]).tolist(),
                        positive_frames=int((labels[idx]==1).any(axis=1).sum())) for p,idx in indices.items()},
          fits={c:{p:dict(gain=f['gain'], median_offset=float(np.median(f['offset'])),
                         median_delta=float(np.median(f['delta']))) for p,f in d.items()} for c,d in fit.items()}))
    return fit


def infer_unit(u, fit, modules, models, bias):
    torch, se, g, L, R, smooth = modules
    split, records, _ = se.unit_records(DATA/'geometry', DATA/'sensor', u, -10, 1)
    keys = [(c[0], arm) for c in CONDITIONS for arm in FEATURE_ARMS]
    rows = {k: [] for k in ('logits','spread','y','main','w','config','frame')}
    for rec in records:
        noisy = R.noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        # One common geometry transport for all response/calibration arms.
        pos = torch.as_tensor(np.asarray(noisy), dtype=g.D64, device=g.DEV)
        pairs = [(i,j) for i in range(1,12) for j in range(max(0,i-3),i)]
        rel = torch.stack([torch.linalg.inv(pos[i]) @ pos[j] for i,j in pairs])
        mat = g.transport(rel,1).to(g.DT)
        residual, variance = [], []
        for c in CONDITIONS:
            h, a = perturb(rec['hist'], rec['ambient'], c, u, rec['config'])
            for arm in FEATURE_ARMS:
                if arm == 'NONE':
                    hh, aa, bb = h, a, bias
                else:
                    method, pool = arm.split('_')
                    f = fit[c[0]][pool]
                    if method == 'SIMPLE':
                        hh, aa, bb = h, a, bias+f['delta']
                    else:
                        hh, aa, bb = (h-f['offset'])/f['gain'], a/f['gain'], bias
                residual.append(hh-bb)
                variance.append(16*aa[...,None]+np.maximum(bb,0))
        rf = torch.as_tensor(np.asarray(residual), dtype=g.DT, device=g.DEV).reshape(len(keys),12,1024)
        vf = torch.as_tensor(np.asarray(variance), dtype=g.DT, device=g.DEV).reshape(len(keys),12,1024)
        total, var = rf.clone(), vf.clone()
        for k,(i,j) in enumerate(pairs):
            total[:,i] += rf[:,j] @ mat[k].T
            var[:,i] += vf[:,j] @ (mat[k]*mat[k]).T
        z4 = (total/var.clamp_min(1e-9).sqrt()).reshape(len(keys),12,8,8,16)
        z1 = (rf/vf.clamp_min(1e-9).sqrt()).reshape(len(keys),12,8,8,16)
        # Match frozen float16 feature storage round-trip.
        z = torch.stack([z4,z1],dim=2).half().float()
        x = (z.sign()*z.abs().log1p()).reshape(-1,2,8,8,16)
        sup = g.query_weights(torch.as_tensor(np.asarray(rec['tq']), dtype=g.D64, device=g.DEV)) >= .75
        sup = sup.reshape(12,6,8,8,16).repeat(len(keys),1,1,1,1)
        predictions = []
        with torch.no_grad():
            for net in models:
                # Modest batch avoids [batch,query,channel,grid] pooling memory blowup.
                predictions.append(torch.cat([net(x[i:i+48],sup[i:i+48]) for i in range(0,len(x),48)]).cpu().numpy())
        predictions = np.asarray(predictions).reshape(3,len(keys),12,6)
        rows['logits'].append(predictions.mean(axis=0))
        rows['spread'].append(predictions.std(axis=0))
        for k,v in dict(y=rec['labels'],main=rec['main'],w=rec['witness'],
                        config=np.full(12,rec['config']),frame=np.arange(12)).items():
            rows[k].append(v)
    d = {k:np.concatenate(v,axis=1 if k in ('logits','spread') else 0) for k,v in rows.items()}
    d['strata'] = se.strata_of(records)
    d['split'] = np.array(split)
    d['keys'] = np.array([f'{c}|{a}' for c,a in keys])
    d['scores'] = np.stack([smooth(x,d['config'],d['frame'],alpha=.5,window=5) for x in d['logits']])
    np.savez_compressed(OUT/f'unit{u:03d}.npz',**d)
    with np.load(DATA/'predictions'/f'unit{u:03d}.npz') as old:
        diff = float(np.max(np.abs(d['logits'][0]-old['NN'])))
    if diff > 2e-3:
        raise AssertionError(('nominal frozen inference mismatch',u,diff))
    return diff


def generate(limit=None):
    start = time.time()
    modules = setup()
    torch, se, g, L, R, smooth = modules
    if not torch.cuda.is_available():
        raise RuntimeError('GPU expected; inspect backend before CPU fallback')
    models, hashes = [], []
    expected = ['b7ffb180ddde81a6c28e1e7c3922d9653e948a9023ba88fea9d1b8d7a9ee0209',
                '7bafc3201a395f60790dbdfc95b834e5cea43580acd15a61611c972656e33c8d',
                '9e8296de5a0af23ecb752299c4ea1f9f729237b560c3549c6bd472be34e13083']
    for i in range(3):
        p = ROOT/f'artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed{i}.pt'
        hashes.append(sha(p))
        assert hashes[-1] == expected[i]
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(p,map_location=L.DEV,weights_only=True))
        models.append(net.eval())
    frozen_decisions=OUT/'DECISIONS_FROZEN.md'
    if not frozen_decisions.exists():
        frozen_decisions=OUT/'DECISIONS.md'
    receipt = dict(source_manifest_sha256=sha(DATA/'source_manifest.json'),
                   decisions_sha256=sha(frozen_decisions),script_sha256=sha(__file__),
                   generator_sha256=generator_hash(Path(__file__).read_text()),
                   models=hashes,backend=str(L.DEV),device=torch.cuda.get_device_name(),
                   calib=CALIB,audit=AUDIT,conditions=CONDITIONS)
    if (OUT/'receipt.json').exists():
        old=json.loads((OUT/'receipt.json').read_text())
        prior_generator=old.get('generator_sha256')
        if prior_generator is None:
            snapshot=(OUT/'generation_source.py').read_text()
            assert hashlib.sha256(snapshot.encode()).hexdigest()==old['script_sha256']
            prior_generator=generator_hash(snapshot)
        assert prior_generator==receipt['generator_sha256'], 'generation changed; use a fresh output, do not reuse cached units'
        for k in ('source_manifest_sha256','decisions_sha256','models','calib','audit','conditions'):
            assert json.dumps(old[k]) == json.dumps(receipt[k]), ('resume_identity',k)
    # Preserve the original execution receipt/source; resume has its own record.
    if not (OUT/'receipt.json').exists():
        write(OUT/'receipt.json',receipt)
        (OUT/'generation_source.py').write_bytes(Path(__file__).read_bytes())
    else:
        write(OUT/'resume_receipt.json',receipt)
    fit=collect_calibration(se)
    bias=np.load(DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    units=(CALIB+AUDIT)[:limit] if limit else CALIB+AUDIT
    diffs={}
    try:
        for i,u in enumerate(units):
            if not (OUT/f'unit{u:03d}.npz').exists():
                diffs[str(u)]=infer_unit(u,fit,modules,models,bias)
            elapsed=time.time()-start
            write(OUT/'progress.json',dict(completed=i+1,total=len(units),last_unit=u,elapsed_s=elapsed,nominal_max_abs_error=diffs))
            print(f'unit {u}: {i+1}/{len(units)}, {elapsed:.1f}s',flush=True)
        write(OUT/'generation_terminal.json',dict(status='pilot_complete' if limit else 'complete',elapsed_s=time.time()-start,nominal_max_abs_error=diffs))
    except BaseException as exc:
        write(OUT/'generation_terminal.json',dict(status='failed',error=repr(exc),elapsed_s=time.time()-start))
        raise


def paired_ci(a,b,metric):
    av,bv=np.array(a[metric],float),np.array(b[metric],float)
    d=av-bv
    d=d[np.isfinite(d)]
    if not len(d):
        return None
    draws=np.random.default_rng(20260929).integers(0,len(d),(2000,len(d)))
    return dict(mean=float(d.mean()),ci95=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist(),units=len(d))


def evaluate():
    from sklearn.metrics import average_precision_score
    # This evaluator is a separate current file, imports no live geometry.
    import importlib.util
    spec=importlib.util.spec_from_file_location('proposal_v5_evaluator',HERE/'cnh_v5_evaluate.py')
    E=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    units={}
    for u in CALIB+AUDIT:
        with np.load(OUT/f'unit{u:03d}.npz') as f:
            units[u]={k:f[k] for k in f.files}
    keys=units[CALIB[0]]['keys'].tolist()
    spread_thr={g:float(np.quantile(np.concatenate([units[u]['spread'][0][units[u]['frame']>=3][:,ix].ravel() for u in CALIB]),.99)) for g,ix in E.GROUPS}
    result=dict(scope='consumed Development; finite post-observation response assumptions; no training',
                units=dict(calib=CALIB,audit=AUDIT),spread_thresholds=spread_thr,rows=[])
    nominal_thr=json.loads((DATA/'analysis/demo_thresholds.json').read_text())['A2_thresholds']
    result['old_threshold_source']='frozen v5 32-calib-unit 10% thresholds, analysis/demo_thresholds.json'
    for condition in [c[0] for c in CONDITIONS]:
        for arm in ('NONE','THRESHOLD','SIMPLE_SAFE','SIMPLE_MIXED','PROPOSED_SAFE','PROPOSED_MIXED','PROPOSED_SAFE_UNGATED','PROPOSED_MIXED_UNGATED'):
            feature_arm='NONE' if arm=='THRESHOLD' else arm.replace('_UNGATED','')
            j=keys.index(f'{condition}|{feature_arm}')
            ds={u:dict(d,A0=d['scores'][j],A1=d['scores'][j],A2=d['scores'][j]) for u,d in units.items()}
            for g,ix in E.GROUPS:
                cal=E.pack(ds,CALIB,ix)
                aud=E.pack(ds,AUDIT,ix)
                threshold=nominal_thr[g] if arm=='NONE' else E.threshold(E.empty_peaks(cal,'A2'),.1)
                # Pack spread with the same exact temporal/query order.
                sp={u:dict(d,A0=d['spread'][j],A1=d['spread'][j],A2=d['spread'][j]) for u,d in units.items()}
                spread=E.pack(sp,AUDIT,ix)['A2']
                valid=np.isfinite(aud['A2'])
                if arm.startswith('PROPOSED') and not arm.endswith('UNGATED'):
                    valid &= spread <= spread_thr[g]
                alarms=(aud['A2']>=threshold)&valid
                stats=E.alert_stats(aud,alarms)
                aps=[]
                for u in AUDIT:
                    d=ds[u]; y=d['y'][d['main']][:,ix].ravel()
                    aps.append(float(average_precision_score(y,d['A2'][d['main']][:,ix].ravel())))
                per_unit={'timely':[],'false_pair_rate':[]}
                for u in AUDIT:
                    select=aud['unit']==u
                    pp={k:v[select] for k,v in aud.items()}
                    s=E.alert_stats(pp,alarms[select])
                    per_unit['timely'].append(s['all']['timely'])
                    per_unit['false_pair_rate'].append(s['false_pair_rate'])
                row=dict(condition=condition,arm=arm,group=g,threshold=threshold,macro_ap=float(np.mean(aps)),ap_units=len(aps),
                         valid_frames=int(valid.sum()),query_frames=int(valid.size),coverage=float(valid.mean()),
                         unknown_positive_frames=int(((aud['y']==1)&~valid).sum()),stats=stats,per_unit=per_unit)
                result['rows'].append(row)
    lookup={(r['condition'],r['arm'],r['group']):r for r in result['rows']}
    for row in result['rows']:
        c,a,g=row['condition'],row['arm'],row['group']
        ref=lookup['nominal',a,g]
        row['restored']=bool(row['macro_ap']>=ref['macro_ap']-.02 and row['stats']['all']['timely']>=ref['stats']['all']['timely']-.03
             and row['stats']['false_pair_rate']<=min(.15,ref['stats']['false_pair_rate']+.02) and row['coverage']>=.90)
        if a in ('PROPOSED_SAFE','PROPOSED_MIXED'):
            simple=lookup[c,a.replace('PROPOSED','SIMPLE'),g]
            ci=paired_ci(row['per_unit'],simple['per_unit'],'timely')
            fpci=paired_ci(row['per_unit'],simple['per_unit'],'false_pair_rate')
            td=row['stats']['all']['timely']-simple['stats']['all']['timely']
            fd=row['stats']['false_pair_rate']-simple['stats']['false_pair_rate']
            row['comparison_simple']=dict(timely_delta=td,false_pair_delta=fd,unit_timely_ci=ci,unit_false_pair_ci=fpci)
            row['advance_candidate']=bool(c!='nominal' and not simple['restored'] and row['coverage']>=.90 and row['stats']['false_pair_rate']<=.15 and
                ((td>=.05 and ci['ci95'][0]>0 and fd<=.01) or (fd<=-.03 and fpci['ci95'][1]<0 and td>=-.02)))
    # The frozen protocol excludes a gain that only comes from abstention.
    for row in result['rows']:
        if row.get('advance_candidate'):
            c,a,g=row['condition'],row['arm'],row['group']
            ungated=lookup[c,a+'_UNGATED',g]
            simple=lookup[c,a.replace('PROPOSED','SIMPLE'),g]
            td=ungated['stats']['all']['timely']-simple['stats']['all']['timely']
            fd=ungated['stats']['false_pair_rate']-simple['stats']['false_pair_rate']
            row['advance_candidate']=bool(td>0 or fd<0)
            row['abstention_only_excluded']=not row['advance_candidate']
    write(OUT/'results.json',result)
    import csv
    with (OUT/'full_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['condition','arm','group','macro_ap','valid_frames','query_frames','coverage',
                'timely','late','never','near','false_pairs','empty_pairs','false_episodes_per_min','restored','advance_candidate']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in result['rows']:
            s=r['stats']; n=s['all']
            writer.writerow(dict(condition=r['condition'],arm=r['arm'],group=r['group'],macro_ap=r['macro_ap'],
                valid_frames=r['valid_frames'],query_frames=r['query_frames'],coverage=r['coverage'],timely=n['timely_count'],
                late=n['late_count'],never=n['never_count'],near=n['near'],false_pairs=s['false_pairs'],empty_pairs=s['empty_pairs'],
                false_episodes_per_min=s['false_episodes_per_simulated_empty_minute'],restored=r['restored'],advance_candidate=r.get('advance_candidate','')))
    write(OUT/'evaluation_receipt.json',dict(script_sha256=sha(__file__),decisions_sha256=sha(OUT/'DECISIONS.md'),
          results_sha256=sha(OUT/'results.json'),status='complete',rows=len(result['rows'])))
    for r in result['rows']:
        if r['arm'] in ('NONE','THRESHOLD','SIMPLE_MIXED','PROPOSED_MIXED'):
            s=r['stats']; n=s['all']
            print(r['condition'],r['group'],r['arm'],f"AP={r['macro_ap']:.4f} coverage={r['coverage']:.3f} timely={n['timely_count']}/{n['near']} never={n['never_count']} FP={s['false_pairs']}/{s['empty_pairs']} restored={r['restored']} advance={r.get('advance_candidate')}")


def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['generate','evaluate'],required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--out',type=Path,help='Fresh output directory with a prewritten DECISIONS.md')
    a=p.parse_args()
    if a.out is not None:
        OUT=a.out
    OUT.mkdir(parents=True,exist_ok=True)
    generate(a.limit) if a.stage=='generate' else evaluate()


if __name__=='__main__':
    main()
