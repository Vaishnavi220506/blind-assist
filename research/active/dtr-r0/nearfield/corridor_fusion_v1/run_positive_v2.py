"""One nested Development comparison, unchanged head with an optional peak loss."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import json
import numpy as np
import torch
from torch.nn import functional as F
from prepare_public_positive import ROOT, HERE, WORK, read, write, sha
from public_positive import PositiveHead, spatial_features, add_positive, select_threshold
from public_positive_v2 import nested_folds, fitting_weights, negative_peak_loss
from run_public_positive import predict, score_changes
from tolerance_eval import metric, temporal
from research_backend import BackendCandidate, DeviceObservation, select_backend

HOME = WORK/'corridor-public-positive-v2-20260917'
PREP = HOME/'preparation'
EPOCHS, BATCH, SEED = 120, 16, 184017


def placement(out, features, valid, target):
    """The new loss gets an equivalent CPU/CUDA measurement, not a GPU assertion."""
    models = {}
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        models[device] = (PositiveHead().to(device), torch.tensor(features[:16], device=device),
            torch.tensor(valid[:16], device=device), torch.tensor(target[:16], device=device))
    def step(device):
        model,x,v,y = models[device]
        model.zero_grad(set_to_none=True)
        z = model(x,v)
        loss = F.binary_cross_entropy_with_logits(z,y) + negative_peak_loss(z,v,v.any(1)/16)
        loss.backward()
        return loss
    def candidate(device):
        return BackendCandidate('positive-peak-'+device, device, lambda:step(device),
            lambda z:DeviceObservation(z.device.type, torch.cuda.get_device_name(0) if device == 'cuda' else 'host CPU', torch.__version__),
            torch.cuda.synchronize if device == 'cuda' else lambda:None)
    select_backend('model-inference', cpu=candidate('cpu'),
        gpu=candidate('cuda') if 'cuda' in models else None,
        warmups=2, repeats=5, record_path=out/'backend.json')
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return read(out/'backend.json')['selected_device_type']


def reduce_results(meta, logits, thresholds, valid, target, known):
    a = np.array([m['A'] for m in meta], bool)
    y = np.array([m['truth'] for m in meta], bool)
    states = np.array([m['stratum'] for m in meta])
    cohorts = np.array([m['cohort'] for m in meta])
    witness = ((target > 0) & known).any(1)
    p, scores, branch = add_positive(a, logits, valid, thresholds)
    z, _, zb = add_positive(a, logits, valid, 0.)
    assert not (a & ~p).any() and not (a & ~z).any()
    assert np.array_equal(p[~valid.any(1)], a[~valid.any(1)])
    assert np.array_equal(z[~valid.any(1)], a[~valid.any(1)])
    summary = {}
    for c in sorted(set(cohorts)):
        ii = np.flatnonzero(cohorts == c)
        rows = [meta[i] for i in ii]
        yy, ss = y[ii], states[ii]
        clear = ss != 'boundary'
        fam = np.array([m['family'] for m in rows])
        methods = {}
        for name, flags in [('A',a), ('calibrated',p), ('fixed_zero',z), ('oracle_reference',a|witness)]:
            pp = flags[ii]
            t = temporal(rows, ss, pp)
            st = temporal(rows, np.where(yy,'positive','negative'), pp)
            added = flags & ~a
            methods[name] = dict(clear=metric(yy[clear],pp[clear]), coverage=float(clear.mean()),
                strict=metric(yy,pp), boundary=metric(yy[~clear],pp[~clear]),
                clear_changes=score_changes(yy[clear],a[ii][clear],pp[clear]),
                temporal=t, strict_events=dict(total=st['core_events'],detected=st['core_events_detected']),
                families={f:metric(yy[clear & (fam==f)],pp[clear & (fam==f)]) for f in sorted(set(fam))},
                added_clear_episodes={kind:sorted({meta[i]['episode_id'] for i in ii
                    if states[i]!='boundary' and added[i] and bool(y[i])==positive})
                    for kind,positive in [('rescued',True),('false',False)]},
                added_true_frames_with_sampled_witness=int(sum(added[ii]&yy&witness[ii])),
                added_true_frames_without_sampled_witness=int(sum(added[ii]&yy&~witness[ii])))
            events_a = {e['episode']:e for e in methods['A']['temporal']['events']}
            changes = []
            for e in t['events']:
                before = events_a[e['episode']]['first_in_core_delay_s']
                after = e['first_in_core_delay_s']
                if before is not None:
                    assert after is not None and after <= before
                if before != after:
                    changes.append(dict(episode=e['episode'], A_first_s=before, branch_first_s=after))
            methods[name]['first_alert_changes'] = changes
        methods['return_at_zero'] = metric((target[ii]>0)[known[ii]],(logits[ii]>=0)[known[ii]])
        methods['zero_return_frames'] = int(sum(~valid[ii].any(1)))
        methods['supported_clear_A_misses'] = int(sum(clear&yy&~a[ii]&witness[ii]))
        summary[c] = methods
    cases = [dict(**m, score=float(scores[i]), tau=float(thresholds[i]),
        calibrated=bool(p[i]), fixed_zero=bool(z[i]), branch_calibrated=bool(branch[i]),
        branch_zero=bool(zb[i]), sampled_witness=bool(witness[i]), usable_returns=int(valid[i].sum()))
        for i,m in enumerate(meta)]
    return summary,cases


def main(arm):
    out = HOME/arm
    assert not out.exists(), 'Do not overwrite a run; inspect/resume mechanical failures explicitly'
    out.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seal = read(PREP/'data-seal.json')
    for name,h in seal['outputs'].items():
        assert sha(PREP/name)==h,name
    data,lab = np.load(PREP/'public-tokens.npz'),np.load(PREP/'offline-targets.npz')
    meta = read(PREP/'metadata.json')
    features = spatial_features(data['tokens'])
    vn = data['valid'][:,:128]
    tn,kn = lab['target'][:,:128],lab['known'][:,:128]
    assert len(meta)==1344 and not (kn&~vn).any()
    assert list(data['ids'])==[m['id'] for m in meta]
    if arm == 'bce':
        prior = WORK/'corridor-public-positive-20260917/run-v1/backend.json'
        device = read(prior)['selected_device_type']
        write(out/'backend.json',dict(selected_device_type=device, inherited_path=str(prior),
            inherited_sha256=sha(prior), reason='UNCHANGED_HEAD_AND_RETURN_LOSS_PLACEMENT_REUSED'))
    else:
        base_summary = read(HOME/'bce/summary.json')
        assert base_summary['stage2_triggered'], 'Peak arm is unnecessary under the specified condition'
        device = placement(out,features,vn,tn)
    folds = nested_folds(meta)
    sources = {str(p):sha(p) for p in [Path(__file__),HERE/'public_positive_v2.py',
        HERE/'PUBLIC_POSITIVE_V2_PROTOCOL_20260917.md',HERE/'public_positive.py',
        HERE/'run_public_positive.py',HERE/'tolerance_eval.py',ROOT/'tools/research_backend.py']}
    write(out/'freeze.json',dict(arm=arm, peak_lambda=1 if arm=='peak' else 0,
        sources=sources,data_seal_sha256=sha(PREP/'data-seal.json'),folds=folds,
        epochs=EPOCHS,batch=BATCH,seed=SEED,device=device,parameters=1537,
        authority='NESTED_SCENE_GROUPED_CONSUMED_DEVELOPMENT'))
    x,v,y = [torch.tensor(q,device=device) for q in [features,vn,tn]]
    model = optimizer = None
    def fit(fi, seed, tag):
        nonlocal model,optimizer
        fi = np.array(fi)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        fitx = features[fi][vn[fi]]
        model = PositiveHead(fitx.mean(0),fitx.std(0).clip(.01)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        weights,peaks = fitting_weights(meta,fi,vn,tn,kn)
        w,pw = torch.tensor(weights,device=device),torch.tensor(peaks,device=device)
        rng = np.random.default_rng(seed)
        history,orders = [],[]
        tick = time.perf_counter()
        for epoch in range(EPOCHS):
            model.train()
            order = rng.permutation(fi)
            orders.append(order.tolist())
            totals = np.zeros(2)
            for j in range(0,len(fi),BATCH):
                jj = order[j:j+BATCH]
                optimizer.zero_grad(set_to_none=True)
                z = model(x[jj],v[jj])
                bce = (F.binary_cross_entropy_with_logits(z,y[jj],reduction='none')*w[jj]).sum()*len(fi)/len(jj)
                peak = negative_peak_loss(z,v[jj],pw[jj])*len(fi)/len(jj)
                loss = bce + (peak if arm=='peak' else 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                totals += [float(bce.detach())*len(jj)/len(fi),float(peak.detach())*len(jj)/len(fi)]
            history.append(dict(epoch=epoch+1,bce=float(totals[0]),peak=float(totals[1])))
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,epoch=EPOCHS),out/f'{tag}.pt')
        write(out/f'{tag}-fit.json',dict(indices=fi.tolist(),seed=seed,orders=orders,history=history,
            peak_eligible_indices=np.flatnonzero(peaks).tolist(),normalization_fit_only=True))
        print(json.dumps(dict(stage='fit_complete',arm=arm,tag=tag,frames=len(fi),peak_frames=int(sum(peaks>0)),
            loss=history[-1],seconds=time.perf_counter()-tick)),flush=True)
        return model
    report_ix = np.array([i for i,m in enumerate(meta) if m['cohort']!='anchor'])
    oof = np.full((len(meta),128),np.nan,np.float32)
    taus = np.full(len(meta),np.nan)
    fold_id = np.full(len(meta),-1)
    try:
        for f in folds:
            k=f['fold']
            cal_logits = np.full((len(meta),128),np.nan,np.float32)
            for inner in f['inner']:
                j=inner['inner']
                model=fit(inner['fit'],SEED+10*k+j,f'outer{k}-inner{j}')
                ci=np.array(inner['calibration'])
                cal_logits[ci]=predict(model,x,v,ci)
            ci=np.array([i for i in f['fit'] if meta[i]['cohort']!='anchor'])
            assert np.isfinite(cal_logits[ci]).all()
            ca=np.array([meta[i]['A'] for i in ci],bool)
            cy=np.array([meta[i]['truth'] for i in ci],bool)
            cc=np.array([meta[i]['stratum']!='boundary' for i in ci])
            chosen,curve=select_threshold(ca,cal_logits[ci],vn[ci],cy,cc)
            np.savez_compressed(out/f'outer{k}-calibration.npz',indices=ci,logits=cal_logits[ci])
            witness=((tn[ci]>0)&kn[ci]).any(1)
            write(out/f'outer{k}-selection.json',dict(chosen=chosen,curve=curve,
                calibration_frames=len(ci),clear_A_FN=int(sum(cc&cy&~ca)),
                clear_supported_A_FN=int(sum(cc&cy&~ca&witness)),
                supported_miss_groups=sorted({meta[i]['group'] for n,i in enumerate(ci) if cc[n] and cy[n] and not ca[n] and witness[n]})))
            model=fit(f['fit'],SEED+10*k+3,f'outer{k}-final')
            ri=np.array(f['report'])
            oof[ri]=predict(model,x,v,ri)
            taus[ri]=chosen['threshold']
            fold_id[ri]=k
        assert np.isfinite(oof[report_ix]).all() and np.isfinite(taus[report_ix]).all()
        np.savez_compressed(out/'predictions.npz',indices=report_ix,logits=oof[report_ix],thresholds=taus[report_ix],folds=fold_id[report_ix])
        write(out/'prediction-seal.json',dict(outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            authority='ALL_OUTER_PREDICTIONS_SEALED_BEFORE_RESULT_REDUCTION'))
        summary,cases=reduce_results([meta[i] for i in report_ix],oof[report_ix],taus[report_ix],vn[report_ix],tn[report_ix],kn[report_ix])
        triggered=any(s['fixed_zero']['clear_changes']['FP_added']>0 or
            s['fixed_zero']['boundary']['FP']>s['A']['boundary']['FP'] or
            s['calibrated']['clear_changes']['FN_rescued']<s['supported_clear_A_misses'] for s in summary.values())
        write(out/'summary.json',dict(cohorts=summary,stage2_triggered=triggered,arm=arm,
            seconds=time.perf_counter()-start,device=device,authority='CONSUMED_DEVELOPMENT',
            fixed_zero_prespecified=True,no_veto=True,no_capture=True))
        write(out/'cases.json',cases)
        assert all(sha(Path(p))==h for p,h in sources.items())
        write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'),resources='No persistent allocation'))
        print(json.dumps({c:{m:s[m]['clear'] for m in ['A','calibrated','fixed_zero']} for c,s in summary.items()}),flush=True)
    finally:
        del model,optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['bce','peak'],required=True)
    args=parser.parse_args()
    main(args.arm)
