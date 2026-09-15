"""MZ142: original checkpoints, identical MZ141 recipe except fixed BN stats."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from mz140_depthor import make_input, predict
from mz141_depthor_training import loss_contribution
from mz142_fixed_bn import load_fixed_bn, training_mode, bn_digest
import run_mz141_depthor_training as old
from run_mz139_surface_fit import read, selected_jsonl, local_dependencies, CODE
from run_mz107_four_sensor import ROOT, sha, write
from evaluate_mz140_depthor import evaluate_frame, rasterize_native_bounds
from evaluate_mz141_depthor import extra_metrics, _pooled, _admission, _surface_extra_gain

PREVIOUS = ROOT/'artifacts.local/work/mz141-depthor-training-20260915/run-v1'


def forward(model, data):
    raw = model(data)[1]
    value = raw.detach()
    finite = torch.isfinite(value)
    n = value.numel()
    low, high = int((value < .001).sum()), int((value > 10.).sum())
    stats = dict(pixels=n, finite_pixels=int(finite.sum()),
        minimum_m=float(value.min()), maximum_m=float(value.max()),
        below_minimum_pixels=low, above_maximum_pixels=high,
        outside_fraction=(low+high)/n, shape=list(value.shape),
        authority='NETWORK_FINAL_BEFORE_RESAMPLING_OR_CLIPPING')
    if stats['finite_pixels'] != n:
        raise FloatingPointError(f'Nonfinite network final: {stats}')
    return raw, stats


def setup(out):
    assert read(PREVIOUS/'completion.json')['status'] == 'PASS'
    rows, metadata, images, labels, cached = old.prepared(out)
    frozen = read(out/'freeze.json')
    previous = read(PREVIOUS/'freeze.json')
    assert metadata == previous['metadata'] and frozen['inputs'] == previous['inputs']
    sources = local_dependencies(__file__)
    for name in ('test_mz142_fixed_bn.py',):
        sources[str((CODE/name).resolve())] = sha(CODE/name)
    for name, digest in sources.items():
        path = Path(name)
        relative = path.relative_to(ROOT.resolve())
        dest = out/'source-snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    frozen['sources'].update(sources)
    shutil.copyfile(CODE/'MZ142_PROTOCOL_20260915.md', out/'protocol-before-fit.md')
    frozen.update(protocol_sha256=sha(out/'protocol-before-fit.md'), experiment='MZ142',
        intervention='BN_EVAL_PRETRAINED_RUNNING_STATS_FIXED_AFFINE_TRAINABLE',
        previous_run_freeze_sha256=sha(PREVIOUS/'freeze.json'))
    groups=sorted({m['scene_group'] for m in metadata.values() if m['partition']=='fit'})
    monitor_ids=[sorted(key for key,m in metadata.items() if m['scene_group']==group)[2] for group in groups]
    assert len(monitor_ids)==12
    frozen['monitor_ids']=monitor_ids
    write(out/'freeze.json', frozen)
    native={e['id']:e for e in selected_jsonl(old.CAP/'evaluator.jsonl',set(monitor_ids))}
    return rows, metadata, images, labels, cached, native


def smoke(out, rows, images, labels):
    old.seed()
    model,info=load_fixed_bn(old.BASE/'upstream',old.BASE/'depthor-zju-small.pt')
    initial_bn=bn_digest(model)
    row=next(r for r in rows if r['id'] in labels and 'near_rod_farwall' in r['id'])
    data,_=make_input(row,images[row['id']])
    raw,stats=forward(model,data)
    depth=F.interpolate(raw,size=(360,640),mode='bilinear',align_corners=False)[0,0]
    loss=loss_contribution(depth,labels[row['id']],'surface',labels[row['id']]['audit']['known_pixels'],1)
    loss.backward()
    groups={}
    for name in ('img_encoder','SpEncoder','decoder','depth_head','conv_out','refine','up','align_mde'):
        params=list(getattr(model,name).parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        g=sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        assert g>0,name
        groups[name]=g
    affine=[p for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for p in m.parameters(recurse=False)]
    assert all(p.requires_grad for p in affine)
    affine_gradient=sum(float(p.grad.abs().sum()) for p in affine if p.grad is not None)
    assert affine_gradient>0 and initial_bn==bn_digest(model)
    assert all(p.grad is None for p in model.depth_anything.parameters())
    model.eval()
    with torch.no_grad():
        raw,_=forward(model,data)
        expected=predict(model,data)
        actual=F.interpolate(raw.clamp(.001,10.),size=(360,640),mode='bilinear',align_corners=False)[0,0]
        difference=float((expected-actual).abs().max())
        assert difference==0
    write(out/'gradient-smoke.json',dict(status='PASS',info=info,bn_unchanged=True,
        bn_affine_gradient_l1=affine_gradient,group_gradients_l1=groups,
        inference_equivalence_max_m=difference,raw=stats,no_optimizer_updates=True))
    del model,raw,depth,loss,data,expected,actual
    old.release()


def monitor(out, arm, epoch, model, rows, metadata, images, cached, native):
    before=old.module_digest(model,'')
    records=[]
    # Eval must not perturb future stochastic-depth draws in the paired training.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.eval()
        for row in rows:
            if row['id'] not in native:
                continue
            data,_=make_input(row,images[row['id']])
            with torch.no_grad():
                raw,stats=forward(model,data)
                depth=F.interpolate(raw.clamp(.001,10.),size=(360,640),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            ref=rasterize_native_bounds(row,native[row['id']],cached[row['id']])
            metric=evaluate_frame(row,native[row['id']],depth,cached[row['id']])
            metric.update(extra_metrics(depth,ref))
            records.append(dict(id=row['id'],family=metadata[row['id']]['family'],metrics=metric,raw=stats))
    training_mode(model)
    assert before==old.module_digest(model,'')
    families={f:_pooled([r['metrics'] for r in records if r['family']==f]) for f in sorted({r['family'] for r in records})}
    write(out/f'monitor/{arm}-epoch{epoch}.json',dict(epoch=epoch,frames=12,records=records,
        families=families,model_state_unchanged=True,torch_cuda_rng_preserved=True,selection_used=False))
    print(json.dumps(dict(stage='fit-monitor',arm=arm,epoch=epoch,
        target_mae_m={f:round(v['target_mae_m'],3) for f,v in families.items()})),flush=True)


def train(out, arm, rows, metadata, images, labels, cached, native):
    old.seed()
    model,info=load_fixed_bn(old.BASE/'upstream',old.BASE/'depthor-zju-small.pt')
    initial=old.module_digest(model,'');backbone=old.module_digest(model);initial_bn=bn_digest(model)
    affine={name:p.detach().clone() for mn,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
        for pn,p in m.named_parameters(recurse=False) for name in [mn+'.'+pn]}
    optimizer=torch.optim.Adam(model.get_lr_params(),lr=1e-4)
    fit_rows=[r for r in rows if r['id'] in labels]
    schedule=np.random.default_rng(old.SEED)
    logs=[];step=0;started=time.perf_counter()
    with (out/f'{arm}-training-raw.jsonl').open('w',encoding='utf-8') as stream:
        for epoch in range(old.EPOCHS):
            training_mode(model)
            order=schedule.permutation(len(fit_rows)).tolist()
            losses=[]
            for offset in range(0,len(order),old.ACCUM):
                batch=[fit_rows[j] for j in order[offset:offset+old.ACCUM]]
                denominator=sum(labels[r['id']]['audit']['known_pixels'] for r in batch)
                optimizer.zero_grad(set_to_none=True);batch_loss=0.
                for row in batch:
                    data,_=make_input(row,images[row['id']])
                    raw,stats=forward(model,data)
                    stream.write(json.dumps(dict(epoch=epoch+1,step=step+1,id=row['id'],**stats))+'\n')
                    depth=F.interpolate(raw,size=(360,640),mode='bilinear',align_corners=False)[0,0]
                    loss=loss_contribution(depth,labels[row['id']],arm,denominator,len(batch))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();batch_loss+=float(loss.detach())
                for p in model.get_lr_params():
                    if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
                optimizer.step();step+=1;losses.append(batch_loss)
                if step%12==0:
                    progress=dict(stage='train',arm=arm,epoch=epoch+1,step=step,total_steps=288,loss=batch_loss,seconds=time.perf_counter()-started)
                    write(out/'progress.json',progress);stream.flush();print(json.dumps(progress),flush=True)
            assert initial_bn==bn_digest(model)
            monitor(out,arm,epoch+1,model,rows,metadata,images,cached,native)
            record=dict(epoch=epoch+1,steps=step,loss_mean=float(np.mean(losses)),fit_order=[fit_rows[j]['id'] for j in order],
                bn_sha256=initial_bn,seconds=time.perf_counter()-started)
            logs.append(record);write(out/f'{arm}-training.json',logs)
            checkpoint=out/f'checkpoints/{arm}-epoch{epoch+1}.pt';checkpoint.parent.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,step=step,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_schedule_rng=schedule.bit_generator.state,
                freeze_sha256=sha(out/'freeze.json')),checkpoint)
    assert step==288 and backbone==old.module_digest(model) and initial_bn==bn_digest(model)
    named=dict(model.named_parameters())
    changed=sum(not torch.equal(named[name].detach(),value) for name,value in affine.items())
    assert changed>0
    previous=read(PREVIOUS/f'{arm}-fit-receipt.json')
    assert initial==previous['initial_state_sha256']
    assert [e['fit_order'] for e in logs]==[e['fit_order'] for e in read(PREVIOUS/f'{arm}-training.json')]
    write(out/f'{arm}-fit-receipt.json',dict(status='PASS',info=info,initial_state_sha256=initial,
        final_state_sha256=old.module_digest(model,''),backbone_unchanged=True,bn_unchanged=True,bn_sha256=initial_bn,
        affine_tensors_changed=changed,optimizer_steps=step,elapsed_seconds_including_monitoring=time.perf_counter()-started,
        final_checkpoint_sha256=sha(checkpoint),sample_order_identical_to_mz141=True))
    model.eval();predictions={};raw_records=[];timings=[]
    for i,row in enumerate(rows):
        data,_=make_input(row,images[row['id']]);torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            raw,stats=forward(model,data)
            depth=F.interpolate(raw.clamp(.001,10.),size=(360,640),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        path=out/f'predictions/{arm}/{i:03d}.npy';path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,depth,allow_pickle=False);predictions[row['id']]=dict(path=str(path),sha256=sha(path))
        raw_records.append(dict(id=row['id'],**stats))
    assert initial_bn==bn_digest(model)
    write(out/f'{arm}-predictions.json',predictions);write(out/f'{arm}-prediction-raw.json',raw_records);write(out/f'{arm}-timings.json',timings)
    del model,optimizer,affine,named,raw,depth,loss,data
    old.release()
    return predictions


def run(out):
    out=out.resolve();assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True);torch.set_num_threads(4);assert torch.cuda.is_available()
    (out/'monitor').mkdir(exist_ok=True)
    try:
        rows,metadata,images,labels,cached,native=setup(out)
        sys.path.insert(0,str(ROOT/'tools'))
        from research_backend import runtime_capabilities
        write(out/'runtime.json',dict(capabilities=runtime_capabilities(),device='cuda',placement='GPU_FIRST_REUSE_VERIFIED_MZ141_BACKEND'))
        smoke(out,rows,images,labels)
        frozen_predictions=read(PREVIOUS/'frozen-predictions.json')
        assert set(frozen_predictions)=={r['id'] for r in rows}
        for r in frozen_predictions.values():assert sha(Path(r['path']))==r['sha256']
        predictions={'frozen':frozen_predictions}
        for arm in ('pixel','surface'):predictions[arm]=train(out,arm,rows,metadata,images,labels,cached,native)
        assert read(out/'pixel-fit-receipt.json')['initial_state_sha256']==read(out/'surface-fit-receipt.json')['initial_state_sha256']
        write(out/'prediction-seal.json',dict(arms=predictions,freeze_sha256=sha(out/'freeze.json'),
            authority='NEW_FINAL_PREDICTIONS_SEALED_BEFORE_HELDOUT_EVALUATOR_PARSE; FROZEN_BASELINE_REUSED_CONSUMED'))
        summary=old.evaluate(out,rows,metadata,cached,predictions)
        fit=summary['partitions']['fit']['arms']
        summary['fit_gates']={a:_admission(fit[a]['families'],fit['frozen']['families']) for a in ('pixel','surface')}
        eligible=[a for a in ('pixel','surface') if summary['gates'][a]['eligible'] and summary['fit_gates'][a]['eligible']]
        selected=('surface' if 'surface' in eligible and ('pixel' not in eligible or summary['surface_extra_gain']['passed']) else 'pixel' if 'pixel' in eligible else None)
        summary.update(experiment='MZ142_FIXED_PRETRAINED_BN_STATS',selected_arm=selected,geometry_admitted=selected is not None,
            decision='GEOMETRY_ADMITTED_'+selected.upper()+'_ALERT_NOT_YET_RUN' if selected else 'STOP_FIXED_BN_FINETUNE_NO_JOINT_FIT_HELDOUT_GEOMETRY_KEEP_MZ129',
            previous_summary_sha256=sha(PREVIOUS/'summary.json'),heldout_fresh=False)
        write(out/'summary.json',summary)
        freeze=read(out/'freeze.json')
        for label in ('inputs','sources'):assert freeze[label]=={p:sha(Path(p)) for p in freeze[label]}
        write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'),sources_inputs_unchanged=True,
            dev_accessed=False,original_test_accessed=False,alert_evaluated=False,resources='PROCESS_LOCAL_GPU_RELEASE_ON_EXIT'))
        print(json.dumps(dict(decision=summary['decision'],selected_arm=selected)),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),progress=read(out/'progress.json') if (out/'progress.json').exists() else None,
            automatic_restart=False,preserve_existing_steps_and_checkpoints=True))
        raise
    finally:old.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
