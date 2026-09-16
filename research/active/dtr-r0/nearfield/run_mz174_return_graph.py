"""Frozen three-fold FIT-only graph correspondence feasibility check."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from mz174_return_graph import ReturnGraph

ROOT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
WORK = ROOT / 'artifacts.local/work/mz174-return-graph-20260916'
OLD = ROOT / 'artifacts.local/work/mz161-dense-task-20260916/run-v1'
PREP = ROOT / 'artifacts.local/work/mz171-return-witness-20260916/preparation'
PARENT = PREP.parent / 'run-v1'
ARMS = ('natural', 'shuffled')
EPOCHS, BATCH, SEED, LIMIT = 120, 16, 174016, 1200.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def admit():
    inputs = {}
    def checked(path, expected=None, *, decode=False):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        inputs[str(path)] = digest
        return read(path) if decode else None
    done = checked(OLD/'completion.json', decode=True)
    assert done['status'] == 'PASS'
    pred = checked(OLD/'prediction-seal.json', done['prediction_seal_sha256'], decode=True)
    model = checked(OLD/'model-seal.json', pred['model_seal_sha256'], decode=True)
    public = checked(OLD/'input-seal.json', pred['input_seal_sha256'], decode=True)
    freeze = checked(OLD/'freeze.json', public['freeze_sha256'], decode=True)
    fitseal = checked(OLD/'fit-label-seal.json', model['fit_label_seal_sha256'], decode=True)
    checked(OLD/'public-tokens.npz', public['tokens_sha256'])
    checked(OLD/'fit-targets.npz', fitseal['targets_sha256'])
    audit = checked(OLD/'input-audit.json', decode=True)
    ids = [r['id'] for r in audit]
    assert len(ids) == len(set(ids)) == 192
    parent_done = checked(PARENT/'completion.json', decode=True)
    assert parent_done['status'] == 'PASS'
    parent_pred = checked(PARENT/'prediction-seal.json', parent_done['prediction_seal_sha256'], decode=True)
    parent_model = checked(PARENT/'model-seal.json', parent_pred['model_seal_sha256'], decode=True)
    parent_freeze = checked(PARENT/'freeze.json', parent_model['freeze_sha256'], decode=True)
    witness = checked(PREP/'fit-witness-seal.json', parent_freeze['inputs'][str(PREP/'fit-witness-seal.json')], decode=True)
    for group in ('inputs', 'sources'):
        for path, digest in witness[group].items():
            checked(path, digest)
    for name, digest in witness['outputs'].items():
        checked(PREP/name, digest)
    assert witness['native_records_decoded'] == 144 and witness['held_native_records_decoded'] == 0
    assert witness['train_ids'] == ids
    with np.load(OLD/'public-tokens.npz', allow_pickle=False) as cache:
        tokens, valid = cache['tokens'], cache['valid']
    with np.load(PREP/'fit-witness-targets.npz', allow_pickle=False) as cache:
        indices, target, known = cache['indices'], cache['target'], cache['known']
    assert target.shape == known.shape == (144, 132)
    assert tokens.shape == (192, 132, 21) and valid.shape == (192, 132)
    assert not known[:, 128:].any() and not (known & ~valid[indices]).any()
    metadata = [freeze['metadata'][ids[i]] for i in indices]
    assert all(v['partition'] == 'fit' for v in metadata)
    assert witness['fit_ids'] == [ids[i] for i in indices]
    folds = np.array([int(v['scene_group'].rsplit('_scene', 1)[1]) for v in metadata])
    assert Counter(folds.tolist()) == {0: 48, 1: 48, 2: 48}
    assert not set(witness['fit_ids']) & {key for key,v in freeze['metadata'].items() if v['partition']=='heldout'}
    return tokens[indices], valid[indices], target, known, folds, metadata, [ids[i] for i in indices], inputs


def metrics(logits, target, known, indices):
    ix = np.asarray(indices)
    p, t, k = logits[ix] >= 0, target[ix] > 0, known[ix]
    tp, fp = int((p & t & k).sum()), int((p & ~t & k).sum())
    fn, tn = int((~p & t & k).sum()), int((~p & ~t & k).sum())
    recall, specificity = tp/(tp+fn), tn/(tn+fp)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, recall=recall, specificity=specificity,
                precision=tp/(tp+fp) if tp+fp else None,
                balanced_accuracy=(recall+specificity)/2,
                unknown_tof_positive=int((p[:,:128] & ~k[:,:128]).sum()),
                radar_outputs_role='UNSUPERVISED_DIAGNOSTIC_NOT_READOUT')


def run(output):
    output = Path(output).resolve()
    assert not output.exists() and output.is_relative_to(WORK.resolve())
    output.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = read(WORK/'backend.json')
    preflight = read(WORK/'preflight.json')
    assert preflight['status'] == 'PASS'
    device = backend['selected_device_type']
    assert device in ('cpu', 'cuda')
    if device == 'cuda':
        assert torch.cuda.is_available() and preflight['cuda_device'] == torch.cuda.get_device_name(0)
    def within():
        if time.perf_counter()-start > LIMIT:
            raise TimeoutError('Frozen1200s allocation reached; no automatic restart')
    tokens, valid, target, known, folds, metadata, ids, inputs = admit()
    inputs.update({str(WORK/n):sha(WORK/n) for n in ('backend.json', 'preflight.json', 'tests.log')})
    sources = {str(CODE/n):sha(CODE/n) for n in ('MZ174_PROTOCOL_20260916.md',
        'mz174_return_graph.py', 'test_mz174_return_graph.py', 'run_mz174_return_graph.py')}
    sources[str(ROOT/'tools/research_backend.py')] = sha(ROOT/'tools/research_backend.py')
    orders = {}
    for fold in range(3):
        rng = np.random.default_rng(SEED+fold)
        train = np.flatnonzero(folds != fold)
        orders[str(fold)] = [rng.permutation(train).tolist() for _ in range(EPOCHS)]
    write(output/'orders.json', orders)
    write(output/'freeze.json', dict(sources=sources, inputs=inputs, ids=ids,
        metadata=metadata, folds=folds.tolist(), epochs=EPOCHS, batch=BATCH, seed=SEED,
        optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001),
        orders_sha256=sha(output/'orders.json'), budget_s=LIMIT, backend=device,
        input_authority='ONLY_PUBLIC_TOKENS_AND_VALIDITY',
        label_authority='FIT144_KNOWN_TOF_TARGETS_ONLY;FOLD_LABELS_EXCLUDED_FROM_OWN_FIT',
        original_dev_test_access=False, original_held48_label_access=False))
    for source in sources:
        dest = output/'source-snapshot'/Path(source).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    print('MZ174 FROZEN', flush=True)
    x = torch.from_numpy(tokens).to(device)
    mask = torch.from_numpy(valid).to(device)
    y = torch.from_numpy(target).to(device)
    oof = {arm:np.full((144,132), np.nan, np.float32) for arm in ARMS}
    train_metrics, timings, model_files, initials, assignments = {}, {}, {}, {}, []
    model = optimizer = None
    try:
        for fold in range(3):
            ti = np.flatnonzero(folds != fold)
            vi = np.flatnonzero(folds == fold)
            positive = known[ti] & (target[ti]>0)
            negative = known[ti] & ~(target[ti]>0)
            weight = np.zeros((144,132), np.float32)
            weight[ti] = .5*positive/positive.sum() + .5*negative/negative.sum()
            assert np.all(weight[vi] == 0) and not np.any(weight[:,128:])
            w = torch.from_numpy(weight).to(device)
            assignments.append(dict(fold=fold, fit_ids=[ids[i] for i in ti],
                validation_ids=[ids[i] for i in vi], known_positive=int(positive.sum()),
                known_negative=int(negative.sum()), positive_mass=float(weight[ti][positive].sum()),
                negative_mass=float(weight[ti][negative].sum())))
            for arm in ARMS:
                torch.manual_seed(SEED+fold)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED+fold)
                model = ReturnGraph(arm=arm).to(device)
                tag = f'fold{fold}-{arm}'
                initials[tag] = state_hash(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                tick = time.perf_counter()
                history = []
                for epoch, order in enumerate(orders[str(fold)], 1):
                    model.train()
                    losses = []
                    for j in range(0,len(order),BATCH):
                        ii = order[j:j+BATCH]
                        optimizer.zero_grad(set_to_none=True)
                        out = model(x[ii], mask[ii])['logits']
                        loss = (F.binary_cross_entropy_with_logits(out,y[ii],reduction='none')*w[ii]).sum()*len(ti)/len(ii)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    within()
                    history.append(dict(epoch=epoch, loss=float(np.mean(losses))))
                    if epoch % 30 == 0:
                        write(output/(tag+'-training.json'), history)
                        print(f'{tag} epoch={epoch} loss={history[-1]["loss"]:.5f}', flush=True)
                checkpoint = output/(tag+'-model.pt')
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=EPOCHS,torch_rng=torch.get_rng_state()), checkpoint)
                model_files[checkpoint.name] = sha(checkpoint)
                model.eval()
                prediction = np.empty((144,132),np.float32)
                with torch.no_grad():
                    for j in range(0,144,BATCH):
                        ii = list(range(j,min(j+BATCH,144)))
                        prediction[ii] = model(x[ii],mask[ii])['logits'].cpu().numpy()
                assert np.isfinite(prediction).all() and np.all(prediction[~valid] == -30)
                np.save(output/(tag+'-logits.npy'), prediction)
                oof[arm][vi] = prediction[vi]
                # Fitting metrics are diagnostics; never select checkpoint or schedule.
                train_metrics[tag] = metrics(prediction,target,known,ti)
                timings[tag] = time.perf_counter()-tick
                del model, optimizer
                model = optimizer = None
                if device == 'cuda': torch.cuda.empty_cache()
            assert initials[f'fold{fold}-natural'] == initials[f'fold{fold}-shuffled']
        for arm in ARMS:
            assert np.isfinite(oof[arm]).all()
            np.save(output/(arm+'-oof-logits.npy'),oof[arm])
        write(output/'fold-assignments.json', assignments)
        write(output/'prediction-seal.json',dict(freeze_sha256=sha(output/'freeze.json'),
            models=model_files, initial_states=initials,
            outputs={p.name:sha(p) for p in output.glob('*-logits.npy')},
            assignments_sha256=sha(output/'fold-assignments.json'),
            authority='ALL_OOF_PREDICTIONS_SEALED_BEFORE_VALIDATION_REDUCTION'))
        print('MZ174 OOF SEALED', flush=True)
        overall = {a:metrics(oof[a],target,known,np.arange(144)) for a in ARMS}
        by_fold = {str(f):{a:metrics(oof[a],target,known,np.flatnonzero(folds==f)) for a in ARMS} for f in range(3)}
        by_family = {f:{a:metrics(oof[a],target,known,[i for i,m in enumerate(metadata) if m['family']==f]) for a in ARMS}
                     for f in sorted({m['family'] for m in metadata})}
        natural, shuffled = overall['natural'], overall['shuffled']
        checks = dict(adequate_discrimination=all(natural[k]>=.75 for k in ('recall','specificity','balanced_accuracy')),
            balanced_accuracy_gain_at_least_5pp=natural['balanced_accuracy']>=shuffled['balanced_accuracy']+.05,
            every_fold_recall_noninferior=all(by_fold[str(f)]['natural']['recall']>=by_fold[str(f)]['shuffled']['recall'] for f in range(3)),
            every_fold_specificity_noninferior=all(by_fold[str(f)]['natural']['specificity']>=by_fold[str(f)]['shuffled']['specificity'] for f in range(3)))
        passed = all(checks.values())
        summary = dict(overall=overall, folds=by_fold, families=by_family,fit_metrics=train_metrics,
            checks=checks, gate_passed=passed, timings=timings, seconds=time.perf_counter()-start,
            known_slot_counts=dict(positive=int((known&(target>0)).sum()),negative=int((known&~(target>0)).sum())),
            native_return_witness_frames=int((known&(target>0)).any(1).sum()),
            usable_tof=int(valid[:,:128].sum()), usable_radar=int(valid[:,128:].sum()),
            decision='MZ174_RELATIONAL_COMPONENT_READY_FOR_HELD_CHECK' if passed else 'MZ174_RELATIONAL_COMPONENT_FIT_GATE_NOT_MET',
            warning_effect='NOT_EVALUATED_IN_SLOT_COMPONENT_STAGE', retained_baseline='MZ129',
            original_held48_label_access=False, original_dev_test_access=False, default_changed=False)
        write(output/'summary.json',summary)
        for path,digest in (sources|inputs).items():assert sha(path)==digest,path
        within()
        write(output/'completion.json',dict(status='PASS',decision=summary['decision'],
            summary_sha256=sha(output/'summary.json'),prediction_seal_sha256=sha(output/'prediction-seal.json'),
            seconds=time.perf_counter()-start, resources='Process-local model released; no active allocation'))
        print(json.dumps(dict(decision=summary['decision'],overall=overall,checks=checks)),flush=True)
    finally:
        del model,optimizer
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=WORK/'run-v1')
    args=parser.parse_args()
    try:
        run(args.output)
    except Exception as exc:
        if args.output.exists():
            write(args.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),resume='NO_AUTOMATIC_RESTART'))
        raise
