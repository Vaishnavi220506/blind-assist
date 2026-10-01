"""Frozen CVR+PARTIAL max-rank fusion on freshly generated corridor units.

No training. Reuses the corridor generator unchanged (new unit ids only), the
CVR v2 projector and fold models, and the LOFO / equal-count A2 fold models.
Plan: artifacts.local/work/cnh-fresh-fusion-20260929/FRESH_FUSION_PLAN.md
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'cnh-fresh-fusion-20260929'
PLAN = OUT/'FRESH_FUSION_PLAN.md'
SPLITS = {'calib': list(range(40000, 40024)), 'evaluation': list(range(50000, 50048))}
USED = set(range(0, 18)) | set(range(1000, 1096)) | set(range(2000, 2024)) | set(range(3000, 3048))
FAMILIES = ('boundary', 'mixed_surface', 'sidewall', 'general')
GROUPS = ('HEAD', 'BODY')
FRAMES = np.arange(11, 16)
ARMS = ('LOFO', 'CVR', 'MOTION8_PARTIAL', 'FUSION', 'EQUAL_COUNT')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    for attempt in range(7):
        try:
            tmp.replace(path); return
        except PermissionError:
            if attempt == 6: raise
            time.sleep(.02*2**attempt)


def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def generate():
    assert PLAN.exists(), 'freeze plan first'
    assert not USED & set(SPLITS['calib']+SPLITS['evaluation'])
    import cnh_corridor_diagnostic as D
    D.OUT = OUT
    D.SPLITS = SPLITS
    D.generate()


def materialize():
    """CVR v2 voxel features for frames 11-15, identical projector and ordering."""
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    from cnh_proposal_attribution_scenes import make_scenes
    torch.set_num_threads(2)
    projector = BatchedProjector()
    for split, units in SPLITS.items():
        folder = OUT/'data'/split
        if (folder/'metadata.npz').exists(): continue
        folder.mkdir(parents=True, exist_ok=True)
        n = len(units)*22*len(FRAMES)
        feats = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float16, shape=(n, 3, *SHAPE))
        meta = {k: [] for k in ['unit', 'config', 'frame', 'family', 'labels']}
        for ui, u in enumerate(units):
            d = read(OUT/'features'/split/f'unit{u}.npz')
            scenes = make_scenes(u)
            for config in range(22):
                ids = np.flatnonzero(d['scene'] == config)
                assert np.array_equal(d['frame'][ids], np.arange(16))
                sensor, travel, noisy = motion_metadata(u, config)
                # Motion replay must match the generator's poses exactly.
                assert np.allclose(sensor, scenes[config]['poses'], atol=1e-9)
                assert np.allclose(travel, scenes[config]['travel'], atol=1e-9)
                offset = (ui*22+config)*len(FRAMES)
                for fi, f in enumerate(FRAMES):
                    voxel = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                    assert bool(torch.isfinite(voxel).all())
                    feats[offset+fi] = voxel.cpu().numpy().astype(np.float16)
                meta['unit'] += [u]*len(FRAMES); meta['config'] += [config]*len(FRAMES)
                meta['frame'] += FRAMES.tolist(); meta['family'] += [str(d['family'][config])]*len(FRAMES)
                meta['labels'] += d['labels'][ids[FRAMES], 2:4].tolist()
            print('materialize', split, u, flush=True)
        feats.flush(); del feats
        np.savez_compressed(folder/'metadata.npz', **{k: np.asarray(v) for k, v in meta.items()})


def infer():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import cnh_corridor_retrain as T
    import cnh_cvr_v2_train as V
    T.frozen_module()
    from cnh_learned_memory_fusion import causal_ewma
    dest = OUT/'predictions'; dest.mkdir(exist_ok=True)
    hashes = {}
    # A2 family models: LOFO and equal-count, same inference as cnh_corridor_lofo.infer_all.
    for arm, base in [('LOFO', WORK/'cnh-corridor-lofo-20260929'), ('EQUAL_COUNT', WORK/'cnh-corridor-equal-count-20260929')]:
        for fold in FAMILIES:
            path = dest/f'{arm}_{fold}.npz'
            if path.exists(): continue
            files = [base/fold/f'model_seed{s}.pt' for s in range(3)]
            for f in files: hashes[str(f.relative_to(ROOT))] = sha(f)
            models = T.load_models(files, 'cuda')
            out = {}
            for split, units in SPLITS.items():
                for u in units:
                    d = read(OUT/'features'/split/f'unit{u}.npz')
                    nn = T.predict(models, d['z4'], d['z1'], d['support'], 48)
                    sm = causal_ewma(nn, d['scene'], d['frame'], alpha=.5, window=5)
                    out[str(u)] = sm.reshape(22, 16, 6)[:, -1, [2, 3]]
            np.savez_compressed(path, **out)
            del models; torch.cuda.empty_cache(); print('infer', arm, fold, flush=True)
    # CVR v2 fold models: frames 11-15 weighted 1,2,4,8,16 as cnh_cvr_v2_train.infer.
    for fold in FAMILIES:
        path = dest/f'CVR_{fold}.npz'
        if path.exists(): continue
        files = [WORK/'cnh-cvr-v2-20260929/models/CVR'/fold/f'model_seed{s}.pt' for s in range(3)]
        for f in files: hashes[str(f.relative_to(ROOT))] = sha(f)
        nets = []
        for f in files:
            net = V.model_for('CVR').cuda(); net.load_state_dict(torch.load(f, map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for split in SPLITS:
            data = V.Data(OUT, split); m = data.meta
            logits = []
            with torch.no_grad():
                for i in range(0, len(data.x), 64):
                    args, _ = data.batch(np.arange(i, min(i+64, len(data.x))))
                    logits.append(torch.stack([n(*args) for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(logits)
            for u in SPLITS[split]:
                scores = []
                for config in range(22):
                    loc = np.flatnonzero((m['unit'] == u) & (m['config'] == config))
                    loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], FRAMES)
                    scores.append((raw[loc].astype(np.float64)*np.array([1, 2, 4, 8, 16])[:, None]).sum(0)/31)
                out[str(u)] = np.asarray(scores)
            del data
        np.savez_compressed(path, **out)
        del nets; torch.cuda.empty_cache(); print('infer CVR', fold, flush=True)
    prior = json.loads((OUT/'model_hashes.json').read_text()) if (OUT/'model_hashes.json').exists() else {}
    save(OUT/'model_hashes.json', {**prior, **hashes})


def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks, threshold
    from cnh_corridor_statistics import paired_cluster_ber
    from sklearn.metrics import roc_auc_score
    assert PLAN.exists() and not (OUT/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    feats = {s: {u: read(OUT/'features'/s/f'unit{u}.npz') for u in us} for s, us in SPLITS.items()}
    pred = {f'{a}_{f}': read(OUT/'predictions'/f'{a}_{f}.npz') for a in ('LOFO', 'EQUAL_COUNT', 'CVR') for f in FAMILIES}

    def rows(split, fold, keep_family):
        out = []
        for u in SPLITS[split]:
            d = feats[split][u]
            labels = d['labels'].reshape(22, 16, 6)[:, -1, [2, 3]]
            for c in range(22):
                fam = str(d['family'][c])
                if not keep_family(fam): continue
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, config=c, group=g, family=fam, label=int(labels[c, q]),
                        target_group=int(d['group'][c]), margin=float(d['margin'][c]),
                        scores=dict(LOFO=float(pred[f'LOFO_{fold}'][str(u)][c, q]),
                                    CVR=float(pred[f'CVR_{fold}'][str(u)][c, q]),
                                    MOTION8_PARTIAL=float(d['motion'][c, 2, q]),
                                    EQUAL_COUNT=float(pred[f'EQUAL_COUNT_{fold}'][str(u)][c, q]))))
        return out

    ledger, calibration, comparisons, depth, aucs = [], [], [], [], []
    for fold in FAMILIES:
        cal_all = rows('calib', fold, lambda f: f != fold)
        ev_all = rows('evaluation', fold, lambda f: f == fold)
        for q, g in enumerate(GROUPS):
            cal = [r for r in cal_all if r['group'] == g and r['label'] == 0]
            ev = [r for r in ev_all if r['group'] == g]
            n = len(cal); cr, er = {}, {}
            for arm in ('LOFO', 'CVR', 'MOTION8_PARTIAL', 'EQUAL_COUNT'):
                ref = [r['scores'][arm] for r in cal]
                cr[arm] = ranks(ref, ref); er[arm] = ranks(ref, [r['scores'][arm] for r in ev])
            cr['FUSION'] = np.maximum(cr['CVR'], cr['MOTION8_PARTIAL'])
            er['FUSION'] = np.maximum(er['CVR'], er['MOTION8_PARTIAL'])
            p = {}
            for arm in ARMS:
                k = threshold(cr[arm], n); fp = int((cr[arm] >= k).sum())
                assert 10*fp <= n and (k == 0 or 10*int((cr[arm] >= k-1).sum()) > n)
                calibration.append(dict(family=fold, group=g, arm=arm, rank_threshold=k, negative=n, fp=fp, fpr=fp/n))
                p[arm] = er[arm] >= k
            y = np.array([r['label'] for r in ev]); units = np.array([r['unit'] for r in ev])
            assert len(np.unique(units)) == 48
            m = {a: metrics(y, p[a]) for a in ARMS}
            vs_lofo = paired_cluster_ber(y, p['LOFO'], p['FUSION'], units)
            vs_cvr = paired_cluster_ber(y, p['CVR'], p['FUSION'], units)
            cvr_vs_lofo = paired_cluster_ber(y, p['LOFO'], p['CVR'], units)
            if fold in ('sidewall', 'mixed_surface'):
                passed = bool(vs_lofo['point'] <= -.05 and vs_lofo['ci'] is not None and vs_lofo['ci'][1] < 0)
            else:
                passed = bool(vs_lofo['point'] <= .02 and vs_cvr['point'] <= .02)
            comparisons.append(dict(family=fold, group=g, metrics=m, fusion_vs_lofo=vs_lofo, fusion_vs_cvr=vs_cvr,
                                    cvr_vs_lofo=cvr_vs_lofo, passed=passed))
            for arm in ARMS:
                s = er[arm].astype(float)
                aucs.append(dict(family=fold, group=g, arm=arm, auc_calib_rank=float(roc_auc_score(y, s))))
            for i, r in enumerate(ev):
                r['predictions'] = {a: int(p[a][i]) for a in ARMS}
                r['rank_numerators'] = {a: int(er[a][i]) for a in ARMS}
                r['rank_denominator'] = n
                r['role'] = 'positive_intrusion' if r['label'] else ('target_group_outside' if r['target_group'] == q else 'other_height_negative')
                r['band'] = '<=5cm' if abs(r['margin']) <= .05 else ('5-15cm' if abs(r['margin']) <= .15 else '>15cm')
                ledger.append(r)
            for role in ('positive_intrusion', 'target_group_outside', 'other_height_negative'):
                for band in ('<=5cm', '5-15cm', '>15cm', 'all'):
                    rr = [r for r in ev if r['role'] == role and (band == 'all' or r['band'] == band)]
                    for arm in ARMS:
                        depth.append(dict(family=fold, group=g, role=role, band=band, arm=arm,
                                          **metrics([r['label'] for r in rr], [r['predictions'][arm] for r in rr])))
    assert len(ledger) == len({(r['unit'], r['config'], r['group']) for r in ledger}) == 48*22*2
    verdict = 'PASS' if all(c['passed'] for c in comparisons) else 'FAIL'
    save(OUT/'calibration.json', calibration)
    save(OUT/'results.json', dict(verdict=verdict, plan_sha256=sha(PLAN), script_sha256=sha(__file__),
                                  comparisons=comparisons, auc=aucs, deviations=[]))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    with (OUT/'depth_table.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(depth[0])); w.writeheader(); w.writerows(depth)
    print(verdict)
    for c in comparisons:
        m = c['metrics']
        print(f"{c['family']:13s} {c['group']} " + ' '.join(f"{a}={m[a]['ber']*100:5.2f}" for a in ARMS)
              + f" | F-L {c['fusion_vs_lofo']['point']*100:+.2f} {np.round(np.array(c['fusion_vs_lofo']['ci'])*100,2).tolist()}"
              + f" F-C {c['fusion_vs_cvr']['point']*100:+.2f} pass={c['passed']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['generate', 'materialize', 'infer', 'evaluate'], required=True)
    stage = p.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        {'generate': generate, 'materialize': materialize, 'infer': infer, 'evaluate': evaluate}[stage]()
        save(OUT/f'terminal_{stage}.json', dict(status='complete'))
    except BaseException as e:
        save(OUT/f'terminal_{stage}.json', dict(status='failed', error=repr(e)))
        raise
