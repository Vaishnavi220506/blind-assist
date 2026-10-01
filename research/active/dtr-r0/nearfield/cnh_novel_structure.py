"""Zero-training unseen-structure diagnostic with frozen all-family A2 and CVR.

Plan: artifacts.local/work/cnh-novel-structure-20260929/NOVEL_STRUCTURE_PLAN.md
Calibration uses known four families only; evaluation adds six novel families.
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
OUT = WORK/'cnh-novel-structure-20260929'
PLAN = OUT/'NOVEL_STRUCTURE_PLAN.md'
SPLITS = {'calib': list(range(60000, 60048)), 'evaluation': list(range(70000, 70096))}
USED = set(range(0, 18)) | set(range(1000, 1096)) | set(range(2000, 2024)) | set(range(3000, 3048)) \
    | set(range(40000, 40024)) | set(range(50000, 50048))
KNOWN = ('boundary', 'mixed_surface', 'sidewall', 'general')
NOVEL = ('pole', 'railing', 'low_beam', 'opposite_wall', 'bollards', 'vehicle')
GROUPS = ('HEAD', 'BODY')
FRAMES = np.arange(11, 16)
ARMS = ('A2', 'CVR', 'MOTION8_PARTIAL', 'FUSION')


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


def scenes_for(unit):
    import cnh_novel_structure_scenes as N
    return N._KNOWN(unit) if unit in SPLITS['calib'] else N.make_all_scenes(unit)


def generate():
    assert PLAN.exists(), 'freeze plan first'
    assert not USED & set(SPLITS['calib']+SPLITS['evaluation'])
    import cnh_proposal_attribution_scenes as S
    import cnh_novel_structure_scenes as N
    import cnh_corridor_diagnostic as D
    save(OUT/'source_manifest.json', {p.name: sha(p) for p in [PLAN, Path(__file__), HERE/'cnh_novel_structure_scenes.py',
         HERE/'cnh_proposal_attribution_scenes.py', HERE/'cnh_corridor_diagnostic.py', HERE/'cnh_corridor_labels.py']})
    S.make_scenes = scenes_for  # D.generate imports make_scenes from S at call time
    D.OUT = OUT
    D.SPLITS = SPLITS
    D.generate()


def materialize():
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    import cnh_novel_structure_scenes as N
    torch.set_num_threads(2)
    projector = BatchedProjector()
    for split, units in SPLITS.items():
        folder = OUT/'data'/split
        if (folder/'metadata.npz').exists(): continue
        folder.mkdir(parents=True, exist_ok=True)
        counts = [len(read(OUT/'features'/split/f'unit{u}.npz')['family']) for u in units]
        n = sum(counts)*len(FRAMES)
        feats = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float16, shape=(n, 3, *SHAPE))
        meta = {k: [] for k in ['unit', 'config', 'frame', 'family', 'labels']}
        offset = 0
        for u in units:
            d = read(OUT/'features'/split/f'unit{u}.npz')
            ref = N._KNOWN(u)[0]
            for config in range(len(d['family'])):
                ids = np.flatnonzero(d['scene'] == config)
                assert np.array_equal(d['frame'][ids], np.arange(16))
                sensor, travel, noisy = motion_metadata(u, config)
                assert np.allclose(sensor, ref['poses'], atol=1e-9) and np.allclose(travel, ref['travel'], atol=1e-9)
                for f in FRAMES:
                    voxel = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                    assert bool(torch.isfinite(voxel).all())
                    feats[offset] = voxel.cpu().numpy().astype(np.float16); offset += 1
                meta['unit'] += [u]*len(FRAMES); meta['config'] += [config]*len(FRAMES)
                meta['frame'] += FRAMES.tolist(); meta['family'] += [str(d['family'][config])]*len(FRAMES)
                meta['labels'] += d['labels'][ids[FRAMES], 2:4].tolist()
            print('materialize', split, u, flush=True)
        assert offset == n
        feats.flush(); del feats
        np.savez_compressed(folder/'metadata.npz', **{k: np.asarray(v) for k, v in meta.items()})


def infer():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import cnh_corridor_retrain as T
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    T.frozen_module()
    from cnh_learned_memory_fusion import causal_ewma
    dest = OUT/'predictions'; dest.mkdir(exist_ok=True)
    a2files = [WORK/'cnh-corridor-retrain-20260929/models'/f'model_seed{s}.pt' for s in range(3)]
    cvrfiles = [WORK/'cnh-cvr-v2-20260929/models/CVR/allfamily'/f'model_seed{s}.pt' for s in range(3)]
    save(OUT/'model_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in a2files+cvrfiles})
    if not (dest/'A2.npz').exists():
        models = T.load_models(a2files, 'cuda'); out = {}
        for split, units in SPLITS.items():
            for u in units:
                d = read(OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
                nn = T.predict(models, d['z4'], d['z1'], d['support'], 48)
                out[str(u)] = causal_ewma(nn, d['scene'], d['frame'], alpha=.5, window=5).reshape(k, 16, 6)[:, -1, [2, 3]]
        np.savez_compressed(dest/'A2.npz', **out); del models; torch.cuda.empty_cache(); print('infer A2', flush=True)
    if not (dest/'CVR.npz').exists():
        nets = []
        for f in cvrfiles:
            net = CVR().cuda(); net.load_state_dict(torch.load(f, map_location='cuda', weights_only=True)); nets.append(net.eval())
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        out = {}
        for split in SPLITS:
            x = np.load(OUT/'data'/split/'features.npy', mmap_mode='r')
            m = read(OUT/'data'/split/'metadata.npz')
            raw = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    b = torch.as_tensor(np.array(x[i:i+64], dtype=np.float32), device='cuda')
                    # Same transform as cnh_cvr_v2_train.Data.batch.
                    b[:, 0] = b[:, 0].sign()*b[:, 0].abs().log1p()
                    b[:, 2] = b[:, 2].sign()*b[:, 2].abs().log1p(); b[:, 1] /= 8
                    b = torch.cat((b, masks[None].expand(len(b), -1, -1, -1, -1)), 1)
                    assert torch.isfinite(b).all()
                    raw.append(torch.stack([n(b) for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(raw)
            for u in SPLITS[split]:
                sel = np.flatnonzero(m['unit'] == u); scores = []
                for config in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == config]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], FRAMES)
                    scores.append((raw[loc].astype(np.float64)*np.array([1, 2, 4, 8, 16])[:, None]).sum(0)/31)
                out[str(u)] = np.asarray(scores)
        np.savez_compressed(dest/'CVR.npz', **out); print('infer CVR', flush=True)


def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks, threshold
    from cnh_corridor_statistics import paired_cluster_ber
    from sklearn.metrics import roc_auc_score
    assert PLAN.exists() and not (OUT/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    pred = {a: read(OUT/'predictions'/f'{a}.npz') for a in ('A2', 'CVR')}

    def rows(split):
        out = []
        for u in SPLITS[split]:
            d = read(OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, config=c, group=g, family=str(d['family'][c]), label=int(labels[c, q]),
                        target_group=int(d['group'][c]), margin=float(d['margin'][c]),
                        scores=dict(A2=float(pred['A2'][str(u)][c, q]), CVR=float(pred['CVR'][str(u)][c, q]),
                                    MOTION8_PARTIAL=float(d['motion'][c, 2, q]))))
        return out

    cal_all, ev_all = rows('calib'), rows('evaluation')
    assert {r['family'] for r in cal_all} == set(KNOWN)
    assert {r['family'] for r in ev_all} == set(KNOWN) | set(NOVEL)
    calibration, table, ledger, depth = [], [], [], []
    for q, g in enumerate(GROUPS):
        cal = [r for r in cal_all if r['group'] == g and r['label'] == 0]; n = len(cal)
        ev = [r for r in ev_all if r['group'] == g]
        cr, er = {}, {}
        for arm in ('A2', 'CVR', 'MOTION8_PARTIAL'):
            ref = [r['scores'][arm] for r in cal]
            cr[arm] = ranks(ref, ref); er[arm] = ranks(ref, [r['scores'][arm] for r in ev])
        cr['FUSION'] = np.maximum(cr['CVR'], cr['MOTION8_PARTIAL']); er['FUSION'] = np.maximum(er['CVR'], er['MOTION8_PARTIAL'])
        p = {}
        for arm in ARMS:
            k = threshold(cr[arm], n); fp = int((cr[arm] >= k).sum())
            assert 10*fp <= n and (k == 0 or 10*int((cr[arm] >= k-1).sum()) > n)
            calibration.append(dict(group=g, arm=arm, rank_threshold=k, negative=n, fp=fp, fpr=fp/n))
            p[arm] = er[arm] >= k
        for i, r in enumerate(ev):
            r['predictions'] = {a: int(p[a][i]) for a in ARMS}
            r['rank_numerators'] = {a: int(er[a][i]) for a in ARMS}; r['rank_denominator'] = n
            r['role'] = 'positive_intrusion' if r['label'] else ('target_group_outside' if r['target_group'] == q else 'other_height_negative')
            r['band'] = '<=5cm' if abs(r['margin']) <= .05 else ('5-15cm' if abs(r['margin']) <= .15 else '>15cm')
            ledger.append(r)
        for fam in KNOWN + NOVEL:
            idx = [i for i, r in enumerate(ev) if r['family'] == fam]
            y = np.array([ev[i]['label'] for i in idx]); units = np.array([ev[i]['unit'] for i in idx])
            row = dict(family=fam, group=g, known=fam in KNOWN, positive=int(y.sum()), negative=int((1-y).sum()))
            for arm in ARMS:
                pp = p[arm][idx]; m = metrics(y, pp)
                row[arm] = dict(m, auc=float(roc_auc_score(y, er[arm][idx].astype(float))) if 0 < y.sum() < len(y) else None,
                                fpr=m['fp']/m['negative'] if m['negative'] else None)
            for arm in ('CVR', 'MOTION8_PARTIAL', 'FUSION'):
                row[f'{arm}_minus_A2'] = paired_cluster_ber(y, p['A2'][idx], p[arm][idx], units)
            table.append(row)
            for role in ('positive_intrusion', 'target_group_outside', 'other_height_negative'):
                for band in ('<=5cm', '5-15cm', '>15cm', 'all'):
                    rr = [ev[i] for i in idx if ev[i]['role'] == role and (band == 'all' or ev[i]['band'] == band)]
                    for arm in ARMS:
                        depth.append(dict(family=fam, group=g, role=role, band=band, arm=arm,
                                          **metrics([r['label'] for r in rr], [r['predictions'][arm] for r in rr])))
    # Family-level summaries (mean over HEAD/BODY) and pre-stated reading rules.
    family = []
    for fam in KNOWN + NOVEL:
        rr = [t for t in table if t['family'] == fam]
        s = dict(family=fam, known=fam in KNOWN)
        for arm in ARMS:
            s[f'{arm}_ber'] = float(np.mean([t[arm]['ber'] for t in rr]))
            s[f'{arm}_auc'] = float(np.mean([t[arm]['auc'] for t in rr]))
            s[f'{arm}_fpr_max'] = float(max(t[arm]['fpr'] for t in rr))
        s['learned_ranking_failure'] = bool(s['A2_auc'] <= .70 and s['CVR_auc'] <= .70)
        s['geometry_transfers'] = bool(s['MOTION8_PARTIAL_auc'] >= .75)
        s['threshold_shift_arms'] = [a for a in ARMS if s[f'{a}_fpr_max'] > .20]
        family.append(s)
    worst = {grp: {arm: max((f for f in family if f['known'] == (grp == 'known')), key=lambda f: f[f'{arm}_ber'])['family']
                   for arm in ARMS} for grp in ('known', 'novel')}
    save(OUT/'calibration.json', calibration)
    save(OUT/'results.json', dict(status='DESCRIPTIVE', plan_sha256=sha(PLAN), script_sha256=sha(__file__),
        novel_learned_ranking_failures=[f['family'] for f in family if not f['known'] and f['learned_ranking_failure']],
        family=family, worst_family=worst, table=table))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    with (OUT/'depth_table.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(depth[0])); w.writeheader(); w.writerows(depth)
    for f in family:
        print(f"{f['family']:13s} {'K' if f['known'] else 'N'} " + ' '.join(
            f"{a}:BER{f[f'{a}_ber']*100:5.1f}/AUC{f[f'{a}_auc']:.3f}/FPmax{f[f'{a}_fpr_max']*100:4.1f}" for a in ARMS)
            + f" fail={f['learned_ranking_failure']} geo={f['geometry_transfers']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['generate', 'materialize', 'infer', 'evaluate'], required=True)
    stage = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        {'generate': generate, 'materialize': materialize, 'infer': infer, 'evaluate': evaluate}[stage]()
        save(OUT/f'terminal_{stage}.json', dict(status='complete'))
    except BaseException as e:
        save(OUT/f'terminal_{stage}.json', dict(status='failed', error=repr(e)))
        raise
