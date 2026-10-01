"""Observable shift/deferral precheck on consumed structure-space Development.

No training, scene generation, new-cohort access, or tuning on evaluation.
Three fixed indicators: small ensemble logit magnitude, seed logit spread,
and shrinkage Mahalanobis distance of 36 observation-only voxel summaries.
Each indicator is calibrated to <=10% triggering on in-support calibration
rows. The existing EXTRAP/PARTIAL operating points are retained unchanged.
All truth/area/scene attributes are evaluator-only, used after scoring.
This diagnostic does not establish deployable OOD detection or a switch.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from cnh_corridor_late_fusion import metrics, ranks, threshold
from cnh_cvr_pilot import CVR
from cnh_cvr_projection import query_masks

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/cnh-structure-space-20260929'
DEFAULT_OUT = ROOT / 'artifacts.local/work/cnh-observable-shift-dev-20260930'
GROUPS = ('HEAD', 'BODY')
INDICATORS = ('low_confidence', 'seed_spread', 'input_distance')
WEIGHTS = np.array([1, 2, 4, 8, 16], dtype=np.float64) / 31


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_meta(split):
    with np.load(SOURCE / 'data' / split / 'metadata.npz', allow_pickle=False) as z:
        # Metadata labels are deliberately not loaded during scoring.
        return {k: z[k] for k in ('unit', 'config', 'frame')}


def transform(x):
    z = np.array(x, dtype=np.float32)
    for c in (0, 2):
        z[:, c] = np.sign(z[:, c]) * np.log1p(np.abs(z[:, c]))
    z[:, 1] /= 8
    if not np.isfinite(z).all():
        raise ValueError('nonfinite model input')
    return z


def summaries(z, masks):
    regions = np.stack([np.ones_like(masks[0]), masks[0], masks[1],
                        1 - np.maximum(masks[0], masks[1])]).reshape(4, -1)
    flat = z.reshape(len(z), 3, -1)
    result = []
    for region in regions:
        w = region / region.sum()
        result.extend([(flat * w).sum(-1),
                       (np.abs(flat) * w).sum(-1),
                       np.sqrt((flat * flat * w).sum(-1))])
    return np.concatenate(result, axis=1).astype(np.float64)


def scene_rows(meta, arrays):
    """Pool exactly frames 11..15, preserving unit/config identifiers."""
    pairs = sorted(set(zip(meta['unit'].tolist(), meta['config'].tolist())))
    out = {name: [] for name in arrays}
    keys = []
    for unit, config in pairs:
        ix = np.flatnonzero((meta['unit'] == unit) & (meta['config'] == config)
                            & np.isin(meta['frame'], np.arange(11, 16)))
        ix = ix[np.argsort(meta['frame'][ix])]
        if not np.array_equal(meta['frame'][ix], np.arange(11, 16)):
            raise ValueError(f'incomplete window {unit}/{config}')
        keys.append((unit, config))
        for name, values in arrays.items():
            out[name].append(np.tensordot(WEIGHTS, values[ix], axes=(0, 0)))
    return keys, {k: np.asarray(v) for k, v in out.items()}


def auc(y, score):
    y = np.asarray(y, dtype=bool)
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return None
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def family_metrics(rows, prediction):
    groups = {}
    for g in GROUPS:
        rr = [r for r in rows if r['group'] == g]
        m = metrics([r['label'] for r in rr], [prediction(r) for r in rr])
        m['rows'] = len(rr)
        m['units'] = len({r['unit'] for r in rr})
        m['fpr'] = m['fp'] / m['negative'] if m['negative'] else None
        m['fnr'] = m['fn'] / m['positive'] if m['positive'] else None
        groups[g] = m
    bers = [m['ber'] for m in groups.values()]
    return dict(groups=groups, macro_ber=float(np.mean(bers))
                if all(v is not None for v in bers) else None)


def paired_macro_delta(rows, prediction, draws=2000):
    """Resample whole units once per draw for both height groups."""
    units = sorted({r['unit'] for r in rows})
    index = {u: i for i, u in enumerate(units)}
    counts = np.zeros((len(units), 2, 6), dtype=np.int64)
    for r in rows:
        y, b, m = bool(r['label']), bool(r['prediction']['EXTRAP']), bool(prediction(r))
        counts[index[r['unit']], GROUPS.index(r['group'])] += np.array(
            [y, not y, y and not b, not y and b, y and not m, not y and m])

    def delta(c):
        return (.5 * ((c[..., 4] - c[..., 2]) / c[..., 0]
                      + (c[..., 5] - c[..., 3]) / c[..., 1])).mean(-1)

    total = counts.sum(0)
    result = dict(point=float(delta(total)), ci=None, total_draws=draws,
                  valid_draws=0, units=len(units))
    pick = np.random.default_rng(2026093007).integers(0, len(units), (draws, len(units)))
    pooled = counts[pick].sum(1)
    valid = (pooled[..., :2] > 0).all(axis=(1, 2))
    result['valid_draws'] = int(valid.sum())
    if valid.any():
        result['ci'] = np.quantile(delta(pooled[valid]), [.025, .975]).tolist()
    return result


def main(out, device):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() or (out / 'request.json').exists():
        raise FileExistsError('use a fresh output directory; do not overwrite diagnostic evidence')
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.20)
    masks = query_masks()
    model_paths = [SOURCE / 'models/EXTRAP' / f'model_seed{s}.pt' for s in range(3)]
    frozen_hashes = json.loads((SOURCE / 'model_hashes.json').read_text())
    for path in model_paths:
        if sha(path) != frozen_hashes[str(path.relative_to(ROOT))]:
            raise ValueError('frozen model identity changed')
    inputs = []
    for split in ('train', 'calib', 'evaluation'):
        for name in ('features.npy', 'metadata.npz'):
            p = SOURCE / 'data' / split / name
            st = p.stat()
            inputs.append(dict(path=str(p.relative_to(ROOT)), bytes=st.st_size,
                               mtime_ns=st.st_mtime_ns))
    request = dict(scope='consumed Development diagnostic, fixed indicators, no new training',
                   pid=os.getpid(), source_script_sha256=sha(Path(__file__)),
                   python=sys.executable, device=device,
                   device_name=torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
                   backend_reason='GPU faster in representative 16-frame benchmark'
                   if device == 'cuda' else 'explicit CPU override; no speed claim',
                   inputs=inputs, model_hashes={str(p.relative_to(ROOT)): sha(p) for p in model_paths},
                   choices=dict(ensemble='EXTRAP seeds 0/1/2; average logits',
                                frame_pooling='11..15, weights 1/2/4/8/16',
                                input_reference='EXTRAP training panels only, no labels',
                                summaries='transformed voxel global/head/body/outside mean, mean_abs, RMS; 36 dims',
                                covariance_shrinkage=.1, trigger_calibration='in-support calib all rows, <=10%',
                                geometry_operating_point='existing PARTIAL predictions unchanged',
                                analysis_subsets=['all', 'out_of_support', 'near_same_side', 'near_same_side_out'],
                                uncertainty='2000 whole-unit paired bootstrap, descriptive, no multiplicity correction'))
    save(out / 'request.json', request)

    # Fit an observation-only distribution reference, never reading scene truth.
    train_meta = load_meta('train')
    chosen = (train_meta['config'] >= 22) & (train_meta['config'] < 34) & (train_meta['frame'] >= 11)
    ix = np.flatnonzero(chosen)
    selected_meta = {k: v[ix] for k, v in train_meta.items()}
    x = np.load(SOURCE / 'data/train/features.npy', mmap_mode='r')
    small = np.concatenate([summaries(transform(x[ids]), masks)
                            for ids in np.array_split(ix, max(1, (len(ix) + 63) // 64))])
    train_keys, pooled = scene_rows(selected_meta, {'summary': small})
    reference = pooled['summary']
    assert len(train_keys) == 96 * 12
    center = reference.mean(0)
    scale = reference.std(0)
    scale = np.maximum(scale, 1e-6)
    standard = (reference - center) / scale
    cov = np.cov(standard, rowvar=False)
    precision = np.linalg.inv(.9 * cov + .1 * np.eye(len(center)))
    np.savez_compressed(out / 'reference.npz', center=center, scale=scale,
                        precision=precision, train_units=96, train_scenes=len(reference))
    del x, small, reference

    nets = []
    for path in model_paths:
        net = CVR().to(device).eval()
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        nets.append(net)
    mask_tensor = torch.as_tensor(masks, device=device)
    tables = {}
    for split in ('calib', 'evaluation'):
        meta = load_meta(split)
        x = np.load(SOURCE / 'data' / split / 'features.npy', mmap_mode='r')
        obs, logits = [], []
        with torch.inference_mode():
            for begin in range(0, len(x), 32):
                z = transform(x[begin:begin + 32])
                obs.append(summaries(z, masks))
                xx = torch.as_tensor(z, device=device)
                xx = torch.cat([xx, mask_tensor[None].expand(len(z), -1, -1, -1, -1)], 1)
                logits.append(torch.stack([n(xx) for n in nets], 1).cpu().numpy())
                if begin % 1024 == 0:
                    save(out / 'progress.json', dict(stage='score', split=split,
                         completed=min(begin + 32, len(x)), total=len(x), elapsed_s=time.monotonic() - start))
        keys, pooled = scene_rows(meta, dict(summary=np.concatenate(obs), logits=np.concatenate(logits)))
        v = (pooled['summary'] - center) / scale
        distance = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', v, precision, v), 0))
        average = pooled['logits'].mean(1)
        spread = pooled['logits'].std(1)
        tables[split] = {(u, c, g): dict(low_confidence=float(-abs(average[i, q])),
                         seed_spread=float(spread[i, q]), input_distance=float(distance[i]),
                         learned_logit=float(average[i, q]))
                         for i, (u, c) in enumerate(keys) for q, g in enumerate(GROUPS)}
        np.savez_compressed(out / f'pooled_{split}.npz', keys=np.asarray(keys),
                            seed_logits=pooled['logits'], summary=pooled['summary'])
        print(f'scored {split}: {len(keys)} scenes', flush=True)
    del nets, net, mask_tensor, xx, x
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Scoring is complete. Evaluator metadata/truth enter only from here onward.
    manifest = json.loads((SOURCE / 'scene_manifest.json').read_text())
    meta = {(r['unit'], r['config']): r for r in manifest if r['split'] != 'train'}
    calibration = {}
    for name in INDICATORS:
        calibration[name] = {}
        for g in GROUPS:
            ref = [v[name] for (u, c, gg), v in tables['calib'].items()
                   if gg == g and meta[u, c]['bin'] <= 2]
            k = threshold(ranks(ref, ref), len(ref))
            calibration[name][g] = dict(rank_threshold=k, n=len(ref),
                 triggered=int((ranks(ref, ref) >= k).sum()), reference=ref)
    rows = [json.loads(s) for s in (SOURCE / 'sample_ledger.jsonl').read_text().splitlines()]
    max_error = 0.
    for r in rows:
        obs = tables['evaluation'][r['unit'], r['config'], r['group']]
        max_error = max(max_error, abs(obs['learned_logit'] - r['scores']['EXTRAP']))
        r['observable'] = obs
        m = meta[r['unit'], r['config']]
        r['outside'] = m['bin'] > 2
        r['corner'] = m['same_side'] > 0 and m['gap'] < .15
        r['trigger'] = {name: bool(ranks(calibration[name][r['group']]['reference'], [obs[name]])[0]
                                   >= calibration[name][r['group']]['rank_threshold']) for name in INDICATORS}
        r['learned_error'] = r['prediction']['EXTRAP'] != r['label']
        r['geometry_only_correct'] = r['learned_error'] and r['prediction']['PARTIAL'] == r['label']
    if max_error > 1e-4:
        raise AssertionError(f'cached inference mismatch: {max_error}')
    for name in INDICATORS:
        for g in GROUPS:
            calibration[name][g].pop('reference')
    subsets = dict(all=rows, out_of_support=[r for r in rows if r['outside']],
                   near_same_side=[r for r in rows if r['corner']],
                   near_same_side_out=[r for r in rows if r['corner'] and r['outside']])
    result = dict(scope=request['scope'], calibration=calibration, inference_max_abs_error=max_error,
                  training_reference=dict(units=96, scenes=1152, dimensions=36), subsets={})
    for subset, rr in subsets.items():
        base = family_metrics(rr, lambda r: r['prediction']['EXTRAP'])
        geometric = family_metrics(rr, lambda r: r['prediction']['PARTIAL'])
        details = {}
        for name in INDICATORS:
            switch = lambda r, name=name: r['prediction']['PARTIAL'] if r['trigger'][name] else r['prediction']['EXTRAP']
            per_group = {}
            for g in GROUPS:
                gg = [r for r in rr if r['group'] == g]
                per_group[g] = {target: auc([r[target] for r in gg], [r['observable'][name] for r in gg])
                               for target in ('outside', 'learned_error', 'geometry_only_correct')}
            triggered = [r for r in rr if r['trigger'][name]]
            details[name] = dict(auc_by_group=per_group, triggered=len(triggered),
                trigger_rate=len(triggered) / len(rr),
                corrected=sum(r['geometry_only_correct'] for r in triggered),
                broken=sum(not r['learned_error'] and r['prediction']['PARTIAL'] != r['label'] for r in triggered),
                useful_total=sum(r['geometry_only_correct'] for r in rr),
                switched=family_metrics(rr, switch), paired_delta=paired_macro_delta(rr, switch))
        result['subsets'][subset] = dict(rows=len(rr), units=len({r['unit'] for r in rr}),
                positive=sum(r['label'] for r in rr), negative=sum(not r['label'] for r in rr),
                learned=base, geometry=geometric, indicators=details)
    result['elapsed_s'] = time.monotonic() - start
    result['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in
                             [SOURCE / 'scene_manifest.json', SOURCE / 'sample_ledger.jsonl']}
    save(out / 'results.json', result)
    (out / 'diagnostic_ledger.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    save(out / 'terminal.json', dict(status='complete', elapsed_s=result['elapsed_s'], trained_models=0,
                                    generated_scenes=0, gpu_models_released=True))
    print(json.dumps({s: {n: round(v['paired_delta']['point'], 4)
                         for n, v in d['indicators'].items()} for s, d in result['subsets'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    try:
        main(args.output, args.device)
    except BaseException as error:
        if args.output.exists() and not (args.output / 'terminal.json').exists():
            save(args.output / 'terminal.json', dict(status='failed', error=repr(error)))
        raise
