"""Privileged segmentation Development evaluation on repaired v2 only.

NPZ contract: common config/frame/labels/witness/strata/split; M0 and
M1__condition/M2__condition scores [N,6]. Optional per-policy diagnostic
suffixes: __candidate, __eligible, __distance, __truth_distance, all [N,6].
Distances are radial, describing selected candidate per query/frame, not every object.
Truth distance is evaluator-only and must never enter score generation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_position_prior_analysis as metrics

FAMILY = 'cnh-track-a-scale-v2-20260926'
EXTRAS = ('candidate', 'eligible', 'distance', 'truth_distance')
COHORT = {'calib': list(range(96, 128)), 'eval': [u for u in range(128, 192) if u != 143]}


def load(root, require_complete=True):
    seqs, units, policies = {'calib': [], 'eval': []}, {'calib': [], 'eval': []}, None
    for path in sorted(Path(root).glob('unit*.npz'), key=lambda p: int(p.stem[4:])):
        unit = int(path.stem[4:])
        with np.load(path, allow_pickle=False) as f:
            split = {'calib': 'calib', 'audit': 'eval'}.get(str(f['split']))
            if split is None or unit not in COHORT[split]:
                raise ValueError(f'{path}: outside repaired-v2 authorized cohort')
            if 'family' in f and str(f['family']) != FAMILY:
                raise ValueError(f'{path}: wrong source family')
            arms = sorted(k for k in f.files if k == 'M0' or
                          (k.startswith(('M1__', 'M2__')) and len(k.split('__')) == 2))
            if 'M0' not in arms or not any(a.startswith('M1__') for a in arms) or not any(a.startswith('M2__') for a in arms):
                raise ValueError(f'{path}: missing method family')
            if policies is not None and policies != arms:
                raise ValueError(f'{path}: inconsistent policies')
            policies = arms
            frame, cfg = f['frame'], f['config']
            if cfg.shape != frame.shape:
                raise ValueError('Config/frame shape mismatch')
            for key in ('labels', 'witness', 'strata', *arms):
                if f[key].shape != (len(frame), 6):
                    raise ValueError(f'{path}: invalid {key} shape')
            for arm in arms:
                if np.isnan(f[arm]).any() or np.isposinf(f[arm]).any():
                    raise ValueError(f'{path}: nonfinite score {arm}')
            extra = {}
            for arm in arms:
                present = [f'{arm}__{k}' in f for k in EXTRAS]
                if any(present) and not all(present):
                    raise ValueError(f'{path}: incomplete diagnostics for {arm}')
                if all(present):
                    extra[arm] = {k: f[f'{arm}__{k}'] for k in EXTRAS}
                    if any(v.shape != (len(frame), 6) for v in extra[arm].values()):
                        raise ValueError(f'{path}: diagnostic shape mismatch')
            units[split].append(unit)
            for c in np.unique(cfg):
                ix = np.flatnonzero((cfg == c) & (frame >= 3))
                ix = ix[np.argsort(frame[ix])]
                if not np.array_equal(frame[ix], np.arange(3, 12)):
                    raise ValueError(f'{path}: missing/duplicate evaluated frame')
                seqs[split].append(dict(unit=unit, y=f['labels'][ix], w=f['witness'][ix],
                    strata=f['strata'][ix].astype(str), s={a: f[a][ix] for a in arms},
                    extra={a: {k: v[ix] for k, v in values.items()} for a, values in extra.items()}))
    if require_complete and units != COHORT:
        raise ValueError('Require all 32 calib and 63 eligible audit units; exclude incomplete143')
    if not all(seqs.values()):
        raise ValueError('Both splits required')
    return seqs, units, policies


def pack(seqs, boxes, policies):
    aliases = [dict(sq, s={a: sq['s']['M0'] for a in metrics.ARMS}) for sq in seqs]
    p = metrics.pack(aliases, boxes)
    p['s'] = {a: np.concatenate([sq['s'][a][:, boxes].T for sq in seqs]) for a in policies}
    p['extra'] = {}
    for a in policies:
        has = [a in sq.get('extra', {}) for sq in seqs]
        if any(has) and not all(has):
            raise ValueError(f'Inconsistent diagnostic availability: {a}')
        if all(has):
            p['extra'][a] = {k: np.concatenate([sq['extra'][a][k][:, boxes].T for sq in seqs]) for k in EXTRAS}
    return p


def diagnostics(p, arm, threshold):
    if arm not in p.get('extra', {}):
        return None
    x = p['extra'][arm]
    eligible, candidate = x['eligible'].astype(bool), x['candidate'].astype(bool)
    confirmed = eligible & candidate & (p['s'][arm] >= threshold) & np.isfinite(x['distance'])
    out = {}
    for st in ('all', 'tiny', 'realistic', 'wide'):
        mask = np.ones_like(p['y']) if st == 'all' else p['frame_strata'] == st
        attempted = mask & eligible & candidate
        positive = mask & p['y']
        good = mask & confirmed & np.isfinite(x['truth_distance'])
        error = x['distance'][good]-x['truth_distance'][good]
        out[st] = dict(unconfirmed_candidate_query_frames=metrics.rate((attempted & ~confirmed).sum(), attempted.sum()),
            positive_frames_without_confirmation=metrics.rate((positive & ~confirmed).sum(), positive.sum()),
            unattempted_query_frames=metrics.rate((mask & ~eligible).sum(), mask.sum()),
            fallback_query_frames=metrics.rate((mask & ~eligible).sum(), mask.sum()) if arm.startswith('M2__') else None,
            confirmed_without_truth=metrics.rate((mask & confirmed & ~np.isfinite(x['truth_distance'])).sum(), (mask & confirmed).sum()),
            distance_error=dict(count=int(error.size), mae_m=float(np.mean(np.abs(error))) if error.size else None,
                median_abs_m=float(np.median(np.abs(error))) if error.size else None,
                bias_m=float(np.mean(error)) if error.size else None))
    return out


def paired_delta(p, arm, threshold, baseline_threshold, stratum='tiny', draws=2000):
    """Descriptive paired unit bootstrap; never treat query frames as independent."""
    eligible = p['near'] & ((p['strata'] == stratum) if stratum != 'all' else True)
    units = np.unique(p['unit'])
    den = np.array([np.sum(eligible & (p['unit'] == u)) for u in units])
    if not den.sum():
        return dict(delta=None, ci95=None, units=int(len(units)))
    nums = []
    for a, t in ((arm, threshold), ('M0', baseline_threshold)):
        hit = (p['s'][a] >= t) & p['y']
        first = np.where(hit.any(1), hit.argmax(1), hit.shape[1])
        timely = metrics.timely(p, first) & eligible
        nums.append(np.array([np.sum(timely & (p['unit'] == u)) for u in units]))
    difference = nums[0]-nums[1]
    ix = np.random.default_rng(20260928).integers(len(units), size=(draws, len(units)))
    d = den[ix].sum(1)
    boot = difference[ix].sum(1)[d > 0]/d[d > 0]
    return dict(delta=float(difference.sum()/den.sum()), ci95=np.quantile(boot, [.025, .975]).tolist(), units=int(len(units)))


def load_objects(root, units, policies):
    """Optional separate object rows, including dropped true masks and fake masks."""
    paths = {u: Path(root)/f'objects{u:02d}.npz' for u in units}
    exists = [p.exists() for p in paths.values()]
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError('Object diagnostics require all eligible audit units')
    fields = ('score', 'distance', 'truth', 'frame', 'class', 'id')
    out = {a: {k: [] for k in fields} for a in policies if a != 'M0'}
    for unit, path in paths.items():
        with np.load(path, allow_pickle=False) as f:
            for arm, values in out.items():
                data = {k: f[f'{arm}__object_{k}'] for k in fields}
                n = len(data['frame'])
                for k in fields:
                    if data[k].shape != ((n, 6) if k in ('score', 'distance') else (n,)):
                        raise ValueError(f'{path}/{arm}: invalid object {k} shape')
                if np.isnan(data['score']).any() or np.isposinf(data['score']).any():
                    raise ValueError('Invalid object score')
                if not np.isin(data['class'], ('tiny', 'realistic', 'wide', 'false')).all():
                    raise ValueError('Unknown object class')
                if not np.isin(data['frame'], np.arange(12)).all():
                    raise ValueError('Invalid object frame')
                if np.any((data['class'] == 'false') & np.isfinite(data['truth'])):
                    raise ValueError('Fake object must not have truth distance')
                cfgkey = f'{arm}__object_config'
                if cfgkey in f:
                    cfg = f[cfgkey]
                    if cfg.shape != (n,) or len(set(zip(cfg.tolist(), data['frame'].tolist(), data['id'].tolist()))) != n:
                        raise ValueError('Invalid or duplicate config/frame/object diagnostic row')
                mask = data['frame'] >= 3
                for k in fields:
                    values[k].append(data[k][mask])
    return {a: {k: np.concatenate(v) for k, v in values.items()} for a, values in out.items()}


def object_diagnostics(data, boxes, threshold):
    score = data['score'][:, boxes]
    winner = score.argmax(1)
    selected = score[np.arange(len(score)), winner]
    distance = data['distance'][:, boxes][np.arange(len(score)), winner]
    confirmed = (selected >= threshold) & np.isfinite(distance)
    true = data['class'] != 'false'
    out = {}
    for st in ('all', 'tiny', 'realistic', 'wide'):
        mask = true & (True if st == 'all' else data['class'] == st)
        good = mask & confirmed & np.isfinite(data['truth'])
        error = np.abs(distance[good]-data['truth'][good])
        out[st] = dict(unconfirmed=metrics.rate((mask & ~confirmed).sum(), mask.sum()),
            confirmed_missing_truth=metrics.rate((mask & confirmed & ~np.isfinite(data['truth'])).sum(), (mask & confirmed).sum()),
            distance_error=dict(count=int(error.size), mae_m=float(error.mean()) if error.size else None,
                median_abs_m=float(np.median(error)) if error.size else None,
                p90_abs_m=float(np.quantile(error, .9)) if error.size else None))
    out['fake_confirmation'] = metrics.rate((~true & confirmed).sum(), (~true).sum())
    return out


def analyze(root):
    seqs, units, policies = load(root)
    objects = load_objects(root, units['eval'], policies)
    out = dict(scope='Consumed repaired-v2 Development; privileged segmentation only for method development',
        unit_ids=units, selection='Per HEAD/BODY and policy: lowest feasible threshold on linspace(1,12,441)+infinity using calib empty-pair false-alert budget; freeze for audit',
        event_definition='Unmodified v4-compatible helpers: frames3..11; near if positive witness<=1m; timely if first positive-frame alert witness>=1m; empty sequence-query any alert is false',
        diagnostic_definition='Selected candidate per query-frame; confirmed only above selected threshold. Missing truth excluded from distance errors, counted separately; size-specific false-alert undefined. Unknown is not negative.',
        object_definition='Distinct visible frame/object rows including dropped masks; frame>=3; true unknown denominator includes every true row. Confirmation takes max query score in HEAD/BODY; radial distance from same winning query. Fake confirmations separate. Rows outside a group still remain in all-visible-object denominator; this is not group-specific obstacle recall.',
        results={}, compact_table=[])
    for group, boxes in metrics.GROUPS.items():
        ca, ev = [pack(seqs[s], boxes, policies) for s in ('calib', 'eval')]
        for budget in metrics.BUDGETS:
            base_threshold = metrics.select(ca, 'M0', budget)[0]
            for arm in policies:
                threshold = metrics.select(ca, arm, budget)[0]
                row = dict(threshold=metrics.threshold_json((threshold,))[0],
                    calib=metrics.evaluate(ca, arm, (threshold,)), eval=metrics.evaluate(ev, arm, (threshold,)),
                    diagnostics=diagnostics(ev, arm, threshold),
                    objects=object_diagnostics(objects[arm], boxes, threshold) if objects is not None and arm in objects else None,
                    tiny_timely_delta_M0=paired_delta(ev, arm, threshold, base_threshold))
                out['results'][f'{group}/{arm}/{budget:.2f}'] = row
                out['compact_table'].append(dict(group=group, policy=arm, budget=budget,
                    threshold=row['threshold'], tiny_timely=row['eval']['tiny']['timely'],
                    actual_false_alert=row['eval']['all']['false_alert'], delta=row['tiny_timely_delta_M0']))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores')
    parser.add_argument('output')
    args = parser.parse_args()
    result = analyze(args.scores)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([r for r in result['compact_table'] if r['budget'] == .1], indent=2))


if __name__ == '__main__':
    main()
