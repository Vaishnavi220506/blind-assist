"""Expanded repeated-view stress check, never fresh scene confirmation.

Reconstructs all eligible edges from the original 500-frame inventory without
the former one-instance/one-frame subsampling. Prediction quality does not
select edges. The original oracle cell/floor/clean-edge limitations remain.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

from cnh_rgb_clearance_probe import (
    ROOT, CACHE, context, frame_candidates, read, save, sha, summarize,
)
from cnh_rgb_clearance_geometry import reference_frame
from cnh_rgb_clearance_edge import zone_map, extract_boundary
from cnh_rgb_zone_range import summarize_distribution

PRIOR = ROOT / 'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
DEFAULT_OUT = ROOT / 'artifacts.local/work/cnh-rgb-association-expanded-20261002'


def cpu_backend(out):
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    return select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),
            lambda _:DeviceObservation('cpu','host CPU','numpy/scipy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},
        record_path=out/'backend.json')


def prepare(out):
    if (out / 'selection.json').exists():
        raise FileExistsError('Preserve existing selection; choose a new directory')
    inventory = read(PRIOR / 'inventory.json')
    original = read(PRIOR / 'selection.json')
    admitted = {row['id']: row['events'] for row in inventory['frames'] if row['events']}
    save(out / 'PLAN.json', dict(
        question='Do candidate mechanisms survive all original eligible repeated views?',
        role='consumed synthetic Development; same 8 scenes, not independent confirmation',
        selection='Reconstruct every eligible edge from the original 19 positive frames; '
                  'exact original geometric rules, no instance/frame subsampling, no prediction selection',
        source_inventory_sha256=sha(PRIOR / 'inventory.json'),
        source_selection_sha256=sha(PRIOR / 'selection.json'),
        primary='All-case <=2cm local clearance, missing counted as failure; scene paired intervals',
        secondary='P50/P95 absolute error, sign disagreement, original-vs-additional records',
        comparisons='Original Depth Pro, cell q10, strongest mode, nearest RGB prior; '
                    'new distribution alignment and spatial support mechanisms',
        limitations=['oracle target cell, floor and clean visible edge',
                     'repeated edges/views/instances are correlated',
                     'geometry-derived cell aggregates are not measured ToF',
                     'not grazing-contact recall or hardware safety'],
        code_sha256=sha(Path(__file__)),
    ))
    manifest, cameras, boxes = context()
    events = []
    started = time.perf_counter()
    for row in manifest:
        if row['id'] not in admitted:
            continue
        status, candidates = frame_candidates((row, cameras[row['id']], boxes))
        assert status['events'] == admitted[row['id']], (row['id'], status)
        events.extend(candidates)
        print('reconstructed', row['id'], len(candidates), flush=True)
    events.sort(key=lambda e: e['id'])
    assert len(events) == inventory['candidate_events'] == 52
    by_id = {e['id']: e for e in events}
    assert all(by_id[e['id']] == e for e in original)
    save(out / 'selection.json', events)
    save(out / 'inventory.json', dict(
        records=len(events), frames=len({e['frame_id'] for e in events}),
        scenes=len({e['scene'] for e in events}),
        known_scene_instances=len({(e['scene'], e['instance_id']) for e in events if e['instance_id'] >= 0}),
        original_records=len(original), additional_records=len(events)-len(original),
        within_body_line_2cm=sum(abs(e['gt_clearance_m']) <= .02 for e in events),
        categories=dict(Counter(e['category'] for e in events)),
        ranges=dict(Counter(e['range_bin'] for e in events)),
        frame_counts=dict(Counter(e['frame_id'] for e in events)),
        prediction_access='NONE', selection_sha256=sha(out / 'selection.json'),
        seconds=time.perf_counter()-started,
    ))


def paired_difference(ledger, candidate, baseline):
    scenes = sorted({r['scene'] for r in ledger})
    counts = np.array([sum(r['scene'] == s for r in ledger) for s in scenes])
    def hit(row, arm):
        value = row['errors_m'][arm]
        return int(value is not None and abs(value) <= .02)
    differences = np.array([sum(hit(r, candidate)-hit(r, baseline)
                               for r in ledger if r['scene'] == s) for s in scenes])
    rng = np.random.default_rng(2026100201)
    sampled = rng.integers(len(scenes), size=(2000, len(scenes)))
    rates = differences[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
    return dict(delta_pp=float(differences.sum()/counts.sum()*100),
                ci95_pp=(np.percentile(rates, [2.5, 97.5])*100).tolist(),
                per_scene={s:dict(n=int(n),net_hits=int(d)) for s,n,d in zip(scenes, counts, differences)})


def coarse_histograms(radial, zones, bin_width=.05, maximum=12.8):
    """Destroy pixel correspondence before exposing simulated range counts."""
    bins = int(round(maximum/bin_width))
    valid = np.isfinite(radial) & (radial > 0) & (radial < maximum) & (zones >= 0)
    indices = zones[valid]*bins + np.floor(radial[valid]/bin_width).astype(int)
    return np.bincount(indices, minlength=64*bins).reshape(64, bins)


def ordinal_transport(predicted_radial, target_prior, counts, bin_width=.05):
    """Map a predicted range's local rank to an unlabelled coarse histogram.

    No geometry truth, target identity, semantic label, or per-pixel sensor
    distance enters this function. It assumes the RGB ordinal ordering holds.
    """
    values = np.asarray(predicted_radial)
    values = values[np.isfinite(values) & (values > 0)]
    counts = np.asarray(counts, dtype=float)
    if not values.size or not counts.sum() or not np.isfinite(target_prior) or target_prior <= 0:
        return None, dict(status='MISSING')
    rank = float((np.sum(values < target_prior)+.5*np.sum(values == target_prior))/values.size)
    cumulative = np.cumsum(counts)
    mass = rank*cumulative[-1]
    # rank==1 must terminate at the last occupied bin, not an empty tail.
    index = min(int(np.searchsorted(cumulative, mass, side='right')), int(np.flatnonzero(counts)[-1]))
    before = cumulative[index-1] if index else 0.
    fraction = (mass-before)/counts[index] if counts[index] else .5
    radius = float((index+np.clip(fraction, 0, 1))*bin_width)
    if radius <= 0:
        radius = .5*bin_width
    return radius, dict(status='OK', predicted_rank=rank, bin_index=index)


def self_check():
    source = np.arange(1., 101.)
    counts = np.zeros(256)
    counts[20:120] = 1
    radius, diag = ordinal_transport(source, 50.5, counts)
    assert np.isclose(radius, 3.5) and diag['predicted_rank'] == .5
    assert ordinal_transport(source*7+2, 50.5*7+2, counts)[0] == radius
    assert ordinal_transport(source, 50.5, np.zeros(256))[0] is None
    assert np.isclose(ordinal_transport(source, 200., counts)[0], 6.)
    assert np.isclose(ordinal_transport(source, .5, counts)[0], 1.)
    sensor = np.array([[.03, .04, np.nan, -1., 12.8, .08]])
    zones = np.array([[0, 0, 0, 0, 0, 1]])
    hist = coarse_histograms(sensor, zones)
    assert hist.sum() == 3 and hist[0, 0] == 2 and hist[1, 1] == 1
    assert np.array_equal(hist, coarse_histograms(sensor[:, ::-1], zones[:, ::-1]))
    return dict(status='PASS', checks=['monotone prediction transform invariance',
        'missing histogram abstains', 'invalid sensor samples excluded',
        'unordered coarse aggregates erase within-cell pixel correspondence',
        'rank endpoints stay inside occupied histogram extent'])


def evaluate_ordinal(out):
    if (out / 'ordinal-result.json').exists():
        raise FileExistsError('Preserve completed results')
    selection = read(out / 'selection.json')
    cpu_backend(out)
    original_ids = {r['id'] for r in read(PRIOR / 'selection.json')}
    save(out / 'ordinal-plan.json', dict(
        method='Local empirical predicted radial CDF -> unlabelled 5cm coarse-range histogram quantile',
        selection_sha256=sha(out / 'selection.json'),
        code_sha256=sha(Path(__file__)),
        role='EXPLORE on all original eligible records; no target truth in rank mapping',
        bins='256 bins, uniform-pixel geometry, [0,12.8m), not photon or hardware histogram',
        comparisons=['depthpro','zone_q10','strongest_mode','rgb_guided_mode'],
        selection_policy='All arms reported; no threshold selected from resulting errors',
    ))
    source = out/'source-ordinal'; source.mkdir(exist_ok=True)
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(out / 'ordinal-check.json', self_check())
    manifest, cameras, _ = context()
    by_id = {r['id']: r for r in manifest}
    groups = {}
    for e in selection:
        groups.setdefault(e['frame_id'], []).append(e)
    ledger = []
    for frame_id, events in groups.items():
        row, camera = by_id[frame_id], cameras[frame_id]
        ref = reference_frame(ROOT, row, camera)
        zones = zone_map(camera)
        with np.load(CACHE/'predictions/native'/f'{frame_id}.npz', allow_pickle=False) as cached:
            predicted = cached['native_depth']
        predicted_radial = predicted/ref['optical_z_per_radial']
        histograms = coarse_histograms(ref['radial'], zones)
        for event in events:
            p = extract_boundary(predicted, ref['lateral_factor'], zones, event['zone_id'])
            arms = ('depthpro','zone_q10','strongest_mode','rgb_guided_mode','ordinal_transport')
            estimates = dict.fromkeys(arms)
            details = dict(status=p['status'])
            if p['status'] == 'OK':
                y, xhalf = p['edge_pixel']; x = int(np.floor(xhalf))
                optical_factor = float(np.mean(ref['optical_z_per_radial'][y,x:x+2]))
                factor = p['side']*p['lateral_factor']*optical_factor
                prior = p['foreground_depth_m']/optical_factor
                cell = zones == event['zone_id']
                distribution = summarize_distribution(ref['radial'][cell], radial_prior=prior)
                transported, details = ordinal_transport(predicted_radial[cell], prior, histograms[event['zone_id']])
                radii = dict(zone_q10=distribution['q10_radial_m'],
                             strongest_mode=distribution['strongest_radial_m'],
                             rgb_guided_mode=distribution['guided_radial_m'],
                             ordinal_transport=transported)
                estimates['depthpro'] = p['clearance_m']
                estimates.update({a:float(factor*r-.30) if r is not None else None for a,r in radii.items()})
                details.update(radial_factor=factor, prior_radial_m=prior, radii=radii)
            # Ground-truth join occurs only after every object-blind estimate.
            record = {k:v for k,v in event.items() if k != 'support_yx'}
            record.update(prediction=p, estimates=estimates, details=details,
                          original_record=event['id'] in original_ids,
                          errors_m={a: v-event['gt_clearance_m'] if v is not None else None for a,v in estimates.items()})
            ledger.append(record)
        print('evaluated ordinal', frame_id, len(ledger), '/', len(selection), flush=True)
        del ref, zones, predicted_radial, predicted
    save(out / 'ordinal-ledger.json', ledger)
    result = dict(status='COMPLETE', n=len(ledger), scenes=len(groups),
        summaries={subset:{arm:summarize(rows,arm) for arm in arms}
                   for subset,rows in [('all',ledger),('original15',[r for r in ledger if r['original_record']]),
                                       ('additional37',[r for r in ledger if not r['original_record']])]},
        paired={a:paired_difference(ledger,'ordinal_transport',a) for a in arms if a != 'ordinal_transport'},
        role='consumed synthetic repeated-view diagnostic, not fresh confirmation')
    result['scenes'] = len({r['scene'] for r in ledger})
    save(out / 'ordinal-result.json', result)
    lines = ['# 原候选全量边缘上的局部排序传递', '',
             '从 RGB 深度的格内排序推算无身份粗格直方图分位；52条边缘/19帧/8场景，含旧15例。'
             '重复视图检查，不是新场景确认；保留已知目标格/地面/干净边缘条件。', '',
             '|范围|方法|≤2cm|P50/P95 cm|', '|---|---|---|---|']
    for subset, summaries in result['summaries'].items():
        for arm,s in summaries.items():
            q=s['absolute_error_cm_quantiles']
            lines.append(f"|{subset}|{arm}|{round(s['within_cm_all']['2']*s['n'])}/{s['n']}|{q['p50']:.2f}/{q['p95']:.2f}|")
    lines += ['', '距离观测为5cm均匀像素几何直方图，未经硬件标定；不能称光子直方图或真实ToF性能。',
              'GT只在各估计结束后评价；所有臂保留，未按结果修改候选或阈值。']
    (out / 'ordinal-REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps({k:{a:round(s['within_cm_all']['2']*s['n']) for a,s in v.items()}
                      for k,v in result['summaries'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','ordinal','check'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.out)
    elif args.action == 'ordinal': evaluate_ordinal(args.out)
    else: print(json.dumps(self_check()))
