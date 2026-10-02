"""Frozen-threshold context transfer on consumed Development simulator units.

Only infer the six existing context families (configs 22..57); no calibration,
training, generation, voxelization, heading scan, or sequence selection.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_novel_structure as B
import cnh_structure_space as SS
import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME

OUT = SS.WORK / 'cnh-margin-context-transfer-20261002'
SOURCE_RESULT = MC.OUT / 'result.json'
UNITS = list(range(70000, 70096))
CONFIGS = list(range(22, 58))
FAMILIES = ('pole', 'railing', 'low_beam', 'opposite_wall', 'bollards', 'vehicle')
ARMS = ('NEAR', 'M3', 'M8')
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 2026100210
RULE = ('All six context families pooled, frozen source-calibration HEAD/BODY thresholds: '
        'M3 minus NEAR contact 0-2 cm paired 96-unit bootstrap 1000 percentile 95% lower >0; '
        'contact >5 cm difference >=-0.02; merged-clear difference <=0.02; and M3 actual '
        'merged-clear rate <=0.12. All four required for SUPPORTED_AT_FROZEN_THRESHOLDS, '
        'otherwise NOT_SUPPORTED_AT_FROZEN_THRESHOLDS. M8 independently uses the same '
        'rule, no arm or family selection. Sigma=0 only; no target calibration.')


def require_plan():
    path = OUT / 'PLAN.json'
    if not path.is_file():
        raise RuntimeError('Write PLAN.json and the run record before inference/evaluation')
    plan = json.loads(path.read_text(encoding='utf8'))
    source = json.loads(SOURCE_RESULT.read_text(encoding='utf8'))
    if source.get('status') != 'COMPLETE':
        raise ValueError('Source thresholds require a COMPLETE result')
    source_hash = ME.sha(SOURCE_RESULT)
    if 'source_threshold_sha256' in plan and plan['source_threshold_sha256'] != source_hash:
        raise ValueError('Frozen source threshold result changed')
    thresholds = {a: {str(q): float(source['thresholds'][a][str(q)]) for q in (0, 1)} for a in ARMS}
    if not np.isfinite([v for d in thresholds.values() for v in d.values()]).all():
        raise ValueError('Nonfinite source threshold')
    if 'ids' in plan and plan['ids'] != UNITS:
        raise ValueError('Planned unit identities differ')
    if 'families' in plan and list(plan['families']) != list(FAMILIES):
        raise ValueError('Planned context families differ')
    if 'model_sha256' in plan:
        actual = {key: value for arm in ARMS for key, value in model_hashes(arm).items()}
        if actual != plan['model_sha256']:
            raise ValueError('Frozen model checkpoints changed from plan')
    return thresholds, plan, source_hash


def load_input():
    """Validate identities and truth without reading predictions/performance."""
    folder = B.OUT / 'data/evaluation'
    m = SS.read(folder / 'metadata.npz')
    x = np.load(folder / 'features.npy', mmap_mode='r')
    if len(x) != 96 * 58 * 5 or len(m['unit']) != len(x):
        raise ValueError('Expected 96 units x 58 configurations x 5 voxel frames')
    if set(map(int, m['unit'])) != set(UNITS):
        raise ValueError('Unexpected metadata units')
    selected, records, feature_hashes = [], [], {}
    for u in UNITS:
        path = B.OUT / 'features/evaluation' / f'unit{u}.npz'
        with np.load(path, allow_pickle=False) as z:
            d = {key: z[key] for key in ('family', 'margin', 'group', 'scene', 'frame', 'labels')}
        if len(d['family']) != 58:
            raise ValueError(f'{u}: expected 58 configurations')
        feature_hashes[str(u)] = ME.sha(path)
        for c in CONFIGS:
            family = str(d['family'][c])
            if family != FAMILIES[(c - 22) // 6]:
                raise ValueError(f'{u}/{c}: unexpected family {family}')
            ids = np.flatnonzero(d['scene'] == c)
            if not np.array_equal(d['frame'][ids], np.arange(16)):
                raise ValueError(f'{u}/{c}: original frame identity mismatch')
            labels = np.asarray(d['labels'][ids[-1], 2:4])
            group, off = int(d['group'][c]), -float(d['margin'][c])
            if group not in (0, 1) or not np.isfinite(off):
                raise ValueError(f'{u}/{c}: invalid target group or offset')
            if not np.array_equal(labels, np.eye(2, dtype=int)[group] * int(off > 0)):
                raise ValueError(f'{u}/{c}: target label != off>0 or other-height label !=0: {labels}, {off}')
            loc = np.flatnonzero((m['unit'] == u) & (m['config'] == c))
            loc = loc[np.argsort(m['frame'][loc])]
            if not np.array_equal(m['frame'][loc], SS.EVAL_FRAMES):
                raise ValueError(f'{u}/{c}: voxel frame identity mismatch')
            if not np.array_equal(m['labels'][loc], d['labels'][ids[SS.EVAL_FRAMES], 2:4]):
                raise ValueError(f'{u}/{c}: cached voxel labels differ from source features')
            if not np.all(m['family'][loc] == family):
                raise ValueError(f'{u}/{c}: cached family differs from source features')
            selected.extend(loc.tolist())
            records.append(dict(unit=u, config=c, family=family, group=group, off=off, labels=labels.tolist()))
    selected = np.asarray(selected)
    if len(selected) != 96 * 36 * 5 or len(np.unique(selected)) != len(selected):
        raise ValueError('Duplicate or missing selected voxel rows')
    provenance = dict(metadata_sha256=ME.sha(folder / 'metadata.npz'), feature_unit_sha256=feature_hashes,
                      voxel_file=str(folder / 'features.npy'), voxel_bytes=(folder / 'features.npy').stat().st_size,
                      voxel_shape=list(x.shape), source_generator_sha256=ME.sha(Path(B.__file__)),
                      source_scene_sha256=ME.sha(Path(B.__file__).with_name('cnh_novel_structure_scenes.py')))
    return x, selected, records, provenance


def model_hashes(arm):
    return {str(p.relative_to(SS.ROOT)): ME.sha(p) for p in MC.model_paths(arm)}


def cache_paths(arm):
    return OUT / f'frame_scores_{arm}.npz', OUT / f'scores_{arm}.npz', OUT / f'inference_{arm}.json'


def read_cache(arm, provenance, models):
    raw_path, score_path, marker_path = cache_paths(arm)
    if not marker_path.exists():
        if raw_path.exists() or score_path.exists():
            raise RuntimeError(f'{arm}: cache without completion marker requires inspection')
        return None
    marker = json.loads(marker_path.read_text(encoding='utf8'))
    if marker.get('status') != 'COMPLETE' or marker['input'] != provenance or marker['models'] != models:
        raise ValueError(f'{arm}: incomplete or changed-input cache; do not silently recompute')
    scores = SS.read(score_path)
    raw = SS.read(raw_path)
    if set(scores) != set(map(str, UNITS)) or set(raw) != set(scores):
        raise ValueError(f'{arm}: unexpected cache units')
    for key in scores:
        if scores[key].shape != (36, 2) or raw[key].shape != (36, 5, 2):
            raise ValueError(f'{arm}/{key}: invalid cache shape')
        expected = (raw[key].astype(float) * SS.WEIGHTS[None, :, None]).sum(1) / SS.WEIGHTS.sum()
        if not np.isfinite(expected).all() or not np.array_equal(scores[key], expected):
            raise ValueError(f'{arm}/{key}: final scores do not equal deployment smoothing')
    if marker['raw_sha256'] != ME.sha(raw_path) or marker['scores_sha256'] != ME.sha(score_path):
        raise ValueError(f'{arm}: cache hash mismatch')
    return scores


def save_npz(path, values):
    tmp = path.with_suffix('.partial.npz')
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    _, _, _ = require_plan()
    x, selected, _, provenance = load_input()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    for arm in ARMS:
        models = model_hashes(arm)
        if read_cache(arm, provenance, models) is not None:
            print('Validated COMPLETE cache; not repeating inference:', arm, flush=True)
            continue
        nets = []
        started = time.monotonic()
        try:
            for p in MC.model_paths(arm):
                net = CVR().cuda()
                net.load_state_dict(torch.load(p, map_location='cuda', weights_only=True))
                nets.append(net.eval())
            predictions = []
            with torch.no_grad():
                for start in range(0, len(selected), 128):
                    batch = SS.prep(torch, x[selected[start:start + 128]], masks)
                    predictions.append(torch.stack([n(batch) for n in nets]).mean(0).cpu().numpy())
                    if start % 2048 == 0:
                        SS.save(OUT / 'progress_infer.json', dict(arm=arm, rows=min(start + 128, len(selected)),
                                                                total_rows=len(selected), elapsed_s=time.monotonic() - started))
            raw = np.concatenate(predictions).reshape(96, 36, 5, 2)
            if not np.isfinite(raw).all() or model_hashes(arm) != models:
                raise ValueError('Nonfinite predictions or model changed during inference')
            smooth = (raw.astype(float) * SS.WEIGHTS[None, None, :, None]).sum(2) / SS.WEIGHTS.sum()
            raw_path, score_path, marker = cache_paths(arm)
            save_npz(raw_path, {str(u): raw[i] for i, u in enumerate(UNITS)})
            save_npz(score_path, {str(u): smooth[i] for i, u in enumerate(UNITS)})
            SS.save(marker, dict(status='COMPLETE', arm=arm, input=provenance, models=models,
                                 raw_sha256=ME.sha(raw_path), scores_sha256=ME.sha(score_path),
                                 evaluator_sha256=ME.sha(__file__), elapsed_s=time.monotonic() - started,
                                 device=torch.cuda.get_device_name(), seeds=5, frames=SS.EVAL_FRAMES.tolist(),
                                 weights=SS.WEIGHTS.tolist(), configuration_order=CONFIGS))
            print('COMPLETE inference:', arm, flush=True)
        finally:
            nets.clear()
            torch.cuda.empty_cache()


def comparison(differences, actual_clear):
    shallow = differences['contact_0_2cm']['ci95'][0]
    deep = differences['contact_gt5cm']['delta']
    clear = differences['clear_merged']['delta']
    checks = dict(shallow_ci_lower_gt_zero=shallow is not None and shallow > 0,
                  deep_drop_le_2pp=deep is not None and deep >= -.02,
                  clear_increase_le_2pp=clear is not None and clear <= .02,
                  actual_clear_le_12pct=actual_clear is not None and actual_clear <= .12)
    return dict(verdict='SUPPORTED_AT_FROZEN_THRESHOLDS' if all(checks.values()) else
                'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS', checks=checks)


def evaluate_arrays(records, scores, thresholds):
    unit_index, offsets, labels, families, groups, values = [], [], [], [], [], []
    for row in records:
        u, c = row['unit'], row['config'] - 22
        for q in (0, 1):
            unit_index.append(u - UNITS[0])
            offsets.append(row['off'] if q == row['group'] else np.nan)
            labels.append(row['labels'][q]); families.append(row['family']); groups.append(q)
            values.append([scores[a][str(u)][c, q] for a in ARMS])
    unit_index, groups, values, families = map(np.asarray, (unit_index, groups, values, families))
    categories = ME.classifications(offsets, labels)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.asarray([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(BOOTSTRAPS)])
    alarms = {a: values[:, ai] >= np.asarray([thresholds[a][str(q)] for q in groups]) for ai, a in enumerate(ARMS)}
    statistics, comparisons = {}, {}
    for scope in ('pooled',) + FAMILIES:
        mask = np.ones(len(groups), bool) if scope == 'pooled' else families == scope
        statistics[scope], rates = {}, {}
        for a in ARMS:
            statistics[scope][a], rates[a] = {}, {}
            for name, membership in categories.items():
                statistics[scope][a][name], rates[a][name] = ME.summarize(membership * mask, alarms[a], unit_index, boot)
        comparisons[scope] = {}
        for a in ('M3', 'M8'):
            differences = {}
            for name in categories:
                va, vn = statistics[scope][a][name]['rate'], statistics[scope]['NEAR'][name]['rate']
                differences[name] = dict(delta=va - vn if va is not None and vn is not None else None,
                                         ci95=ME.ci95(rates[a][name] - rates['NEAR'][name]))
            comparisons[scope][a] = dict(differences=differences)
            if scope == 'pooled':
                comparisons[scope][a].update(comparison(differences, statistics[scope][a]['clear_merged']['rate']))
            else:
                comparisons[scope][a]['role'] = 'DESCRIPTIVE_ONLY; no family selection'
    return dict(status='COMPLETE', verdict=comparisons['pooled']['M3']['verdict'], rule=RULE,
                statistics=statistics, comparisons=comparisons, thresholds=thresholds,
                n=dict(evaluation_units=96, configurations_per_unit=36, target_scenes=len(records), query_rows=len(groups)),
                units=UNITS, families=FAMILIES, configuration_order=CONFIGS,
                bootstrap=dict(draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED, resampling='paired whole evaluation units',
                               conditioning='fixed source thresholds; no source calibration uncertainty resampling'),
                limits=['Consumed Development simulator units and six existing background structures; not fresh confirmation',
                        'Frozen source thresholds; no target calibration, training, rendering or voxelization',
                        'Sigma=0 final-frame scores only; no heading scan, sequence or hardware claim',
                        'Family descriptions cannot select a favorable subset; all six are in the primary comparison',
                        'Old failed verdicts and prior fresh-unit results remain unchanged'])


def write_report(result):
    lines = [result['verdict'], '', '# 冻结阈值跨背景结构迁移', '',
             '96个既有模拟单位70000–70095，每单位六背景族各6场景，共3456目标、6912查询行。',
             '直接沿用新单位确认的NEAR/M3/M8各HEAD/BODY阈值，不做目标域校准；固定五种子与末5帧1/2/4/8/16平滑。',
             '主比较合并全部六族；1000次配对单位bootstrap95%区间，条件于源阈值。浅擦碰差区间下界>0、深擦碰差≥−2pp、清晰差≤2pp、该臂实际清晰≤12%四项同时通过才支持。', '',
             '| 臂 | 判读 | 浅擦碰差95%区间 | 深擦碰差 | 清晰差 | 实际清晰率 |',
             '| --- | --- | --- | --- | --- | --- |']
    for a in ('M3', 'M8'):
        c = result['comparisons']['pooled'][a]; d = c['differences']; s = d['contact_0_2cm']
        lines.append(f"| {a} | {c['verdict']} | {ME.pp(s['delta'])} [{ME.pp(s['ci95'][0])}, {ME.pp(s['ci95'][1])}] | {ME.pp(d['contact_gt5cm']['delta'])} | {ME.pp(d['clear_merged']['delta'])} | {ME.percentage(result['statistics']['pooled'][a]['clear_merged']['rate'])} |")
    for scope in ('pooled',) + FAMILIES:
        lines += ['', f'## {scope}：报警数/分母、率及95%区间', '',
                  '| 指标 | NEAR | M3 | M8 |', '| --- | --- | --- | --- |']
        for category in result['statistics'][scope]['NEAR']:
            lines.append('| ' + category + ' | ' + ' | '.join(ME.metric_text(result['statistics'][scope][a][category]) for a in ARMS) + ' |')
        lines += ['', '| 配对差 | M3−NEAR | M8−NEAR |', '| --- | --- | --- |']
        for category in result['statistics'][scope]['NEAR']:
            text = []
            for a in ('M3', 'M8'):
                d = result['comparisons'][scope][a]['differences'][category]
                text.append(f"{ME.pp(d['delta'])} [{ME.pp(d['ci95'][0])}, {ME.pp(d['ci95'][1])}]")
            lines.append('| ' + category + ' | ' + ' | '.join(text) + ' |')
    lines += ['', '## 解释边界', '',
              '本轮复用已消费Development旧模拟单位和六种背景结构，不是fresh confirmation、保护测试或实机证据。各族只作描述，不据此选点。',
              '擦身身体外0–10cm只报告、不计误报；清晰为同高度身体外>10cm目标与另一高度label0查询的并集。只用σ0；旧判读不变。',
              '完整阈值、单位、分母、区间、数据和模型哈希见result.json。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    thresholds, _, source_hash = require_plan()
    if (OUT / 'result.json').exists():
        previous = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        if previous.get('status') == 'COMPLETE':
            print('Existing COMPLETE result; evaluation not repeated:', previous['verdict'], flush=True)
            return
        raise RuntimeError('Incomplete result requires inspection, not overwrite')
    x, _, records, provenance = load_input()
    del x
    scores, hashes = {}, {}
    for a in ARMS:
        hashes[a] = model_hashes(a)
        scores[a] = read_cache(a, provenance, hashes[a])
        if scores[a] is None:
            raise RuntimeError(f'Missing COMPLETE inference for {a}')
    result = evaluate_arrays(records, scores, thresholds)
    result['provenance'] = dict(input=provenance, model_sha256=hashes,
        scores_sha256={a: ME.sha(cache_paths(a)[1]) for a in ARMS},
        source_threshold_result=str(SOURCE_RESULT), source_threshold_sha256=source_hash,
        plan_sha256=ME.sha(OUT / 'PLAN.json'), evaluator_sha256=ME.sha(__file__),
        metric_helper_sha256=ME.sha(Path(ME.__file__)), preprocessing_sha256=ME.sha(Path(SS.__file__)))
    write_report(result)
    SS.save(OUT / 'result.json', result)
    print(result['verdict'], flush=True)
    for a in ('M3', 'M8'):
        print(a, json.dumps(result['comparisons']['pooled'][a]), flush=True)


def check():
    """Synthetic only: boundaries, all four guards, pooled and paired statistics."""
    d = {name: dict(delta=value, ci95=[.001, .1]) for name, value in
         [('contact_0_2cm', .02), ('contact_gt5cm', -.02), ('clear_merged', .02)]}
    assert comparison(d, .12)['verdict'] == 'SUPPORTED_AT_FROZEN_THRESHOLDS'
    assert comparison(d, .12001)['verdict'] == 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'
    for key, value in [('contact_gt5cm', -.02001), ('clear_merged', .02001)]:
        other = {k: dict(v) for k, v in d.items()}; other[key]['delta'] = value
        assert comparison(other, .1)['verdict'] == 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'
    other = {k: dict(v) for k, v in d.items()}; other['contact_0_2cm']['ci95'] = [0, .1]
    assert comparison(other, .1)['verdict'] == 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'
    cats = ME.classifications([-.11, -.10, 0, .02, .05, .051, np.nan], [0, 0, 0, 1, 1, 1, 0])
    assert cats['clear_merged'].tolist() == [1, 0, 0, 0, 0, 0, 1]
    records, scores = [], {a: {} for a in ARMS}
    for u in UNITS:
        for a in ARMS:
            scores[a][str(u)] = np.zeros((36, 2))
        for c in CONFIGS:
            off = [-.11, -.10, .01, .04, .08, 0][(c - 22) % 6]
            q = (u + c) % 2
            records.append(dict(unit=u, config=c, family=FAMILIES[(c - 22) // 6], off=off, group=q,
                                labels=(np.eye(2, dtype=int)[q] * int(off > 0)).tolist()))
            for a in ARMS:
                scores[a][str(u)][c - 22, q] = int(off > 0)
    r = evaluate_arrays(records, scores, {a: {'0': .5, '1': .5} for a in ARMS})
    assert r['verdict'] == 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'
    assert r['statistics']['pooled']['NEAR']['contact_0_2cm']['denominator'] == 576
    assert r['statistics']['pooled']['NEAR']['clear_merged']['denominator'] == 4032
    for family in ('pooled',) + FAMILIES:
        for category in r['comparisons'][family]['M3']['differences'].values():
            assert category['delta'] == 0 and category['ci95'] == [0., 0.]
    print('PASS synthetic boundaries, four frozen guards, 96-unit pairing and six-family pooling; no real scores read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'infer', 'evaluate'))
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    else:
        started = time.monotonic()
        require_plan()
        try:
            infer() if args.stage == 'infer' else evaluate()
            SS.save(OUT / f'terminal_{args.stage}.json', dict(status='COMPLETE', elapsed_s=time.monotonic() - started,
                                                           source_sha256=ME.sha(__file__)))
        except BaseException as error:
            SS.save(OUT / f'terminal_{args.stage}.json', dict(status='FAILED', error=repr(error), elapsed_s=time.monotonic() - started))
            raise
