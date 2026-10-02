"""Frozen frame-history representation inference and nominal sequence evaluation.

Only the new models receive normal source48 calibration. Retained M3 uses its
old raw scores and threshold. Inference consumes existing old3 and per-exposure caches; no projection or training.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE
import cnh_yaw_envelope_evaluate as YE
import cnh_yaw_source_transfer_evaluate as ST

OUT = MC.SS.WORK/'cnh-frame-history-20261002'
NEW_ARMS = ('AGG', 'HIST')
ARMS = ('M3', 'AGG', 'HIST')
UNITS = MC.SPLITS['calib']+MC.SPLITS['evaluation']
FRAMES = np.arange(3, 16)
M3_THRESHOLD = .8557642486787612
BOOT_SEED = 2026100223
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')
PRIMARY_GROUP = 'same_height_nominal_outside10_20cm'


def file_snapshot(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stable_hash(path):
    before = file_snapshot(path)
    digest = ME.sha(path)
    if before != file_snapshot(path):
        raise ValueError('Input changed while hashing: '+str(path))
    return digest, before


def verify_inputs(receipt):
    """SHA small evidence; bulk bytes were hashed once and are stat-guarded."""
    YE.verify_hashes(receipt['input_sha256'])
    for path, expected in receipt['bulk_stat'].items():
        if file_snapshot(path) != expected:
            raise ValueError('Bulk input changed since full hash: '+path)


def history_inputs():
    plan_sha = ME.sha(OUT/'PLAN.json')
    aggregate_path, request_path = OUT/'cache_receipt.json', OUT/'request.json'
    aggregate = OE.read(aggregate_path)
    request_sha = ME.sha(request_path)
    if aggregate.get('status') != 'COMPLETE' or aggregate.get('plan_sha256') != plan_sha or aggregate.get('request_sha256') != request_sha:
        raise ValueError('History cache aggregate is incomplete or belongs to another request')
    hashes = {str(p): ME.sha(p) for p in (aggregate_path, request_path)}
    request = OE.read(request_path)
    YE.verify_hashes(request['inputs']['source_sha256'])
    hashes.update(request['inputs']['source_sha256'])
    bulk, snapshots = {}, {}
    for split in ('calib', 'evaluation'):
        if aggregate['splits'][split]['units'] != MC.SPLITS[split]:
            raise ValueError('History cache split identity changed')
        for unit in MC.SPLITS[split]:
            folder = OUT/'cache'/split/f'unit{unit}'
            receipt_path = folder/'receipt.json'
            digest = ME.sha(receipt_path)
            if aggregate['unit_receipt_sha256'].get(f'{split}/unit{unit}/receipt.json') != digest:
                raise ValueError('Unit receipt differs from aggregate')
            receipt = OE.read(receipt_path)
            expected = dict(status='COMPLETE', split=split, unit=unit, plan_sha256=plan_sha,
                request_sha256=request_sha, shape=[40, 13, 8, 24, 17, 33], frames=FRAMES.tolist(),
                configs=40, slots=8, old3_windows_bit_exact=520,
                latest_exact=True, coverage_recovery_exact=True,
                mass_dtype='float16', coverage_dtype='uint8')
            if any(receipt.get(k) != v for k, v in expected.items()):
                raise ValueError('History receipt contract mismatch: '+str(receipt_path))
            hashes[str(receipt_path)] = digest
            for name, dtype in (('mass.npy', np.float16), ('coverage.npy', np.uint8)):
                path = folder/name
                bulk[str(path)], snapshots[str(path)] = stable_hash(path)
                if receipt['output_sha256'].get(name) != bulk[str(path)]:
                    raise ValueError('History cache SHA mismatch: '+str(path))
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    if array.shape != (40, 13, 8, 24, 17, 33) or array.dtype != dtype:
                        raise ValueError('History array shape/dtype mismatch: '+str(path))
                finally:
                    array._mmap.close()
    return hashes, bulk, snapshots


def validate_plan(plan):
    expected = dict(run='CNH_FRAME_HISTORY_20261002', calibration_units=MC.SPLITS['calib'],
        evaluation_units=MC.SPLITS['evaluation'], decision_frames=FRAMES.tolist(),
        evaluation_configurations_per_unit=40, arms=list(ARMS), train_arms=list(NEW_ARMS),
        seeds=list(range(5)), frozen_M3_threshold=M3_THRESHOLD)
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('Frozen PLAN differs from evaluator: '+key)
    if plan['bootstrap']['replicates'] != N_BOOT or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('Frozen bootstrap differs from evaluator')
    YE.verify_hashes(plan['prior_sha256'])


def row_indices(metadata, units, allowed_frames):
    """Map existing chunk metadata to [unit,config,frame] without assuming order."""
    unit_map = {u: i for i, u in enumerate(units)}
    columns = [np.asarray(metadata[k]) for k in ('unit', 'config', 'frame')]
    if any(c.ndim != 1 for c in columns) or len({len(c) for c in columns}) != 1:
        raise ValueError('Voxel metadata columns must be equal-length vectors')
    u, c, f = columns
    if not np.isin(u, units).all() or not np.isin(f, allowed_frames).all() or (c < 0).any() or (c >= 40).any():
        raise ValueError('Voxel metadata unit/config/frame out of range')
    if any(not np.array_equal(v, v.astype(np.int64)) for v in columns):
        raise ValueError('Voxel identity fields must be integer valued')
    pos = np.asarray([unit_map[int(x)] for x in u])
    flat = (pos*40+c.astype(int))*13+(f.astype(int)-3)
    if len(np.unique(flat)) != len(flat):
        raise ValueError('Duplicate unit/config/frame within a voxel chunk')
    return pos, c.astype(int), f.astype(int)-3, flat


def inference_inputs():
    plan_path, receipt_path, request_path = OUT/'PLAN.json', OUT/'training_receipt.json', OUT/'training_request.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    validate_plan(plan)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path) or receipt.get('request_sha256') != ME.sha(request_path):
        raise ValueError('Training is incomplete or receipt plan/request hash differs')
    declared = {k.replace('\\', '/'): v for k, v in receipt['models_sha256'].items()}
    expected = {f'models/{a}/model_seed{s}.pt' for a in NEW_ARMS for s in range(5)}
    if set(declared) != expected:
        raise ValueError('Training receipt must name five models per AGG/HIST arm')
    hashes = {str(p): ME.sha(p) for p in (plan_path, receipt_path, request_path)}
    hashes.update(plan['prior_sha256'])
    request = OE.read(request_path)
    if receipt['source_sha256'] != request['source_sha256']:
        raise ValueError('Training receipt and request bind different sources')
    YE.verify_hashes(receipt['source_sha256'])
    hashes.update(receipt['source_sha256'])
    for relative, digest in declared.items():
        path = OUT/relative
        if ME.sha(path) != digest:
            raise ValueError('Deployment model changed: '+relative)
        hashes[str(path)] = digest
    hashes[str(Path(__file__).resolve())] = ME.sha(__file__)
    from cnh_frame_history_train import __file__ as train_source
    hashes[str(Path(train_source).resolve())] = ME.sha(train_source)
    for source in (SE.__file__, OE.__file__, ST.__file__, XE.__file__, MC.__file__, ME.__file__, YE.__file__):
        hashes[str(Path(source).resolve())] = ME.sha(source)
    parts, snapshots, seen = [], {}, np.zeros(len(UNITS)*40*13, dtype=bool)
    for split in ('calib', 'evaluation'):
        for early in (True, False):
            allowed = np.arange(3, 11) if early else np.arange(11, 16)
            folder = MC.OUT/'data'/(split+('_early' if early else ''))
            files = sorted(folder.glob('features_c*.npy'))
            if not files or any('.partial' in p.name for p in files):
                raise ValueError('Missing/incomplete retained voxel chunks: '+str(folder))
            for feature in files:
                metadata = feature.with_name(feature.name.replace('features_', 'metadata_').replace('.npy', '.npz'))
                with np.load(metadata, allow_pickle=False) as cache:
                    m = {k: cache[k] for k in ('unit', 'config', 'frame')}
                if not np.isin(m['unit'], MC.SPLITS[split]).all():
                    raise ValueError('Voxel split contains a unit from another split')
                _, _, _, flat = row_indices(m, UNITS, allowed)
                if seen[flat].any():
                    raise ValueError('Duplicate unit/config/frame across retained chunks')
                seen[flat] = True
                array = np.load(feature, mmap_mode='r', allow_pickle=False)
                try:
                    if array.shape != (len(flat), 3, 24, 17, 33) or array.dtype != np.float16:
                        raise ValueError('Retained voxel feature shape/dtype does not match metadata')
                finally:
                    array._mmap.close()
                hashes[str(feature)], snapshots[str(feature)] = stable_hash(feature)
                hashes[str(metadata)] = ME.sha(metadata)
                parts.append(dict(split=split, feature=str(feature), metadata=str(metadata), allowed_frames=allowed.tolist()))
    if not seen.all():
        raise ValueError('Retained chunks do not cover every144x40x13 inference slot')
    bulk = {p: hashes.pop(p) for p in list(hashes) if p.endswith('.npy')}
    cache_hashes, cache_bulk, cache_snapshots = history_inputs()
    hashes.update(cache_hashes); bulk.update(cache_bulk)
    snapshots.update(cache_snapshots)
    request = OE.read(OUT/'request.json')
    actual = {str(Path(p).resolve()): digest for p, digest in {**hashes, **bulk}.items()}
    for split in ('calib', 'evaluation'):
        for part in request['inputs']['old_caches'][split]:
            for record in part.values():
                if actual.get(str(Path(record['path']).resolve())) != record['sha256']:
                    raise ValueError('Retained old3/metadata differs from history-cache parity inputs')
    verify_inputs(dict(input_sha256=hashes, bulk_stat=snapshots))
    return plan, receipt, hashes, bulk, snapshots, parts


def infer():
    receipt_path = OUT/'scores_receipt.json'
    if receipt_path.exists():
        prior = OE.read(receipt_path)
        if prior.get('status') != 'COMPLETE':
            raise ValueError('Inspect incomplete inference receipt before resuming')
        verify_inputs(prior)
        for name, digest in prior['output_sha256'].items():
            if ME.sha(OUT/name) != digest:
                raise ValueError('Completed score cache changed: '+name)
        print('Existing COMPLETE scores verified; no repeated inference', flush=True)
        return prior
    plan, training, hashes, bulk, snapshots, parts = inference_inputs()
    import torch
    from cnh_frame_history_train import HistoryCVR, HistoryStore, prepare_old, prepare_history
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Validated CUDA inference runtime is required')
    started = time.monotonic()
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    outputs, receipts = {}, {}
    stores = {}
    try:
        for split in ('calib', 'evaluation'):
            stores[split] = HistoryStore(split, units=MC.SPLITS[split], verify_hashes=False)
        for arm in NEW_ARMS:
            output, arm_receipt_path = OUT/f'frame_scores_{arm}.npz', OUT/f'scores_receipt_{arm}.json'
            if output.exists() or arm_receipt_path.exists():
                prior = OE.read(arm_receipt_path)
                if prior.get('status') != 'COMPLETE' or prior['input_sha256'] != hashes or prior.get('bulk_sha256') != bulk or prior.get('bulk_stat') != snapshots or ME.sha(output) != prior['output_sha256']:
                    raise ValueError('Partial/changed arm output cannot be silently replaced: '+arm)
                outputs[output.name], receipts[arm] = prior['output_sha256'], prior
                print('Reuse completed', arm, flush=True)
                continue
            models = []
            net = batch = prediction = state = None
            try:
                for seed in range(5):
                    net = HistoryCVR().cuda()
                    state = torch.load(OUT/'models'/arm/f'model_seed{seed}.pt', weights_only=True, map_location='cuda')
                    if set(state) != set(net.state_dict()):
                        raise ValueError('History checkpoint key mismatch: '+arm)
                    net.load_state_dict(state, strict=True)
                    models.append(net.eval())
                values = np.full((len(UNITS), 40, 13, 2), np.nan, np.float32)
                with torch.no_grad():
                    for part in parts:
                        with np.load(part['metadata'], allow_pickle=False) as data:
                            metadata = {k: data[k] for k in ('unit', 'config', 'frame')}
                        ui, ci, fi, _ = row_indices(metadata, UNITS, part['allowed_frames'])
                        array = np.load(part['feature'], mmap_mode='r', allow_pickle=False)
                        try:
                            for begin in range(0, len(array), 64):
                                end = min(begin+64, len(array))
                                old = torch.as_tensor(np.array(array[begin:end]), device='cuda')
                                batch = prepare_old(old, masks)
                                if arm == 'HIST':
                                    mass, coverage = stores[part['split']].batch(*(metadata[k][begin:end] for k in ('unit', 'config', 'frame')))
                                    history = prepare_history(arm, old, torch.as_tensor(mass, device='cuda'), torch.as_tensor(coverage, device='cuda'))
                                else:
                                    history = prepare_history(arm, old)
                                prediction = torch.stack([model(batch, history) for model in models]).mean(0).cpu().numpy()
                                values[ui[begin:end], ci[begin:end], fi[begin:end]] = prediction
                        finally:
                            array._mmap.close()
                        print('infer', arm, Path(part['feature']).parent.name, Path(part['feature']).name, round(time.monotonic()-started, 1), flush=True)
                if not np.isfinite(values).all():
                    raise ValueError('Inference did not fill every planned frame score')
                verify_inputs(dict(input_sha256=hashes, bulk_stat=snapshots))
                with output.open('xb') as file:
                    np.savez_compressed(file, **{str(u): values[i] for i, u in enumerate(UNITS)})
                arm_receipt = dict(status='COMPLETE', arm=arm, input_sha256=hashes, bulk_sha256=bulk, bulk_stat=snapshots, output_sha256=ME.sha(output),
                    units=UNITS, frames=FRAMES.tolist(), shape_per_unit=[40, 13, 2], seeds=5,
                    schema='npz keys are string unit IDs; raw mean logits before causal smoothing')
                OE.save(arm_receipt_path, arm_receipt)
                outputs[output.name], receipts[arm] = arm_receipt['output_sha256'], arm_receipt
            finally:
                net = batch = prediction = state = old = history = mass = coverage = None
                models.clear()
                torch.cuda.empty_cache()
    finally:
        for store in stores.values():
            store.close()
        masks = None
        torch.cuda.empty_cache()
    result = dict(status='COMPLETE', plan_sha256=ME.sha(OUT/'PLAN.json'), training_receipt_sha256=ME.sha(OUT/'training_receipt.json'),
        input_sha256=hashes, bulk_sha256=bulk, bulk_stat=snapshots, output_sha256=outputs, arms=receipts, elapsed_s=time.monotonic()-started,
        cache_summary=OE.read(OUT/'cache_receipt.json'),
        bulk_validation='One full SHA read at inference startup; size/mtime_ns checked between arms and during evaluation; unchanged stat is not a new content hash',
        operations=dict(existing_voxels=True, projection=False, rendering=False, training=False),
        runtime=dict(torch=torch.__version__, device=torch.cuda.get_device_name(), allocated_after_release=torch.cuda.memory_allocated()),
        evaluator_sha256=ME.sha(__file__))
    OE.save(receipt_path, result)
    print('COMPLETE frame-history frozen-model inference', flush=True)
    return result


def load_evaluation_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'scores_receipt.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    validate_plan(plan)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Inference is incomplete or belongs to a different plan')
    verify_inputs(receipt)
    g, retained, old_provenance = OE.load_inputs()
    rows_path = OE.OUT/'rows.json'
    rows = OE.read(rows_path)
    keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    if keys != list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist())):
        raise ValueError('All-object nominal row identity differs')
    g['target_group'] = np.asarray([r['target_group'] for r in rows])
    g['target_off'] = np.asarray([r['target_off'] for r in rows])
    hashes = dict(receipt['input_sha256']); hashes.update(old_provenance['input_sha256'])
    hashes.update({str(p): ME.sha(p) for p in (plan_path, receipt_path, rows_path)})
    scores = {'M3': retained['M3']}
    for arm in NEW_ARMS:
        path = OUT/f'frame_scores_{arm}.npz'
        digest = ME.sha(path)
        if receipt['output_sha256'].get(path.name) != digest:
            raise ValueError('New score cache hash changed: '+arm)
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != set(map(str, UNITS)):
                raise ValueError('New score cache has unexpected unit keys: '+arm)
            raw = np.stack([cache[str(u)] for u in UNITS])
        if raw.shape != (144, 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('New score cache shape/values differ: '+arm)
        scores[arm] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)] = digest
    prior_path = OE.OUT/'result.json'
    prior = OE.read(prior_path)
    if prior.get('status') != 'COMPLETE' or prior['cells']['M3']['threshold'] != M3_THRESHOLD:
        raise ValueError('Original M3 threshold/result changed')
    hashes[str(prior_path)] = ME.sha(prior_path)
    training = OE.read(OUT/'training_receipt.json')
    return g, scores, prior, dict(input_sha256=hashes, bulk_sha256=receipt['bulk_sha256'],
        bulk_stat=receipt['bulk_stat'], plan=plan, scores_receipt=receipt,
        training_summary=training,
        evaluator_sha256=ME.sha(__file__), helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, ST.__file__, XE.__file__)})


def comparison_checks(comparison):
    outside_ci = comparison['primary_outside_clear']['paired_unit_ci95']
    shallow_ci = comparison['contact0-2cm']['paired_unit_ci95']
    deep, clear = comparison['contact>5cm']['delta_timely'], comparison['clear']['delta_per_min']
    return dict(outside_clear_ci_upper_lt_zero=outside_ci[1] is not None and outside_ci[1] < 0,
        shallow_timely_ci_lower_ge_minus2pp=shallow_ci[0] is not None and shallow_ci[0] >= -.02,
        deep_timely_decline_le_2pp=deep is not None and deep >= -.02,
        overall_clear_increase_le_0p2_per_min=clear is not None and clear <= .2)


def analyze(g, scores):
    cal, ev = g['split'] == 'calib', g['split'] == 'evaluation'
    fields = ('unit', 'query', 'target_group', 'target_off', 'frame_ranges', 'covered', 'clear_all', 'ref_category')
    if any(g[k].shape[0] != len(ev) for k in fields):
        raise ValueError('Required geometry rows do not align with split')
    e = {key: g[key][ev] for key in fields}
    units, ui = np.unique(e['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    thresholds, calibration = {'M3': M3_THRESHOLD}, {'M3': dict(role='Retained old threshold; no new calibration')}
    for arm in NEW_ARMS:
        thresholds[arm], calibration[arm] = SE.calibrate(scores[arm], cal & g['clear_all'], 1.)
    weights = {c: (e['covered'] & (e['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = e['clear_all'].astype(float)
    groups = ST.clear_groups(e)
    cells, sampled, flags, subgroup_samples = {}, {}, {}, {}
    for arm in ARMS:
        score = scores[arm][ev]
        if score.shape != e['frame_ranges'].shape or not np.isfinite(score).all():
            raise ValueError('Score/geometry mismatch for '+arm)
        cells[arm], sampled[arm], flags[arm] = XE.summarize_cell(e, score, thresholds[arm], weights, ui, boot)
        cells[arm]['calibration'] = calibration[arm]
        cells[arm]['clear_subgroups'], subgroup_samples[arm] = ST.summarize_clear_groups(groups, flags[arm]['stopped'], ui, boot)
    comparisons, decisions = {}, {}
    for control in ('AGG', 'M3'):
        comparison = XE.paired_changes(cells, sampled, flags, weights, 'HIST', control)
        group_changes = {}
        for name, keep in groups.items():
            a, b = (cells[arm]['clear_subgroups'][name]['false_stops_per_min']['value'] for arm in ('HIST', control))
            na, ca = flags['HIST']['stopped'], flags[control]['stopped']
            group_changes[name] = dict(delta_per_min=a-b if a is not None and b is not None else None,
                paired_unit_ci95=SE.interval(subgroup_samples['HIST'][name]-subgroup_samples[control][name]),
                added_first_stops=int((keep & na & ~ca).sum()), removed_first_stops=int((keep & ca & ~na).sum()))
        comparison['clear_subgroups'] = group_changes
        comparison['primary_outside_clear'] = group_changes[PRIMARY_GROUP]
        comparisons[control] = comparison
        decisions[control] = comparison_checks(comparison)
    verdict = ('FRAME_HISTORY_SUPPORTED_DEV' if all(all(checks.values()) for checks in decisions.values())
               else 'FRAME_HISTORY_NOT_ESTABLISHED_DEV')
    return dict(status='COMPLETE', verdict=verdict, comparisons_HIST_minus_control=comparisons,
        per_control_decision_checks=decisions, cells=cells, thresholds=thresholds, calibration=calibration,
        evaluation_units=units.tolist(), clear_subgroup_counts={k: int(m.sum()) for k, m in groups.items()},
        n=dict(calibration_units=len(np.unique(g['unit'][cal])), evaluation_units=len(units), query_episodes=int(ev.sum()),
            covered=int(e['covered'].sum()), right_censored=int((~e['covered']).sum()), all_clear=int(e['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()),
            shallow_contributing_units=len(np.unique(e['unit'][weights['contact0-2cm'] > 0]))),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole evaluation units',
            pairing='one common bootstrap matrix for every arm/subgroup', conditional_on='trained checkpoints and calibrated AGG/HIST thresholds'),
        primary_group_definition='query==target_group AND -0.20<=nominal target_off<-0.10 AND all13frame clear_all; nominal metadata, not dynamic gap',
        rule='HIST outside-clear first-stops/min CI upper<0 versus BOTH AGG and fixed M3; versus each, shallow timely CI lower>=-0.02,deep point>=-0.02,overall clear point increase<=0.2/min')


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def write_report(result):
    lines = [result['verdict'], '', '# 逐帧历史表示的名义序列选择性', '',
        'AGG与HIST采用相同新增分支；AGG仅用旧聚合三通道构造伪历史，HIST保留每次历史曝光的投影质量与覆盖。各五seed均值和原五分数因果平滑；各自在48校准单位按1次/代理分钟定阈值，M3冻结。', '',
        '| 比较 | 外10–20cm清晰首停差及95%区间，次/分钟 | 浅及时差及95%区间，pp | 深及时差，pp | 整体清晰差，次/分钟 |', '| --- | --- | --- | --- | --- |']
    for control, comparison in result['comparisons_HIST_minus_control'].items():
        p, s, d, c = (comparison[k] for k in ('primary_outside_clear', 'contact0-2cm', 'contact>5cm', 'clear'))
        lines.append(f"| HIST−{control} | {number(p['delta_per_min'])} [{number(p['paired_unit_ci95'][0])}, {number(p['paired_unit_ci95'][1])}] | {number(s['delta_timely'],100)} [{number(s['paired_unit_ci95'][0],100)}, {number(s['paired_unit_ci95'][1],100)}] | {number(d['delta_timely'],100)} | {number(c['delta_per_min'])} |")
    lines += ['', '| 指标 | M3 | AGG | HIST |', '| --- | --- | --- | --- |']
    for category in CATEGORIES:
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][arm]['metrics'][category], category == 'clear', category == 'pass0-10cm') for arm in ARMS)+' |')
    lines += ['', '| 清晰子组 | M3 | AGG | HIST |', '| --- | --- | --- | --- |']
    for group in result['clear_subgroup_counts']:
        texts = []
        for arm in ARMS:
            metric = result['cells'][arm]['clear_subgroups'][group]
            rate = metric['false_stops_per_min']
            texts.append(f"{metric['first_stops']}/{metric['clear_minutes']:.2f}={number(rate['value'])} [{number(rate['ci95'][0])}, {number(rate['ci95'][1])}]（n={metric['n']}）")
        lines.append('| '+group+' | '+' | '.join(texts)+' |')
    lines += ['', '| 模型/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
    for arm in ARMS:
        for category in OE.CONTACTS:
            metric = result['cells'][arm]['metrics'][category]['median_lead_to_0p5m_s']
            lines.append(f"| {arm}/{category} | {number(metric['value'])} [{number(metric['ci95'][0])}, {number(metric['ci95'][1])}] |")
    lines += ['', '阈值与校准：', '']
    for arm in ARMS:
        lines.append(f"- {arm}: {result['thresholds'][arm]!r}; {result['calibration'][arm]}")
    lines += ['', '配对补回/丢失及逐比较判读：', '', '```json',
        json.dumps(result['comparisons_HIST_minus_control'], ensure_ascii=False, indent=2), '```', '',
        json.dumps(result['per_control_decision_checks'], ensure_ascii=False), '',
        '分母与删失：'+json.dumps(result['n'], ensure_ascii=False), '',
        '1000次整单位共同bootstrap，seed2026100223；区间条件于固定检查点和已校准阈值。两个比较都必须通过，不新增评估集绝对1次/分钟门槛。', '',
        'HIST同时引入历史质量变化、覆盖和年龄结构，收益不能独立归因为时间一致性或更多光子。仅有归一化z1，负质量不是自由空间证据；两通道adapter仍是容量瓶颈。训练名义偏移簇不代表连续10–20cm支持。', '',
        '已消费Development、同名义仿真源，非新盲测、跨yaw或硬件安全证据。清晰2.6秒/查询包含首停后暴露，13采样点清晰不保证帧间连续清晰；不是现场步行分钟。右删失不作漏停；提前量条件于已报警，按0.8m/s相对0.5m换算，不是人已安全停止。', '',
        '旧M3 38点复现：'+json.dumps(result['nominal_M3_point_reproduction'], ensure_ascii=False), '',
        '训练、缓存、推理耗时及字节绑定见result.json provenance；大输入启动时一次全SHA，后续使用size/mtime_ns快照防止常规并发修改，不声称快照等同重新哈希。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    prior_output = OE.read(path) if path.exists() else None
    if prior_output is not None:
        if prior_output.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete result requires inspection')
        verify_inputs(prior_output['provenance'])
        print('Existing COMPLETE result verified; evaluation not repeated', flush=True)
        return prior_output
    g, scores, prior, provenance = load_evaluation_inputs()
    result = analyze(g, scores)
    result['nominal_M3_point_reproduction'] = ST.nominal_parity({'0|BASE': result['cells']['M3']}, prior)
    result['old_M3_vs_NEAR_verdict_retained'] = prior['verdict']
    result['provenance'] = provenance
    verify_inputs(provenance)
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], json.dumps(result['per_control_decision_checks']), flush=True)
    return result


def check():
    # Out-of-order metadata must restore exact unit/config/frame slots; duplicates fail.
    meta = dict(unit=np.array([95001, 95000]), config=np.array([20, 0]), frame=np.array([15, 3]))
    u, c, f, flat = row_indices(meta, [95000, 95001], FRAMES)
    assert u.tolist() == [1, 0] and c.tolist() == [20, 0] and f.tolist() == [12, 0]
    try:
        row_indices(dict(unit=np.array([95000]*2), config=np.array([0]*2), frame=np.array([3]*2)), [95000], FRAMES)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate frame route was not rejected')
    comparison = {'primary_outside_clear': {'paired_unit_ci95': [-.2, -.001]},
        'contact0-2cm': {'paired_unit_ci95': [-.02, .1]}, 'contact>5cm': {'delta_timely': -.02}, 'clear': {'delta_per_min': .2}}
    assert all(comparison_checks(comparison).values())
    comparison['primary_outside_clear']['paired_unit_ci95'][1] = 0.
    assert not comparison_checks(comparison)['outside_clear_ci_upper_lt_zero']
    # Full statistic path: original M3 is not calibrated; evaluation scores cannot move new thresholds.
    ref = np.array(['clear', 'clear', 'contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'clear', 'censored'])
    g = dict(split=np.array(['calib']*2+['evaluation']*6), unit=np.array([95000, 95001, 96000, 96000, 96001, 96001, 96001, 96001]),
        query=np.array([0, 0, 0, 0, 0, 0, 1, 0]), target_group=np.zeros(8, int),
        target_off=np.array([-.15, -.15, .01, .1, -.05, -.15, .1, .01]), frames=FRAMES,
        covered=np.array([True]*7+[False]), clear_all=ref == 'clear', ref_category=ref,
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (8, 1)))
    g['frame_ranges'][-1] += .3
    scores = {a: np.zeros((8, 13)) for a in ARMS}
    for arm in ARMS:
        scores[arm][2:5, 1] = 2.
    scores['M3'][5, 1] = scores['AGG'][5, 1] = 2.
    result = analyze(g, scores)
    assert result['thresholds']['M3'] == M3_THRESHOLD
    assert result['comparisons_HIST_minus_control']['M3']['primary_outside_clear']['delta_per_min'] < 0
    assert result['cells']['HIST']['clear_subgroups']['remaining_same_height']['false_stops_per_min']['value'] is None
    changed = {a: s.copy() for a, s in scores.items()}
    for s in changed.values():
        s[2:] += 100
    again = analyze(g, changed)
    assert again['thresholds'] == result['thresholds']
    assert 'episode_draw_pairs' not in result['cells']['M3']['metrics']['contact0-2cm']
    json.dumps(result, allow_nan=False)
    print('PASS: complete routing slots/duplicate rejection, strict benefit and NI boundaries, split-only calibration, empty strata and frozen M3 threshold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'infer', 'evaluate'))
    args = parser.parse_args()
    {'check': check, 'infer': infer, 'evaluate': evaluate}[args.stage]()
