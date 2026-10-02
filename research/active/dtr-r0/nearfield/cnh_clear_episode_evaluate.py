"""Frozen nominal cached-voxel inference and clear-episode objective evaluation.

Only the new models receive normal source48 calibration. Retained M3 uses its
old raw scores and threshold. No projection, training or old-result mutation.
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

OUT = MC.SS.WORK/'cnh-clear-episode-20261002'
NEW_ARMS = ('MEAN', 'MAX')
ARMS = ('M3', 'MEAN', 'MAX')
UNITS = MC.SPLITS['calib']+MC.SPLITS['evaluation']
FRAMES = np.arange(3, 16)
M3_THRESHOLD = .8557642486787612
BOOT_SEED = 2026100222
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')
PRIMARY_GROUP = 'same_height_nominal_outside10_20cm'


def validate_plan(plan):
    expected = dict(run='CNH_CLEAR_EPISODE_20261002', calibration_units=MC.SPLITS['calib'],
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
        raise ValueError('Training receipt must name exactly five deployment CVR models for MEAN/MAX')
    hashes = {str(p): ME.sha(p) for p in (plan_path, receipt_path, request_path)}
    hashes.update(plan['prior_sha256'])
    for relative, digest in declared.items():
        path = OUT/relative
        if ME.sha(path) != digest:
            raise ValueError('Deployment model changed: '+relative)
        hashes[str(path)] = digest
    parts, seen = [], np.zeros(len(UNITS)*40*13, dtype=bool)
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
                    if len(array) != len(flat) or array.ndim != 5 or array.shape[1] != 3 or array.dtype != np.float16:
                        raise ValueError('Retained voxel feature shape/dtype does not match metadata')
                finally:
                    array._mmap.close()
                hashes[str(feature)], hashes[str(metadata)] = ME.sha(feature), ME.sha(metadata)
                parts.append(dict(feature=str(feature), metadata=str(metadata), allowed_frames=allowed.tolist()))
    if not seen.all():
        raise ValueError('Retained chunks do not cover every144x40x13 inference slot')
    return plan, receipt, hashes, parts


def infer():
    receipt_path = OUT/'scores_receipt.json'
    if receipt_path.exists():
        prior = OE.read(receipt_path)
        if prior.get('status') != 'COMPLETE':
            raise ValueError('Inspect incomplete inference receipt before resuming')
        YE.verify_hashes(prior['input_sha256'])
        for name, digest in prior['output_sha256'].items():
            if ME.sha(OUT/name) != digest:
                raise ValueError('Completed score cache changed: '+name)
        print('Existing COMPLETE scores verified; no repeated inference', flush=True)
        return prior
    plan, training, hashes, parts = inference_inputs()
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Validated CUDA inference runtime is required')
    started = time.monotonic()
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    outputs, receipts = {}, {}
    try:
        for arm in NEW_ARMS:
            output, arm_receipt_path = OUT/f'frame_scores_{arm}.npz', OUT/f'scores_receipt_{arm}.json'
            if output.exists() or arm_receipt_path.exists():
                prior = OE.read(arm_receipt_path)
                if prior.get('status') != 'COMPLETE' or prior['input_sha256'] != hashes or ME.sha(output) != prior['output_sha256']:
                    raise ValueError('Partial/changed arm output cannot be silently replaced: '+arm)
                outputs[output.name], receipts[arm] = prior['output_sha256'], prior
                print('Reuse completed', arm, flush=True)
                continue
            models = []
            net = batch = prediction = state = None
            try:
                for seed in range(5):
                    net = CVR().cuda()
                    state = torch.load(OUT/'models'/arm/f'model_seed{seed}.pt', weights_only=True, map_location='cuda')
                    if set(state) != set(net.state_dict()):
                        raise ValueError('Deployment checkpoint must contain pure original CVR keys: '+arm)
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
                            for begin in range(0, len(array), 128):
                                end = min(begin+128, len(array))
                                batch = MC.SS.prep(torch, array[begin:end], masks)
                                prediction = torch.stack([model(batch) for model in models]).mean(0).cpu().numpy()
                                values[ui[begin:end], ci[begin:end], fi[begin:end]] = prediction
                        finally:
                            array._mmap.close()
                        print('infer', arm, Path(part['feature']).parent.name, Path(part['feature']).name, round(time.monotonic()-started, 1), flush=True)
                if not np.isfinite(values).all():
                    raise ValueError('Inference did not fill every planned frame score')
                YE.verify_hashes(hashes)
                with output.open('xb') as file:
                    np.savez_compressed(file, **{str(u): values[i] for i, u in enumerate(UNITS)})
                arm_receipt = dict(status='COMPLETE', arm=arm, input_sha256=hashes, output_sha256=ME.sha(output),
                    units=UNITS, frames=FRAMES.tolist(), shape_per_unit=[40, 13, 2], seeds=5,
                    schema='npz keys are string unit IDs; raw mean logits before causal smoothing')
                OE.save(arm_receipt_path, arm_receipt)
                outputs[output.name], receipts[arm] = arm_receipt['output_sha256'], arm_receipt
            finally:
                net = batch = prediction = state = None
                models.clear()
                torch.cuda.empty_cache()
    finally:
        masks = None
        torch.cuda.empty_cache()
    result = dict(status='COMPLETE', plan_sha256=ME.sha(OUT/'PLAN.json'), training_receipt_sha256=ME.sha(OUT/'training_receipt.json'),
        input_sha256=hashes, output_sha256=outputs, arms=receipts, elapsed_s=time.monotonic()-started,
        operations=dict(existing_voxels=True, projection=False, rendering=False, training=False),
        runtime=dict(torch=torch.__version__, device=torch.cuda.get_device_name(), allocated_after_release=torch.cuda.memory_allocated()),
        evaluator_sha256=ME.sha(__file__))
    OE.save(receipt_path, result)
    print('COMPLETE clear-episode frozen-model inference', flush=True)
    return result


def load_evaluation_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'scores_receipt.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    validate_plan(plan)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Inference is incomplete or belongs to a different plan')
    YE.verify_hashes(receipt['input_sha256'])
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
    support_path = OUT/'train_support.json'
    support = OE.read(support_path)
    if support.get('status') != 'COMPLETE':
        raise ValueError('Training clear-support description is incomplete')
    hashes[str(support_path)] = ME.sha(support_path)
    return g, scores, prior, dict(input_sha256=hashes, plan=plan, scores_receipt=receipt,
        training_summary={k: training[k] for k in ('clear_support', 'recipe', 'runtime')},
        train_support=support,
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
    for control in ('MEAN', 'M3'):
        comparison = XE.paired_changes(cells, sampled, flags, weights, 'MAX', control)
        group_changes = {}
        for name, keep in groups.items():
            a, b = (cells[arm]['clear_subgroups'][name]['false_stops_per_min']['value'] for arm in ('MAX', control))
            na, ca = flags['MAX']['stopped'], flags[control]['stopped']
            group_changes[name] = dict(delta_per_min=a-b if a is not None and b is not None else None,
                paired_unit_ci95=SE.interval(subgroup_samples['MAX'][name]-subgroup_samples[control][name]),
                added_first_stops=int((keep & na & ~ca).sum()), removed_first_stops=int((keep & ca & ~na).sum()))
        comparison['clear_subgroups'] = group_changes
        comparison['primary_outside_clear'] = group_changes[PRIMARY_GROUP]
        comparisons[control] = comparison
        decisions[control] = comparison_checks(comparison)
    verdict = ('CLEAR_EPISODE_SUPPORTED_DEV' if all(all(checks.values()) for checks in decisions.values())
               else 'CLEAR_EPISODE_NOT_ESTABLISHED_DEV')
    return dict(status='COMPLETE', verdict=verdict, comparisons_MAX_minus_control=comparisons,
        per_control_decision_checks=decisions, cells=cells, thresholds=thresholds, calibration=calibration,
        evaluation_units=units.tolist(), clear_subgroup_counts={k: int(m.sum()) for k, m in groups.items()},
        n=dict(calibration_units=len(np.unique(g['unit'][cal])), evaluation_units=len(units), query_episodes=int(ev.sum()),
            covered=int(e['covered'].sum()), right_censored=int((~e['covered']).sum()), all_clear=int(e['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()),
            shallow_contributing_units=len(np.unique(e['unit'][weights['contact0-2cm'] > 0]))),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole evaluation units',
            pairing='one common bootstrap matrix for every arm/subgroup', conditional_on='trained checkpoints and calibrated MEAN/MAX thresholds'),
        primary_group_definition='query==target_group AND -0.20<=nominal target_off<-0.10 AND all13frame clear_all; nominal metadata, not dynamic gap',
        rule='MAX outside-clear first-stops/min CI upper<0 versus BOTH MEAN and fixed M3; versus each, shallow timely CI lower>=-0.02,deep point>=-0.02,overall clear point increase<=0.2/min')


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def write_report(result):
    lines = [result['verdict'], '', '# 清晰序列峰值训练的名义序列选择性', '',
        'MEAN与MAX使用各自五seed纯CVR部署分数，复用原有13帧体素；各自在source48全帧清晰查询上按1次/代理分钟校准单一阈值。旧M3阈值和分数保留，旧失败不改。', '',
        '| 主比较 | 同高度外10–20cm清晰首停差及95%区间，次/分钟 | 浅及时差及95%区间，pp | 深及时差，pp | 整体清晰差，次/分钟 |', '| --- | --- | --- | --- | --- |']
    for control, comparison in result['comparisons_MAX_minus_control'].items():
        p, s, d, c = (comparison[k] for k in ('primary_outside_clear', 'contact0-2cm', 'contact>5cm', 'clear'))
        lines.append(f"| MAX−{control} | {number(p['delta_per_min'])} [{number(p['paired_unit_ci95'][0])}, {number(p['paired_unit_ci95'][1])}] | {number(s['delta_timely'],100)} [{number(s['paired_unit_ci95'][0],100)}, {number(s['paired_unit_ci95'][1],100)}] | {number(d['delta_timely'],100)} | {number(c['delta_per_min'])} |")
    lines += ['', '| 指标 | M3 | MEAN | MAX |', '| --- | --- | --- | --- |']
    for category in CATEGORIES:
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][arm]['metrics'][category], category == 'clear', category == 'pass0-10cm') for arm in ARMS)+' |')
    lines += ['', '| 名义清晰子组 | M3首停/分钟 | MEAN首停/分钟 | MAX首停/分钟 |', '| --- | --- | --- | --- |']
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
    lines += ['', '## 阈值与判读', '']
    for arm in ARMS:
        lines.append(f"- {arm}：阈值{result['thresholds'][arm]!r}；校准记录{result['calibration'][arm]}。")
    for control, checks in result['per_control_decision_checks'].items():
        lines.append(f'- MAX相对{control}：{checks}。')
    n = result['n']
    lines += ['', '两对照都要求同高度外侧清晰差的95%上界严格<0；浅及时差95%下界≥−2pp、深点差≥−2pp、整体清晰增加≤0.2次/代理分钟。MAX在训练图中惩罚全窗最高因果平滑分数；MEAN对相同清晰序列取逐时刻负类BCE均值。固定系数0.5、五epoch，不事后调阈值或配方。', '',
        '训练额外负例为原训练单位全物体半宽0.40m、13帧标签均为0的查询；两臂使用相同序列、主M3逐帧BCE、初始化和shuffle。额外项均除以batch_scene×2，不按该batch清晰数归一。推理仍是原CVR与原因果平滑，没有新增投票、等待或max部署策略。', '',
        '同一序列下MAX额外损失≥MEAN额外损失。MEAN控制了相同清晰序列和标称额外权重，但没有等化损失或梯度强度；只能解释固定MAX目标整体收益，不能声称纯时间聚焦的独立因果效果。训练名义横向偏移仍为六簇，不是10–20cm连续覆盖；这里检验既有观测上的序列负例目标，不能证明新增空间信息。', '',
        f"浅擦碰仅{n['shallow_episodes']}条、来自{n['shallow_contributing_units']}单位；非劣性未建立不等于证明方法无用。source评估{n['query_episodes']}条查询中{n['covered']}覆盖截止点、{n['right_censored']}右删失；其无报警不计漏报。", '',
        '清晰子组按名义target_group及负向target_off分层，并与全13采样时刻清晰相交，不是动态真实净距。整体清晰仍唯一校准预算，未按子组重新平衡。空子组显示无分母，不编造0%性能。', '',
        '1000次整单位配对bootstrap，seed2026100222；区间条件于已训练检查点和已定阈值，不含再训练或校准不确定性。已消费Development、同仿真生成器，不是新盲测、实机或跨yaw验证。', '',
        '清晰2.6秒/查询暴露包含首停后时间，采样点清晰不保证帧间连续清晰；不是现场步行负担。提前量条件于已报警，相对0.5m按0.8m/s名义换算，不是人已经安全停住。完整分母、配对补回丢失、检查点和缓存哈希见result.json。', '']
    if 'provenance' in result:
        support = result['provenance']['train_support']
        total = support['overall']['total_query_episodes']
        clear = support['overall']['clear_query_episodes']
        other = support['tables']['same_target_query']['False']['clear_query_episodes']
        same = support['tables']['same_target_query']['True']['clear_query_episodes']
        lines += [f'训练全清晰查询{clear}/{total}，其中另一高度{other}（{100*other/clear:.2f}%）、同目标高度{same}。同目标高度清晰支持仅名义外侧12cm±0.6cm簇；不按子组重权或扩大支持来救本轮。', '']
        lines += ['训练清晰支持与固定配方：', '',
                  '```json', json.dumps(result['provenance']['training_summary'], ensure_ascii=False, indent=2), '```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    prior_output = YE.existing_complete(path)
    if prior_output is not None:
        print('Existing COMPLETE result verified; evaluation not repeated', flush=True)
        return prior_output
    g, scores, prior, provenance = load_evaluation_inputs()
    result = analyze(g, scores)
    result['nominal_M3_point_reproduction'] = ST.nominal_parity({'0|BASE': result['cells']['M3']}, prior)
    result['old_M3_vs_NEAR_verdict_retained'] = prior['verdict']
    result['provenance'] = provenance
    YE.verify_hashes(provenance['input_sha256'])
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
    scores['M3'][5, 1] = scores['MEAN'][5, 1] = 2.
    result = analyze(g, scores)
    assert result['thresholds']['M3'] == M3_THRESHOLD
    assert result['comparisons_MAX_minus_control']['M3']['primary_outside_clear']['delta_per_min'] < 0
    assert result['cells']['MAX']['clear_subgroups']['remaining_same_height']['false_stops_per_min']['value'] is None
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
