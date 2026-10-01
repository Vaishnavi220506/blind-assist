"""Cached RGB depth local-clearance assay on task-aligned Hypersim edges.

GT chooses a coarse nominal angular cell and clean visible boundary support.
The local extractor receives prediction, public calibration and that cell only.
Separate oracle-boundary and oracle-range controls expose the granted priors.
No new inference, rendering, training, or actual ToF-gating claim.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage

from cnh_rgb_clearance_geometry import reference_frame, camera_geometry, infer_floor, self_check
from cnh_rgb_clearance_edge import zone_map, extract_boundary, self_check as edge_check

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
LABELS = ROOT/'artifacts.local/work/cnh-rgb-candidate-supplement-20261001/hypersim'
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
NAMES = {1:'wall', 3:'cabinet', 4:'bed', 5:'chair', 6:'sofa', 7:'table',
         8:'door', 9:'window', 10:'bookshelf', 12:'counter', 14:'desk',
         15:'shelves', 17:'dresser', 24:'refrigerator', 32:'nightstand',
         33:'toilet', 34:'sink', 35:'floor-supported narrow lamp',
         36:'bathtub', 38:'otherstructure', 39:'otherfurniture'}
ARMS = ['oracle_boundary_depthpro', 'local_edge_depthpro', 'local_edge_oracle_range']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            HERE/'cnh_rgb_clearance_geometry.py', HERE/'cnh_rgb_clearance_edge.py',
            CACHE/'manifest.json', CACHE/'observations.json', CACHE/'protocol.json',
            LABELS/'visible-scene-instances.json', LABELS/'verified-frames.json']}


def prepare(out):
    assert not (out/'PLAN.json').exists(), 'Preserve old run; choose a new output directory'
    plan = dict(scope='consumed Hypersim500 synthetic Development; conditional local precision assay',
                question='Can cached native Depth Pro supply local clearance error near the ideal 1-2cm regime?',
                coordinates='public complete ray matrix + official world_from_camera pose; horizontal heading, worldZ up',
                query='nominal half-width0.30m, forward1.15-2.1m, height0.75-1.85m above known visible floor; not a calibrated wearer',
                geometry_quality='unchanged public pose max|R.T R-I| and |detR-1|<=1e-4; no pose fitting; floor>=256pixels at forward0.3-6m, q95-q05<=0.10m; roll<=10deg for horizontal-edge extractor',
                source_targets=NAMES,
                target_rule='visible vertical-ish foreground depth discontinuities>=0.15m at lateral abs0.15-0.45m; semantic allowed, known instance except wall; ordinary otherprop40 excluded; lamp35 requires bbox height>=0.5m, minXY<=0.15m, bottom within0.25m of known floor',
                support='per connected edge and angular cell: >=16pixels, vertical span>=16pixels, horizontal span<=max(4,0.25*vertical_span), GT lateral q95-q05<=0.03m; >=80percent one foreground semantic/instance',
                roi='public ray-angle45deg,8x8 nominal cells; truth tells which cell contains target; no observed current-model ToF gate available',
                selection='max128 events; stable SHA256 order round-robin range x intrusion/pass x target-family; max4 per scene,1 per frame,1 per scene/instance (wall one per scene/semantic); no prediction loaded before selection',
                predictor='one directed logdepth jump,15pixel within-cell valid-pair mean,>=8valid pair support,thresholdlog1.02; strongest in provided cell; foreground predicted depth median at2-5pixels inward,y+-3',
                target_clearance='median(side*horizontalX-0.30) on selected clean visible boundary support, not full hidden object-surface distance',
                controls=dict(oracle_boundary_depthpro='uses correct GT edge pixels and side; isolates metric-depth contribution',
                              local_edge_depthpro='predicted edge + predicted foreground depth; no semantic/instance truth enters extraction',
                              local_edge_oracle_range='same predicted edge + GT median foreground opticalZ; perfect range/association control, not measured ToF'),
                metrics='all-case within1/2/5cm including missing as failures; valid-only bias,std,RMSE,P50/P90/P95 abs error; sign disagreements; nearest GT edge pixel distance; missing and exclusion counts',
                bootstrap='1000 scene-cluster resamples seed2026100121; conditional precision interval, no independent confirmation',
                stop='complete this assay and report; no training/fusion or additional model inference',
                input_sha256=source_hashes())
    save(out/'PLAN.json', plan)


def context():
    manifest = read(CACHE/'manifest.json')
    camera = {r['id']: r['camera_matrix'] for r in read(CACHE/'observations.json')}
    boxes = {(r['scene'], r['instance_id']): r for r in read(LABELS/'visible-scene-instances.json')}
    assert len(manifest) == 500 and all(r['source'] == 'hypersim' for r in manifest)
    return manifest, camera, boxes


def frame_candidates(args):
    row, camera, boxes = args
    ref = reference_frame(ROOT, row, camera)
    floor = infer_floor(ref)
    pose = ref['camera_world_from_camera']
    right = ref['right_world']
    roll = float(np.degrees(np.arctan2(right @ pose[:, 1], right @ pose[:, 0]))) if ref['heading_valid'] else None
    status = dict(id=row['id'], scene=row['scene'], floor=floor,
                  pose_error=ref['pose_orthogonality_error'], roll_deg=roll,
                  events=0, reason=floor['reason'])
    if floor['floor_height_m'] is None:
        return status, []
    if abs(roll) > 10:
        status['reason'] = 'ROLL_OUTSIDE_HORIZONTAL_EXTRACTOR_SCOPE'
        return status, []
    height = ref['world_height']-floor['floor_height_m']
    z, lateral, forward = (ref[k] for k in ('optical_z', 'lateral', 'forward'))
    semantic, instance = ref['semantic'], ref['instance']
    valid = np.isfinite(z) & (z > 0) & (semantic > 0)
    allowed = np.isin(semantic, list(NAMES)) & ((instance >= 0) | (semantic == 1))
    # Small table lamps cannot stand in for floor-supported poles.
    for identity in np.unique(instance[semantic == 35]):
        b = boxes.get((row['scene'], int(identity)))
        narrow = (b is not None and b['extents_m'][2] >= .5
                  and min(b['extents_m'][:2]) <= .15
                  and abs(b['position_m'][2]-b['extents_m'][2]/2-floor['floor_height_m']) <= .25)
        if not narrow:
            allowed[(semantic == 35) & (instance == identity)] = False
    zones = zone_map(camera)
    all_events = []
    component_count = 0
    for side in (-1, 1):
        fore_slice = (slice(None), slice(None, -1)) if side < 0 else (slice(None), slice(1, None))
        back_slice = (slice(None), slice(1, None)) if side < 0 else (slice(None), slice(None, -1))
        fgz, bgz = z[fore_slice], z[back_slice]
        fgx, fgf, fgh = lateral[fore_slice], forward[fore_slice], height[fore_slice]
        fgzone = zones[fore_slice]
        mask = (valid[fore_slice] & valid[back_slice] & allowed[fore_slice]
                & (bgz-fgz >= .15) & (side*fgx >= .15) & (side*fgx <= .45)
                & (fgf >= 1.15) & (fgf < 2.1) & (fgh >= .75) & (fgh <= 1.85)
                & (fgzone >= 0))
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        component_count += n
        for ci, bounds in enumerate(ndimage.find_objects(labels), 1):
            if bounds is None:
                continue
            ry, rx = np.nonzero(labels[bounds] == ci)
            ry += bounds[0].start
            rx += bounds[1].start
            fgcols = rx+(1 if side > 0 else 0)
            for zone_id in np.unique(zones[ry, fgcols]):
                ids = zones[ry, fgcols] == zone_id
                yy, xx = ry[ids], fgcols[ids]
                if len(yy) < 16:
                    continue
                yspan, xspan = int(np.ptp(yy))+1, int(np.ptp(xx))+1
                if yspan < 16 or xspan > max(4, .25*yspan):
                    continue
                truth_x = lateral[yy, xx]
                if np.ptp(np.percentile(truth_x, [5, 95])) > .03:
                    continue
                # The zone's public lateral side must agree with the target.
                public_side = int(np.sign(np.median(ref['lateral_factor'][zones == zone_id])))
                if public_side != side:
                    continue
                identities = np.stack([semantic[yy, xx], instance[yy, xx]], axis=1)
                keys, counts = np.unique(identities, axis=0, return_counts=True)
                best = int(np.argmax(counts))
                if counts[best]/len(yy) < .8:
                    continue
                sem, inst = map(int, keys[best])
                own = (identities[:, 0] == sem) & (identities[:, 1] == inst)
                yy, xx = yy[own], xx[own]
                if len(yy) < 16:
                    continue
                true_forward = float(np.median(forward[yy, xx]))
                clearance = float(np.median(side*lateral[yy, xx]-.30))
                family = 'narrow_floor_structure' if sem == 35 else ('structural' if sem in (1, 8, 9, 38) else 'furniture')
                event_id = f'{row["id"]}|zone{int(zone_id)}|s{side}|c{ci}|i{inst}'
                fields = dict(id=event_id, frame_id=row['id'], scene=row['scene'], zone_id=int(zone_id),
                              side=side, semantic=sem, category=NAMES[sem], instance_id=inst, family=family,
                              gt_clearance_m=clearance, gt_forward_m=true_forward,
                              gt_optical_z_m=float(np.median(z[yy, xx])),
                              gt_height_median_m=float(np.median(height[yy, xx])),
                              gt_height_range_m=[float(height[yy, xx].min()), float(height[yy, xx].max())],
                              support_pixels=len(yy), support_yx=np.stack([yy, xx], axis=1).tolist(),
                              gt_roi_bbox=[int(yy.min()), int(xx.min()), int(yy.max())+1, int(xx.max())+1],
                              floor_height_m=floor['floor_height_m'],
                              range_bin='1.15-1.6m' if true_forward < 1.6 else '1.6-2.1m',
                              stratum=f'{family}|{true_forward >= 1.6}|{clearance < 0}')
                all_events.append(fields)
    status.update(reason='OK' if all_events else 'NO_ELIGIBLE_VISIBLE_EDGE',
                  events=len(all_events), raw_components=component_count)
    return status, all_events


def select_events(events):
    buckets = {}
    for e in events:
        buckets.setdefault(e['stratum'], []).append(e)
    for b in buckets.values():
        b.sort(key=lambda e: hashlib.sha256(e['id'].encode()).hexdigest())
    selected, frames, instances = [], set(), set()
    scenes = Counter()
    while any(buckets.values()) and len(selected) < 128:
        for name in sorted(buckets):
            while buckets[name]:
                event = buckets[name].pop(0)
                key = (event['scene'], event['instance_id']) if event['instance_id'] >= 0 else (event['scene'], -1, event['semantic'])
                if (event['frame_id'] in frames or key in instances or scenes[event['scene']] >= 4):
                    continue
                selected.append(event)
                frames.add(event['frame_id'])
                instances.add(key)
                scenes[event['scene']] += 1
                break
            if len(selected) >= 128:
                break
    return selected


def inventory(out, workers):
    started = time.perf_counter()
    plan = read(out/'PLAN.json')
    assert plan['input_sha256'] == source_hashes()
    assert not (out/'selection.json').exists(), 'Use existing selection for evaluation, do not replace it'
    manifest, cameras, boxes = context()
    statuses, events = [], []
    with ThreadPoolExecutor(workers) as pool:
        for i, (status, candidates) in enumerate(pool.map(frame_candidates,
                [(r, cameras[r['id']], boxes) for r in manifest])):
            statuses.append(status)
            events.extend(candidates)
            if (i+1) % 25 == 0:
                save(out/'progress.json', dict(stage='GT_inventory', completed=i+1, total=500, candidate_events=len(events)))
                print('inventory', i+1, 'events', len(events), flush=True)
    selection = select_events(events)
    save(out/'inventory.json', dict(frames=statuses, frame_reasons=dict(Counter(r['reason'] for r in statuses)),
                                   candidate_events=len(events), candidate_scenes=len({e['scene'] for e in events}),
                                   candidate_categories=dict(Counter(e['category'] for e in events)),
                                   selected_events=len(selection), selected_scenes=len({e['scene'] for e in selection}),
                                   selected_categories=dict(Counter(e['category'] for e in selection)),
                                   selected_strata=dict(Counter(e['stratum'] for e in selection)),
                                   prediction_access='NONE', seconds=time.perf_counter()-started))
    save(out/'selection.json', selection)
    save(out/'geometry-check.json', self_check(ROOT))
    save(out/'extractor-check.json', edge_check())
    print('selected', len(selection), 'scenes', len({e['scene'] for e in selection}), flush=True)


def evaluate_event(event, row, camera):
    geom = camera_geometry(ROOT, row, camera)
    with np.load(CACHE/'predictions/native'/f'{row["id"]}.npz', allow_pickle=False) as z:
        prediction = z['native_depth']
    assert prediction.shape == (768, 1024)
    pixels = np.asarray(event['support_yx'])
    yy, xx = pixels.T
    side = event['side']
    lateral_factor = geom['lateral_factor']
    oracle_values = side*lateral_factor[yy, xx]*prediction[yy, xx]-.30
    oracle_valid = np.isfinite(oracle_values) & np.isfinite(prediction[yy, xx]) & (prediction[yy, xx] > 0)
    oracle_clear = float(np.median(oracle_values[oracle_valid])) if oracle_valid.any() else None
    estimated = extract_boundary(prediction, lateral_factor, zone_map(camera), event['zone_id'])
    local_clear = estimated.get('clearance_m') if estimated['status'] == 'OK' else None
    range_clear = None
    pixel_distance = None
    if estimated['status'] == 'OK':
        assert estimated['side'] == side
        range_clear = float(side*estimated['lateral_factor']*event['gt_optical_z_m']-.30)
        pixel_distance = float(np.min(np.linalg.norm(pixels-np.asarray(estimated['edge_pixel'])[None, :], axis=1)))
    result = dict(event)
    result.pop('support_yx')
    result.update(prediction=estimated, predicted_to_gt_edge_distance_px=pixel_distance,
                  estimates={arm: value for arm, value in zip(ARMS, (oracle_clear, local_clear, range_clear))},
                  errors_m={arm: (value-event['gt_clearance_m']) if value is not None else None
                            for arm, value in zip(ARMS, (oracle_clear, local_clear, range_clear))})
    return result


def summarize(ledger, arm, mask=None):
    rows = ledger if mask is None else [r for r in ledger if mask(r)]
    errors = np.array([r['errors_m'][arm] for r in rows if r['errors_m'][arm] is not None])
    n, valid = len(rows), len(errors)
    values = dict(n=n, valid=valid, missing=n-valid,
                  within_cm_all={str(cm): int(np.sum(np.abs(errors) <= cm/100))/n if n else None for cm in (1, 2, 5)},
                  absolute_error_cm_quantiles=None, signed_bias_cm=None, centered_std_cm=None, rmse_cm=None,
                  sign_counts=dict(contact=0, pass_by=0, contact_wrongly_clear=0, pass_by_wrongly_contact=0))
    if valid:
        values.update(absolute_error_cm_quantiles=dict(zip(('p50', 'p90', 'p95'), (np.percentile(abs(errors), [50, 90, 95])*100).tolist())),
                      signed_bias_cm=float(errors.mean()*100), centered_std_cm=float(errors.std()*100),
                      rmse_cm=float(np.sqrt(np.mean(errors**2))*100))
    for r in rows:
        true_contact = r['gt_clearance_m'] < 0
        estimate = r['estimates'][arm]
        values['sign_counts']['contact' if true_contact else 'pass_by'] += 1
        if estimate is not None:
            if true_contact and estimate >= 0:
                values['sign_counts']['contact_wrongly_clear'] += 1
            elif not true_contact and estimate < 0:
                values['sign_counts']['pass_by_wrongly_contact'] += 1
    if n:
        scenes = sorted({r['scene'] for r in rows})
        denominators = np.array([sum(r['scene'] == s for r in rows) for s in scenes])
        successes = np.array([[sum(r['scene'] == s and r['errors_m'][arm] is not None
                                   and abs(r['errors_m'][arm]) <= cm/100 for r in rows)
                               for cm in (1, 2, 5)] for s in scenes])
        rng = np.random.default_rng(2026100121)
        weights = np.array([np.bincount(rng.integers(len(scenes), size=len(scenes)), minlength=len(scenes)) for _ in range(1000)])
        rates = (weights @ successes)/(weights @ denominators)[:, None]
        values['within_cm_all_ci95'] = {str(cm): np.percentile(rates[:, i], [2.5, 97.5]).tolist() for i, cm in enumerate((1, 2, 5))}
        values['scenes'] = len(scenes)
    return values


def evaluate(out):
    started = time.perf_counter()
    plan = read(out/'PLAN.json')
    assert plan['input_sha256'] == source_hashes()
    assert not (out/'result.json').exists(), 'Preserve complete result'
    manifest, cameras, _ = context()
    by_id = {r['id']: r for r in manifest}
    selection = read(out/'selection.json')
    # Bind selected immutable inputs before opening model predictions.
    bound = {}
    for event in selection:
        row = by_id[event['frame_id']]
        paths = [CACHE/'predictions/native'/f'{row["id"]}.npz',
                 CACHE/'predictions/native'/f'{row["id"]}.json',
                 ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['rgb'],
                 ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth']]
        paths += [paths[3].with_name(paths[3].name.replace('depth_meters', name))
                  for name in ('semantic', 'semantic_instance')]
        geom = camera_geometry(ROOT, row, cameras[row['id']])
        paths += [Path(p) for p in geom['metadata_paths']]
        pred_hash = sha(paths[0])
        assert pred_hash == read(paths[1])['sha256']
        assert sha(paths[2]) == row['rgb_sha256'] and sha(paths[3]) == row['depth_sha256']
        for p in paths:
            if str(p.relative_to(ROOT)) not in bound:
                bound[str(p.relative_to(ROOT))] = sha(p)
    save(out/'evaluation-input-seal.json', dict(plan_sha256=sha(out/'PLAN.json'), selection_sha256=sha(out/'selection.json'), input_sha256=bound))
    sys.path.insert(0, str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    backend = select_backend(Workload.SCALAR_SCORING,
                cpu=BackendCandidate('numpy-cpu', 'cpu', lambda: np.arange(128).mean(),
                                     lambda _: DeviceObservation('cpu', 'host CPU', 'numpy/scipy', ('CPU',))),
                capabilities={'python_executable':sys.executable, 'reason_code':'TASK_NOT_GPU_SUITABLE'},
                record_path=out/'backend.json')
    ledger = []
    for i, event in enumerate(selection):
        ledger.append(evaluate_event(event, by_id[event['frame_id']], cameras[event['frame_id']]))
        if (i+1) % 16 == 0:
            save(out/'progress.json', dict(stage='cached_prediction_evaluation', completed=i+1, total=len(selection)))
            print('evaluated', i+1, '/', len(selection), flush=True)
    save(out/'case-ledger.json', ledger)
    result = dict(status='COMPLETE' if selection else 'INSUFFICIENT_TARGET_OPPORTUNITY',
                  scope=plan['scope'], plan_sha256=sha(out/'PLAN.json'), inventory=read(out/'inventory.json'),
                  arms={arm: dict(overall=summarize(ledger, arm),
                      by_range={name:summarize(ledger, arm, lambda r,n=name:r['range_bin']==n) for name in ('1.15-1.6m','1.6-2.1m')},
                      by_family={name:summarize(ledger, arm, lambda r,n=name:r['family']==n) for name in sorted({r['family'] for r in ledger})}) for arm in ARMS},
                  prediction_status=dict(Counter(r['prediction']['status'] for r in ledger)),
                  localization=dict(estimated=sum(r['predicted_to_gt_edge_distance_px'] is not None for r in ledger),
                                    within3px=sum(r['predicted_to_gt_edge_distance_px'] is not None and r['predicted_to_gt_edge_distance_px']<=3 for r in ledger)),
                  limits=['GT supplies a cell with a clean eligible visible edge and known floor; actual ToF gate/registration not tested',
                          'public calibrated camera, no head mounting or real collision truth',
                          'oracle boundary arm has perfect target/edge pixels; oracle range arm has perfect target-associated range',
                          'physical obstacles are represented only by selected visible boundary support; occluded full extent and severe occlusion untested',
                          'clean vertical-edge, flat-floor, low-roll and semantic subset; missing source/target support explicitly excluded',
                          'cached monocular model only, no new training/inference; pretraining overlap unexcluded',
                          'scene-bootstrap describes this consumed sample; systematic bias/tails cannot be called iid Gaussian sigma'],
                  backend=backend, evaluation_seconds=time.perf_counter()-started)
    save(out/'result.json', result)
    report(out, result, ledger, by_id, cameras)
    verify(out)
    save(out/'terminal.json', dict(status='complete', evaluation_seconds=time.perf_counter()-started))


def report(out, result, ledger, by_id, cameras):
    labels = dict(oracle_boundary_depthpro='真值边界位置 + Depth Pro深度',
                  local_edge_depthpro='粗格内预测边界 + Depth Pro深度',
                  local_edge_oracle_range='同一预测边界 + 真值对应距离')
    lines = ['# Hypersim局部边界净距核查（已消费Development）', '',
             '真值指明目标所在的45°/8×8名义粗格，并提供局部地面与干净可见结构边缘；预测边界提取器只读取Depth Pro缓存、公共光线/姿态和粗格编号。没有实际ToF门控记录，不能称融合方法效果。', '',
             f"500帧中筛出{result['inventory']['candidate_events']}个可用边界记录，固定选取{len(ledger)}个、{result['inventory']['selected_scenes']}场景；普通桌面小物件类别40已排除。类别与排除原因见inventory.json。", '',
             '|条件臂|分母/有效|≤1cm全部占比|≤2cm全部占比及95%场景区间|≤5cm全部占比|绝对误差P50/P95(cm)|有符号偏差(cm)|',
             '|---|---|---|---|---|---|---|']
    for arm in ARMS:
        s = result['arms'][arm]['overall']
        if not s['n']:
            lines.append(f'|{labels[arm]}|0|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|')
            continue
        rate = s['within_cm_all']
        ci = s['within_cm_all_ci95']['2']
        q = s['absolute_error_cm_quantiles']
        qs = f"{q['p50']:.2f}/{q['p95']:.2f}" if q else 'UNKNOWN'
        bias = f"{s['signed_bias_cm']:.2f}" if s['signed_bias_cm'] is not None else 'UNKNOWN'
        lines.append(f"|{labels[arm]}|{s['n']}/{s['valid']}|{rate['1']:.1%}|{rate['2']:.1%} [{ci[0]:.1%},{ci[1]:.1%}]|{rate['5']:.1%}|{qs}|{bias}|")
    lines += ['', '缺失边界计入全部分母的精度失败；P50/P95、偏差和标准差只描述有估计样本。关联错位和符号错误未从主结果排除。', '',
              '局限：']+['- '+text for text in result['limits']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    if not ledger:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, label in zip(ARMS, ('oracle edge / Depth Pro', 'predicted edge / Depth Pro', 'predicted edge / oracle range')):
        err = np.array([r['errors_m'][arm] for r in ledger if r['errors_m'][arm] is not None])*100
        if len(err):
            xx = np.sort(abs(err)); yy = np.arange(1, len(err)+1)/len(ledger)
            axes[0].step(xx, yy, where='post', label=label)
            axes[1].hist(err, bins=np.linspace(-50, 50, 31), histtype='step', label=label)
    axes[0].axvline(2, color='black', ls='--', lw=.8)
    axes[0].set(xlabel='absolute clearance error (cm)', ylabel='fraction of all selected cases', xlim=(0, 30), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='signed clearance error (cm)', ylabel='estimated cases within plotted range')
    axes[0].grid(alpha=.25); axes[1].grid(alpha=.25)
    fig.suptitle('Cached monocular depth; oracle coarse-cell localization and known floor, consumed Development')
    fig.tight_layout(); fig.savefig(out/'clearance-errors.png', dpi=160); plt.close(fig)
    # Six fixed-rank visual diagnostics; selection of previews never changes metrics.
    ordered = sorted(ledger, key=lambda r: abs(r['errors_m']['local_edge_depthpro']) if r['errors_m']['local_edge_depthpro'] is not None else np.inf)
    indices = sorted(set(np.linspace(0, len(ordered)-1, min(6, len(ordered))).astype(int)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    from PIL import Image
    for ax, idx in zip(axes.ravel(), indices):
        r = ordered[idx]
        selection = next(e for e in read(out/'selection.json') if e['id']==r['id'])
        coords = np.asarray(selection['support_yx'])
        rgb = np.asarray(Image.open(ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/by_id[r['frame_id']]['rgb']))
        zones = zone_map(cameras[r['frame_id']])
        yy, xx = np.nonzero(zones == r['zone_id'])
        box = [max(0, yy.min()-10), max(0, xx.min()-10), min(768, yy.max()+11), min(1024, xx.max()+11)]
        ax.imshow(rgb)
        ax.scatter(coords[:, 1], coords[:, 0], s=3, c='lime', label='GT edge')
        if r['prediction']['status'] == 'OK':
            py, px = r['prediction']['edge_pixel']; ax.plot(px, py, 'mx', ms=10, mew=2)
        ax.set_xlim(box[1], box[3]); ax.set_ylim(box[2], box[0])
        error = r['errors_m']['local_edge_depthpro']
        es = f'{error*100:+.1f} cm' if error is not None else r['prediction']['status']
        ax.set_title(f"{r['category']}; error {es}\n{r['frame_id']}", fontsize=8)
        ax.axis('off')
    for ax in axes.ravel()[len(indices):]: ax.axis('off')
    fig.suptitle('Green: GT support; magenta: predicted edge. Fixed error-rank previews only.', fontsize=11)
    fig.tight_layout(); fig.savefig(out/'case-previews.png', dpi=160); plt.close(fig)


def verify(out):
    plan = read(out/'PLAN.json')
    seal = read(out/'evaluation-input-seal.json')
    assert source_hashes() == plan['input_sha256']
    assert sha(out/'PLAN.json') == seal['plan_sha256'] and sha(out/'selection.json') == seal['selection_sha256']
    assert all(sha(ROOT/path) == expected for path, expected in seal['input_sha256'].items())
    selection = read(out/'selection.json')
    ledger, result = read(out/'case-ledger.json'), read(out/'result.json')
    assert len(selection) == len(ledger) == len({r['frame_id'] for r in ledger})
    assert max(Counter(r['scene'] for r in ledger).values(), default=0) <= 4
    for arm in ARMS:
        got = summarize(ledger, arm)
        assert got == result['arms'][arm]['overall']
        for r in ledger:
            if r['estimates'][arm] is not None:
                assert abs(r['errors_m'][arm]-(r['estimates'][arm]-r['gt_clearance_m'])) < 1e-12
    save(out/'verification.json', dict(status='PASS', rows=len(ledger), arms=len(ARMS),
              checks=['source/selection hashes', 'one frame/event and max4 per scene',
                      'all-case denominators retain missing predictions', 'ledger arithmetic and scene-bootstrap recomputation'],
              geometry=self_check(), extractor=edge_check()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare','inventory','evaluate','verify'], required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare': prepare(args.output)
    elif args.stage == 'inventory': inventory(args.output, args.workers)
    elif args.stage == 'verify': verify(args.output)
    else:
        try: evaluate(args.output)
        except BaseException as error:
            save(args.output/'terminal.json', dict(status='failed', error=repr(error))); raise


if __name__ == '__main__': main()
