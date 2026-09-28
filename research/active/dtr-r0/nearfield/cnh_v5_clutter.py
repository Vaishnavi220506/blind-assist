"""Development-only frozen v5 clutter diagnostic; no training or audit calibration.

Artifacts own the request, progress, per-unit checks and resumable outputs. Frozen
sensor/readout/evaluation functions are imported rather than reimplemented.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import frozen snapshots without adding bytecode files to their source tree.
sys.dont_write_bytecode = True

import numpy as np
import cnh_v5_robustness as R


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def init_source(data):
    R.setup(data)
    import cnh_track_a_v13_sensor as S
    S.FAMILY = json.loads((Path(data)/'request.json').read_text())['family']
    return S


def angular_support(tq):
    """Exact ray/box angular projection, including rays hitting beyond box depth."""
    from cnh_route_sensor import angular_rays
    from cnh_track_a_readout import BOXES
    rays = angular_rays(16)[0] @ tq[:3, :3].T
    origin = tq[:3, 3]
    masks = []
    for low, high in BOXES:
        near = np.zeros(rays.shape[:-1])
        far = np.full_like(near, np.inf)
        for axis in range(3):
            d = rays[..., axis]
            parallel = np.abs(d) < 1e-12
            a = np.divide(low[axis]-origin[axis], d, out=np.full_like(d, -np.inf), where=~parallel)
            b = np.divide(high[axis]-origin[axis], d, out=np.full_like(d, np.inf), where=~parallel)
            near = np.maximum(near, np.minimum(a, b))
            far = np.minimum(far, np.maximum(a, b))
            if not low[axis] <= origin[axis] <= high[axis]:
                far = np.where(parallel, -np.inf, far)
        masks.append(far >= near)
    return np.asarray(masks)


def render_one(c, S, params):
    # The frozen allocator has three SNR slots. Only slot 0 is synthesized when
    # LEVELS=(6,); repack that into canonical slot 1 for frozen unit_records.
    obs, oracle = S.render_config(c, -10, params, 5, True)
    for key in ('hist', 'ambient'):
        first = obs[key][0].copy()
        obs[key].fill(0)
        obs[key][1] = first
    return obs, oracle


def sensor_job(job):
    data, out, u, doses = job
    data, out = Path(data), Path(out)
    S = init_source(data)
    params, _ = S.reference_parameters()
    S.LEVELS = (6,)
    raw = json.loads((data/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json').read_text(encoding='utf-8-sig'))
    started = time.monotonic()
    baseline = []
    max_error = 0.
    with np.load(data/'sensor'/f'unit{u:02d}-mount-10-observations.npz') as stored:
        for c in raw['configs']:
            obs, oracle = render_one(c, S, params)
            sel = np.flatnonzero(stored['config'] == c['config'])
            for key in ('hist', 'ambient'):
                err = float(np.max(np.abs(obs[key][1].astype(float)-stored[key][1, sel].astype(float))))
                max_error = max(max_error, err)
                if err > 1e-9:
                    raise RuntimeError(('K0_resynthesis', u, c['config'], key, err))
            baseline.append(oracle['object_id'])
    save_json(out/'checks'/f'unit{u:03d}-resynthesis.json', dict(unit=u, max_abs_error=max_error, pass_gate=True,
        configs=len(raw['configs']), frames=sum(len(c['labels']) for c in raw['configs']), elapsed_s=time.monotonic()-started))
    if not doses:
        return u
    # Generate one nested ten-slot draw, then select dose prefixes.
    modified, geometry_receipt = build_unit_clutter(raw, exempt_ground_contact=True)
    save_json(out/'checks'/f'unit{u:03d}-geometry.json', geometry_receipt)
    for k in doses:
        folder = out/f'K{k}'
        target = folder/'sensor'/f'unit{u:02d}-mount-10-observations.npz'
        if target.exists() and (folder/'diagnostics'/f'unit{u:03d}.npz').exists():
            continue
        dose_unit = modified[k]
        save_json(folder/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json', dose_unit)
        parts, diagnostics = [], {key: [] for key in ('class_visible','any_clutter_visible','occlusion_loss','baseline_target_visible')}
        for ci, c in enumerate(dose_unit['configs']):
            obs, oracle = render_one(c, S, params)
            ids = oracle['object_id']
            added = [o for o in c['objects'] if 'clutter_class' in o]
            class_ids = [[o['id'] for o in added if o['clutter_class'] == j] for j in range(1, 6)]
            any_clutter = np.isin(ids, [o['id'] for o in added]).any(axis=(1,2,3))
            visible = np.zeros((len(ids),6,5), bool)
            loss = np.zeros((len(ids),6), bool)
            base_visible = np.zeros_like(loss)
            for t in range(len(ids)):
                support = angular_support(obs['T_Q_tof'][t])
                for q in range(6):
                    for j, group in enumerate(class_ids):
                        visible[t,q,j] = bool((np.isin(ids[t], group) & support[q]).any())
                    contributors = c['contributors'][t][q]
                    before = int(np.isin(baseline[ci][t], contributors).sum())
                    after = int(np.isin(ids[t], contributors).sum())
                    base_visible[t,q] = before > 0
                    loss[t,q] = before > 0 and after < .5*before
            for key, value in zip(diagnostics, (visible, any_clutter, loss, base_visible)):
                diagnostics[key].append(value)
            parts.append(obs)
        stacked = {key: np.concatenate([p[key] for p in parts], axis=1 if key in ('hist','ambient') else 0) for key in parts[0]}
        stacked.update(config=np.repeat([c['config'] for c in dose_unit['configs']],12), frame=np.tile(np.arange(12),len(parts)),rate=np.array(5))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **stacked)
        tmp.replace(target)
        diag = folder/'diagnostics'/f'unit{u:03d}.npz'
        diag.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(diag, **{key:np.concatenate(v) for key,v in diagnostics.items()})
    return u


def prediction_job(job):
    data, out, u, k, models = job
    S = init_source(data)
    folder = Path(out)/f'K{k}'
    # R.unit_job accepts a root containing geometry/sensor; setup uses original
    # frozen source already loaded, and its absent local source is harmless.
    return R.unit_job((str(folder),str(folder/'predictions'),u,'snr6',S.FAMILY,models))


def run_batch(a, stage, fn, jobs, workers):
    start = time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, j) for j in jobs]
        for i, f in enumerate(as_completed(futures),1):
            u = f.result()
            receipt = dict(stage=stage,completed=i,total=len(jobs),unit=u,elapsed_s=time.monotonic()-start)
            save_json(a.out/'progress.json',receipt)
            print(json.dumps(receipt),flush=True)


def fit_bias(a, doses, units):
    # Match robustness's SNR6 branch exactly: reuse frozen clean calib bias.
    # Dose-specific recalibration changes alarm thresholds only, never readouts.
    bias=np.load(a.data/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    for k in doses:
        folder = a.out/f'K{k}'
        pred = folder/'predictions'
        pred.mkdir(parents=True,exist_ok=True)
        np.save(pred/'bias.npy',bias)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',type=Path,nargs=3,required=True)
    p.add_argument('--stage',choices=('precheck','pilot','full','evaluate'),required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu-workers',type=int,default=4)
    a=p.parse_args()
    a.data=a.data.resolve(); a.out=a.out.resolve()
    root_out=a.out
    if a.stage=='pilot':
        a.out=root_out/'pilot'
    a.out.mkdir(parents=True,exist_ok=True)
    assert 1 <= a.workers <= 4
    assert 1 <= a.cpu_workers <= (os.cpu_count() or 1)
    assert [R.digest(m) for m in a.models] == list(R.MODEL_HASHES)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    init_source(a.data)
    manifest=json.loads((a.data/'source_manifest.json').read_text())
    for name,h in manifest['sha256'].items():
        assert R.digest(a.data/'source'/name)==h, ('source_identity',name)
    contract=dict(data=str(a.data),source_manifest=R.digest(a.data/'source_manifest.json'),
                  models=list(R.MODEL_HASHES),script_sha256=R.digest(__file__),
                  bias='frozen clean v5 calib SNR6',support_contact_exempt=True,
                  family='cnh-v5-clutter-20260928',doses=[0,3,6,10],
                  cpu_workers=a.cpu_workers,gpu_workers=a.workers,
                  sensor_backend='CPU: frozen NumPy raycast/synthesizer; GPU_BACKEND_UNAVAILABLE',
                  prediction_backend='CUDA')
    contract_path=a.out/(a.stage+'_request.json')
    if contract_path.exists():
        assert json.loads(contract_path.read_text())==contract, 'Resume identity differs'
    save_json(contract_path,contract)
    start=time.monotonic()
    units=[0,1,32,33] if a.stage in ('precheck','pilot') else list(range(96))
    doses=[] if a.stage=='precheck' else [10] if a.stage=='pilot' else [3,6,10]
    lock=a.out/'RUNNING.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        if a.stage!='evaluate':
            if a.stage!='precheck':
                assert json.loads((root_out/'precheck_terminal.json').read_text())['status']=='complete'
            if a.stage=='full':
                assert json.loads((a.out/'pilot_terminal.json').read_text())['status']=='complete'
            run_batch(a,a.stage+'-sensor',sensor_job,[(str(a.data),str(a.out),u,doses) for u in units],a.cpu_workers)
            if doses:
                fit_bias(a,doses,units)
                run_batch(a,a.stage+'-readout',prediction_job,[(str(a.data),str(a.out),u,k,[str(m) for m in a.models]) for k in doses for u in units],a.workers)
        if a.stage!='precheck':
            evaluate_clutter(a.data,a.out,units,report_name='pilot_results.json' if a.stage=='pilot' else 'results.json')
        assert R.digest(__file__)==contract['script_sha256'], 'Script changed during execution'
        save_json(root_out/(a.stage+'_terminal.json'),dict(status='complete',elapsed_s=time.monotonic()-start,units=units,doses=doses,contract=contract))
    except BaseException as e:
        save_json(root_out/(a.stage+'_terminal.json'),dict(status='STOP',error=repr(e),elapsed_s=time.monotonic()-start))
        raise
    finally:
        lock.unlink()


# Geometry and reporting helpers below are task-specific; inference remains frozen.
"""Clutter helpers: requires frozen data/source first on sys.path."""
import copy
import hashlib
import numpy as np

CLUTTER_CLASSES = ('side_pole', 'side_shelf', 'floor_box', 'overhead', 'beyond_range')
CLUTTER_FAMILY = 'cnh-v5-clutter-20260928'

def clutter_seed(unit, config, slot):
    return int(hashlib.sha256(f'{CLUTTER_FAMILY}|{unit}|{config}|clutter|{slot}'.encode()).hexdigest()[:16], 16)

def aabb_gap(a, b):
    a, b = np.asarray(a).reshape(-1, 3), np.asarray(b).reshape(-1, 3)
    d = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0.)
    return float(np.linalg.norm(d))

def clutter_candidate(c, rng, kind, slot):
    from cnh_track_a_geometry import box_mesh, cylinder_mesh
    anchor = np.asarray(c['world_from_Q'])[7]
    height, sign, z = float(c['height']), rng.choice([-1, 1]), rng.uniform(.5, 5.)
    if kind == 0:
        r, h = rng.uniform(.03, .08), rng.uniform(.8, 2.5)
        center = np.array([sign*rng.uniform(.75, 1.6), height-h/2, z])
        mesh = cylinder_mesh(center, r, h, axis=1, sides=64)
    elif kind == 1:
        h, length = rng.uniform(.8, 2.), rng.uniform(.5, 2.)
        center = np.array([sign*rng.uniform(.8, 1.6), height-h/2-.1, z])
        half = np.array([.025, h/2, length/2])
        mesh = box_mesh(center-half, center+half)
    elif kind == 2:
        h = rng.uniform(.05, min(.35, height-.9-.10))
        half = np.array([rng.uniform(.1, .5)/2, h/2, rng.uniform(.1, .5)/2])
        center = np.array([rng.uniform(-.6, .6), height-h/2, z])
        mesh = box_mesh(center-half, center+half)
    elif kind == 3:
        half = np.array([rng.uniform(.2, 1.)/2, rng.uniform(.05, .2)/2, .025])
        center = np.array([rng.uniform(-.9, .9), -.30-rng.uniform(0., .5)-half[1], z])
        mesh = box_mesh(center-half, center+half)
    else:
        half = np.array([rng.uniform(.1, .5)/2, rng.uniform(.2, 1.)/2, rng.uniform(.1, .5)/2])
        center = np.array([rng.uniform(-.6, .6), rng.uniform(.0, .6), 0.])
        mesh = box_mesh(center-half, center+half)
    world = mesh@anchor[:3, :3].T+anchor[:3, 3]
    if kind == 4:
        world[..., 2] += np.asarray(c['world_from_Q_10hz'])[-1, 2, 3]+3.3+rng.uniform(.05, 1.)-world[..., 2].min()
    return dict(id=1000+slot, triangles_world=world.tolist(), rho=float(rng.uniform(.1, .9)),
                family='controlled-clutter', category='CLUTTER', clutter_class=kind+1,
                clutter_class_name=CLUTTER_CLASSES[kind],
                clutter_slot=slot, seed=clutter_seed(c['unit'], c['config'], slot))

def frozen_label_check(c, additions, clean=None):
    """Frozen label function with certified bounds far outside queries.

    Missing stored 10Hz truth is regenerated from frozen objects and true poses.
    Fast bounds certify exact labels, contributors and boundary, not margin value.
    """
    import cnh_track_a_v13_generate as gen
    from cnh_track_a_fastmargin import fast_margin
    old = gen.signed_margin
    gen.signed_margin = fast_margin
    result = {}
    try:
        for suffix, warmup in (('', 3), ('_10hz', 6)):
            poses = c['world_from_Q'+suffix]
            def calc(objects):
                _, lab, contrib, boundary = gen.labels_for_poses(objects, poses)
                return dict(labels=lab.tolist(), contributors=contrib, boundary=boundary.tolist(),
                            main=((np.arange(len(poses)) >= warmup) & ~boundary).tolist())
            baseline = clean[suffix] if clean is not None else calc(c['objects'])
            for key, val in baseline.items():
                if key+suffix in c and val != c[key+suffix]:
                    raise AssertionError(('clean_label_mismatch', c['unit'], c['config'], key+suffix))
            changed = calc(c['objects']+additions) if additions else baseline
            for key in baseline:
                if changed[key] != baseline[key]:
                    raise AssertionError(('injected_label_mismatch', c['unit'], c['config'], key+suffix))
            result[suffix] = baseline
    finally:
        gen.signed_margin = old
    return result

def inject_config(c, max_k=10, exempt_ground_contact=False):
    """K means requested slots; accepted slot prefix ensures dose nesting.

    A class is drawn uniformly once per slot; up to 50 geometry redraws within it.
    User authorized support contact exemption only (ground id100, pole/floor box).
    """
    from cnh_track_a_generate import BOXES
    from cnh_track_a_fastmargin import fast_margin
    poses = np.concatenate([np.asarray(c['world_from_Q']), np.asarray(c['world_from_Q_10hz'])])
    accepted, logs = [], []
    for slot in range(max_k):
        rng = np.random.default_rng(clutter_seed(c['unit'], c['config'], slot))
        kind = int(rng.integers(5))
        reasons = {'aabb': 0, 'query_margin': 0}
        for attempt in range(1, 51):
            obj = clutter_candidate(c, rng, kind, slot)
            tri = np.asarray(obj['triangles_world'])
            others = c['objects']+accepted
            if exempt_ground_contact and kind in (0, 2):
                others = [o for o in others if not (o['id'] == 100 and o.get('category') == 'BACKGROUND')]
            if any(aabb_gap(tri, o['triangles_world']) < .02-1e-12 for o in others):
                reasons['aabb'] += 1
                continue
            margins = []
            for pose in poses:
                local = (tri-pose[:3, 3])@pose[:3, :3]
                margins.extend(fast_margin(local, *box) for box in BOXES)
            if max(margins) >= -.05:
                reasons['query_margin'] += 1
                continue
            obj['max_margin_certified_bound'] = float(max(margins))
            accepted.append(obj)
            break
        logs.append(dict(slot=slot, category=CLUTTER_CLASSES[kind], accepted=bool(accepted and accepted[-1]['clutter_slot']==slot),
                         attempts=attempt, rejection_reasons=reasons))
    clean = frozen_label_check(c, [])
    for k in (3, 6, 10):
        if k <= max_k:
            frozen_label_check(c, [o for o in accepted if o['clutter_slot'] < k], clean)
    return accepted, logs, clean

def config_at_dose(c, accepted, k):
    out = copy.deepcopy(c)
    extras = [o for o in accepted if o['clutter_slot'] < k]
    out['objects'] += extras
    out['object_ids'] += [o['id'] for o in extras]
    return out

def build_unit_clutter(raw, exempt_ground_contact=True):
    """Return ({K: unit metadata}, receipt), without writing input or output files."""
    import time
    started = time.time()
    modified = {k: {**copy.deepcopy(raw), 'configs': []} for k in (0, 3, 6, 10)}
    receipt = dict(unit=raw['unit'], split=raw['split'], support_contact_exempt=exempt_ground_contact,
                   class_names={i+1: name for i, name in enumerate(CLUTTER_CLASSES)}, configs=[],
                   checks=dict(labels=True, contributors=True, main=True, boundary=True, rates=[5, 10]),
                   stored_10hz_labels=True)
    for c in raw['configs']:
        additions, logs, clean = inject_config(c, exempt_ground_contact=exempt_ground_contact)
        receipt['stored_10hz_labels'] &= 'labels_10hz' in c
        for k in modified:
            out = config_at_dose(c, additions, k)
            for suffix, fields in clean.items():
                for key, value in fields.items():
                    out[key+suffix] = value
            modified[k]['configs'].append(out)
        receipt['configs'].append(dict(config=c['config'], slots=logs,
            requested={k:k for k in modified},
            accepted={k:sum(o['clutter_slot']<k for o in additions) for k in modified},
            shortfall={k:k-sum(o['clutter_slot']<k for o in additions) for k in modified},
            compared_rows={suffix:{field:len(vals) for field,vals in fields.items()}
                           for suffix,fields in clean.items()}))
    receipt['wall_s'] = time.time()-started
    return modified, receipt

"""Evaluation helpers, to be incorporated into the task-owned clutter script."""
import json
from pathlib import Path
import numpy as np


def evaluate_clutter(data, out, units=None, report_name='results.json'):
    import cnh_v5_robustness as R
    R.setup(data)
    import cnh_v5_evaluate as E
    import cnh_v5_episode_calibration as C
    from sklearn.metrics import average_precision_score
    data, out = Path(data), Path(out)
    clean = R.load(data/'predictions')
    selected = sorted(clean) if units is None else sorted(units)
    clean = {u: clean[u] for u in selected}
    splits = {s: [u for u in selected if str(clean[u]['split']) == s] for s in ('calib', 'audit')}
    assert splits['calib'] and splits['audit']
    clean_packs = {g: {s: E.pack(clean, splits[s], ix) for s in splits} for g, ix in E.GROUPS}
    arms = {'S2': 'A0', 'A2': 'A2'}
    frozen_v5 = json.loads((data/'analysis/v5_results.json').read_text())
    classes = ('side_pole', 'side_panel', 'floor_clutter', 'overhead', 'beyond_range')
    nominal = {g: {name: {str(t): C.calibrate(clean_packs[g]['calib'], arm, t)
                for t in C.TARGETS} for name, arm in arms.items()} for g, _ in E.GROUPS}
    report = dict(scope='EXPLORE Development reuse; controlled generator; no training or audit threshold selection',
                  split_units=splits, bootstrap_draws=10000, targets=list(C.TARGETS), doses={},
                  attribution_definition='Audit query-frames t=3..11, labels==0, including boundary negatives. '
                      'Class visible means a first-hit ray from that class in the query angular projection; classes may overlap. '
                      'Reference means no first-hit clutter ray anywhere in the sensor field of view. '
                      'These are conditional alarm proportions, not causal attribution.',
                  occlusion_definition='Audit positive query-frames t=3..11; loss >50% relative to clean target-visible rays; '
                      'zero clean visibility reported separately and is not a loss event.')
    for k in (0, 3, 6, 10):
        folder = out/f'K{k}'/'predictions'
        if k and not folder.exists():
            continue
        ds = clean if k == 0 else R.load(folder)
        assert set(selected).issubset(ds), ('missing_predictions', k, sorted(set(selected)-set(ds)))
        for u in selected:
            for field in ('y', 'main', 'config', 'frame', 'strata'):
                assert np.array_equal(ds[u][field], clean[u][field]), ('prediction_identity', k, u, field)
        diagnostics = {}
        if k:
            for u in splits['audit']:
                with np.load(out/f'K{k}'/'diagnostics'/f'unit{u:03d}.npz') as z:
                    diagnostics[u] = {key: z[key] for key in
                        ('class_visible', 'any_clutter_visible', 'occlusion_loss', 'baseline_target_visible')}
                d, z = ds[u], diagnostics[u]
                assert np.array_equal(np.lexsort((d['frame'], d['config'])), np.arange(len(d['frame'])))
                assert z['class_visible'].shape == (*d['y'].shape, 5)
                assert z['any_clutter_visible'].shape == (len(d['frame']),)
                assert all(z[key].dtype == bool for key in z)
                assert z['occlusion_loss'].shape == z['baseline_target_visible'].shape == d['y'].shape
        groups = {}
        for g, ix in E.GROUPS:
            cal, aud = (E.pack(ds, splits[s], ix) for s in ('calib', 'audit'))
            per_unit = {}
            for u in splits['audit']:
                d = ds[u]
                yy = d['y'][d['main']][:, ix].ravel()
                assert 0 < yy.sum() < len(yy), ('ap_unit_not_evaluable', u, g)
                per_unit[u] = dict(query_frames=len(yy), positives=int(yy.sum()), negatives=int((yy == 0).sum()),
                    **{name: float(average_precision_score(yy, d[arm][d['main']][:, ix].ravel())) for name, arm in arms.items()})
            delta = E.paired([v['A2']-v['S2'] for v in per_unit.values()])
            result = dict(AP={name: float(np.mean([v[name] for v in per_unit.values()])) for name in arms},
                          AP_pair=delta, AP_per_unit=per_unit, nominal={}, recalibrated={}, attribution={})
            for name, arm in arms.items():
                result['nominal'][name], result['recalibrated'][name] = {}, {}
                result['attribution'][name] = {}
                for t in C.TARGETS:
                    key = str(t)
                    thr = nominal[g][name][key]
                    stat = E.alert_stats(aud, aud[arm] >= thr)
                    clean_aud = clean_packs[g]['audit']
                    base = E.alert_stats(clean_aud, clean_aud[arm] >= thr)
                    result['nominal'][name][key] = dict(threshold=thr, audit=stat,
                        clean_audit=base, drift_episodes_per_minute=stat['false_episodes_per_simulated_empty_minute']-
                            base['false_episodes_per_simulated_empty_minute'])
                    recal = C.calibrate(cal, arm, t)
                    result['recalibrated'][name][key] = dict(threshold=recal,
                        calib=E.alert_stats(cal, cal[arm] >= recal), audit=E.alert_stats(aud, aud[arm] >= recal))
                    counts = {c: [0, 0] for c in (*classes, 'no_clutter_visible')}
                    for u in splits['audit']:
                        d = ds[u]
                        eligible = (d['frame'] >= 3)[:, None] & (d['y'][:, ix] == 0)
                        alarms = d[arm][:, ix] >= thr
                        masks = {c: diagnostics[u]['class_visible'][:, ix, ci] for ci, c in enumerate(classes)} if k else {
                            c: np.zeros_like(eligible) for c in classes}
                        masks['no_clutter_visible'] = (~diagnostics[u]['any_clutter_visible'][:, None]) if k else np.ones_like(eligible)
                        for c, visible in masks.items():
                            mask = eligible & visible
                            counts[c][0] += int((alarms & mask).sum())
                            counts[c][1] += int(mask.sum())
                    result['attribution'][name][key] = {c: dict(alarm_query_frames=v[0], negative_query_frames=v[1],
                        proportion=v[0]/v[1] if v[1] else None) for c, v in counts.items()}
            # Preserve the literal original-v5 nominal thresholds as well as
            # the clean-calib episode-budget thresholds above. No new fitting.
            result['nominal_original_v5'] = {}
            for name, arm in arms.items():
                result['nominal_original_v5'][name] = {}
                for budget in E.BUDGETS:
                    key = f'{budget:.2f}'
                    thr = frozen_v5['working_points'][f'{g}|{arm}|{key}']['threshold']
                    clean_aud = clean_packs[g]['audit']
                    base = E.alert_stats(clean_aud, clean_aud[arm] >= thr)
                    stat = E.alert_stats(aud, aud[arm] >= thr)
                    result['nominal_original_v5'][name][key] = dict(
                        threshold=thr, audit=stat, clean_audit=base,
                        drift_episodes_per_minute=stat['false_episodes_per_simulated_empty_minute']-
                            base['false_episodes_per_simulated_empty_minute'])
            positive, loss, baseline_zero = 0, 0, 0
            for u in splits['audit']:
                d = ds[u]
                pos = (d['frame'] >= 3)[:, None] & (d['y'][:, ix] == 1)
                positive += int(pos.sum())
                if k:
                    diag = diagnostics[u]
                    loss += int((pos & diag['occlusion_loss'][:, ix]).sum())
                    baseline_zero += int((pos & ~diag['baseline_target_visible'][:, ix]).sum())
            result['occlusion'] = dict(loss_gt50_query_frames=loss, positive_query_frames=positive,
                fraction=loss/positive if positive else None,
                clean_zero_visible_positive_query_frames=baseline_zero if k else None)
            groups[g] = result
            p2 = result['recalibrated']
            print(f"K={k} {g}: AP S2={result['AP']['S2']:.6f} A2={result['AP']['A2']:.6f} "
                  f"delta CI={delta['ci95']} wins={delta['wins']}/{delta['units']}; "
                  f"T2 tiny S2={p2['S2']['2.0']['audit']['tiny']} A2={p2['A2']['2.0']['audit']['tiny']}", flush=True)
        report['doses'][str(k)] = groups
    if '10' in report['doses']:
        report['advantage_retained'] = bool(all(
            r['AP_pair']['ci95'][0] > 0 and
            r['recalibrated']['A2']['2.0']['audit']['tiny']['timely_count'] >=
            r['recalibrated']['S2']['2.0']['audit']['tiny']['timely_count']
            for r in report['doses']['10'].values()))
    E.save(out/report_name, report)
    return report


if __name__ == '__main__':
    main()
