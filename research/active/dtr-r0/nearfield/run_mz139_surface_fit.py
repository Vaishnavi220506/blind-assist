"""Seal observation-only MZ139 fits before scoring selected TRAIN or dev rows.

No test option, automatic successor, threshold selection, or mandatory second fit.
Native volume containment uses a disclosed diagnostic tolerance, never an alert
dilation. Partial TRAIN pilots are implementation diagnostics, not acceptance.
"""
import argparse
import ast
import copy
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
import torch

from mz136_incumbent import public_observations
from mz115_spatial_allocation import possible, certain
from evaluate_mz136_corridor_pair import score, retention, pair_metrics
from run_mz107_four_sensor import ROOT, sha, write, truth, metrics

CODE = Path(__file__).resolve().parent
WORK = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914'
AUDIT_TOLERANCE_M = .04
NUMERICAL_TOLERANCE_M = 1e-6
PRIOR_SAMPLE_MISS_IDS = (
    'mz136_shallow_boundary_stress_scene4_enter_05',
    'mz136_shallow_boundary_stress_scene4_exit_01',
    'mz136_shallow_boundary_stress_scene4_exit_02',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def serial(value):
    return json.loads(json.dumps(value, allow_nan=False))


def replay_payload(prediction):
    """Only nondeterministic elapsed fit time is outside replay equality."""
    value = copy.deepcopy(prediction)
    value.get('fit', {}).pop('seconds', None)
    return value


def selected_jsonl(path, ids):
    """Decode only selected records; unmatched evaluator records stay unparsed."""
    selected = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            match = re.search(r'"id"\s*:\s*"([^"]+)"', line)
            assert match, 'Missing join ID'
            if match[1] in ids:
                selected.append(json.loads(line))
    assert len(selected) == len(ids)
    assert {r['id'] for r in selected} == ids
    return selected


def select_ids(spec, split, pilot=False, limit=None):
    if split not in ('train', 'dev'):
        raise ValueError('Only train and consumed dev are admitted')
    if split != 'train' and (pilot or limit is not None):
        raise ValueError('Pilot and limit are TRAIN only; dev must contain all48')
    frames = [f for f in spec['frames'] if f['split'] == split]
    assert len(frames) == (192 if split == 'train' else 48)
    if pilot:
        counts = Counter()
        chosen = []
        for frame in frames:
            if counts[frame['family']] < 3:
                chosen.append(frame)
                counts[frame['family']] += 1
        frames = chosen
    if limit is not None:
        if not 1 <= limit <= len(frames):
            raise ValueError('TRAIN limit must be within the selected cohort')
        frames = frames[:limit]
    return {f['id'] for f in frames}


def local_dependencies(entry):
    """Hash the local model import closure without loading data or modules."""
    pending = [Path(entry).resolve()]
    found = {}
    while pending:
        path = pending.pop()
        if str(path) in found:
            continue
        found[str(path)] = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        for name in names:
            relative = Path(*name.split('.')).with_suffix('.py')
            for directory in (path.parent, CODE, ROOT/'tools'):
                candidate = (directory/relative).resolve()
                if candidate.is_file() and candidate.is_relative_to(ROOT.resolve()):
                    pending.append(candidate)
                    break
    return found


def verify_train_freeze(path, method, model_sources):
    path = Path(path).resolve()
    if path.is_dir():
        path = path/'freeze.json'
    frozen = read(path)
    assert frozen['split'] == 'train', 'Expected a TRAIN freeze'
    assert frozen['method'] == method, 'Model settings changed after TRAIN freeze'
    assert frozen['model_sources'] == model_sources, 'Model/dependency code changed after TRAIN freeze'
    completion = read(path.parent/'completion.json')
    assert completion['status'] == 'PASS', 'TRAIN run did not complete'
    assert completion['freeze_sha256'] == sha(path)
    return dict(path=str(path), sha256=sha(path), completion_sha256=sha(path.parent/'completion.json'))


def paired_indices(rows, spec, split):
    episodes = {}
    for i, row in enumerate(rows):
        episodes.setdefault(row['episode_id'], []).append(i)
    pairs = []
    for pair in spec['pairs']:
        if pair['split'] != split:
            continue
        a, b = [episodes[ep] for ep in pair['episodes']]
        assert len(a) == len(b)
        assert [rows[i]['time_s'] for i in a] == [rows[i]['time_s'] for i in b]
        pairs.extend(dict(a=i, b=j) for i, j in zip(a, b))
    assert sorted(i for p in pairs for i in (p['a'], p['b'])) == list(range(len(rows)))
    return pairs


def native_audit(row, evaluation, prediction, corrected, point_in_surface):
    """Only evaluator phase calls this function or point_in_surface."""
    zones = {z['zone_id']: z for z in evaluation['zonal_tof_native']}
    replaced = {tuple(k) for k in prediction['replacement_keys']}
    points, returns = [], []
    for item in corrected['spatial_evidence']:
        key = (item['zone_id'], item['target_slot'])
        zone = zones.get(key[0])
        lineage = [] if zone is None else [v for v in zone['returned_lineage'] if v['target_index'] == key[1]]
        rays = {} if zone is None else {r['subray']: r for r in zone['private_rays']}
        hits = [] if len(lineage) != 1 else [rays.get(i) for i in lineage[0]['hit_indices']]
        available = bool(hits) and all(h is not None and 'hit_point_m' in h for h in hits)
        old_possible = possible(item['localized_xyz'])
        old_certain = certain(item['coarse_xyz'])
        record = dict(id=row['id'], zone=key[0], slot=key[1], status=item['status'],
            replaced=key in replaced, native_lineage_available=available,
            incumbent_possible=old_possible, incumbent_certain=old_certain,
            modeled_surface_alert=prediction['surface_candidate'] if key in replaced else None)
        returns.append(record)
        if not available:
            continue
        for hit in hits:
            point = (np.asarray(hit['hit_point_m'])-evaluation['body_origin_m']).tolist()
            corridor = possible([[v, v] for v in point])
            contained = all(lo-1e-9 <= v <= hi+1e-9 for (lo, hi), v in zip(item['localized_xyz'], point))
            modeled_contains = proximity = None
            if key in replaced:
                modeled_contains = any(point_in_surface(point, surface, tolerance=NUMERICAL_TOLERANCE_M)
                                       for surface in prediction['surfaces'])
                proximity = any(point_in_surface(point, surface, tolerance=AUDIT_TOLERANCE_M)
                                for surface in prediction['surfaces'])
            points.append(dict(id=row['id'], zone=key[0], slot=key[1], subray=hit['subray'],
                actor=hit.get('actor_id'), body_point_m=point, corridor=corridor,
                replaced=key in replaced, incumbent_contains=contained,
                incumbent_alert_support=bool(old_possible or old_certain),
                modeled_contains=modeled_contains,
                modeled_sigma_scale_proximity=proximity,
                geometry_retained=bool(modeled_contains) if key in replaced else contained,
                original_support_unchanged=key not in replaced,
                modeled_surface_alert=prediction['surface_candidate'] if key in replaced else None))
    return returns, points


def evaluate(rows, es, predictions, corrected, baseline, radars, spec, split, complete, out, point_in_surface):
    assert [r['id'] for r in rows] == [e['id'] for e in es] == [p['id'] for p in predictions]
    gt = np.asarray([truth(e) for e in es], bool)
    flags = {'mz129': np.asarray([p['candidate'] for p in baseline], bool),
             'surface_fit': np.asarray([p['candidate'] for p in predictions], bool)}
    indices = list(range(len(rows)))
    arms = {name: score(rows, es, gt, values, indices, flags['mz129']) for name, values in flags.items()}
    if split == 'dev':
        assert complete and len(rows) == 48
        assert arms['mz129']['metrics'] == dict(TP=24, FP=13, FN=0, TN=11, UNKNOWN=11)
    cases, returns, points, strata = [], [], [], {}
    for i, (row, e, pred, cached, radar) in enumerate(zip(rows, es, predictions, corrected, radars)):
        target = next(o for o in e['native_bounds'] if o['name'] == 'shape0')
        low = np.asarray(target['center_m'])-target['extent_m']-e['body_origin_m']
        high = np.asarray(target['center_m'])+target['extent_m']-e['body_origin_m']
        overlap = float(min(high[1], .3)-max(low[1], -.3))
        stratum = 'LT_1CM_ABS_GAP_OR_OVERLAP' if abs(overlap) < .01 else 'GE_1CM_ABS_GAP_OR_OVERLAP'
        for label in (stratum, 'shallow_family' if e['family'] == 'shallow_boundary_stress' else 'other_families'):
            strata.setdefault(label, []).append(i)
        rr, pp = native_audit(row, e, pred, cached, point_in_surface)
        returns.extend(rr)
        points.extend(pp)
        cases.append(dict(id=row['id'], family=e['family'], truth=bool(gt[i]),
            baseline=bool(flags['mz129'][i]), candidate=pred['candidate'], state=pred['state'],
            surface_candidate=pred['surface_candidate'], tof_candidate=pred['tof_candidate'],
            native_target_overlap_m=overlap, stratum=stratum,
            native_corridor_contributors=sum(p['corridor'] for p in pp),
            replaced_returns=len(pred['replacement_keys']), surfaces=pred['surfaces'], fit=pred['fit'],
            replacement_keys=pred['replacement_keys'], frozen_radar=dict(candidate=radar['candidate'],
                raw=radar.get('raw_center_support'), current=radar.get('corrected_current_radar'),
                carry=radar.get('inherited_carry_support'), guards=radar.get('guard_events')),
            removed_baseline_fp=bool(not gt[i] and flags['mz129'][i] and not flags['surface_fit'][i]),
            new_fp=bool(not gt[i] and not flags['mz129'][i] and flags['surface_fit'][i]),
            lost_baseline_tp=bool(gt[i] and flags['mz129'][i] and not flags['surface_fit'][i])))
    pairs = paired_indices(rows, spec, split) if complete else None
    for name, result in arms.items():
        m = result['metrics']
        result.update(precision=m['TP']/max(1, m['TP']+m['FP']), recall=m['TP']/max(1, m['TP']+m['FN']),
            retention=retention(result, arms['mz129']),
            strata={s: dict(frames=len(ix), **metrics(gt[ix], flags[name][ix])) for s, ix in strata.items()})
        result['paired_binary'] = pair_metrics(gt, flags[name].astype(float), flags[name], pairs) if pairs is not None else None
        result['joint_target_met'] = bool(complete and result['retention']['pass_retention'] and
                                         arms['mz129']['metrics']['FP'] > 0 and m['FP'] <= .8*arms['mz129']['metrics']['FP'])
    candidate = arms['surface_fit']
    candidate['tof_only'] = score(rows, es, gt, np.asarray([p['tof_candidate'] for p in predictions], bool), indices)
    available = [i for i, p in enumerate(predictions) if p['surface_candidate'] is not None]
    surface_flags = np.asarray([bool(p['surface_candidate']) for p in predictions], bool)
    # Conditional surface-only confusion; unavailable outputs are reported separately.
    surface_only = dict(available=len(available), unavailable=len(rows)-len(available),
        conditional_metrics=metrics(gt[available], surface_flags[available]),
        unavailable_positive=int(sum(gt[i] for i in indices if i not in available)))
    audit = dict(returns=len(returns), replaced_returns=sum(r['replaced'] for r in returns),
        missing_native_lineage_returns=sum(not r['native_lineage_available'] for r in returns),
        contributors=len(points), corridor_contributors=sum(p['corridor'] for p in points),
        original_support_unchanged_contributors=sum(p['original_support_unchanged'] for p in points),
        replaced_contributors=sum(p['replaced'] for p in points),
        replaced_outside_fitted_volume=sum(p['replaced'] and not p['modeled_contains'] for p in points),
        corridor_outside_fitted_volume=sum(p['replaced'] and p['corridor'] and not p['modeled_contains'] for p in points),
        previously_contained_corridor_excluded=sum(p['replaced'] and p['corridor'] and p['incumbent_contains'] and not p['modeled_contains'] for p in points),
        corridor_prior_support_replaced_by_nonalert_surface=sum(p['replaced'] and p['corridor'] and p['incumbent_alert_support'] and p['modeled_surface_alert'] is False for p in points),
        numerical_tolerance_m=NUMERICAL_TOLERANCE_M,
        sigma_scale_proximity=dict(tolerance_m=AUDIT_TOLERANCE_M,
            replaced_outside=sum(p['replaced'] and not p['modeled_sigma_scale_proximity'] for p in points),
            corridor_outside=sum(p['replaced'] and p['corridor'] and not p['modeled_sigma_scale_proximity'] for p in points),
            semantics='SIGMA_SCALE_PROXIMITY_ONLY_NOT_NATIVE_EVIDENCE_RETENTION_OR_ALERT_DILATION'))
    summary = dict(split=split, frames=len(rows), complete_partition=complete,
        authority='CONSUMED_DEVELOPMENT_ONLY' if split == 'dev' else 'TRAIN_IMPLEMENTATION_DIAGNOSTIC',
        arms=arms, surface_only=surface_only, native_retention=audit,
        states=dict(Counter(p['state'] for p in predictions)),
        three_prior_sample_miss_cases=[c for c in cases if c['id'] in PRIOR_SAMPLE_MISS_IDS],
        suppressed_fp_geometry_trace=[c for c in cases if c['removed_baseline_fp']],
        decision=('TRAIN_IMPLEMENTATION_DIAGNOSTIC_NO_DEV_PROMOTION' if split == 'train' else
                  'CONSUMED_DEV_JOINT_GAIN_REQUIRES_INDEPENDENT_CONFIRMATION' if candidate['joint_target_met'] else
                  'KEEP_MZ129_NO_CONSUMED_DEV_JOINT_GAIN'),
        limits='Partial TRAIN event summaries describe only selected fragments. Binary pair ordering is not continuous ranking. Missing modeled surfaces are unavailable, not clear. The .04m proximity audit is not native evidence retention or a calibrated enclosure.')
    write(out/'native-returns.json', returns)
    write(out/'native-contributors.json', points)
    write(out/'cases.json', cases)
    write(out/'summary.json', summary)
    return summary


def run(args):
    ids = select_ids(read(WORK/'source/returned-v1/capture-v1/spec.json'), args.split, args.pilot, args.limit)
    out = args.output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    model = importlib.import_module('mz139_surface_fit')
    method = serial(model.METHOD)
    model_sources = local_dependencies(model.__file__)
    train_record = verify_train_freeze(args.require_train_freeze, method, model_sources) if args.require_train_freeze else None
    cap = WORK/'source/returned-v1/capture-v1'
    prep = WORK/'incumbent/fresh-v1'
    spec, receipt = read(cap/'spec.json'), read(cap/'receipt.json')
    assert receipt['status'] == 'PASS' and sha(cap/'spec.json') == receipt['spec_sha256']
    # Hash evaluator bytes now; parse selected evaluator records only after fitting.
    for name in ('raw.jsonl', 'evaluator.jsonl'):
        assert sha(cap/name) == receipt['hashes'][name]
    rows = public_observations(selected_jsonl(cap/'raw.jsonl', ids))
    complete = len(rows) == (192 if args.split == 'train' else 48)
    done, seal = read(prep/'completion.json'), read(prep/'prediction-seal.json')
    assert done['status'] == 'PASS' and sha(prep/'prediction-seal.json') == done['prediction_seal_sha256']
    assert sha(prep/'nominal/predictions.json') == seal['predictions_sha256']['nominal']
    assert sha(prep/'observation-seal.json') == seal['observation_seal_sha256']
    assert sha(prep/'nominal/raw.jsonl') == read(prep/'observation-seal.json')['hashes']['nominal']
    assert rows == selected_jsonl(prep/'nominal/raw.jsonl', ids)
    for path, digest in seal['source_hashes'].items():
        assert sha(Path(path)) == digest
    cache = read(prep/'nominal/predictions.json')
    mapping = {p['id']: i for i, p in enumerate(cache['predictions'])}
    assert len(mapping) == len(cache['predictions'])
    baseline, corrected, radars = [[cache[key][mapping[r['id']]] for r in rows]
                                   for key in ('predictions', 'corrected', 'radar')]
    paths = [cap/n for n in ('spec.json', 'receipt.json', 'raw.jsonl', 'evaluator.jsonl')]
    paths += [prep/n for n in ('completion.json', 'prediction-seal.json', 'observation-seal.json',
                               'nominal/raw.jsonl', 'nominal/predictions.json')]
    images = []
    for row in rows:
        path = (cap/row['rgb_path']).resolve()
        assert path.is_relative_to(cap.resolve()) and sha(path) == receipt['hashes'][row['rgb_path']]
        image = cv2.imread(str(path))
        assert image is not None and image.shape == (360, 640, 3)
        images.append(image)
        paths.append(path)
    inputs = {str(path): sha(path) for path in paths}
    sources = dict(model_sources)
    sources.update(local_dependencies(__file__))
    backend_source = (ROOT/'tools/research_backend.py').resolve()
    sources[str(backend_source)] = sha(backend_source)
    protocol = CODE/'MZ139_SURFACE_FIT_20260915.md'
    assert protocol.is_file(), 'MZ139 protocol must exist before fit'
    requested = torch.device(args.device)
    if requested.type == 'cuda':
        assert torch.cuda.is_available(), 'Requested CUDA runtime is unavailable'
        torch.cuda.set_device(requested)
        device_name = torch.cuda.get_device_name(requested)
    elif requested.type == 'cpu':
        device_name = 'CPU'
    else:
        raise ValueError('Supported fitting devices: cpu or cuda')
    out.mkdir(parents=True)
    for path in sources:
        destination = out/'source-snapshot'/Path(path).relative_to(ROOT.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(protocol, out/'protocol-before-fit.md')
    write(out/'freeze.json', dict(split=args.split, complete_partition=complete, ids=[r['id'] for r in rows],
        method=method, model_sources=model_sources, sources=sources, inputs=inputs,
        protocol_sha256=sha(out/'protocol-before-fit.md'), train_freeze=train_record,
        native_containment_numerical_tolerance_m=NUMERICAL_TOLERANCE_M,
        sigma_scale_proximity_tolerance_m=AUDIT_TOLERANCE_M,
        selection=dict(pilot=args.pilot, limit=args.limit), replay_requested=args.replay))
    write(out/'backend.json', dict(requested_device=str(requested), device_type=requested.type,
        device_name=device_name, python=sys.executable, torch=torch.__version__,
        torch_cuda=torch.version.cuda, opencv=cv2.__version__, numpy=np.__version__,
        placement='USER_SELECTED_RUNTIME; CPU_CUDA_RENDERER_BENCHMARK_RECORDED_SEPARATELY',
        no_automatic_backend_benchmark=True))

    def fit_one(i):
        result = serial(model.fit_frame(copy.deepcopy(rows[i]), images[i].copy(), copy.deepcopy(corrected[i]),
            copy.deepcopy(baseline[i]), copy.deepcopy(radars[i]), device=str(requested)))
        assert result['id'] == rows[i]['id']
        for key in ('candidate', 'tof_candidate'):
            assert isinstance(result[key], bool), key
        assert result['surface_candidate'] is None or isinstance(result['surface_candidate'], bool)
        assert result['candidate_state'] == ('ALERT' if result['candidate'] else 'UNKNOWN')
        keys = [tuple(k) for k in result['replacement_keys']]
        original = {(e['zone_id'], e['target_slot']) for e in corrected[i]['spatial_evidence']}
        assert len(keys) == len(set(keys)) and set(keys).issubset(original)
        assert not radars[i]['candidate'] or result['candidate'], 'Frozen Radar support was lost'
        assert isinstance(result['surfaces'], list) and isinstance(result['fit'], dict)
        assert isinstance(result['state'], str)
        return result

    try:
        if requested.type == 'cuda':
            torch.cuda.synchronize(requested)
        started = time.perf_counter()
        predictions = []
        for i in range(len(rows)):
            predictions.append(fit_one(i))
            # Unsealed progress retains successful fits if a later frame fails.
            with (out/'fit-progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(predictions[-1], allow_nan=False)+'\n')
            print(json.dumps(dict(completed=i+1, frames=len(rows), id=rows[i]['id'], state=predictions[-1]['state'])), flush=True)
        if requested.type == 'cuda':
            torch.cuda.synchronize(requested)
        elapsed = time.perf_counter()-started
        write(out/'predictions.json', predictions)
        write(out/'prediction-seal.json', dict(predictions_sha256=sha(out/'predictions.json'),
            freeze_sha256=sha(out/'freeze.json'), seconds=elapsed,
            authority='PUBLIC_OBSERVATION_PREDICTIONS_SAVED_BEFORE_SELECTED_NATIVE_PARSE'))
        es = selected_jsonl(cap/'evaluator.jsonl', ids)
        summary = evaluate(rows, es, predictions, corrected, baseline, radars, spec, args.split,
                           complete, out, model.point_in_surface)
        replay = None
        if args.replay:
            replay = [replay_payload(fit_one(i)) for i in range(len(rows))] == [replay_payload(p) for p in predictions]
            assert replay, 'Explicit replay differs from sealed fit'
        assert inputs == {p: sha(Path(p)) for p in inputs}
        assert sources == {p: sha(Path(p)) for p in sources}
        write(out/'completion.json', dict(status='PASS', freeze_sha256=sha(out/'freeze.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'), summary_sha256=sha(out/'summary.json'),
            inputs_unchanged=True, model_sources_unchanged=True, replay_equal_excluding_fit_seconds=replay,
            replay_performed=args.replay, seconds=elapsed,
            resources='Foreground process only; no persistent worker or allocation; durable fits and evidence retained'))
        print(json.dumps(dict(decision=summary['decision'], arms={a: r['metrics'] for a, r in summary['arms'].items()},
                              native=summary['native_retention']), indent=2))
        return summary
    except Exception as exc:
        write(out/'failure.json', dict(status='FAIL', exception=type(exc).__name__, message=str(exc),
            predictions_sealed=(out/'prediction-seal.json').exists(), resources='No persistent workers; diagnostic files retained'))
        raise
    finally:
        if requested.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'dev'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument('--pilot', action='store_true')
    subset.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--require-train-freeze', type=Path)
    parser.add_argument('--replay', action='store_true')
    run(parser.parse_args())
