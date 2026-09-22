"""Sealed angular/quantization information diagnostic, with zero inference calls."""
from collections import Counter
import argparse
import json
from pathlib import Path
import shutil
import time

import active_view as av
import boundary_observability as probe

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'artifacts.local/work/ba-bias-drift-20260922/run-v1'
PARENT = ROOT/'artifacts.local/work/ba-boundary-observability-20260922'
DOCS = ROOT/'research/active/dtr-r0/nearfield'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate(output):
    source = read(output/'source.json')
    observations = read(output/'observations.json')
    lookup = {(r['id'], r['schedule']): r['views'] for r in observations}
    pairs = {}
    for row in source:
        if row['pair']:
            pairs.setdefault(row['pair'], []).append(row)
    rows = []
    for name, pair in pairs.items():
        assert len(pair) == 2 and {r['truth'] for r in pair} == {True, False}
        assert len({r['stratum'] for r in pair}) == 1
        for schedule in probe.schedules():
            rows.append(dict(pair=name, stratum=pair[0]['stratum'], ids=[r['id'] for r in pair],
                             schedule=schedule,
                             comparison=probe.compare(lookup[pair[0]['id'], schedule], lookup[pair[1]['id'], schedule])))
    av.write_json(output/'pair-evaluation.json', rows)
    metrics, contrasts = {}, {}
    for face in ('all', 'boundary_left', 'boundary_right', 'boundary_far'):
        metrics[face], contrasts[face] = {}, {}
        for schedule in probe.schedules():
            selected = [r for r in rows if r['schedule'] == schedule and (face == 'all' or r['stratum'] == face)]
            metrics[face][schedule] = dict(pairs=len(selected),
                raw_distinct_pairs=[r['pair'] for r in selected if r['comparison']['raw_distinct']],
                hit_distinct_pairs=[r['pair'] for r in selected if r['comparison']['hit_mismatch_rays']],
                separated={p: [r['pair'] for r in selected if r['comparison']['changed_bins'][p]] for p in probe.STEPS})
        for schedule in probe.schedules():
            contrasts[face][schedule] = {}
            for p in probe.STEPS:
                base = set(metrics[face]['fixed']['separated'][p])
                new = set(metrics[face][schedule]['separated'][p])
                contrasts[face][schedule][p] = dict(gained=sorted(new-base), lost=sorted(base-new), retained=sorted(new&base))
            coarse = set(metrics[face][schedule]['separated']['coarse'])
            fine = set(metrics[face][schedule]['separated']['fine'])
            contrasts[face][schedule]['fine_vs_coarse'] = dict(gained=sorted(fine-coarse), lost=sorted(coarse-fine))
    counterexamples = []
    for row in source:
        if row['stratum'] not in ('boundary_left', 'boundary_right'):
            continue
        witness = probe.lateral_counterexample(row)
        alternate = probe.scene(witness['alternative'])
        captures, checks = {}, {}
        for schedule in probe.schedules():
            captures[schedule] = probe.capture(alternate, schedule, witness['bias']['pose_x_m'])
            checks[schedule] = probe.compare(lookup[row['id'], schedule], captures[schedule])
        valid = (witness['domain_valid'] and witness['opposite_label'] and
                 abs(witness['bias']['pose_x_m']) <= .001 and
                 all(c['max_raw_difference_m'] <= probe.RAW_TOL and not any(c['changed_bins'].values()) and
                     c['hit_mismatch_rays'] == 0 for c in checks.values()))
        counterexamples.append(dict(**witness, checks=checks, valid=valid, captures=captures))
    av.write_json(output/'counterexamples.json', counterexamples)
    allchanges = {}
    for schedule in probe.schedules():
        counters = Counter()
        for row in source:
            c = probe.compare(lookup[row['id'], 'fixed'], lookup[row['id'], schedule])
            for p in probe.STEPS:
                counters[p+'_changed_histories'] += bool(c['changed_bins'][p])
                counters[p+'_changed_bins'] += c['changed_bins'][p]
        allchanges[schedule] = dict(counters)
    result = dict(kind='CONSUMED_BOUNDARY_OBSERVATION_INFORMATION_DIAGNOSTIC', scenes=180, boundary_pairs=len(pairs),
                  schedules={s: list(v) for s, v in probe.schedules().items()}, quantization_m=probe.STEPS,
                  pair_metrics=metrics, pair_contrasts=contrasts, all_scene_changes=allchanges,
                  counterexamples=len(counterexamples), valid_counterexamples=sum(r['valid'] for r in counterexamples),
                  invalid_counterexample_ids=[r['id'] for r in counterexamples if not r['valid']],
                  max_counterexample_raw_residual_m=max(c['max_raw_difference_m'] for r in counterexamples for c in r['checks'].values()),
                  base_view_records=len(observations)*13, base_ray_returns=len(observations)*13*8,
                  counterexample_view_replays=len(counterexamples)*3*13,
                  counterexample_ray_replays=len(counterexamples)*3*13*8,
                  solver_calls=0, recognition_metrics_not_evaluated=True)
    av.write_json(output/'summary.json', result)
    return result


def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite')
    output.mkdir(parents=True)
    start, stages = time.perf_counter(), []

    def stage(name):
        stages.append(dict(name=name, seconds=time.perf_counter()-start))
        print(json.dumps(stages[-1]), flush=True)

    try:
        av.verify_seal(OLD/'completion-seal.json')
        hashes = {str(p.resolve()): av.file_hash(p) for p in OLD.iterdir() if p.is_file()}
        av.write_json(output/'old-input-hashes.json', hashes)
        files = [Path(__file__), Path(probe.__file__), Path(av.__file__), Path(__file__).with_name('bent_path_observation.py'),
                 Path(__file__).with_name('test_boundary_observability.py'), DOCS/'BOUNDARY_OBSERVABILITY_PROTOCOL_20260922.md']
        for p in files:
            shutil.copyfile(p, output/p.name)
        av.seal(output/'design-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('design_sealed_before_capture')
        source = read(OLD/'source.json')
        assert len(source) == 180
        av.write_json(output/'source.json', source)
        observations = []
        for row in source:
            value = probe.scene(row)
            for schedule in probe.schedules():
                observations.append(dict(id=row['id'], schedule=schedule, views=probe.capture(value, schedule)))
        av.write_json(output/'observations.json', observations)
        av.seal(output/'observation-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('all_observations_sealed_before_pair_scoring_and_counterexamples')
        av.verify_seal(output/'observation-seal.json')
        oldnominal = {r['id']: r['views'] for r in read(OLD/'observations.json') if r['condition'] == 'nominal'}
        identity_rays = 0
        for row in observations:
            if row['schedule'] == 'fixed':
                for old, new in zip(oldnominal[row['id']], row['views'], strict=True):
                    assert old['bins'] == new['bins']['coarse'] and old['raw_ranges'] == new['raw_ranges']
                    identity_rays += len(old['bins'])
        av.write_json(output/'baseline-replay.json', dict(exact_raw_and_coarse_identity_rays=identity_rays))
        result = evaluate(output)
        assert all(av.file_hash(Path(p)) == h for p, h in hashes.items())
        stage('evaluation_complete')
        av.write_json(output/'stage-order.json', stages)
        av.seal(output/'completion-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status='COMPLETE', pair_counts={s: {p: len(v) for p, v in r['separated'].items()}
                          for s, r in result['pair_metrics']['all'].items()},
                          valid_counterexamples=result['valid_counterexamples'], seconds=time.perf_counter()-start)), flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), stages=stages))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if PARENT.resolve() not in args.output.resolve().parents:
        parser.error('Output must be a new child of the canonical boundary observability root')
    run(args.output)
