"""Authenticate old captures; separate public tokens from native supervision."""
from pathlib import Path
import sys
import time
import json
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parent), str(ROOT/'tools')]
from public_return_tokens import encode_tokens
from mz136_incumbent import public_observations
from run_mz139_surface_fit import selected_jsonl
from mz171_return_labels import make_witness_labels
from tolerance_eval import geometry, classify

WORK = ROOT/'artifacts.local/work'
OUT = WORK/'corridor-public-positive-20260917/preparation'
E1 = WORK/'corridor-depth-e1-20260917/run-v2'
CAPTURES = {
    'anchor': WORK/'mz136-corridor-pair-20260914/source/returned-v1/capture-v1',
    'old': WORK/'mz170-mean-confirmation-20260916/source/returned-v1/capture-v1',
    'new': WORK/'corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'data-seal.json').exists()
    start = time.perf_counter()
    bindings = {}
    def bind(p, expected=None):
        h = sha(p)
        if expected is not None:
            assert h == expected, str(p)
        bindings[str(p)] = h
    for p in [Path(__file__), HERE/'public_return_tokens.py', HERE/'public_positive.py',
              HERE/'PUBLIC_POSITIVE_PROTOCOL_20260917.md', HERE.parent/'mz171_return_labels.py',
              HERE.parent/'mz143_corridor_features.py', HERE.parent/'mz115_spatial_allocation.py',
              HERE.parent/'mz136_boundary_geometry.py', HERE/'tolerance_eval.py']:
        bind(p)
    bind(OUT/'token-parity-audit.json')
    assert read(OUT/'token-parity-audit.json')['status'] == 'PASS'
    source_token = WORK/'mz161-dense-task-20260916/run-v1'
    bind(source_token/'public-tokens.npz', read(source_token/'input-seal.json')['tokens_sha256'])
    cached = np.load(source_token/'public-tokens.npz')
    cached_ids = [r['id'] for r in read(source_token/'input-audit.json')]
    e1seal = read(E1/'prediction-seal.json')
    bind(E1/'report-scores.npz', e1seal['scores_sha256'])
    bind(E1/'selection-seal.json', e1seal['selection_sha256'])
    sel = read(E1/'selection-seal.json')
    threshold = sel['selection']['A']['dev_best']['threshold']
    bind(E1/'A-model.pkl', 'd1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef')
    bind(E1/'freeze.json')
    oldscores = np.load(E1/'report-scores.npz')['A']
    oldids = read(E1/'freeze.json')['ids']['report']
    conf = WORK/'corridor-depth-confirmation-20260917/evaluation-v1'
    bind(conf/'predictions.json', read(conf/'prediction-seal.json')['predictions_sha256'])
    newpred = read(conf/'predictions.json')
    amap = dict(zip(oldids, oldscores)) | {p['id']: p['A_score'] for p in newpred}
    xs, masks, ys, ks, metadata, audits, times = [], [], [], [], [], [], []
    for cohort, cap in CAPTURES.items():
        spec, receipt = read(cap/'spec.json'), read(cap/'receipt.json')
        assert receipt['status'] == 'PASS'
        bind(cap/'spec.json', receipt['spec_sha256'])
        bind(cap/'receipt.json')
        for name in ['raw.jsonl', 'evaluator.jsonl']:
            bind(cap/name, receipt['hashes'][name])
        frames = {r['id']: r for r in spec['frames'] if cohort != 'anchor' or r['split'] == 'train'}
        rows = public_observations(selected_jsonl(cap/'raw.jsonl', set(frames)))
        es = selected_jsonl(cap/'evaluator.jsonl', set(frames))
        assert [r['id'] for r in rows] == [e['id'] for e in es]
        yaw, episode = 0., None
        for ri, (row, e) in enumerate(zip(rows, es)):
            if row['episode_id'] != episode:
                yaw = 0.
            if row['imu_valid']:
                yaw += row['delta_yaw']
            episode = row['episode_id']
            tick = time.perf_counter()
            public = encode_tokens(row, yaw)
            times.append(time.perf_counter()-tick)
            if cohort == 'anchor':
                assert cached_ids[ri] == row['id']
                assert np.array_equal(public['tokens'], cached['tokens'][ri])
                assert np.array_equal(public['valid'], cached['valid'][ri])
            lab = make_witness_labels(row, e)
            assert not np.any(lab['known'] & ~public['valid'])
            assert not lab['known'][128:].any()
            f = frames[row['id']]
            g = geometry(e)
            metadata.append(dict(id=row['id'], cohort=cohort, family=f['family'],
                group=f['scene_group'], scene_index=int(f['scene_group'].rsplit('_scene', 1)[1]),
                episode_id=row['episode_id'], time_s=row['time_s'], packet=row['tof_packet_received'],
                truth=g['strict'], stratum=classify(g, .05),
                A_score=None if cohort == 'anchor' else float(amap[row['id']]),
                A=None if cohort == 'anchor' else bool(amap[row['id']] >= threshold)))
            xs.append(public['tokens']); masks.append(public['valid'])
            ys.append(lab['target']); ks.append(lab['known'])
            audits.append(dict(id=row['id'], public=public['audit'], **{k: v for k, v in lab['audit'].items() if k != 'slots'}))
        print(json.dumps(dict(stage='prepared', cohort=cohort, frames=len(rows), seconds=time.perf_counter()-start)), flush=True)
    x, valid = np.stack(xs), np.stack(masks)
    y, known = np.stack(ys), np.stack(ks)
    assert x.shape == (768, 132, 21)
    # Reconstructed anchor targets must also match the complete old supervised cache.
    oldlab = WORK/'mz171-return-witness-20260916/run-v1'
    bind(oldlab/'evaluation-witness-labels.npz', read(oldlab/'completion.json')['outputs']['evaluation-witness-labels.npz'])
    z = np.load(oldlab/'evaluation-witness-labels.npz')
    assert np.array_equal(y[:192], z['target']) and np.array_equal(known[:192], z['known'])
    # Existing changed-domain sampled oracle is a label audit, never a feature.
    oracle = WORK/'corridor-surface-oracle-20260917/oracle-returns.json'
    records = {r['id']: r for r in read(oracle)}
    for i, m in enumerate(metadata):
        if m['cohort'] == 'new':
            assert bool((y[i] > 0).any()) == records[m['id']]['sampled_point_reachable']
    bind(oracle)
    np.savez_compressed(OUT/'public-tokens.npz', tokens=x, valid=valid, ids=[m['id'] for m in metadata])
    np.savez_compressed(OUT/'offline-targets.npz', target=y, known=known)
    write(OUT/'metadata.json', metadata)
    write(OUT/'label-audit.json', audits)
    folds = []
    for k in range(6):
        report = [i for i,m in enumerate(metadata) if m['cohort'] != 'anchor' and m['scene_index'] == k]
        cal = [i for i,m in enumerate(metadata) if m['cohort'] != 'anchor' and m['scene_index'] == (k+1)%6]
        fit = [i for i,m in enumerate(metadata) if m['cohort'] == 'anchor' or m['scene_index'] not in (k, (k+1)%6)]
        assert [len(fit), len(cal), len(report)] == [576, 96, 96]
        groups = [{metadata[i]['group'] for i in ix} for ix in [fit,cal,report]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        folds.append(dict(fold=k, fit=fit, calibration=cal, report=report,
                          groups=dict(zip(['fit','calibration','report'], map(sorted, groups)))))
    write(OUT/'folds.json', folds)
    write(OUT/'preparation-summary.json', dict(frames=768, original_test_decoded=False,
        source_token_and_label_parity=True, positive_slots=int(y.sum()), known_slots=int(known.sum()),
        all_public_usable_tof_known=bool(np.array_equal(valid[:, :128], known[:, :128])),
        public_extraction_ms=dict(mean=float(np.mean(times)*1000), p50=float(np.percentile(times,50)*1000), p95=float(np.percentile(times,95)*1000)),
        seconds=time.perf_counter()-start))
    assert all(sha(Path(p)) == h for p,h in bindings.items())
    write(OUT/'data-seal.json', dict(bindings=bindings, A_threshold=threshold,
        outputs={n:sha(OUT/n) for n in ['public-tokens.npz','offline-targets.npz','metadata.json','label-audit.json','folds.json','preparation-summary.json']},
        authority='PUBLIC_TOKENS_SEPARATE_FROM_OFFLINE_NATIVE_TARGETS_CONSUMED_GROUPED_DEVELOPMENT'))


if __name__ == '__main__':
    main()
