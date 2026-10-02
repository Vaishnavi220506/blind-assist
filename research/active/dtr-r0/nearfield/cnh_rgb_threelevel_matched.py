"""Matched ToF-family successor to the frozen BASE-only RGB opportunity screen.

prepare freezes completed margin results, producer scores and model/input hashes.
Existing matched scores are reused; no duplicate GPU inference is launched.
No training, rendering, downloads or automatic physical follow-up. All policies
are selected on consumed Development; bootstrap intervals condition on selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import cnh_rgb_threelevel_gate as G

ROOT, WORK = G.ROOT, G.WORK
OUT = WORK/'cnh-rgb-threelevel-matched-20261002'
ENVELOPE = WORK/'cnh-detectability-envelope-20261001'
MARGIN = WORK/'cnh-margin-labels-20261002'
UNITS = np.arange(92000, 92096)
MARGINS = [None, *range(-30, 21)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def paths(arm):
    import cnh_readout_fix as RF
    if arm == 'BASE':
        return [RF.model_path('BASE', s) for s in range(5)]
    source = WORK/'cnh-near-range-20261001' if arm == 'NEAR' else MARGIN
    return [source/'models'/arm/f'model_seed{s}.pt' for s in range(5)]


def validate_rows():
    """Validate exact (unit, config, query) association, final labels and family."""
    data = load_npz(G.SOURCE/'rows.npz')
    meta = load_npz(ENVELOPE/'data/evaluation/metadata.npz')
    base = load_npz(ENVELOPE/'base_ensemble.npz')
    np.testing.assert_array_equal(np.unique(data['unit']), UNITS)
    np.testing.assert_array_equal(np.unique(meta['unit']), UNITS)
    assert len(data['s']) == 7680 and len(meta['unit']) == 19200
    assert len(set(zip(data['unit'], data['config'], data['q']))) == 7680
    for ui, u in enumerate(UNITS):
        ids = data['unit'] == u
        np.testing.assert_array_equal(data['ui'][ids], ui)
        np.testing.assert_array_equal(data['s'][ids], base[str(u)][data['config'][ids], data['q'][ids]])
        for c in range(40):
            ii = np.flatnonzero((meta['unit'] == u) & (meta['config'] == c))
            ii = ii[np.argsort(meta['frame'][ii])]
            np.testing.assert_array_equal(meta['frame'][ii], np.arange(11, 16))
            for q in (0, 1):
                k = np.flatnonzero(ids & (data['config'] == c) & (data['q'] == q))
                assert len(k) == 1
                assert data['label'][k[0]] == meta['labels'][ii[-1], q]
                assert data['family'][k[0]] == meta['family'][ii[-1]]
    assert np.isfinite(data['s']).all()
    return data, meta


def prepare():
    # Do not create a partial plan while the owner is still training/evaluating.
    result_path = MARGIN/'result.json'
    if not result_path.exists():
        raise RuntimeError('WAIT: completed margin result.json required before prepare')
    margin = read(result_path)
    assert margin['status'] == 'COMPLETE'
    verdicts = {a: margin['primary'][a]['verdict'] for a in ('M3', 'M8')}
    assert all(v in ('HELPS', 'THRESHOLD_SUFFICES') for v in verdicts.values())
    arms = ['BASE', 'NEAR'] + (['M3', 'M8'] if 'HELPS' in verdicts.values() else [])
    data, meta = validate_rows()
    files = [Path(__file__), Path(G.__file__), result_path, MARGIN/'PLAN.json',
             G.SOURCE/'rows.npz', G.SOURCE/'PLAN.json', G.OUT/'PLAN.json', G.OUT/'result.json',
             ENVELOPE/'base_ensemble.npz', ENVELOPE/'data/evaluation/features.npy',
             ENVELOPE/'data/evaluation/metadata.npz']
    for name in ('cnh_structure_space.py', 'cnh_readout_fix.py', 'cnh_cvr_pilot.py',
                 'cnh_cvr_projection.py', 'cnh_detectability_envelope.py'):
        files.append(Path(__file__).with_name(name))
    models = {a: [str(p.relative_to(ROOT)) for p in paths(a)] for a in arms}
    files += [ROOT/p for pp in models.values() for p in pp]
    cached_sources = {a: str((ENVELOPE/'base_ensemble.npz' if a == 'BASE' else MARGIN/f'scores92000_{a}.npz').relative_to(ROOT)) for a in arms}
    files += [ROOT/p for p in cached_sources.values()]
    for rel in cached_sources.values():
        z = load_npz(ROOT/rel)
        assert set(z) == set(map(str, UNITS))
        assert all(v.shape == (40, 2) and np.isfinite(v).all() for v in z.values())
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    source_plan = read(G.OUT/'PLAN.json')
    plan = dict(scope='consumed synthetic Development; matched same-table conditional opportunity',
        units=UNITS.tolist(), rows=len(data['s']), voxel_rows=len(meta['unit']), models=models, cached_sources=cached_sources,
        margin_verdicts=verdicts, inclusion='BASE and NEAR always; BOTH M3/M8 if either original arm HELPS',
        budgets=[.10, .20], heading_deg=[0., 1.], clearance_sigma_cm=2., margins_cm=MARGINS,
        report_ranges=source_plan['report_ranges'], primary_range=[1.2, 2.1], hashes=hashes,
        inference='Reuse producer infer92 caches; inspected exact detectability-envelope recipe: batch128, SS.prep, 5-seed logit mean, float64 causal weights [1,2,4,8,16]/31',
        canary='No new inference. BASE copied exactly, every row keyed to original metadata; NEAR cache recipe inspected and producer source retained.',
        budget='separate query budgets on all-distance other-height label0 clear rows; near-pass unpenalized',
        primary_policy='one ToF model and one oracle model/common margin, each maximizing pooled contact0-2 then contact2-5 then lower summed clear burden then lower frozen index',
        stop_envelope='for EACH contact bin separately compare best oracle over all models/margins against best ToF model. These two pairs need not be simultaneously attainable; preserve primary shared pair too.',
        rule='At heading1 budget10%, point PAUSE iff BOTH separate-bin envelope gains <3pp. Qualified PAUSE_MATCHED_FAMILY requires valid support and both fixed-policy unit-bootstrap upper95 <3pp; otherwise UNCERTAIN. Any >=3pp is CONTINUE_OPPORTUNITY only.',
        support=source_plan['support'], bootstrap='1000 paired 96-unit resamples seed2026100209, selected policies fixed; conditional CI does not correct same-data selection',
        unchanged_original=dict(path=str(G.OUT.relative_to(ROOT)), result_sha256=sha(G.OUT/'result.json')),
        limits=source_plan['limits']+['limited XZ nominal geometry oracle contains invisible surfaces; not achievable RGB or global upper bound',
            'strongest permitted family, not all possible ToF algorithms; no per-row model switching',
            'no physical histogram or new-scene step automatically follows this result'],
        backend='TASK_NOT_GPU_SUITABLE: cache copy, CPU scalar analytic integration and clustered bootstrap')
    OUT.mkdir(exist_ok=False)
    producer = Path(__file__).with_name('cnh_margin_labels.py')
    (OUT/'producer-source.py').write_bytes(producer.read_bytes())
    plan['producer_source_sha256'] = sha(OUT/'producer-source.py')
    G.save(OUT/'PLAN.json', plan)
    G.save(OUT/'margin_result_snapshot.json', margin)
    G.save(OUT/'original_base_result_snapshot.json', read(G.OUT/'result.json'))
    print(json.dumps({'prepared': str(OUT), 'arms': arms}))


def frozen():
    plan = read(OUT/'PLAN.json')
    for rel, digest in plan['hashes'].items():
        assert sha(ROOT/rel) == digest, f'frozen input changed: {rel}'
    return plan


def aggregate(raw, meta, units):
    out = {}
    for u in units:
        sel = np.flatnonzero(meta['unit'] == u)
        values = []
        for c in range(40):
            loc = sel[meta['config'][sel] == c]
            loc = loc[np.argsort(meta['frame'][loc])]
            np.testing.assert_array_equal(meta['frame'][loc], np.arange(11, 16))
            values.append((raw[loc].astype(np.float64)*np.array([1, 2, 4, 8, 16])[:, None]).sum(0)/31)
        out[str(u)] = np.asarray(values)
    return out


def import_scores():
    plan = frozen()
    for arm, rel in plan['cached_sources'].items():
        source = ROOT/rel
        destination = OUT/f'scores_{arm}.npz'
        receipt = OUT/f'inference_{arm}.json'
        assert not destination.exists() and not receipt.exists(), 'preserve imported scores'
        # Exact byte copy retains the producer array identity without rerunning a model.
        destination.write_bytes(source.read_bytes())
        record = dict(status='COMPLETE', arm=arm, execution='REUSED_EXISTING_MATCHED_CACHE',
                      source=rel, source_sha256=sha(source), score_sha256=sha(destination),
                      plan_sha256=sha(OUT/'PLAN.json'), new_inference_rows=0)
        assert record['source_sha256'] == record['score_sha256']
        G.save(receipt, record)
    print('Imported matching frozen score caches; no inference')


def choose(rates, indices, primary, secondary, definitions):
    return min(indices, key=lambda i: (-rates[i, primary], -rates[i, secondary],
                                      sum(definitions[i]['clear_per_query']), i))


def evaluate():
    plan = frozen()
    assert not (OUT/'result.json').exists(), 'preserve prior result'
    data, _ = validate_rows()
    model_scores = {}
    for a in plan['models']:
        receipt = read(OUT/f'inference_{a}.json')
        assert receipt['status'] == 'COMPLETE' and receipt['plan_sha256'] == sha(OUT/'PLAN.json')
        assert receipt['score_sha256'] == sha(OUT/f'scores_{a}.npz')
        z = load_npz(OUT/f'scores_{a}.npz')
        model_scores[a] = np.array([z[str(u)][c, q] for u, c, q in zip(data['unit'], data['config'], data['q'])])
    np.testing.assert_array_equal(model_scores['BASE'], data['s'])
    rng = np.random.default_rng(2026100209)
    boot = np.array([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(1000)])
    result = dict(plan=plan, checks=G.fixtures(), runs={})
    original = read(G.OUT/'result.json')
    ledger, probabilities = {}, {}
    target = np.isfinite(data['offset'])
    for budget in plan['budgets']:
        defs, pp = [], []
        for arm, scores in model_scores.items():
            dd = dict(data, s=scores)
            for margin in MARGINS:
                veto = np.ones(len(target)) if margin is None else ndtr((margin/100-data['vertical_sides_xz'])/.02)
                p, definition = G.best_thresholds(dd, veto, budget)
                definition.update(model=arm, margin_cm=margin)
                pp.append(p); defs.append(definition)
        ps = np.asarray(pp)
        ledger[str(budget)] = defs; probabilities[str(budget)] = ps
        tof_ids = [i for i, d in enumerate(defs) if d['margin_cm'] is None]
        for sig in plan['heading_deg']:
            weights, support = {}, {}
            for lr, hr in plan['report_ranges']:
                mask = target & (data['range'] >= lr) & (data['range'] < hr)
                for name, (lo, hi) in G.BINS.items():
                    key = f'{lr}-{hr}|{name}'
                    w = np.zeros(len(target))
                    w[mask] = G.bin_weights(data['offset'][mask], data['range'][mask], lo, hi, sig)
                    weights[key] = w
                    support[key] = max(G.missing_mass(d, a, sig) for d in (lr, hr) for a in (lo, hi))
            keys = list(weights); ww = np.array(list(weights.values())); den = ww.sum(1)
            assert np.all(den > 0)
            rates = ps@ww.T/den
            k0, k2 = [keys.index(f'1.2-2.1|contact{x}') for x in ('0-2', '2-5')]
            pick = lambda ids, p, s: choose(rates, ids, p, s, defs)
            all_ids = range(len(ps))
            t0, o0 = pick(tof_ids, k0, k2), pick(all_ids, k0, k2)
            t2, o2 = pick(tof_ids, k2, k0), pick(all_ids, k2, k0)

            def comparison(pi, ref):
                metrics = {}
                for ki, key in enumerate(keys):
                    w = ww[ki]
                    cd = np.bincount(data['ui'], weights=w, minlength=96)
                    diff = np.bincount(data['ui'], weights=w*(ps[pi]-ps[ref]), minlength=96)
                    bd = boot@cd
                    assert np.all(bd > 0)
                    metrics[key] = dict(expected_n=float(den[ki]), source_rows=int(np.sum(w > 1e-12)),
                        source_units=int(np.sum(cd > 1e-12)), recall=float(rates[pi, ki]), reference_recall=float(rates[ref, ki]),
                        delta_pp=float(100*(rates[pi, ki]-rates[ref, ki])), delta_pp_ci95=np.percentile(100*(boot@diff)/bd, [2.5, 97.5]).tolist(),
                        maximum_missing_kernel_mass=support[key], support_valid=support[key] <= .05)
                return dict(policy_index=pi, policy=defs[pi], reference_index=ref, reference=defs[ref], metrics=metrics)

            shared = comparison(o0, t0)
            envelopes = {'contact0-2': comparison(o0, t0), 'contact2-5': comparison(o2, t2)}
            selected = [envelopes[b]['metrics'][f'1.2-2.1|{b}'] for b in envelopes]
            point_pause = all(v['delta_pp'] < 3 for v in selected)
            reading = ('NOT_EVALUABLE' if not all(v['support_valid'] for v in selected) else
                       'CONTINUE_OPPORTUNITY' if not point_pause else
                       'PAUSE_MATCHED_FAMILY' if all(v['delta_pp_ci95'][1] < 3 for v in selected) else 'UNCERTAIN')
            by_model = {}
            for arm in model_scores:
                ids = [i for i, d in enumerate(defs) if d['model'] == arm]
                ref = next(i for i in ids if defs[i]['margin_cm'] is None)
                by_model[arm] = dict(tof=comparison(ref, ref), oracle_primary=comparison(pick(ids, k0, k2), ref),
                                     oracle_secondary_envelope=comparison(pick(ids, k2, k0), ref))
            # Numerical inheritance check, not a replacement or edit of old results.
            original_arms = original['runs'][f'{budget}|{sig}']['arms']
            for name, old_name in [('tof', 'ToF'), ('oracle_primary', 'oracle_primary'),
                                   ('oracle_secondary_envelope', 'oracle_secondary_envelope')]:
                new, old = by_model['BASE'][name], original_arms[old_name]
                for field in ('thresholds', 'margin_cm', 'clear_per_query'):
                    assert new['policy'][field] == old['policy'][field]
                for key in keys:
                    for field in ('expected_n', 'recall', 'delta_pp', 'delta_pp_ci95'):
                        np.testing.assert_allclose(new['metrics'][key][field], old['metrics'][key][field], rtol=0, atol=1e-10)
            result['runs'][f'{budget}|{sig}'] = dict(shared_primary=shared, separate_bin_envelopes=envelopes,
                per_model=by_model, point_reading='PAUSE' if point_pause else 'CONTINUE', evidence_reading=reading,
                all_policy_rates=dict(metric_keys=keys, recalls=rates.tolist()))
    result['decision'] = dict(matched_family=result['runs']['0.1|1.0']['evidence_reading'],
        original_base_only_unchanged=read(G.OUT/'result.json')['decision'], downstream='NO_AUTOMATIC_STEP3')
    result['checks']['original_base_only_reproduced'] = True
    G.save(OUT/'policies.json', ledger)
    np.savez_compressed(OUT/'policy_probabilities.npz', **probabilities)
    G.save(OUT/'result.json', result)
    print(json.dumps(result['decision']))


def fixtures():
    checks = G.fixtures()
    meta = dict(unit=np.repeat(92000, 200), config=np.repeat(np.arange(40), 5), frame=np.tile(np.arange(11, 16), 40))
    raw = np.column_stack([meta['frame'], meta['frame']+100]).astype(np.float32)
    got = aggregate(raw, meta, [92000])['92000']
    expected = np.dot(np.arange(11, 16), [1, 2, 4, 8, 16])/31
    np.testing.assert_allclose(got, np.tile([expected, expected+100], (40, 1)))
    # Primary model is shared across bins; separate envelopes may choose another.
    defs = [{'clear_per_query': [.1, .1]} for _ in range(3)]
    rates = np.array([[.8, .6], [.7, .9], [.85, .75]])
    assert choose(rates, [0, 1], 0, 1, defs) == 0
    assert choose(rates, [0, 1], 1, 0, defs) == 1
    assert choose(rates, range(3), 0, 1, defs) == 2
    checks.update(causal_aggregation=True, shared_vs_envelope_selection=True)
    print(json.dumps(checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'import', 'evaluate', 'fixtures'])
    args = parser.parse_args()
    {'prepare': prepare, 'import': import_scores, 'evaluate': evaluate, 'fixtures': fixtures}[args.stage]()
