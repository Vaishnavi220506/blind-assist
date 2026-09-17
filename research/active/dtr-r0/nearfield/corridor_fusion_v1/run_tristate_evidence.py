"""Run the fixed evaluator-only existing-return decision probe; no fitting."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from tolerance_eval import metric, temporal
from tristate_evidence import classify_returns, decisions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK = ROOT / 'artifacts.local/work'
SOURCE = WORK / 'corridor-surface-oracle-20260917'
OUT = WORK / 'corridor-tristate-evidence-20260917'
REPRESENTATIONS = ('sampled_points', 'full_faces')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def transitions(y, a, p, mask):
    return dict(FN_rescued=int(sum(mask & y & ~a & p)),
                TP_lost=int(sum(mask & y & a & ~p)),
                FP_removed=int(sum(mask & ~y & a & ~p)),
                FP_added=int(sum(mask & ~y & ~a & p)))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'prediction-seal.json').exists(), 'Do not overwrite a sealed run'
    parent = read(SOURCE / 'freeze.json')
    for filename, digest in parent['bindings'].items():
        assert sha(Path(filename)) == digest, filename
    parent_seal = read(SOURCE / 'prediction-seal.json')
    for name, key in [('oracle-returns.json', 'returns_sha256'),
                      ('scores.npz', 'scores_sha256'), ('freeze.json', 'freeze_sha256')]:
        assert sha(SOURCE / name) == parent_seal[key]
    assert read(SOURCE / 'independent-audit.json')['status'] == 'PASS'
    raw_path = next(Path(p) for p in parent['bindings'] if Path(p).name == 'raw.jsonl')
    files = [SOURCE / p for p in ('freeze.json', 'prediction-seal.json', 'scores.npz',
             'oracle-returns.json', 'cases.json', 'independent-audit.json')]
    files += [raw_path, Path(__file__), HERE / 'tristate_evidence.py',
              HERE / 'tolerance_eval.py', HERE / 'TRISTATE_EVIDENCE_PROTOCOL_20260917.md']
    bindings = {str(p): sha(p) for p in files}
    ids = parent['ids']
    write(OUT / 'freeze.json', dict(bindings=bindings, ids=ids,
          threshold=parent['threshold'], representations=REPRESENTATIONS,
          arms=list(decisions(False, 'UNKNOWN')), zero_fit=True, zero_capture=True,
          authority='CONSUMED_EVALUATOR_ONLY_READOUT_NOT_DEPLOYABLE_CEILING'))
    raw = {r['id']: r for r in map(json.loads, raw_path.read_text().splitlines())}
    records = read(SOURCE / 'oracle-returns.json')
    scores = np.load(SOURCE / 'scores.npz')
    assert list(scores['ids']) == [r['id'] for r in records] == ids
    a = scores['A'] >= parent['threshold']
    predictions = []
    # No truth, family or error identity is read by the state/decision functions.
    for i, rec in enumerate(records):
        variants = {}
        for rep in REPRESENTATIONS:
            state, reason = classify_returns(rec, raw[rec['id']]['tof_packet_received'], rep)
            variants[rep] = dict(state=state, reason=reason, alerts=decisions(a[i], state))
        predictions.append(dict(id=rec['id'], variants=variants))
    write(OUT / 'predictions.json', predictions)
    write(OUT / 'prediction-seal.json', dict(predictions_sha256=sha(OUT / 'predictions.json'),
          freeze_sha256=sha(OUT / 'freeze.json'),
          authority='PRIVILEGED_NATIVE_GEOMETRY_DECISIONS_SEALED_BEFORE_LABEL_JOIN'))

    cases = read(SOURCE / 'cases.json')
    assert [c['id'] for c in cases] == ids
    assert np.array_equal(a, [c['alerts']['A'] for c in cases])
    y = np.array([c['truth'] for c in cases])
    states = np.array([c['state'] for c in cases])
    family = np.array([c['family'] for c in cases])
    masks = dict(clear=states != 'boundary', boundary=states == 'boundary',
                 strict=np.ones(len(ids), bool))
    rows = [raw[i] for i in ids]
    base_temporal = temporal(rows, states, a)
    old_first = {e['episode']: e for e in base_temporal['events']}
    summary, outputs = {}, []
    for rep in REPRESENTATIONS:
        evidence = np.array([p['variants'][rep]['state'] for p in predictions])
        unknown = evidence == 'UNKNOWN'
        assert np.array_equal(unknown, [r['usable_slots'] == 0 for r in records])
        arms = {}
        for arm in decisions(False, 'UNKNOWN'):
            p = np.array([pr['variants'][rep]['alerts'][arm] for pr in predictions])
            assert np.array_equal(p[unknown], a[unknown])
            if arm == 'A_OR_P':
                assert not np.any(a & ~p)
            t = temporal(rows, states, p)
            t['changed_core_first_alert'] = [dict(episode=e['episode'],
                A_delay_s=old_first[e['episode']]['first_in_core_delay_s'],
                probe_delay_s=e['first_in_core_delay_s']) for e in t['events']
                if e['first_in_core_delay_s'] != old_first[e['episode']]['first_in_core_delay_s']]
            strict_states = np.where(y, 'positive', 'negative')
            strict_t = temporal(rows, strict_states, p)
            native_suppressed = [dict(id=ids[i], corridor_points=sum(
                s.get('corridor_contributors', 0) for s in records[i]['slots']))
                for i in np.flatnonzero(a & ~p)]
            arms[arm] = dict(**{k: metric(y[m], p[m]) for k, m in masks.items()},
                clear_changes=transitions(y, a, p, masks['clear']),
                strict_changes=transitions(y, a, p, masks['strict']),
                families={f: dict(clear=metric(y[masks['clear'] & (family == f)],
                    p[masks['clear'] & (family == f)]),
                    changes=transitions(y, a, p, masks['clear'] & (family == f)))
                    for f in sorted(set(family))},
                temporal=t, strict_events=dict(events=strict_t['core_events'],
                    detected=strict_t['core_events_detected']),
                unknown_alerts_changed=int(sum(unknown & (a != p))),
                unknown_A_TP_retained=int(sum(unknown & y & a & p)),
                suppressed_native_corridor_points=sum(r['corridor_points'] for r in native_suppressed),
                changed_cases=[dict(id=ids[i], family=family[i], truth=bool(y[i]),
                    stratum=states[i], baseline=bool(a[i]), probe=bool(p[i]),
                    evidence=evidence[i], usable_returns=records[i]['usable_slots'],
                    diagnosis=cases[i]['diagnosis']['category']) for i in np.flatnonzero(a != p)])
        summary[rep] = dict(methods=arms, evidence={k: dict(frames=int(sum(evidence == k)),
            **{stratum: metric(y[m & (evidence == k)], a[m & (evidence == k)])
               for stratum, m in masks.items()}) for k in ('POSITIVE', 'OUTSIDE_ONLY', 'UNKNOWN')},
            unknown_reasons=dict(Counter(pr['variants'][rep]['reason'] for pr in predictions
                                        if pr['variants'][rep]['state'] == 'UNKNOWN')))
    for case, prediction in zip(cases, predictions):
        outputs.append(dict(**case, evidence_probe=prediction['variants']))
    write(OUT / 'cases.json', outputs)
    write(OUT / 'clear-cases.json', [c for c in outputs if c['state'] != 'boundary'])
    result = dict(representations=summary, frames=len(ids), clear_coverage=float(masks['clear'].mean()),
        native_points_preserved=sum(s.get('contributors', 0) for r in records for s in r['slots']),
        native_corridor_points_preserved=sum(s.get('corridor_contributors', 0) for r in records for s in r['slots']),
        zero_training=True, zero_capture=True, analysis_seconds=time.perf_counter() - started,
        authority='PRIVILEGED_CONTROLLED_DEVELOPMENT_NOT_PUBLIC_METHOD_OR_UNIVERSAL_CEILING',
        negative_state_limit='Observed returned surfaces outside does not certify unmeasured corridor free')
    write(OUT / 'summary.json', result)
    select_backend('scalar-scoring', cpu=BackendCandidate('tristate-cpu', 'cpu', lambda: len(ids),
        lambda _: DeviceObservation('cpu', 'host CPU', 'Python NumPy scalar geometry and saved decisions')),
        cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=OUT / 'backend.json')
    assert all(sha(Path(p)) == h for p, h in bindings.items())
    write(OUT / 'completion.json', dict(status='PASS', summary_sha256=sha(OUT / 'summary.json'),
        prediction_seal_sha256=sha(OUT / 'prediction-seal.json'),
        zero_return_fallback_preserved=True, source_hashes_unchanged=True,
        resources='Process-local CPU; no worker, GPU allocation or persistent process'))
    print(json.dumps({rep: {arm: dict(clear=v['clear'], changes=v['clear_changes'],
        core_events=v['temporal']['core_events_detected'], first_alert_changes=v['temporal']['changed_core_first_alert'])
        for arm, v in r['methods'].items()} for rep, r in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
