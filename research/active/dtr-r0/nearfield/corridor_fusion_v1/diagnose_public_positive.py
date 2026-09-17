"""Post-fit zero-logit diagnostic and public-only artifact inference check.

This never changes the sealed selected outputs, fits a model or searches a new
threshold. Zero is the natural sigmoid-0.5 diagnostic, declared after outcomes.
"""
import time
import numpy as np
import torch
from prepare_public_positive import ROOT, HERE, WORK, OUT as PREP, CAPTURES, read, write, sha
from run_public_positive import OUT, evaluate
from mz136_incumbent import public_observations
from run_mz139_surface_fit import selected_jsonl
from public_positive_inference import PublicPositivePredictor


def main():
    torch.set_num_threads(4)
    done = read(OUT/'completion.json')
    assert done['status'] == 'PASS' and sha(OUT/'summary.json') == done['summary_sha256']
    pred = np.load(OUT/'oof-predictions.npz')
    data = np.load(PREP/'public-tokens.npz')
    target = np.load(PREP/'offline-targets.npz')
    metadata = read(PREP/'metadata.json')
    ii = pred['indices']
    meta = [metadata[i] for i in ii]
    score, cases = evaluate(meta, pred['logits'], np.zeros(len(ii)), data['valid'][ii,:128],
        target['target'][ii,:128], target['known'][ii,:128], pred['folds'])
    write(OUT/'zero-logit-diagnostic.json', dict(cohorts=score,
        authority='POSTHOC_FIXED_ZERO_LOGIT_DIAGNOSTIC_NOT_CALIBRATED_PRIMARY_RESULT',
        threshold=0., fits=0, threshold_search=False))
    write(OUT/'zero-logit-cases.json', cases)
    public_heads = {k:PublicPositivePredictor(OUT/f'fold{k}-model.pt',
        read(OUT/f'fold{k}-selection.json')['chosen']['threshold']) for k in range(6)}
    selected = read(OUT/'cases.json')
    by_id = {c['id']:c for c in selected}
    exact_agreement = 0
    delta, latency = [], []
    for cohort in ['old','new']:
        ids = {c['id'] for c in selected if c['cohort'] == cohort}
        rows = public_observations(selected_jsonl(CAPTURES[cohort]/'raw.jsonl', ids))
        yaw, previous = 0., None
        for row in rows:
            if row['episode_id'] != previous:
                yaw = 0.
            if row['imu_valid']:
                yaw += row['delta_yaw']
            previous = row['episode_id']
            expected = by_id[row['id']]
            model = public_heads[expected['fold']]
            start = time.perf_counter()
            result = model.predict(row, yaw, expected['A'])
            latency.append((time.perf_counter()-start)*1000)
            delta.append(abs(result['evidence_logit']-expected['evidence_logit']))
            assert result['alert'] == expected['public_OR']
            assert result['positive_evidence'] == expected['evidence_alert']
            exact_agreement += 1
    write(OUT/'public-inference-audit.json', dict(status='PASS', frames=exact_agreement,
        max_logit_abs_difference=float(max(delta)),
        all_frame_and_branch_decisions_match=True,
        additional_token_frontend_plus_head_ms=dict(mean=float(np.mean(latency)),
            p50=float(np.percentile(latency,50)), p95=float(np.percentile(latency,95))),
        scope='Host CPU public-only path, excludes A RGB Radar frontend, capture and transport; no phone claim',
        code_sha256=sha(HERE/'public_positive_inference.py')))
    print({c:{a:score[c][a]['clear'] for a in ['A','A_public_OR','A_oracle_sampled_OR']} for c in score})


if __name__ == '__main__':
    main()
