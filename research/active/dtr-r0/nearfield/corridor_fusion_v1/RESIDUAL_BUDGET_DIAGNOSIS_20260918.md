# Saved residual scores at matched Clear FP budgets

Date: 2026-09-18. Authority: **posthoc consumed-report diagnosis only**.

The fixed B0/B1/B2 residual recipe remains closed. At both requested low-FP
budgets, every residual has lower maximum Clear TP than raw HGB and A*.
Its frozen-threshold failure therefore cannot be explained solely by threshold
selection. This does not establish universal dominance, exhausted sensor signal,
or failure of counterfactual learning in general.

## Inputs and selection

Reuse the authenticated scores from the [texture experiment](BG_INVARIANCE_RESULTS_20260918.md)
and the existing 288-frame report cohort. Clear covers 216/288 frames (75%),
108 positive and 108 negative; strict covers all 288, with 144 positives.
There are 18 core events and 30 strict events. No new data, inference, training,
calibration or deployed threshold changes occurred.

Enumerate every distinct saved score threshold using `score >= threshold`, plus
an all-negative point. For each Clear FP budget, maximize Clear TP, then prefer
fewer FP, then the higher threshold. This last tie-break does **not** optimize
events. Preserve all maximum-TP plateaus and all event/onset details.
HGB uses its exact saved probabilities; residuals use saved logits. Threshold
numbers across these score scales are not directly comparable.

## Selected operating points

| Clear FP budget | Method | Clear TP/FP/FN | Strict TP/FP/FN | Strict events |
| --- | --- | --- | --- | --- |
| <=3 | raw HGB | 99/3/9 | 108/7/36 | 23/30 |
| <=3 | A* | 98/3/10 | 110/6/34 | 25/30 |
| <=3 | B0 | 88/3/20 | 97/4/47 | 24/30 |
| <=3 | B1 | 97/3/11 | 109/6/35 | 25/30 |
| <=3 | B2 | 93/3/15 | 106/6/38 | 26/30 |
| <=5 | raw HGB | 99/3/9 | 108/7/36 | 23/30 |
| <=5 | A* | 99/4/9 | 111/7/33 | 25/30 |
| <=5 | B0 | 91/4/17 | 103/6/41 | 25/30 |
| <=5 | B1 | 98/4/10 | 110/7/34 | 25/30 |
| <=5 | B2 | 95/5/13 | 108/9/36 | 26/30 |

All ten selected points retain 18/18 core events and 76/76 native-ToF-supported
Clear positive frames. Event retention alone does not preserve alert timing.

At budget <=5, the full maximum-TP plateaus have strict-event ranges raw23–25,
A*25–26, B025, B125–27 and B226. For example, B1 can obtain 98/4 with27 strict
events at logit1.447108268737793; its selected higher threshold gives the same
98/4 with25. Raw can obtain99/4 with25 events, and A*99/5 with26. These are
additional posthoc tradeoffs, not new frozen results. At <=3, B2 has two
maximum-TP thresholds, both93/3 with26 events; other arms have one.

## Event identities and timing

The following comparisons use the selected matched-budget A* point. The
patterns hold at both budgets except the explicit B0 exception. Episode IDs
below omit the common `singleconfirm_` prefix. Times are first-alert delays
relative to the start of the corresponding sampled event, at0.25s cadence.

| Residual | Lost strict events | Gained strict events |
| --- | --- | --- |
| B0 <=3 | shallow_boundary_stress_scene4_exit; scene5_exit | shallow_boundary_stress_scene1_enter |
| B0 <=5 | shallow_boundary_stress_scene5_exit | shallow_boundary_stress_scene1_enter |
| B1 | shallow_boundary_stress_scene5_exit | shallow_boundary_stress_scene1_enter |
| B2 | shallow_boundary_stress_scene5_exit | shallow_boundary_stress_scene1_enter; scene2_exit |

All residuals delay `near_rod_farwall_scene1_in` from0 to0.25s. B0 additionally
delays `near_rod_farwall_scene5_in` from0.5 to1.0s. These rod delays also occur
against matched raw HGB. B1/B2 retain the latter rod onset at0.5s.

Against A*, all residuals delay `shallow_boundary_stress_scene2_enter` from0
to0.25s. B1/B2 and B0 at <=5 delay `shallow_boundary_stress_scene4_exit`
from0.25 to0.5s; B0 at <=3 misses it entirely. All residuals detect
`shallow_boundary_stress_scene5_enter` earlier, at0.25 rather than0.5s.
The gained scene1_enter is detected at0.25s; B2's gained scene2_exit at0s.
The lost scene5_exit was detected by both baselines at0.5s.

Raw has fewer selected strict events than A*: it misses scene1_exit and
scene2_enter, but detects scene5_enter at0 rather than0.5s. Thus a residual's
higher event total relative to raw does not establish retention or earlier
warning. Machine-readable comparisons include both frozen baselines and both
matched-budget baselines, with every lost/gained/delayed/earlier event.

## Decision and evidence

No residual threshold supplies a joint improvement over either selected
baseline: at least as many Clear TP, no more actual Clear FP, no lost or delayed
core/strict event, and at least one strict improvement. Every threshold was
checked, not only the selected tie-break. The stronger simple observation is
that residual maximum Clear TP is already lower at both requested budgets.
There is no demonstrated low-FP joint benefit to recover through threshold
transfer in this cohort. Keep raw HGB and A*; do not tune or rename this fixed
residual recipe. The diagnostic supplies no fresh-validation or algorithmic
novelty claim and launches no successor.

Implementation: [diagnose_residual_budgets.py](diagnose_residual_budgets.py).
Durable local outputs are under
`artifacts.local/work/corridor-bg-invariance-20260918/budget-diagnostic/`:
`summary.json` (selected points, plateaus and event comparisons), `curves.json`
(all thresholds with boundary, rod, native and temporal metrics), and
`completion.json` (input/source/output SHA256 receipts).
Independent scalar checks passed for all ten selected confusion matrices and
every event onset. Input hashes remained unchanged; zero training runs and zero
new frames. Existing experiment-ledger/inheritance metadata errors remain
pending as documented in the parent report; no ledger was rewritten.
