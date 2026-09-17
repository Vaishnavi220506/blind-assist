# Minimal CCRL: fixed ranking residual does not retain A* precision

Retain standalone A*. The fixed counterfactual-ranking residual recovers six
clear rod misses but loses one BODY true frame and adds16 clear false alerts.
Matched BCE recovers the same six rod frames with only7 added clear FP. This
run does not establish a ranking-specific alert benefit or a completed innovation.
The exact recipe is a negative control, not evidence against all counterfactual
learning. No tuning, capture, missingness-fusion successor or App change follows.

## Paired result

All methods use the same existing288-frame controlled report cohort. The5cm
clear stratum contains216frames (75% coverage); boundary72 remains reported.
This cohort was previously consumed. It was excluded from fitting, normalization,
checkpoint selection and threshold selection here, but is not fresh confirmation.

| Method | Clear TP/FP/FN | Precision | Recall | F1 | Strict TP/FP/FN | Strict events |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Saved A* | 96/3/12 | 96.97% | 88.89% | **92.75%** | 107/6/37 | 24/30 |
| A* + matched BCE residual | 101/10/7 | 90.99% | 93.52% | 92.24% | 120/17/24 | 27/30 |
| A* + BCE/ranking residual | 101/19/7 | 84.17% | 93.52% | 88.60% | 118/25/26 | 26/30 |

Strict F1:83.27%,85.41%,82.23%. Boundary TP/FP/FN:11/3/25,19/7/17,17/6/19.
Core events remain18/18 for all. Both residuals restore the rod scene5 first
alert from0.5s to0s. Ranking delays an already detected boundary event
`singleconfirm_shallow_boundary_stress_scene2_enter` from0 to0.25s. Increased
event counts therefore do not imply every baseline event retained at its timing.
The fixed low-FP acceptance condition fails for both residuals.

| Clear family | A* TP/FP/FN | BCE | Ranking |
| --- | --- | --- | --- |
| Rod | 26/2/10 | 32/9/4 | 32/15/4 |
| BODY | 35/1/1 | 34/1/2 | 34/4/2 |
| HEAD | 35/0/1 | 35/0/1 | 35/0/1 |

Ranking rescues13 strict FN, loses2 strict TP and adds19 FP. BCE rescues14,
loses1, adds12 and removes1 FP. Pair ordering is A*124/144(86.11%),
BCE131/144(90.97%), ranking129/144(89.58%). Ranking's mean logit gap9.821
versus BCE9.794 does not translate into better ordering or final alerts.
This is not evidence that A* learned background shortcuts: that causal claim
would need an intervention capable of isolating the nuisance variable.

All69 zero-ToF-return frames remain semantically UNKNOWN, never known-free.
Their strict TP/FP/FN are A*21/4/15, BCE22/7/14, ranking22/10/14. The residual
has no missingness veto, but is not a positive-only retention construction.
Among92 native sampled-support-positive frames, A* alerts82, BCE89, ranking86.
BCE retains all82 A* true alerts; ranking loses1 and recovers5. This audits
frame-level native support retention, not individual contributors or real sensors.

## Mechanism and data integrity

The [fixed protocol](CCRL_PROTOCOL_20260918.md) preserves A*'s HGB weights.
Because HGB is not differentiable, each candidate adds the same39,793-parameter
2485->16->1 logit residual. The adapter is an architecture change relative to A*;
only matched BCE versus ranking isolates the objective. Zero initialization
starts at the baseline. Both use120 full-batch AdamW steps, seed187018,
last checkpoint, fit-only normalization and the same public features.

Source auditing admits672 lateral frame pairs across1344rows with no rejected
pairs. Residual fitting uses576pairs/1152non-anchor rows; the192anchor rows
serve HGB only. There are **zero exact background-only intervention pairs**:
invariance loss is NOT_EVALUABLE and was not run. Matching unrelated same-label
scenes would not establish nuisance invariance. Same paired noise seeds establish
common starts, not identical draw alignment after geometry-dependent branches.

Six outer folds keep all members of each held scene index out of fitting.
Each outer residual trains on960rows using base scores from five further
group-disjoint HGB fits. Thirty inner HGBs and14 residual fits complete once.
Final residuals train on1152 saved A* OOF scores, then attach to the saved final
A*. This crossfit-to-final score transfer remains a limitation. Inner HGBs use
the float32 public cache; saved outer/final HGBs retain exact original-feature
replay. This is shared by both residual arms, not an objective difference.

One threshold per arm is chosen from pooled outer-held clear predictions:
BCE0.34591200947761536logit, ranking-0.05963850021362305logit. Selection clear
TP/FP/FN is A*398/12/34, BCE388/30/44, ranking392/34/40 on864clear rows.
These are threshold-selection statistics, not unbiased validation. All report
public predictions were sealed before the report label join; A* probabilities
reproduce the saved288 baseline probabilities bitwise. No report threshold
sweep or outcome-driven checkpoint choice was performed.

The experiment takes141.33s locally. Equivalent real-step probe medians are
CUDA2.081ms and CPU3.565ms; fitting uses actual RTX5060 Laptop CUDA with
Torch2.11/cu130. HGB uses CPU because sklearn HGB has no GPU backend. No
end-to-end optimized latency, phone performance or safety benefit is claimed.

## Verification, disposition and reproduction

Four focused tests pass: initial baseline preservation, rank-gradient direction,
pair authentication and split-boundary removal. The read-only audit verifies
input/model seals, nested group/pair isolation, fit-only normalization, final
checkpoint length, exact threshold selection, bitwise final weight replay,
native truth/support parity, independent scalar confusion counts and event
onsets. It reuses the existing native geometry helper; this is not an independent
geometry implementation. The requested subagent review was unavailable after
two service503 failures; no independent-agent review is claimed.

The first launch stopped before creating artifacts because this Python3.11
runtime lacks `Path.is_junction`; resolved-target validation fixed the mechanical
preflight. No fitting result existed or was retried. Raw failure details remain
in the run's execution-notes.json. Registration through the supported command
still fails at the pre-existing `experiments/index.jsonl:303` input fingerprint
mismatch. Structured inheritance likewise remains pending an unknown terminal;
logs are retained, and no ledger was manually edited or bypassed.

Intended disposition: `NEGATIVE_CONTROL` for this fixed ranking residual at the
low-FP alert role; retain A* as the precision baseline. Preserve BCE's disclosed
recall/event tradeoff as diagnostic evidence, not a promoted version. Revisit
only with a materially changed mechanism or authenticated nuisance interventions
under a separately authorized experiment, not a sweep of this consumed report.

```powershell
$py = 'E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe'
& $py -m unittest discover -s research/active/dtr-r0/nearfield/corridor_fusion_v1 -p test_ccrl.py
& $py research/active/dtr-r0/nearfield/corridor_fusion_v1/audit_ccrl.py
```

`run_ccrl.py` deliberately refuses to overwrite an existing experiment. Retained
payloads under `artifacts.local/work/corridor-ccrl-20260918/` include14 residual
checkpoints,30 inner HGBs, pair audits, fold membership/base scores, model/data
seals, calibration curves, all public report predictions/features, cases and
audits. These support reproduction without another capture. No disposable
dataset copy or task-owned service was created; training/replay processes have
exited. The current host owns the retained evidence; no paid allocation remains.
