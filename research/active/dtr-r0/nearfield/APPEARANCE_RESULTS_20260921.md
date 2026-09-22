# Appearance probe: model-input sensitivity observed; strict source gate not met

2026-09-21. One 1152-frame UE capture and frozen B/N inference completed.
The prospective terminal is **NOT_EVALUABLE for the pure-appearance causal
claim**. All 864 paired public ToF vectors are identical, but 23 native depth
image pairs differ. Preserve the raw model sensitivity as a diagnostic lead;
do not report a passed appearance intervention, model improvement or independent
generalization result.

[Protocol](APPEARANCE_PROTOCOL_20260921.md), [runner](appearance_probe.py),
[capture](appearance_capture.py), [saved-output audit](audit_appearance_probe.py).

## What ran and why admission failed

The lexicographically first consumed TRAIN group of each of four families gives
288 distinct poses: four groups, three complete relations, 24 frames each.
Reference, exact re-render, target-material change and background-material change
give 1152 frames / 48 clips. N has trained on these geometries. No model fitting,
cutoff selection, material search, second capture or protected test occurred.
The two weights, encoder, source specification and executable snapshot were hashed
before capture. Cutoffs remain B_control=6.888704776763916, N=7.03014612197876.

| Versus reference, each 288 pairs | Native depth unequal | Public ToF unequal |
| --- | ---: | ---: |
| Exact re-render | 8 | 0 |
| Target material | 8 | 0 |
| Background material | 7 | 0 |

A subsequent read-only inspection of those saved native arrays finds **51 changed
pixels across the 23 unequal pairs**, 1–5 pixels per affected image. Maximum
depth difference is 1.364730m; recorded examples include distant surfaces.
The exact source of raster differences was not isolated. No tolerance was fitted,
pair discarded, native array overwritten, or source gate relaxed after inspection.

Actual target/background locations, bounds, scales, rotations and mesh paths
match their counterparts exactly; actual materials match the prescribed changes.
Three paired camera Y receipts differ by 1.7347e-18m, also recorded rather than
silently rounded into exact equality. Actual-geometry and declared-geometry
corridor labels agree on all 1152 frames. The strict full-image/native-pose
requirements remain unmet even though public ranges are unchanged.

## Observed output changes, without a causal promotion

All flip counts below use all 288 pairs, including failed native pairs. They
are changes in decisions, not gains. An alert is original A OR the frozen model
supplement. The unchanged nonrecursive hold is one prior sample / 0.2s.

| Model | Intervention | Current flips | Held flips | Supplement flips | Median absolute logit change |
| --- | --- | ---: | ---: | ---: | ---: |
| B | Exact re-render | 1 | 1 | 2 | 0.220 |
| B | Target material | 13 | 14 | 17 | 2.255 |
| B | Background material | 20 | 17 | 30 | 4.020 |
| N | Exact re-render | 0 | 0 | 2 | 0.380 |
| N | Target material | 8 | 3 | 16 | 2.741 |
| N | Background material | 12 | 11 | 20 | 4.110 |

The numeric flip criterion (at least three excess flips and excess in at least
two groups) is true for all four model/intervention comparisons. It is subordinate
to source admission, so the final result remains NOT_EVALUABLE. At the actual
model-input level, unchanged public ToF accompanies changed RGB and outputs;
this warrants retaining an RGB sensitivity diagnostic. It does not isolate
material alone from all rendering nuisance or explain the prior HEAD/BODY trade.

Post-hoc attribution to admission failures does not change denominators: none of
B's current flips occur on a failed-depth pair; one N target flip and one N
background flip do. These are descriptive locations, not a new selected-cohort
success test. This is stronger motivation to retain the lead than to label the
whole idea negative, while leaving its prospective causal claim unaccepted.

## Complete-clip metrics

Each condition has Core192 frames (44 positive /148 negative) and Boundary96
(44 positive /52 negative), with four positive events in each. All negatives
before entry and after exit remain. The accompanying `metrics.json` retains
current/held TP/FP/FN, precision/recall/FPR, UNKNOWN, full event timing, false
segments and subgroup tables for all six arms and all conditions.

**Core; current / held.** P/R/FPR are percentages.

| Model / condition | Current TP/FP/FN | Current P/R/FPR | Held TP/FP/FN | Held P/R/FPR |
| --- | --- | --- | --- | --- |
| B / reference | 44/26/0 | 62.86/100.00/17.57 | 44/32/0 | 57.89/100.00/21.62 |
| B / repeat | 44/27/0 | 61.97/100.00/18.24 | 44/33/0 | 57.14/100.00/22.30 |
| B / target | 44/18/0 | 70.97/100.00/12.16 | 44/23/0 | 65.67/100.00/15.54 |
| B / background | 44/15/0 | 74.58/100.00/10.14 | 44/23/0 | 65.67/100.00/15.54 |
| N / reference | 43/4/1 | 91.49/97.73/2.70 | 44/7/0 | 86.27/100.00/4.73 |
| N / repeat | 43/4/1 | 91.49/97.73/2.70 | 44/7/0 | 86.27/100.00/4.73 |
| N / target | 42/4/2 | 91.30/95.45/2.70 | 43/7/1 | 86.00/97.73/4.73 |
| N / background | 42/5/2 | 89.36/95.45/3.38 | 43/8/1 | 84.31/97.73/5.41 |

**Boundary; current / held.** P/R/FPR are percentages.

| Model / condition | Current TP/FP/FN | Current P/R/FPR | Held TP/FP/FN | Held P/R/FPR |
| --- | --- | --- | --- | --- |
| B / reference | 40/6/4 | 86.96/90.91/11.54 | 42/8/2 | 84.00/95.45/15.38 |
| B / repeat | 40/6/4 | 86.96/90.91/11.54 | 42/8/2 | 84.00/95.45/15.38 |
| B / target | 37/4/7 | 90.24/84.09/7.69 | 39/6/5 | 86.67/88.64/11.54 |
| B / background | 33/4/11 | 89.19/75.00/7.69 | 36/6/8 | 85.71/81.82/11.54 |
| N / reference | 42/3/2 | 93.33/95.45/5.77 | 44/5/0 | 89.80/100.00/9.62 |
| N / repeat | 42/3/2 | 93.33/95.45/5.77 | 44/5/0 | 89.80/100.00/9.62 |
| N / target | 35/3/9 | 92.11/79.55/5.77 | 42/5/2 | 89.36/95.45/9.62 |
| N / background | 32/3/12 | 91.43/72.73/5.77 | 35/5/9 | 87.50/79.55/9.62 |


A is identical across all four conditions: Core current34/4/10, held34/7/10;
Boundary current6/3/38, held8/5/36 (TP/FP/FN). Current UNKNOWN remains explicit:
168 Core and96 Boundary per condition for every arm, including frames where
the model or hold supplies an alert. UNKNOWN silence is never counted as TN.

B and N detect all eight positive events in each condition, hiding meaningful
frame/timing changes. Maximum first-event delay changes for B Boundary from0.2s
to0.4s under either material change; N Boundary changes from0 to0.4s for target
and0.2s for background. N Core changes from0 to0.2s. These are sampled posed
trajectories, not real-time system latency. B Core current false-segment counts
are7/7/6/9 for reference/repeat/target/background; N has4/4/4/5. These numbers are
diagnostics of a consumed controlled source, not evidence for promoting an arm.

## Validation, execution correction and retained evidence

The pre-run three tests passed. Independent review checked frozen cutoffs,
bundle hashes, intervention isolation and evaluator field compatibility. It
also found that the original runner continued inference after failed admission,
although final evaluation downgraded the result. That deviation occurred here:
the saved predictions are diagnostic only. The delivered runner now rejects
failed admission before loading models; its additional rejection test passes
(four focused tests total). The worker's original frozen code was not changed
and no second inference was launched. `postlaunch-code-correction.json` records
old/new hashes; `bundle/source/` retains the exact executed code. Reproducing
this historical run requires that snapshot, not the corrected working copy.

The independent saved-output audit verifies all three seals in order, 72
condition/stratum/arm confusion groups, paired flips, cutoff/OR/hold logic and
the final primary status. It reports the three exact-pose differences above;
audit PASS means the records/counts agree, not source admission. Full native
arrays were inspected on the worker for the read-only pixel diagnostic; only
metadata/ranges and eight original RGB previews were transferred locally.
Two original reference/target previews were visually inspected.

Execution used the authorized Windows worker with RTX3060 Laptop GPU. Actual
torch2.9.1+cu128 CUDA was selected by equivalent CPU/GPU measurements for encoder
and both heads. Encoder batch medians: CPU67.813ms / CUDA4.742ms; B head0.834ms /
0.565ms; N head0.815ms /0.585ms. Feature extraction took9.381s. The capture script
reports775.641s; complete worker job including startup reports899.753s. These are
host run costs, not target-device latency.

Local evidence: `artifacts.local/work/ba-appearance-20260921/`, including the
before-run protocol/input/code bundle, `remote-results/`, diagnostic receipts,
independent audit and correction receipt. Full1152 RGB/native frames, feature
cache and original execution snapshot remain on the worker under
`G:/DevWorkspace/BlindAssist/artifacts/work/ba-appearance-20260921/`; the job receipt
is under `artifacts/evidence/jobs/ba-appearance-20260921/`. These are task-owned
durable decision evidence, retained for audit/reproduction. Release storage only
through a separately authorized evidence-retention cleanup of these exact trees;
do not remove shared map, engine, environment or model caches.

The UE process tree has no survivors, task actors/render targets were released,
map/source assets are unchanged, and the scheduled job has exited successfully
with no remaining task registration. No paid allocation was created.

Local inheritance is diagnostic component only: frozen-input RGB sensitivity
and paired-render admission checks remain useful; no alert policy or pure-material
causal result is promoted. Supported registry receipts accompany the evidence;
the existing ledger303 fingerprint / unknown-terminal errors prevent global
metadata admission. No manual ledger change or inherited-success bypass occurs.
The pilot is complete; no retraining or outcome-based recapture follows.
