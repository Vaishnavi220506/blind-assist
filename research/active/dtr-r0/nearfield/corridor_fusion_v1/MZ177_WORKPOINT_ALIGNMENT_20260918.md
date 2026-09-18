# MZ177 work-point alignment (2026-09-18)

This note audits the consumed 288-frame work point before the continuous
approach capture. It replays saved IDs, labels, model scores and the sealed
oracle arrays. It does not train, refit, tune a threshold, or change the A*
model.

## Finding

The frozen public A* result remains **96/3/12** on clear rows at the frozen
probability threshold `0.5568065433174727`. The current
`run_oracle_surface_ceiling.py` calls `predict_proba` and writes probability
scores. Its unchanged A* arm is bitwise equal to the frozen A* probability,
and remains 96/3/12.

The two native support interventions in the current Oracle run both produce
**93/3/15** at that same probability threshold:

| arm | score domain | clear TP/FP/FN | changed clear rows |
|---|---|---:|---:|
| frozen A* | probability | 96/3/12 | 0 |
| Oracle A* (unchanged features) | probability | 96/3/12 | 0 |
| Oracle returned-point attribution | probability | 93/3/15 | 3 |
| Oracle complete linked surface | probability | 93/3/15 | 3 |

The three changed clear IDs are all positive near-rod frames:

- `singleconfirm_near_rod_farwall_scene0_in_00`
- `singleconfirm_near_rod_farwall_scene0_in_02`
- `singleconfirm_near_rod_farwall_scene1_in_01`

For example, the frozen probabilities are approximately `0.5914`, `0.5803`
and `0.5729`; attribution changes them to approximately `0.5269`, `0.5465`
and `0.5309`, while full-surface values are approximately `0.5248`, `0.5295`
and `0.5138`. Each falls below `0.5568065433174727`. The original model hash
is unchanged (`dffd5c5a...5cd59`), and the Oracle seal reports non-support
features bitwise unchanged.

## Score-domain check

The saved A* diagnostic also contains logits. Comparing the logit directly to
the probability threshold is a domain error, but numerically produces the
same clear 93/3/15 count on the same three IDs. The equivalent logit threshold
is `logit(0.5568065433174727) = 0.22821148907779354`; applying it to the saved
logits restores 96/3/12. This explains how a fixed A* point can appear to
change if probability and logit fields are mixed, but that mismatch is not how
`run_oracle_surface_ceiling.py` computes its A* arm.

The separate posthoc B2 diagnostic also has a 93/3/15 point, using B2 logits
and threshold `1.643951654434204`. It changes 21 clear decisions, a different
set from the three Oracle changes. Therefore 93/3/15 is not attributable only
to B2, and it must not be reported as the frozen A* result.

## Reproducible receipt

The corrected audit script is
[`audit_workpoint_alignment.py`](audit_workpoint_alignment.py). It was run with
`E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe` and
completed `PASS` into the fresh junction-routed artifact directory:

`artifacts.local/work/continuous-approach-20260918/alignment-v2/`

The receipt is in `completion.json`; `audit.json` records all input hashes,
score domains, thresholds, model identity, order/label checks and strict/clear
metrics. `decision-changes.json` contains each changed ID, label, episode,
score and alert transition for Oracle attribution, full surface, the explicit
logit-domain diagnostic, posthoc A*, and B2.

The audit checked 288 ordered IDs, 216 clear rows and 72 boundary rows. The
boundary rows remain in the input and are excluded only from clear confusion
counts. No training, evaluator read, model change or threshold change was
performed.
