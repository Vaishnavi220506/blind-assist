# Core projection stress: events retained, lateral false alerts increase

2026-09-21. One consumed-simulation check is complete. With the original ranges
unchanged, shifting reported zone projections by only two 256-wide lattice
pixels retains all 16 Core events but adds lateral false alerts and delays one
HEAD onset in each direction. **The declared Core usefulness checks pass;
insensitivity to calibration error is not established.** Keep the frozen Core
demo and record this as a diagnostic COMPONENT. No correction or tuning follows.

[Frozen protocol](CORE_PROJECTION_STRESS_PROTOCOL_20260921.md),
[runner](run_core_projection_stress.py), [focused tests](test_core_projection_stress.py),
[independent verifier](verify_core_projection_stress.py).

## Exact intervention and evidence scope

Reuse all 1152 previously consumed transfer frames, 48 complete clips and 16
base layouts. Core contains 768 frames (173 positive / 595 negative), including
all INSIDE pre-entry/post-exit frames and all OUTSIDE clips. Boundary contains
384 frames (173 positive / 211 negative). The source, labels and partition
membership are unchanged. This is paired Development sensitivity, not new
arrangements, physical sensing, natural distribution or a protected final set.

Move both reported horizontal box edges by -2/0/+2 on the 256x192 projection
grid: native-image -5/0/+5 pixels. Range bytes, zone identity, box widths,
vertical coordinates and timestamps remain unchanged. At Z=3 m the horizontal
projection displacement is approximately 5.59 cm. This is one declared
engineering stress magnitude, not a hardware calibration tolerance, rigid
rotation, physical sensor displacement or error-frequency estimate. Neither
sign was selected for performance, and no new returns were generated.

The unchanged geometric scorer, definite-support bypass, T0=.007085703945147101,
T=.4071309640537889 and nonrecursive one-frame .2 s hold run in each condition.
The learned RGB supplement is not run. Native depth and ownership are not read;
RGB identities remain bound, but this geometry-only check establishes no RGB
benefit. Zero-shift scores/current/held flags/UNKNOWN/valid/definite counts
reproduce all 1152 original A records exactly. All three prediction vectors
were sealed before parsing evaluator metadata and labels.

## Core result with the retained hold policy

Counts are TP/FP/FN; sampled durations are count times .2 s, not continuous time.

| Reported projection | TP/FP/FN | Precision | All-negative FPR | Events | Worst first-alert delay | Lowest event coverage | FP segments / seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal | 173/33/0 | 83.98% | 5.55% | 16/16 | 0 s | 100% | 24 / 6.6 |
| Left 2 pixels | 172/36/1 | 82.69% | 6.05% | 16/16 | .2 s | 90.91% | 24 / 7.2 |
| Right 2 pixels | 172/64/1 | 72.88% | 10.76% | 16/16 | .2 s | 90% | 30 / 12.8 |

Relative to nominal, left adds 11 FP and removes 8 (net +3), while right adds
34 and removes 3 (net +31). Both lose exactly one nominal true-positive frame
and add no Core TP. Thus similar aggregate Core recall conceals materially
different false-alert costs. The right condition nearly doubles sampled Core
false-alert duration. Direction asymmetry is conditional on this finite cohort,
not evidence of a universal hardware or algorithmic right-side bias.

| Negative phase | Nominal FP | Left FP | Right FP |
| --- | ---: | ---: | ---: |
| Complete OUTSIDE clips (384 negative frames) | 0 | 11 | 34 |
| INSIDE pre-entry | 9 | 6 | 7 |
| INSIDE post-exit | 24 | 19 | 23 |

Every added Core FP is in an OUTSIDE clip. The apparent improvements on
pre-entry/exit samples do not cancel that new lateral burden. Strong-only FP
is 17/17/39 for nominal/left/right, so the right-side cost already exists in
current spatial decisions; hold increases its duration rather than originating
the entire failure.

The left condition delays `head_horizontal_g03_inside` by one sample
(10/11 positive frames covered). Right delays `head_hanging_plane_g03_inside`
(9/10 covered). All other held Core events retain full coverage. There are no
held-Core internal or terminal gaps; maximum silence is one initial sample.
The full clip names, first-alert and release records remain in `metrics.json`.

Current-observation UNKNOWN is nominal 656/768, left 724/768, right 686/768.
It is recomputed by the unchanged rule under each reported projection and is
never removed from denominators. Relative to nominal, 70/48 Core UNKNOWN flags
change under left/right offsets (including changes in both directions). All
OUTSIDE frames stay UNKNOWN, even when falsely alerting. No negative frame is
declared definite in this cohort; that does not certify definite-support
reliability for other calibration errors.

## Why the prospective checks pass without implying robust calibration

Under the same nominal/left/right inputs, Calibration has 240/211/240 Core FP;
strong+hold has 33/36/64. Every condition therefore retains at least 50% FP
reduction versus its matching permissive Calibration comparator. All 16 events,
<=.2 s onset, >=5/6 coverage, <=1 silent sample, Core false segments and the
separate INSIDE-negative/OUTSIDE FP-frame/segment conditions pass.

These inherited usefulness limits intentionally allow a bounded cost. They
contain no requirement of equal FP to nominal strong+hold. Keep the original
`passed_by_condition` values; do not redefine the gate after seeing outcomes.
The scientific conclusion includes both facts: useful relative-to-Calibration
behavior remains, while absolute lateral specificity is calibration-sensitive.

## Boundary remains a separate challenge

| Held condition | TP/FP/FN | Events | Largest delay among detected events |
| --- | --- | --- | --- |
| Nominal | 14/5/159 | 4/16 | .2 s |
| Left | 25/4/148 | 5/16 | .4 s |
| Right | 55/5/118 | 9/16 | 1.2 s |

All 384 Boundary frames remain UNKNOWN in every condition. The apparent gains
under deliberately wrong geometry are not an admissible calibration update or
learned Boundary improvement. No shifted condition replaces the retained demo.

## Verification, disposition and reproducibility

Four focused projection/causality checks and eleven existing complete-event
metric checks pass. Independent saved-output verification passes 289,527
assertions across four seals, 3456 frame-condition support reconstructions and
10368 policy decisions. It independently reproduces all metric groups, event
details and prospective checks without invoking the production metric module
or rerunning the score integral. Saved continuous scores are audited inputs;
support geometry and causal decisions are independently reconstructed.

Evidence is in `artifacts.local/work/ba-core-projection-stress-20260921/`, with
protocol, prediction and evaluation seals, reported projections, all flags,
complete metrics and changed IDs. Host scalar scoring plus three readouts per
input condition averages 2.682 ms (p95 4.550 ms); this excludes sensing,
RGB processing, I/O and device execution. CPU placement is
TASK_NOT_GPU_SUITABLE. No simulator, GPU job, paid worker or background process
was started or remains allocated.

Retain COMPONENT_OR_CHALLENGER / COMPONENT for this specified projection-error
diagnostic. The user still has no connected 8x8 hardware or measured inputs.
No hardware claim, model fit, threshold/offset retry, automatic successor,
protected-test access or runtime/default change follows this result. Supported
global registration still fails at `experiments/index.jsonl:303` (input fingerprint
mismatch), and inheritance reports an unknown terminal. Local structured
disposition and both command receipts retain this pre-existing metadata gap;
no ledger entry was edited or bypassed.
