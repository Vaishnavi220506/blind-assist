# Existing spatial evidence: useful headroom, asymmetric positive and negative authority

On the existing changed-domain 288 frames, **A OR privileged positive support
reaches 103 TP / 27 FP / 5 FN, F1 86.55%** on the 5 cm clear task. Combining
positive support with an outside-return veto gives **102 / 17 / 6, F1 89.87%**.
It recovers 8 A misses and removes 10 false alerts, but deletes one existing
rod TP and delays that episode's first alert by 0.75 s. These are evaluator-only
decision probes, not trained/public-input method performance or a universal
upper bound. No data were collected, no model was trained or rerun, and no
threshold was adjusted.

## Fixed decision rules and what they actually mean

The [protocol](TRISTATE_EVIDENCE_PROTOCOL_20260917.md) reuses the frozen A score
threshold 0.3917890013717321 and the [audited returned-support records](SURFACE_ORACLE_RESULTS_20260917.md).
All native faces/points must belong to an existing usable ToF return; unreturned
rays, ground-truth class, family and error identity cannot trigger a decision.
The native geometry remains privileged even for the sampled-point variant.

- **P / POSITIVE:** at least one exact returned face intersects the nominal
  corridor. The sampled control tests exact linked points instead.
- **N_candidate / OUTSIDE_ONLY:** a packet was received, at least one usable
  return exists, every usable return is resolved, and no returned support
  intersects. This is the most permissive nonzero-return negative candidate;
  no count or coverage threshold was fitted. It is evidence about the measured
  surfaces, not proof that the entire corridor is empty.
- **U / UNKNOWN:** missing packet, no usable returns, or unresolved support
  without P. Every arm retains A's decision on U. P takes precedence over an
  unrelated unresolved slot.

Exact face unions are used; enclosing boxes cannot silently bridge separate
surfaces. The four arms are A, A OR P, A AND NOT N_candidate, and
(A OR P) AND NOT N_candidate. Predictions/states were sealed before joining
prior labels and diagnostics. The source is already consumed Development;
this ordering does not create new confirmation evidence.

## Main task: 5 cm clear corridor

All arms evaluate the same 216 / 288 frames, **75% coverage**, comprising
108 positive and 108 negative frames. Sampled points and full faces produce
identical clear-task states and decisions.

| Rule | TP / FP / FN | Precision | Recall | F1 | A FN recovered | A TP lost | A FP removed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 95 / 27 / 13 | 77.87% | 87.96% | 82.61% | 0 | 0 | 0 |
| A OR P | 103 / 27 / 5 | 79.23% | 95.37% | 86.55% | 8 | 0 | 0 |
| A AND NOT N_candidate | 94 / 17 / 14 | 84.68% | 87.04% | 85.84% | 0 | 1 | 10 |
| Positive plus veto | 102 / 17 / 6 | 85.71% | 94.44% | 89.87% | 8 | 1 | 10 |

No arm adds FP. The positive-only rule preserves every A alert by construction;
its zero new FP here relies on privileged geometry, not a demonstrated public
predictor. Joint fusion improves F1 by 7.26 percentage points and reduces clear
FP by 37.04%, while the single veto loss remains material.

| Clear family | A | A OR P | Veto only | Positive plus veto |
| --- | --- | --- | --- | --- |
| BODY | 36 / 6 / 0 | 36 / 6 / 0 | 36 / 3 / 0 | 36 / 3 / 0 |
| HEAD | 31 / 8 / 5 | 36 / 8 / 0 | 31 / 3 / 5 | 36 / 3 / 0 |
| Rod / far wall | 28 / 13 / 8 | 31 / 13 / 5 | 27 / 11 / 9 | 30 / 11 / 6 |

## UNKNOWN is preserved; outside-only still contains missed objects

The 288-frame face readout has P=102, OUTSIDE_ONLY=114, U=72. Sampled points
have P=98, OUTSIDE_ONLY=118, U=72. The 72 U frames consist of 29 missing packets
and 43 received packets with no usable return. All 72 decisions match A in
every arm. Across the strict cohort this retains 35 A TP; within clear it retains
18 A TP, together with the 17 false alerts that ToF cannot adjudicate here.

| Clear evidence state | Frames | Positive frames | Negative frames | A TP / FP / FN / TN |
| --- | ---: | ---: | ---: | --- |
| POSITIVE | 84 | 84 | 0 | 76 / 0 / 8 / 0 |
| OUTSIDE_ONLY | 91 | 4 | 87 | 1 / 10 / 3 / 77 |
| UNKNOWN | 41 | 20 | 21 | 18 / 17 / 2 / 4 |

**Four true clear-positive frames have only outside returns.** Three were
already A misses; the fourth is `s1confirm_near_rod_farwall_scene1_in_00`.
It has 50 usable, fully resolved returns, but no native ray hits the corridor
rod. A correctly alerts at t=0; the outside-only veto suppresses it. The next
alert occurs at t=0.75 s, including in the joint arm. Thus a large return count
does not establish sufficient corridor coverage or license a categorical veto.
Keep OUTSIDE_ONLY distinct from certified free space; unmeasured target space
can remain unknown even when the overall frame has many valid returns.

The joint arm's remaining clear FN are the original five missing-measurement
rod frames plus this newly vetoed rod TP. Its 17 remaining FP are all UNKNOWN
frames. Merely fixing zero-return handling resolves neither return coverage
gaps nor false alerts from independent modalities.

## Event detection, first alert and burden

Every main-task arm retains **18 / 18 core events**. Their observed first-sample
counts are A=16, OR=17, veto=15, joint=16. Mean observed first-core delays are
0.111 / 0.014 / 0.153 / 0.056 s respectively; these aggregate averages do not
erase a worsened individual event.

Positive support changes the first HEAD episode from 1.25 to 0 s and the first
rod episode from 0.75 to 0.25 s. Veto changes a different rod episode from 0 to
0.75 s. Both effects remain in the joint result. All 18 core events are
left-censored, and no stable-release opportunity exists in these snippets;
do not infer advance warning before obstacle onset or stable alarm clearing.

Clear-negative sampled alarm burden falls from 27/108=25% to 17/108=15.74%
for either veto arm. With the inherited 4 Hz bin convention this is 6.75 to
4.25 sampled seconds; negative alert segments fall from 16 to 12. Positive-only
OR leaves negative burden unchanged. These are simulated short-trajectory
metrics, not continuous user-experience or deployable latency measurements.

## Strict and boundary appendix

| Full-face rule | Strict 288 TP / FP / FN | Strict F1 | Boundary 72 TP / FP / FN | Strict events |
| --- | --- | ---: | --- | --- |
| A | 120 / 45 / 24 | 77.67% | 25 / 18 / 11 | 29 / 30 |
| A OR P | 139 / 45 / 5 | 84.76% | 36 / 18 / 0 | 30 / 30 |
| Veto only | 118 / 31 / 26 | 80.55% | 24 / 14 / 12 | 29 / 30 |
| Positive plus veto | 137 / 31 / 7 | 87.82% | 35 / 14 / 1 | 30 / 30 |

Sampled-point controls are respectively 136/45/8, 117/31/27 and 133/31/11
for the three changed strict arms, with 29/30, 28/30 and 29/30 strict events.
Their boundary counts are 33/18/3, 23/14/13 and 31/14/5. Full faces add four
boundary support opportunities: three additional A misses are rescued and
one sampled-veto TP loss is avoided. None of this is additional clear-task
surface-completion benefit.

Full-face veto also loses the boundary TP
`s1confirm_shallow_boundary_stress_scene3_enter_04`: one background return,
private target hit but no usable target return. Sampled-only veto additionally
loses `s1confirm_shallow_boundary_stress_scene5_enter_05`, removing that strict
positive event entirely in the veto-only arm. These cases are retained in the
audit rather than hidden by the main-task coverage mask.

## Decision and reproducibility

Retain a **privileged positive result for existing spatial-evidence readout**:
the information can improve the joint task without new sensing, and public
representation/readout is a justified research candidate. This does not prove
that current public inputs can recover the oracle's point/face geometry.
Retain positive-only OR as the no-A-alert-loss comparator. Treat the joint
89.87% result as an attractive measured tradeoff with explicit rod/timing cost,
not as proof of reliable negative evidence or a final method. No tuning is
needed to chase the 90% boundary.

Five focused tests cover missingness fallback, unresolved-support UNKNOWN,
positive priority, disjoint union gaps and sampled-versus-complete support.
The independent audit reconstructs intersections without the probe helper and
checks states, alerts, counts and event changes against the saved results.
All 49,729 returned points, including 5,207 corridor contributors, remain
unchanged. No veto removes an alert backed by a returned native corridor point;
nevertheless it removes independent true evidence where the rod was not sampled.
Native retention alone therefore cannot certify final-alert retention.

Reproduce with `run_tristate_evidence.py` (fresh output directory only),
`python -m unittest test_tristate_evidence -v`, and
`audit_tristate_evidence.py` from this directory. Evidence is retained under
`artifacts.local/work/corridor-tristate-evidence-20260917/`: input freeze,
sealed predictions, all cases, summary, backend/completion and independent
audit. Scalar CPU only; no persistent workers or GPU allocation were started.
The source A, dataset, old oracle and default App are unchanged. This probe
ends with its report; no automatic training, capture or sensor successor.

The supported registration command hit the existing
`experiments/index.jsonl:303 input_fingerprint` validation error; structured
inheritance then returned unknown terminal ID. Both receipts are retained.
Registration and the intended diagnostic COMPONENT_OR_CHALLENGER disposition
remain pending; the shared ledger was not edited or bypassed.
