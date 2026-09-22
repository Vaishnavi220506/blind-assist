# Return lineage: retained foreground can lose the selected return

2026-09-20, consumed Development diagnosis only. The near surface remains in
the sampled zone in real corridor-positive examples while the original proxy
selects a farther non-target bin. This is a measurable representation loss in
the simulator. It does not establish recovery by multi-return, or a hardware
failure rate. Calibration outputs and threshold remain frozen; no alternative
return policy, aggregation, temporal rule, training or capture was run.

## Negative gaps: four competition states, with a simulator range-edge caveat

The five original negative gap frames yield four B states and one retained
target return; none is geometry transport. The four B states are three initial
switches and one continuation, not four independent temporal switches.

| Gap / old winner | Native target / zone pixels | Sampled target / zone samples | Reported m | Classification |
| --- | ---: | ---: | ---: | --- |
| f0267 / z36 | 48/756 | 9/165 | 7.4742 | B, pure backdrop wins |
| f0268 / z36 | 32/756 | 9/165 | 7.8498 | B continues |
| f0304 / z35 | 44/784 | 9/165 | 7.9576 | B, pure backdrop wins |
| f0376 / z43 | 140/784 | 30/165 | 7.9373 | B, pure backdrop wins |
| f0389 / z50 | 163/702 | 37/154 | 3.1353 | Target still selected |

The outgoing target is beyond 3 m in all five gap frames. These are not missed
near corridor obstacles. All three initial B switches coincide with backdrop
samples crossing the original simulator's hard `[0.1,8)` candidate gate:

| Initial switch | Backdrop depth before -> gap, m | Eligible backdrop samples before -> gap |
| --- | ---: | ---: |
| f0266 -> f0267 / z36 | 8.053624 -> 7.933624 | 0 -> 156 |
| f0303 -> f0304 / z35 | 8.060224 -> 7.940224 | 0 -> 156 |
| f0375 -> f0376 / z43 | 8.071424 -> 7.951425 | 0 -> 135 |

The backdrop already exists in the native footprint before each switch. Its
entry into eligible bin 79 allows it to beat a still-present foreground bin.
This is evidence of the proxy's range gate plus competition, not evidence of
spontaneous physical weak-return loss. No negative gap should be repaired.
The complete 13-record flank/gap expansion retains source IDs, occupancy,
all eligible candidate bins and the original selected return.

## Positive audit, with separate frame and zone denominators

B here means eligible target samples remain, a pure non-target bin farther than
all those samples wins, and that winning bin has no sampled corridor support.
A static B state need not be a temporal flip. The table restricts B to zones
with sampled target support inside the actual corridor in positive frames.

| Measure | Core432 | Thin96, original nominal45 observations |
| --- | ---: | ---: |
| Positive frames | 144 | 36 |
| Positive frames with native target in corridor | 108 | 33 |
| Positive frames with sampled target in corridor | 89 | 31 |
| B zone-frames / sampled target-corridor zone-frames | 94/500 (18.8%) | 95/187 (50.8%) |
| B subset with at least 4 eligible target samples | 87 | 87 |
| Positive frames with a corridor B zone | 55/144 | 24/36 |
| B winners: pure backdrop / pure unassigned other | 86 / 8 | 81 / 14 |
| B frames that are Calibration FN | 0 | 6 |
| Unchanged full-cohort TP / FP / FN | 144 / 146 / 0 | 30 / 15 / 6 |

All 189 corridor B records report beyond 3 m, and their original depth support
intervals lie wholly beyond 3 m. The >=4 view is an audit subset; the original
sensor minimum of four hits applies to the whole zone, not each target/bin.
Geometric-contact positives without visible/sampled interior target support
remain in the frame denominator. These curated cohorts must not be pooled into
a natural-distribution rate.

Core HEAD: 51/172 corridor zone-frames are B, across 36/72 positive frames.
Core BODY: 43/328 are B, across 19/72 frames. All still alert. Thin's 4 cm rod
has 41/54 B zone-frames across 6/6 positive frames; the thin plate has 39/54
across 6/6. Both retain all six alerts through other evidence.

The six existing Thin FN, **f0078-f0083**, all belong to the small foreground
inside the same zone as a distant backdrop. All eight sampled target-corridor
zone records in those frames lose to pure backdrop; target samples number
7-14, and the target bin ranks second. One additional f0081 zone has native
target pixels but no sampled target and remains a sampling-loss case.

For example, Thin f0078/z36 has target occupancy 52/756 native pixels and
14/165 sensor samples. Its target bin is at 2.988640 m with proxy weight
1.567442. The backdrop contributes 151 samples at 6.925048 m, weight 3.148700;
the noisy selected return is 7.039585 m. Thus the near surface did not leave
the zone: its candidate evidence was available but was not selected.
This diagnoses selection loss accompanying all six FN; it does not prove
that changing the return would recover six alerts without collateral FP.

## Persistent suppression versus actual near-to-far switching

Core positive frames contain two retained-foreground temporal B transitions
(f0011/z36 and f0379/z51), but only f0011 has a previous reported return <=3 m.
The f0379 switching zone has no sampled target-corridor support. Thin positives
have no temporal B transitions: their selection loss is persistent rather than
an episodic near-to-far dropout.

Core f0010 -> f0011/z36 retains 24 target samples in both frames, while its
reported return changes 2.443724 -> 6.194264 m. At f0011 those target samples
split between adjacent 10 cm bins: 18 at 2.299993 m (weight 3.402669) and six
at 2.322373 m (1.112468). The 141-sample backdrop at 6.270024 m wins with
3.586581. The combined target weight exceeds the backdrop, but neither
individual bin does. This exposes a bin-boundary sensitivity of the toy
constructor; the 22 mm foreground depth difference is not two resolved
physical VL53 targets. No merged-bin or alternative-return counterfactual ran.

The frozen grid-crossing proxy is a change in native target-occupied zone set.
Of Core's 42 positive crossing frames, 14 have corridor B evidence and zero
have a temporal B switch. Of Thin's five, four have corridor B and one is FN;
none has a temporal B switch. The audit does not establish grid crossing as
the cause of near-return dropout.

## Source fidelity and hardware boundary

The original constructor groups optical-Z samples into 10 cm bins, uses
sum(1/max(Z,.3)^2) to select one bin, and then adds original seeded noise.
It does not simulate photon signal, reflectance, waveform peaks or target
confidence. Object identity is authenticated actor/bounds metadata with the
existing +/-2 cm native-depth ownership proxy, not a segmentation-ID buffer.
Unassigned surfaces remain OTHER. Occupancies are discrete native pixel-centre
and sensor-lattice measurements, not exact continuous surface-area integrals.

[ST UM3109 Rev12](https://www.st.com/content/st_com/en/technical-documents/UM3109.html)
documents up to four targets per zone, default one target and Strongest order,
Closest ordering, and 600 mm minimum target separation. That interface makes
return retention a meaningful future hardware question; it does not validate
our toy candidate bins as sensor-detected targets. Meeting the separation bound
alone does not establish detection of a weak foreground return.

## Decision, validation and retained evidence

Retain Calibration as this branch's baseline. Keep RGB/noRGB and Zone-Handoff
closures. Retain this diagnostic as an evaluator-only **COMPONENT_OR_CHALLENGER,
mode COMPONENT**: candidate availability and selection-loss evidence, not an
alarm method. The positive small-foreground result keeps single versus multiple
return representation worth investigating if separately authorized. No successor
is started, and no threshold or overlap formula is changed.

All 528 stored vectors, original winner traces, scores and baseline predictions
reproduce exactly; original source seals pass before and after. The audit covers
33,792 zone-frames and 30,976 immediate same-zone pairs. Four focused synthetic
checks pass. Independent review recounts all records/transitions and reconstructs
133 unique native frames and 335 specified gap/FN/positive-B records with the
original RNG. No mechanical discrepancy was found. CPU audit time is 22.487 s
(TASK_NOT_GPU_SUITABLE), not hardware latency.

Frozen protocol JSON SHA256:
`412d63f98fcff40bbe67dbc31fa3408b12c25e5975b84691d0b9d31f3e63830f`.
Durable payload: `artifacts.local/work/ba-return-lineage-20260920/`, including
all candidate-bin records, gap tracks, FN records, original result seal,
independent review, eligibility/provenance and registry receipts. The derived
delivery seal is separate from the original result seal. All payloads are
decision evidence; no disposable dataset/model or persistent process remains.

Global registration/inheritance is blocked by the existing ledger line303
fingerprint error / unknown terminal. Preserve actual failure receipts and the
local disposition; do not edit around or claim completion of global metadata.

Implementation: [audit runner](run_return_lineage_audit.py),
[lineage reconstruction](return_lineage_core.py),
[focused checks](test_return_lineage_core.py),
[frozen protocol](RETURN_LINEAGE_PROTOCOL_20260920.md).
