# Cross-zone existing-return availability ceiling

2026-09-20. EXPLORE consumed-source diagnostic. Status: COMPLETE.
The user authorized the prerequisite information-ceiling check,
without training, a new visual algorithm, new observations or threshold changes.

Cross-zone existing support is nonzero but narrow in the obstacle-specific
cohort: **1/4 fixed missed interior events** has a same-frame pure near anchor
elsewhere. The other three have no target-owned observed return, even when the
range cutoff is removed. Retain a bounded association opportunity; do not promote
this into a general solution for thin obstacles or a recovered detection.

## Measured availability

The 96-frame / eight-clip camera-corridor audit preserves every original frame,
prediction, label, event and timestamp. Its saved-scalar cutoff is <3m.

| Original missed interior event | Interior frames | Frames with pure cross-zone support | Eligible target pixels |
| --- | ---: | ---: | ---: |
| s00 / g1_center: 4cm center cylinder | 5 | 0 | 0 |
| s02 / g2_intrusion: horizontal bar | 6 | 5 | 88 |
| s05 / g3_thin: 4cm-wide thin plate | 6 | 0 | 0 |
| s06 / g4_inside: small same-zone object | 5 | 0 | 0 |

The s02 witnesses are f0031..f0035, original frames7..11 at1.4..2.2s. Entry was
at1.2s; **none of the four missed events has support at entry**. Even this sole
opportunity first appears0.2s later. These are availability times, not alert times.
The old raw-ToF arm already detects s02; this is not an opportunity demonstrated
to beat that simpler baseline.
Across all22 missed interior frames,88/1,856 sampled target pixels meet the strict
cross-zone condition. Across all29 positive missed frames including boundaries,
371/2,471 pixels in10 frames qualify. Pure/mixed counts remain separate; the
corridor has zero mixed target anchors. All sampled target pixels are in the
ToF zones, so outside-FOV exclusions do not explain the three event zeros.

The original NFO still detects1/5 interior events. Its all-known frame counts
remain TP7, FP5, FN29, TN55 (recall19.44%, precision58.33%, FPR8.33%); one
prediction-UNKNOWN frame remains recorded. No new alert, false-positive or
latency improvement was measured. Pixel IoU is not defined for this alert output.

The separate original500 VAL cohort uses <2m and privileged connected near
regions. In the **far_small** domain, the6,079 old FN pixels partition as follows:

| Existing support for the missed near region | Pixels | Share of old FN |
| --- | ---: | ---: |
| Pure near donor elsewhere; no local contribution at any range | 4,539 | 74.67% |
| Only possible/mixed near donor elsewhere; no local contribution | 116 | 1.91% |
| Local owned return >=2m, with a near donor elsewhere | 95 | 1.56% |
| No observed near donor anywhere for the region | 1,329 | 21.86% |

There are220 frame-local missed connected regions;110 have any pure near donor,
and118 have any possible near donor (including those110). These are region
counts, not independently identified objects. A connected depth<2m region can
include adjoining surfaces, so74.67% is an optimistic support envelope, not
validated same-object association coverage. No outside-ToF pixel enters far_small.

The frozen far_small baseline is TP13,134, FP98,788, FN6,079, TN174,743:
recall68.36%, precision11.73%, IoU11.13%, FPR36.12%. Adding only those4,539 FN
mathematically gives91.98% recall, **not an achieved result**. All98,788 false
positives and the separate212,988 pure_far false positives remain unchanged.
All seven original domain TP/FP/FN/TN totals replay exactly. The350,140 reference
UNKNOWN pixels stay excluded, and domains overlap rather than form a partition.
The full-domain other_pure count includes28,507 outside-ToF pixels, separately
marked in the payload; do not interpret that full count as covered query zones.

Decision: the narrower hypothesis "some missed foreground has an existing
return in another zone" has a witness in controlled simulation. Its proposed
general rescue role is unsupported: the center cylinder, thin plate and small
same-zone object still provide no such return. A later lightweight association
pilot could target the observed horizontal-bar case while retaining negative
controls, but this audit neither implements nor authorizes an automatic successor.
Large-model inference is unnecessary for this audit; practical edge cost and
RGB association feasibility remain unmeasured. Physical weak-peak availability
remains NOT_EVALUABLE without real captures.

Question: among the fixed NFO misses, does the same frame already contain an
observed near return attributable to the missed foreground in another zone?
This tests a necessary input opportunity for a future cross-zone association
method, not its sufficient geometric/visual feasibility.

## Frozen cohorts and attribution

Use all 96 existing camera-corridor frames with saved NFO alerts, original event
intervals and the original target footprint. Retain all eight clips, interior,
boundary and entry distinctions. No G6 clip is substituted: that earlier source
has no saved ToF. A target footprint uses the existing native sampled points
inside authenticated render bounds plus 2 cm, sampled to the exact 256x192
sensor lattice. It is a bounds-derived proxy, not a newly captured instance mask.

Separately, use the same original 500 VAL frames and packed 0.081 NFO predictions
from the frozen-transfer/Depth Pro echo comparisons. This is different from the
500 TEST files inspected in the prior input-availability inventory. Keep all
original near2m domains, families, frame identities and baseline counts. Define
reference components by 8-connected known depth<2m, without pruning or gap
filling. This deliberately optimistic envelope can merge touching objects or
surfaces. A positive result here alone cannot establish same-object anchoring.

Reconstruct the original scalar proxy's winning 10 cm histogram bin, random
dropout and Gaussian noise from its unchanged seed/depth input. Require exact
float32 saved-range parity in every one of the 596 frames before counting its
lineage. This is attribution replay of old observations; no new measurement is
fed to a model. Pixels merely near the saved distance are not assigned to it.

For a finite observed scalar, inspect the actual winning-bin contributor pixels:

- **Pure:** every contributor belongs to the same component/target footprint.
- **Possible/mixed:** at least one contributor belongs, with other contributors
  also present. Possible includes pure; these categories are not added together.
- **No observed anchor:** dropout, out-of-range/missing scalar, or no attributable
  winning-bin contributor. Never replace the saved value with reference distance.

Count anchors below2m for the NFO cohort and below3m for the camera-corridor
cohort; the latter additionally reports all observed ranges diagnostically.
These are point-return opportunity cutoffs, not new uncertainty-aware corridor
alerts. Keep an **other-zone-only** opportunity separate: the queried pixel's
own zone has no observed contributor from that foreground at any range, but
another zone has a near anchor. A local owned return above the near cutoff is
recorded separately, never credited as cross-zone-only.
Outside-ToF pixels are counted separately where applicable. No future frame,
temporal smoothing, depth extrapolation or constant-depth object assumption is used.

## Readout and decision

Report original TP/FP/FN/TN unchanged and missed pixels/components/events with
local, other-zone-only pure, other-zone-only possible and no anchors. Retain
actual donor zone IDs and contributor-label counts for audit. An optimistic
recall ceiling adds only eligible old FN to old TP mathematically; it is not a
new prediction or a measured precision/IoU/alert improvement. Reference identity
and foreground shape are privileged and unavailable to a future RGB method.

Zero pure-anchor opportunity in the missed corridor events closes this proposed
source of recovery for those cases. Nonzero opportunities identify a bounded
subset that may warrant a later implementation; their size, coverage and mixed
ownership remain explicit. A large merged-component ceiling cannot override
missing entity evidence. Do not declare any arbitrary percentage a demonstrated
algorithm success. Finish this audit and its evidence delivery regardless of
outcome; no automatic training, model/threshold sweep or visual successor.

Budget: one scalar/file pass per cohort, CPU, no model calls or new captures.
This is not an edge-runtime benchmark. Synthetic checks verify exact sensor
replay, mixed ownership, original domains, zone boundaries and opportunity
accounting before cohort scoring. The protocol and code/input hashes are frozen
before the new attribution counts are produced; old public predictions remain
sealed. Native access is explicitly evaluator-only throughout this ceiling.

Payload root: `artifacts.local/work/ba-cross-zone-anchor-ceiling-20260920/`.
Implementation: [shared replay](cross_zone_anchor_core.py),
[500-frame audit](audit_cross_zone_nfo.py), [96-frame audit](audit_cross_zone_corridor.py),
[synthetic checks](test_cross_zone_anchor_core.py).

## Execution and retained evidence

The protocol froze at2026-09-19T16:42:50Z (2026-09-20 local), before attribution
counts, with eight code hashes and11 input hashes. SHA256:
`ce421592fb0f37012dc25834e74c702758819b9fa3a148b1bae1729fa177498b`.
The immutable `protocol-before-audit.md` contains the pre-score decision rules.
All596 saved scalar vectors replay exactly, including dropout and noisy float32
distances. The16 synthetic tests pass. Each cohort ran once: corridor3.387s and
NFO9.918s on CPU. These are audit timings, not algorithm/device benchmarks.
Zero model calls, new captures, new observation streams or training updates.

Retain `protocol.json`, `protocol-before-audit.md`, `nfo-results.json`,
`nfo-frames.json`, `corridor-results.json`, backend/test/run logs and verification
receipts under the payload root. These records include actual zone IDs and
winning-bin contributor counts. Preserve old source payloads and prediction seals.
No persistent worker or device session was started. A mistakenly created empty
nested artifact directory was removed after checking its path and empty content;
all evidence uses the canonical artifacts.local junction.

Global registration is blocked by the pre-existing
`experiments/index.jsonl:303 input_fingerprint` mismatch. The supported inheritance
command also cannot register this unknown terminal. `registration.log`,
`inheritance.log` and `local-disposition.json` preserve the pending metadata state;
no global ledger was repaired or bypassed. Local disposition retains this as a
consumed diagnostic component, with no prediction/alert/hardware authority.
