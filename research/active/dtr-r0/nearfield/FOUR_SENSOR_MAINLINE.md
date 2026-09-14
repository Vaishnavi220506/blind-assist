# Four-sensor simulation mainline

User-confirmed architecture, 2026-09-13: **one RGB camera + ToF + Radar + IMU**.
The immediate priority is algorithmic benefit in class-agnostic forward obstacle
awareness. Work remains simulation-only. A second camera, stereo depth, removal
of Radar, or substitution of simulator truth for an observable input changes the
architecture and requires an explicit new user decision.

## Fixed-hardware priority, user correction 2026-09-14

Keep **RGB + 8x8 multi-zone ToF + the same low-cost Radar + IMU** as the
algorithm research input budget. Higher-resolution ToF is an information-capability
comparator only; it is not the mainline solution or evidence of an algorithm
contribution. The later user correction prioritizes direct corridor classification
with lateral intervention pairs over further flow-to-return contraction.
[MZ136](MZ136_RESULTS_20260914.md) completes that one matched BCE/hinge experiment:
pair order does not become a useful common alarm threshold, and no pairing gain
is retained. Keep MZ129. Native evidence stays auditable; a statistical candidate
need not solve unique return identity or OR the full incumbent alarm back in.
The user subsequently authorized a [TRAIN-only fit repair](MZ136_TRAIN_FIT_REPAIR_20260914.md):
frozen existing features plus a 385-parameter standardized direct readout reaches
191/192 frames and 95/96 pairs correct at zero logit, versus 154/192 and 60/96.
The subsequent frozen consumed-dev48 check fails alert transfer:12TP/6FP/12FN
at zero logit, or24TP/20FP/0FN at matched MZ129 recall/timing versus24/13/0.
Retain the fitting diagnostic only; scene-dependent residual offsets need work.
No hardware expansion or capture was performed in this continuation.

[Grouped readout stability](MZ136_GROUPED_READOUT_20260914.md) reduces readout-only
OOF errors with stronger L2 and paired midpoint supervision, but frozen dev48
still needs22FP versus MZ12913 at matched recall/timing. Keep MZ129; this fixed
pooled-feature linear correction is not an alert challenger. No original-test run.

Use measured image/IMU motion and state translation/correspondence assumptions.
IMU rotation does not supply metric translation. An image track need not be the
surface producing its ToF return, and coherent object motion can mimic ego motion.
Raw sensor support must be retained for audit; it does not follow that every old
possible-support branch must permanently own an alert vote. A new conditional
readout may change decisions under declared assumptions, with full missed-alert,
timing, UNKNOWN and actual contributor accounting. Do not claim calibrated
occupancy probabilities from uncalibrated fit or intersection scores.

[MZ135](MZ135_RESULTS_20260914.md) tests one actual five-frame conditional
image-flow candidate: 853 returns narrow and 17 possible bits disappear, but
139TP/93FP/5FN stays unchanged and 31 native corridor samples are newly lost.
Reject this source-ownership/local-flow assumption; no tuning or training follows.
[MZ133](MZ133_RESULTS_20260914.md) is the bounded angular capability control;
[MZ134](MZ134_RESULTS_20260914.md) diagnoses the inadequacy of an unconstrained
single-frame attribution model. All preserve MZ129 as the retained baseline.
Earlier stop points below continue to govern their completed experiments, not
this separately user-authorized fixed-hardware question.

## What the next implementation must answer

1. Does a measured surface occupy the current walking/body/head corridor?
2. Which image region, ToF zone and Radar return can credibly refer to one target?

RGB supplies image extent and angular localization. ToF supplies zonal range and
validity, not exact per-pixel depth. Radar supplies uncertain range, bearing and
radial velocity, not object identity. IMU supplies rotation increments, not
accurate metric translation or the user's future intended path. Time and
extrinsic calibration are explicit inputs. Association ambiguity stays visible.

Use identical ToF/Radar/IMU packets, geometry, history and task labels in the
three-sensor baseline and the four-sensor candidate. Report individual small-pole
and HEAD retention, false alarms, missing support/UNKNOWN, association coverage
and cost. Preserve independently valid ToF evidence. Missing RGB detections or
missing range returns cannot establish clearance. Never suppress an independent
sensor merely because another sensor missed the object.

RGB may refine ToF/Radar support within space it observes, but must never define
either sensor's complete spatial coverage. Preserve independently observed
support outside RGB coverage. This is a geometric coverage constraint.

## Evidence routing and correction

- [MZ132 ideal visible contours](MZ132_RESULTS_20260914.md): anonymous complete
  UE instance boxes/masks create2387/2553new ToF association proxies, but both
  remain139TP/93FP/5FN and BODY FP30. Two old BODY FP removed, two restored by
  retaining independent support outside RGB; all true frames/event times survive.
  Merged and nonvisual supports sustain BODY nuisance. Radar accepts no current
  association, so its mask audit has zero coverage. No native ToF sample lost.
  Keep MZ129; no frontend training, retuning or successor. Scoped negative
  diagnostic with ledger303 metadata pending, not a verdict against segmentation.
- [MZ131-A native angular envelope](MZ131_RESULTS_20260914.md): one fixed,
  explicitly uncalibrated cohort center-span formula narrows triggering
  footprints in 56/93 FP and changes their actual possible bits/scores in 23
  frames. FP93 to84 (six BODY, three rod), but TP139 to138 and 158 native
  corridor samples lost. All15 events/times survive. Reject this exact envelope;
  depth connectivity is not a demonstrated within-zone bound. Keep MZ129;
  intended NEGATIVE_CONTROL pending ledger303, no automatic successor.
- [MZ130 local support](MZ130_RESULTS_20260914.md): all four arms remain
  139/93/5 and 22 false segments, with all true frames/event times retained.
  815 local ToF returns create no new associations; 103 Radar returns gain
  local corroboration but no alert benefit. Image clipping excludes 67 native
  ToF samples outside RGB, including one corridor point. Reject this exact
  complete-mask rule; preserve independent support beyond visual coverage.
  MZ129 remains best on this panel; NEGATIVE_CONTROL metadata pending ledger303.
- [MZ129 Radar extent](MZ129_RESULTS_20260914.md): corrected MZ125 boxes enter
  current Radar geometry with complete original flow, raw and guard protection.
  On the same consumed 288 frames, 139/103/5 becomes 139/93/5; all ten exclusions
  are offroute HEAD, false segments 26 to 22, all true frames/events/timing
  retained. ToF subdivision has no frame gain alone or combined. Retain the
  simpler scoped Radar component; MZ116/default unchanged, metadata pending
  ledger303. BODY, rod and boundary errors remain; no automatic successor.
- [MZ128 FP diagnosis](MZ128_FP_DIAGNOSIS_20260914.md): all103 remaining false
  frames audited;67ToF-only/14Radar-only/22both.101have current real off-corridor
  support,2require inherited carry. Main issue is possible spatial extent
  overlapping the corridor;422ToF support returns have0native hazard hits and
  30current real Radar supports use offroute visual association.95FP are in
  multi-frame segments. The separate ToF angular-depth graph is absent from this
  alert path. Native-range/drop-MERGED controls are evaluator diagnostics only;
  unchanged MZ128/MZ116, no new predictor or automatic successor.
- [Depth connectivity](TOF_DEPTH_CONNECTIVITY_20260914.md), 2026-09-14: same60
  consumed artificial packets, three fixed graph arms. Matched near view improves
  equal21/34/12 to45/10/0 for both depth gates;22/22 known-depth background bridges
  cut,117/117 observed anchors retained. Raw all-support19/36/12 becomes17/38/0;
  keep notice selection separate from topology attribution. Four correct coarse
  directions still have UNKNOWN depth. Retain the simple absolute-gate component,
  not within-zone decomposition, hardware or fusion validation; frozen equal and
  MZ116 unchanged. Registration/inheritance pending ledger303; no successor.
- [ToF direction stress](TOF_DIRECTIONAL_STRESS_20260914.md), 2026-09-14: freeze
  equal zone weights for the additive readout. On 60 hand-constructed packets,
  equal and 1/0.5 both give 19/60 exact near-direction sets, 36 false CENTER
  frames and 12 bilateral merges. Range-blind connectivity merges modes when
  background bridges them; status weights cannot split a connected component.
  This is synthetic logic evidence, not physical echo or RGB fusion validation.
  Keep the old arm only for reproduction, MZ116 unchanged; registration and
  intended inheritance pending ledger303. No automatic successor.
- [MZ128](MZ128_RESULTS_20260913.md) preserves all association context and applies
  fixed zone weights only to alerts. Soft original139/107/5has28false segments;
  soft MZ125 matches139/105/5. Declared binary-four alert-use control gives139/103/5,
  57.44%precision,15events,.25s delay,26segments. Benefit is context retention plus
  two BODY exclusions, not continuous-weight superiority. Keep components and
  MZ116 baseline; structured registration/inheritance remains pending ledger303.
- [MZ127](MZ127_RESULTS_20260913.md) fills the four-column contrast:139/114/5,
  combined with MZ125139/108/5, preserving15events and.25s delay. Full MZ125 has
  fewer FP105 but more false segments27versus25combined. Removing peripheral
  zones restores5HEAD FP through insufficient two-zone association, offset by
  2BODY FP removed. Keep Development components and MZ116 baseline; ledger pending.
- [MZ126](MZ126_RESULTS_20260913.md) tests whole-zone central ToF selection.
  Half image width leaves MZ116 unchanged; quarter target retains16zones and
  only14.07%actual image width:130TP/58FP/14FN,69.15%precision,15/15events but
  first-alert delay increases to1.25s. All9 added FN are boundary stress;
  BODY/HEAD/rod recall unchanged. Retain a Development tradeoff diagnostic,
  not a narrowed-coverage default; MZ116 remains baseline. Ledger303 pending.
- [MZ125](MZ125_RESULTS_20260913.md) recovers pixel foreground extent before
  unchanged positive ToF association. On 288 consumed frames, FP116 becomes105
  with TP139/FN5, all15 events, delay and false segments unchanged. All11 removals
  belong to two offroute HEAD episodes; all1678 native samples in changed regions
  remain contained. Retain a scoped Development component, keep MZ116 baseline;
  no fresh confirmation or default promotion. Ledger303 registration stays pending.
- [MZ124](MZ124_RESULTS_20260913.md) completes the user-authorized five-direction
  Development package on consumed MZ123. Failure diagnosis separates BODY
  readout from weak rod response and low/wrong-distance HEAD output. Interval
  geometry tightens 62 ToF supports, but strict correction leaves MZ116 unchanged;
  conditional 3FP reduction inherits unresolved Radar rejection. Matched ranking
  loses 13TP and 11HEAD frames versus BCE on 96 dev frames. Hold/majority trade
  recall against nuisance. No tested arm replaces MZ116; no automatic successor.
  Intended scoped NEGATIVE_CONTROL; registration/inheritance remain pending.
- [MZ123](MZ123_RESULTS_20260913.md) validates MZ122's same-round early model
  at frozen 0.58 on 288 new controlled frames. MZ116 139/116/5 becomes 122/104/22
  TP/FP/FN, one suspended-HEAD event missed, and only 10.34% FP reduction.
  Alert-frame precision also falls 54.51% to 53.98%; nonstress failure persists.
  Stop this fixed fresh joint-alert role; retain the earlier consumed development
  gain and local-fusion spatial evidence. MZ116 retains high recall with unresolved
  false alerts. All 288 frames are consumed; failure causes remain unresolved. No training,
  threshold rescue or automatic successor; intended NEGATIVE_CONTROL pending
  the existing ledger303 registration/inheritance blocker.
- [MZ122](MZ122_RESULTS_20260913.md) completes paired local-before-pooling fits.
  Training/dev spatial precision envelopes improve, but late frame-recall
  thresholds<=.49and nuisance thresholds>=.60do not intersect. Offroute FP and
  1cm boundary misses prevent joint alert retention. Preserve positive spatial
  evidence, stop this exact alert role; MZ116/MZ121 remain. No MZ119 transfer,
  CNH or automatic data expansion. Intended scoped NEGATIVE_CONTROL, registration
  pending ledger303; extra1296parameters prevent isolated order-only attribution.

- [MZ121](MZ121_JOINT_READOUT_20260913.md) checks the existing development curve
  without training. Joint feasible grid.30–.60exists; the separate.60readout
  gives84TP/16FP/0FN with HEAD142/142and rod114/126. Wrong cells increase899to1130
  versus.71, so threshold correction does not establish improved spatial precision.
  Keep MZ116 and MZ120's old terminal; no MZ119 retuning or automatic model run.
  Intended Development COMPONENT_OR_CHALLENGER, registration pending ledger303.

- [MZ120](MZ120_RESULTS_20260913.md) follows the newly authorized learned spatial
  ownership direction. The independent single-frame candidate can retract old
  alerts; preserving every old output is not its requirement. MZ116 stays fixed.
  Small-data Development FP40to14 hides rod-cell recall102/126. Frozen transfer
  FP62to20 also loses HEAD frame recall42/42to26/42 and delays alerts to1.25s.
  Stop this exact pilot; no promotion, CNH or new capture. The earlier MZ119 stop
  was superseded for this bounded round, not unlimited successors. Structured
  inheritance remains pending the existing ledger303 mismatch; intended
  NEGATIVE_CONTROL, not a conclusion against learned fusion or current inputs.

- [MZ119](MZ119_RESULTS_20260913.md) was the previous final authorized attempt.
  **USER_REQUESTED_STOP_AFTER_MZ119**: no further optimization or successor.
  On240new frames, ToF-scaled causal RGB parallax leaves primary153TP/62FP/2FN
  and timing/nuisance unchanged. Four1.8m gains have mixed correct/false point
  support;13of55depth points put distant context nearby. Moving anchors explain
  17commanded-reference pose containment failures. Retain the fixed recipe as
  intended NEGATIVE_CONTROL, keep MZ116 runnable, and make no promotion claim.
  Source/inference/audit completed and capture resources released; structured
  registration remains pending the pre-existing ledger303 fingerprint error.

- [MZ118](MZ118_RESULTS_20260913.md) recovers border-touching RGB surface regions
  without changing Radar proposals. The interval-plane route removes3MZ115FP,
  but excludes551actual corridor-hit contributors across44MZ117frames despite
  unchanged primary frame recall. Wall-only anchors do not establish a mixed
  target's surface identity. Preserve as a localization negative control; the
  full distance curve also loses warnings. Keep MZ116 as runnable challenger.

- [MZ117](MZ117_RESULTS_20260913.md) admits a complete native multi-actor source:
  35near/far mixed frames, including18weak-near/far-biased frames. Conditional
  plane and nominal proxy both have zero refinement coverage across1168merged
  targets; all ten-distance decisions remain MZ116-equivalent (primary141/64/4).
  Preserve the fixed eligibility recipe as a coverage negative control; zero
  contributor losses are vacuous. Keep MZ116 runnable; source is now consumed.

- [MZ116](MZ116_RESULTS_20260913.md) adds a runnable pixel-resolution Radar guard:
  independent support survives a visual box with no robust interior.1008 consumed
  frames preserve every prior alert and recover one real rod frame (MZ115142/69/2).
  The mixed-ToF oracle exposes29FP potential, but429current merged targets all
  have one actor; future mixture claims need actual multi-actor merged inputs.

- [MZ115](MZ115_RESULTS_20260913.md) implements hypothetical finite ToF footprints
  and strongest2 status-bearing return tuples. Same-tuple RGB allocation on fresh
  240frames changes141TP/71FP/3FN to141/69/3 at3.6m, with no contributor hit
  dropped. Duration falls.5s but false segments18→19;2.6m loses oneTP and delays
  first alert. Retain the interface/allocation as a Development challenger,
  without alert promotion or claims of full-curve/hardware improvement.

- MZ90--100 are range/rotation sensor experiments; their predictors do not consume
  RGB. They do not establish a tested four-sensor fusion system.
- MZ101--106 are a **separate stereo RGB + ToF branch**, with no Radar in the
  evaluated pipeline. Preserve their negative results and baselines within that
  branch. They do not diagnose the four-sensor mainline's algorithmic ceiling.
- MZ107 starts a fresh, bounded, synchronized four-sensor simulation canary.
  Its simple rendered appearance is controlled Development, not a general visual
  detector benchmark. Native geometry and target identities are evaluator-only.

## Advancement and stop points

First verify input provenance and that actual RGB pixels affect association.
Then measure a paired task effect with an RGB-disabled control. A gain must
retain critical baseline positives; fewer reported points alone is not a gain.
MZ109 exposes an additional requirement: also report absolute critical-stratum
recall and positives unique to the active RGB challenger. Retaining100% of a
baseline that detects1/16 true boundary frames does not establish adequate recall.
Keep comparator-specific gains and tradeoffs visible; do not promote a method
solely because a baseline-only gate passes.
If image proposals or associations fail, report that specific mechanism and stop
the bounded attempt. Do not jump to larger depth models or more sensors.

Only after a useful observable association mechanism is fixed should a complete
new scene split test it unchanged, including ambiguity, low contrast, occlusion,
time/calibration errors and sensor gaps. Current-corridor occupancy and future
collision prediction are different tasks; report them separately. Each experiment
ends at its stated budget, followed by evidence and code delivery.

## Revised design direction: spatial evidence before object categories

User proposal, 2026-09-13; not a validated algorithm. Represent coarse forward
regions with local contour/extent candidates, range and extent uncertainty,
known/unknown height, sensor provenance, evidence age and unobserved space.
Do not fill a whole grid cell or clip a thin structure at a cell center. A
ToF-zone range is not automatically the depth of every image pixel in that zone.

ToF supplies zonal distance and validity; Radar supplies independent distance,
coarse bearing and motion with HEIGHT_UNKNOWN where unobserved; RGB supplies
boundaries and angular localization without requiring a named object category;
IMU compensates short-term rotation without inventing metric motion or intent.
Allow separate RGB+ToF and RGB+Radar candidate branches. ToF absence must not
prevent consideration of an RGB+Radar pair. Conflicting ranges can belong to
different objects; preserve alternative explanations and independent support.
Repeated Radar detections alone do not establish independent corroboration.

Use the initial body/walking reference for the first simulation comparison,
explicitly retaining reference uncertainty. Head orientation and future motion
remain distinct. A multizone footprint/multireturn simulator is a separate
sensor-model study: existing center-ray results cannot decide physical thin-pole
detection, and a replacement must not assume that every thin pole is detected.

[MZ110 diagnosis](MZ110_RESULTS_20260913.md) finds six ToF-blocked opportunities
among 30 nominal misses; simple gate removal recovers four but adds a net FP.
All nine pole misses lack target Radar and five new FP have correct diagnostic
identity but wrong geometry. Preserve the diagnostic as a component, keep both
existing comparators, and do not promote the gate-removal counterfactual.

[MZ111](MZ111_RESULTS_20260913.md) and its unchanged new-scene
[MZ112 evaluation](MZ112_RESULTS_20260913.md) establish controlled recall gains
from independent RGB/Radar association plus current-visual-correspondence-gated
short range persistence. Fresh nonstress FN29 to7 at the same29FP and maximum
first alert delay1.0s to0.25s; full-panel FP38 to41 exposes the boundary tradeoff.
Keep these as components/challengers, with explicit age/ambiguity and retained
nominal/interval/hold comparators. Plane/filter complexity shows no fresh gain;
zero-velocity controls show most persistence gains do not require Doppler.
The actual reference remains initial body axes; pose drift, full miss/nuisance
curves and ghost discrimination remain unresolved.

[MZ113](MZ113_RESULTS_20260913.md) tests actual pixel correspondence on a new
240-frame dynamic source with15obstacle exits. The optical-flow combination
improves incumbent84TP/60FP/4FN to87/61/1, recovering three crossing-pole frames;
the same-policy angular control remains84/60/4. All16events start alerting at
the first sampled positive bin. Post-exit false support stays26frames/6.5s.
The one addedFP correctly tracks a pole still7mm outside the corridor. Retain
the controlled tracking component and its ablation, not a nuisance-improvement
claim. Dynamic sampled boxes do not establish natural tracking or full occlusion
robustness; initial body reference and hypothetical RF limits remain.
Post-seal diagnostics identify40independent persistent-ghost FP among61total FP;
the sole FN has range/image evidence but reciprocal association ambiguity. These
are separate responsibilities, not reasons to increase persistence lifetime.

[MZ114](MZ114_RESULTS_20260913.md) rejects a fixed global-assignment/range-prior
alert replacement:768consumed frames375TP/138FP/21FN ->377/142/19. Preserving the
old matched-pair range smoothing restores one lostTP (posthoc378/142/18), but
no net FP or timing benefit appears; ghost-to-object mistakes can masquerade as FP
removal or reinforce new alerts. Keep it as a negative control within that role.
Suppressing unmatched Radar would remove71FP but lose31TP on480frames, including
25correct hazardous-actor supports. Neither absent center rays nor repeated
compatible ranges are enough to identify persistent ghosts. Study finite-footprint zonal
return information next; do not assume a reported range-noise sigma is a true
within-zone spatial depth distribution, or strongest-first is nearest-first.

After fixing an observable method, evaluate complete new scenes with identical
alert distance/corridor/margins and the full miss-versus-reminder relationship.
Report obstacle-event misses, first correct alert timing and irrelevant alert
count/duration per route. Keep centimeter-boundary pressure tests separate from
event-level headline results. If RGB cannot reduce nuisance at matched recall
and timing, retain it for direction display/explanation within that tested scope.
