# MZ132: anonymous visible-contour frontend diagnostic

One consumed simulation Development diagnostic, not deployable performance.
Retain MZ129. Pause new ToF center-span, envelope and weighting recipes.
Question: do accurate visible candidates create associations on returns that
actually sustain false alerts, particularly the 30 BODY false frames?

## Fixed comparison and acceptance, before candidate scoring

The authenticated comparator is MZ129 `replay-v1r2` Radar arm: 139 TP / 93 FP /
5 FN, 15/15 events, maximum delay .25 s, 22 false segments, 288 frames.
All original sensor packets, timestamps, IMU, range uncertainty, grouping rules,
association thresholds, zone weights and alert threshold remain fixed.

MZ129 only accepts boxes. The user authorized choosing the effective approach
after this interface limitation was explained. Use two nested fixed contrasts:

1. Anonymous visible-instance tight boxes replace the current RGB proposals.
2. The same candidates supply pixel masks: replace box overlap area with exact
   mask overlap area in the existing positive ToF signal/cohort matcher, then
   evaluate the union of local mask tiles with the same range/pose enclosures.
   Radar keeps its current box-based association/range filter; its associated
   plane can use the corresponding mask with one unchanged plane anchor.

This second arm changes the geometric carrier in association/readout, so it
must not be reported as a byte-identical fusion system or frontend-only effect.
Do not add MZ130's inherited-association veto: new anonymous indices have a new
namespace. Keep the MZ115 .25 m grouping, two-zone minimum, .8 coverage, .9 cosine,
.1 margin and 2 px padding. No post-score tuning or new cohort.

Both arms preserve raw Radar, complete original optical-flow carry and existing
MZ129 guards. Missing/unresolved candidates preserve the corresponding MZ129
return support. Any refinement preserves the native zone portion outside RGB
coverage; it cannot use image clipping as evidence of absence. SIM_MERGED keeps
its original support/range bounds. No-association is not clearance.

Primary go criterion: BODY FP <=15, every incumbent TP retained, all original
event first-alert times preserved or earlier, no new false frames, no previously
retained native corridor contributor newly excluded. Report all four families,
including all 35 boundary FP; report true-frame retention separately from native
hazard-hit correctness, especially near-rod/far-wall cases. Report native sample
loss outside RGB and on nonhazardous actors too, not just frame counts.

For each original FP report new ToF/Radar associations, changed actual triggering
returns, changed possible bits and zone scores, remaining raw/carry/guard/other
support, and final alert outcome. Pixel IoU or altered return count is not the
acceptance measure. Serialize predictions before scoring candidate labels.

Gain makes a small RGB frontend eligible for a separate next implementation;
this diagnostic does not start training automatically. A sound no-gain result
stops this frontend investment under this association/readout. A source/alignment
failure is NOT_EVALUABLE, not evidence against segmentation. Neither outcome
authorizes another ToF shrinkage recipe or a repair sweep.

## Source provenance and isolation

The original UE capture exports only RGB, no depth or instance masks. AABB
projections and sparse ToF collision hits are not visible pixel truth. Original
renderer includes noncollision albedo tile geometry 1 mm in front of the cubes.
Re-render the frozen scene/camera in a separate mask-only engineering capture,
including those tile children, all objects, background and floor. Do not call
the sensor simulator again. Verify original-appearance RGB alignment against
the frozen images, check categorical decoding and occlusion, and retain receipts.
If renderer provenance or visible boundaries cannot be established, do not score
an ideal-frontend claim.

Pre-score engineering admission clarification: exact original RGB bytes were
initially used as an alignment check. Local smoke showed small photometric
differences, and the original worker reproduced the BODY differences too. Exact
RGB brightness is not the contour requirement. Require the three cross-host
categorical masks to agree exactly. Admit the full categorical pass only with actual UE camera,
base-instance bounds, FOV and render-target size matched to the frozen source,
complete texture-child grouping, and exact nonempty palette decoding on every
frame. Preserve per-frame RGB differences without a fitted MAE tolerance. These
are pixel-center visible geometry masks with AA disabled, not fractional edge
coverage or a bit-identical replacement RGB stream. No candidate outcome was
scored before this source-admission clarification.
The full pass also exposed UE Python Rotator constructor float32 conversion:
compare readback to that exact input representation, then retain the 1e-9 degree
arithmetic-roundoff check. Preserve the first failed decimal comparison; this
is a typed input correction without recapture or outcome-based tolerance.

The predictor receives only anonymous 2D component masks/boxes in image
coordinates. Export every visible logical instance without category, distance,
hazard status, corridor test or echo-source identity; sort by 2D geometry rather
than actor ID. The evaluator-side color/actor map never enters the associator.
Do not convert rendered background or floor into a free-space veto.

Evidence: `artifacts.local/work/mz132-visible-contour-20260914/`.
`readiness-v1` authenticates all 288 original RGB hashes and reproduces each
saved MZ129 ToF allocation/score/alert; it is an engineering audit, not an oracle
result. The first readiness invocation found a partial RGB copy in `returned-v1`;
the complete authenticated `mz125.../rgb-v1` copy resolves that path issue.
Payloads and temporary captures remain under the canonical artifact junction.
Task-owned UE processes must be released after engineering capture.
