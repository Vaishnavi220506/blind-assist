# One CPU contour-parallax mechanism test

Phase: consumed controlled synthetic Development. Status: complete; frozen retain
gate failed. Run started 2026-09-19 and evaluation/delivery completed 2026-09-20.

The CPU contour replacement recovered **0/4** original 2–3 m misses. Algorithm
P95 was **554.568 ms/window**, exceeding the frozen 200 ms allocation; peak RSS
was 67.484 MiB. Close this exact recipe as `NEGATIVE_CONTROL`. This does not
establish that every contour representation is ineffective, and no parameter,
implementation-speed or successor rescue was run.

## Measured result

All twelve three-view windows completed, with predictions sealed before native
scoring. There were 1,407 line candidates, 453 accepted distance intervals and
954 UNKNOWN lines. Only seven lines qualified as near under the inherited G6
geometry; all seven occurred in the forward 1 m clip. None recovered an intended
target under the frozen distributed-support criterion, including the four
originally missed translational 2–3 m clips.

| Original clip | Motion / starting distance | Target candidate samples | Unique target pixels / visible pixels | Maximum target samples on one line | Accepted lines touching target | Recovered |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 01 | Forward / 2 m | 14 | 14 / 2,208 | 5 | 0 | No |
| 02 | Forward / 3 m | 20 | 20 / 1,060 | 17 | 0 | No |
| 05 | Lateral / 2 m | 32 | 32 / 1,821 | 14 | 0 | No |
| 06 | Lateral / 3 m | 8 | 8 / 880 | 6 | 0 | No |
| 03 | Forward / 6 m control | 21 | 21 / 224 | 17 | 0 | Not a near target |
| 07 | Lateral / 6 m control | 3 | 3 / 220 | 3 | 1 | Not a near target |
| 10 | Far wall mark control | 34 | 34 / 280 | 17 | 0 | Not a near target |

Both far bars previously had zero proposals. They now have native-target
candidates, so the minimum coverage gate passes. Coverage is still sparse:
9.375% and 1.364% of visible target pixels, respectively; the mark has 12.143%.
The three controls have zero target false-near samples, but most target-touching
lines abstain. This is not evidence of broadly reliable far-object ranging.
Both pure-yaw windows retain all distances as UNKNOWN (120 and 122 lines), with
14 and 4 target samples present. The empty control produces no near line.

Across all windows there are **104 non-target false-near samples**, all in the
forward 1 m clip and all individually supported by the image matcher. This is
104/23,563 = 0.441% of non-target candidate samples, or 104/17,674 = 0.588% of
native-far non-target samples. Samples can repeat across lines; these are not
unique pixel, frame or event false-positive rates. The accepted intervals contain
native depth at 5,853/7,695 = **76.062%** of valid accepted samples; the score-drop
interval is not a calibrated confidence interval. Twenty-two proposed samples
have UNKNOWN native reference and remain outside the known-reference denominators.

The unchanged old cached outputs replay exactly against both old reports: 41
target candidates, zero target false-near, 45 non-target false-near, of which 16
also meet the old body/head readout. Proposal populations and near-height
semantics differ; the new 104 and old 16/45 are descriptive counts, not a paired
accuracy or precision comparison. No old matcher or dense model was rerun.

## What the saved failures explain

Saved-output diagnosis, without new inference, shows two failures. In clips 01
and 06, the best-covered proposed line has only 5/17 and 6/17 target samples;
the fixed proposal set does not preserve a sufficiently supported foreground
segment. In clip 02, all 17 samples of one line lie on the target, but the best
hypothesis is 14.179 m versus native optical depth 2.730 m. Its score is 0.812
yet its alternative margin is only 0.00262; the saved interval spans 4.455–20 m.
The matcher correctly abstains on that ambiguous score rather than recovering
the target. In clip 05, a line has 14 target samples inside its 1.878–20 m interval,
but that extremely broad interval and margin 0.00378 provide no near-range claim.

Thus this frozen line/profile representation neither resolves the relevant
correspondence ambiguity nor meets its CPU allocation. Preserving a visible
contour alone did not supply reliable foreground distance here. This result does
not test foreground/background echo assignment: the reused source has no ToF.

## Measured cost and execution record

| CPU host measurement, 12 windows | Median | P95 | Maximum |
| --- | ---: | ---: | ---: |
| RGB decode and input verification | 19.651 ms | 21.029 ms | 21.083 ms |
| Algorithm wall time | 527.056 ms | 554.568 ms | 555.831 ms |
| Decode plus algorithm | 547.307 ms | 574.688 ms | 574.894 ms |

Algorithm times range from 448.178 to 555.831 ms. The complete sealed loop,
including result serialization, took 6.781 s. One logical CPU core and one
OpenCV/BLAS thread were used, with OpenCL off; peak RSS was 67.484 MiB. No GPU,
large model, training, new capture or online pose estimator was used. This is
a desktop CPU proxy, not a measured edge-device speed or power budget.

The first inference attempt failed while serializing a NumPy boolean after one
window. The exact failed files and logs remain in `attempt1/`. A scalar-only JSON
conversion repair preserved matcher code and every scientific setting, then
produced the twelve sealed windows. The actual work was **13 matcher calls**
(one unsealed failed call plus twelve sealed calls), not twelve total; the extra
mechanical retry is disclosed separately from the frozen twelve-window cohort.
No timing for that failed serialization is mixed into the twelve-window P95.

The first native evaluation failed because the original historical report lacks
`false_body_head_near`, which its later attribution report supplies. The original
evaluation code, seal and failure log remain in `evaluation-attempt1/`. The
mechanical repair compares every supplied old count and requires the enriched
field to match the attribution report; missing fields are never treated as zero.
Predictions and scientific scoring rules were unchanged. Both repairs have exact
before/after hash receipts. Five matcher tests and ten evaluator tests pass.
An independent recount passes for all 12 IDs, 1,407 lines, 23,919 points, hashes,
counts, intervals and all six gate checks. It recomputes from saved per-point
annotations and predictions without rerunning the matcher or native geometry;
this independently verifies aggregation, not source geometry certification.

The global registration command failed on the pre-existing
`experiments/index.jsonl:303` input-fingerprint mismatch. The inheritance command
then returned `unknown terminal id: ba-contour-parallax-20260919`. Both receipts
are retained. A local structured disposition records `NEGATIVE_CONTROL`; global
metadata remains pending and the unrelated registry was not edited or bypassed.

The source audit, complete line-level results, saved-target diagnostic and all
twelve endpoint overlays remain under the payload root below. Resource release
restored CPU affinity; no task-owned inference worker, model allocation or paid
compute remains. All small task artifacts support provenance, failure diagnosis
or reproduction and are retained; no files were deleted and zero bytes reclaimed.

The following method and gates were frozen before inference. The immutable
human-readable copy is `protocol-before-prediction.md`, and `protocol.json` has
SHA-256 `d9d427f5b76255b2dc938c8c3bed49001cfc5cb8742f3bfc6ce29fd1db3cfc89`.

The user authorized one test of short-window sparse contour ranging under limited
edge compute. Reuse all twelve old NF-G6 windows (36 RGB frames), their ideal
metric poses and calibration; replace only the proposal/matching mechanism. No
dense model, new weights, training, new capture, threshold sweep or actual-pose
estimator is introduced. [Old G6](STRUCTURE_INFORMATION_20260907.md) remains a
negative control, and [the edge-compute constraint](CAMERA_FORWARD_CONTRACT_20260919.md)
remains binding. This tests geometric information with privileged poses, not a
deployable RGB+ToF system or camera-corridor alert improvement.

## Hypothesis and fixed method

G6's isolated 9x9 patches missed all four translational 2–3 m bars; forward
cases had measurable parallax but ambiguous correspondences, and both far
translational bars had zero proposals. A spatially distributed straight contour
with endpoint evidence may improve proposal coverage and distinguish alternative
depths without a large model. It may still fail on appearance, aperture ambiguity,
occlusion, straight-segment extraction or its geometric approximation.

1. Detect native-resolution straight lines with OpenCV LSD standard refinement,
   scale 1.0. Keep lines at least 8 px long and within a 5 px image border.
   Rank by length, with deterministic endpoint ties, in an 8x4 midpoint grid;
   keep four per tile, at most 128. No orientation, target-mask or scene-ID filter.
2. Sample 17 evenly spaced points including both endpoints. Use 9 px normal
   profiles and 9x9 tangent/normal patches at both endpoints. One common optical
   depth per line is a local frontoparallel approximation, not arbitrary 3D-curve
   reconstruction. Retain all selected lines and all rejection reasons.
3. Project these supports into both earlier views using supplied poses. Test the
   same 96 inverse depths as G6, 0.05–2 per metre. For each view, take the smaller
   of mean profile NCC and the weaker endpoint NCC, then take the smaller view
   score. Both views and endpoints must support one interpretation.
4. Keep G6's score >=0.75, alternative-depth margin >=0.05 outside +/-20% inverse
   depth, score-within-0.03 interval and relative inverse-width <=50%. Add half
   a grid step at each interval end for quantization uncertainty; reject best
   hypotheses at the grid limits. At least 13/17 profiles must independently
   reach 0.75 in both views. Normal parallax or both-endpoint displacement must
   be >=1 px. A zero translation baseline always returns UNKNOWN distance.

No score is a calibrated probability. Lack of a line is not evidence of absence.
No ToF returns exist in the old G6 input manifest, so this mechanism test does not
invent them or claim to have implemented the proposed ToF motion-scale source.

## Fixed evaluation and decision

Predictions are sealed before native reference access. Use G6's endpoint native
optical depth and target-object bounds plus 5 mm, with nearest pixel defined as
floor(coordinate+0.5); no GT-nearest-neighbour reassignment. Invalid/out-of-image
native reference is UNKNOWN. Report candidate points and lines, all target/native
support, accepted/near/UNKNOWN, depth interval containment and false-near counts.

A line-level near claim requires its entire depth interval at all 17 rays to lie
within G6's heading-forward (0.08,3] m and floor-relative height [0.65,1.85) m.
This is the inherited G6 mechanism readout, not the newer constant-width camera
corridor. A recovered target additionally requires at least 13 supported points
on the intended target with native optical depth inside the saved interval.

Retain only as an ideal-pose mechanism candidate if all criteria hold:

- Recover at least one of the four original dense-missed translational 2–3 m
  clips (old clip_01/02/05/06), keeping their denominator fixed.
- Both 6 m translational bars and the far wall mark (clip_03/07/10) each have
  a target candidate, with zero target false-near predictions.
- Both pure-yaw clips (clip_08/09) have no accepted distance.
- Meet the declared CPU proxy budget below. Report all non-target false-near
  counts/rates and support denominators; different proposal populations do not
  establish a paired false-positive improvement over G6.

Any failure closes this exact one-pass recipe. A source or execution defect is
NOT_EVALUABLE, not an algorithm negative; only mechanical repairs preserving
scientific settings are allowed and must retain the failed attempt. No automatic
parameter rescue, runtime integration or successor. A pass remains insufficient
for obstacle-alert or device suitability claims.

## CPU proxy and provenance

Freeze one logical CPU core (affinity 0), one OpenCV/BLAS thread, no OpenCL/CUDA,
and at most twelve matcher calls. The development allocation is algorithm P95
<=200 ms/window and process peak resident memory <=256 MiB. This is an explicit
exploratory allocation on the host Intel Core Ultra 7 251HX, not a supplied target
hardware requirement or evidence for an MCU/phone. Include first-call cost and
separately report RGB decoding/input hashing, algorithm wall/CPU time and total
input-to-result time. Acquisition and pose estimation are absent, not free.

The source audit matches all 36 RGB and 36 native payloads to the existing
verification hashes and all twelve old model caches to their prediction receipt.
It preserves gaps: the old verification has no root hash bound by the listed old
outputs; 24 non-endpoint native hashes rely on that file; old per-view shader/
streaming readiness counters do not exist. Reuse is consumed-source Development,
not new independent confirmation or strengthened source certification.

Payload root: `artifacts.local/work/ba-contour-parallax-20260919/`. The public
`observations.json` contains RGB, calibration and disclosed ideal poses only.
The separate `source-audit.json` holds old evaluator-only baseline counts.
Protocol, input/code hashes, prediction seal, all rejected candidates, evaluation,
CPU receipts, tests and failures are retained. No old output is overwritten.

Implementation: [matcher](ba_contour_parallax.py),
[CPU runner](run_ba_contour_parallax.py), [synthetic tests](test_ba_contour_parallax.py),
[evaluator](evaluate_ba_contour_parallax.py),
[evaluation tests](test_ba_contour_parallax_evaluation.py).
