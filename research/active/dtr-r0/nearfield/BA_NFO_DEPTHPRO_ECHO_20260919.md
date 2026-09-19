# Frozen Depth Pro + existing ToF: global scale helps; this layer recipe fails

2026-09-19 · EXPLORE · consumed synthetic Development · completed one round.

Existing single-return ToF contains useful correction information: uniform
scale markedly reduces Depth Pro false positives and improves its recall.
The fixed spatial assignment recipe does **not** beat that simple control and
neither candidate passes the original joint task. Retain original NFO as the
task baseline; retain global scale as a limited diagnostic component, and stop
this exact layer recipe. These results revise the previous priority suggestion:
RGB-only failure is insufficient reason to conclude that new range hardware is
the first requirement. No next experiment, training or App change was started.

## Frozen comparison

Reused all original 500 frames in the same order: 52 scenes, six families, the
sealed native Depth Pro `192×256` point-ray cache, and the original 64 public
zone boxes / single optical-z returns. Depth Pro, NFO and all thresholds remain
unchanged. This adds existing ToF to the failed standalone reference; it does
not reopen that RGB-only replacement claim.

The two new arms share **identical echo pairs** and differ only in how far the
scale correction propagates:

1. Form four-connected regions by increasing adjacent log-depth difference.
   Merge only if the **entire merged region** has max/min ≤ 1.25. This prevents
   an arbitrarily long smooth-depth chain joining foreground and background.
   Stable row-major ties; no size pruning; regions may cross ToF zones.
2. Within each valid zone, use the existing proxy's strongest 10 cm depth bin
   under `1/max(z,0.3)^2` energy, over predicted `0.1≤z<8 m`. Select the region
   contributing most energy to that bin; take the ordinary mean of its pixels
   in that bin and zone. This is predicted association, not measured identity.
3. **Global:** multiply all pixels by the exponential median of the paired
   log return/prediction ratios. **Layered:** apply that same estimator within
   each region. Unanchored regions retain their original Depth Pro values and
   receive an explicit no-ToF-support flag. No repeated pairing or clipping.

The 1.25 boundary ratio was inherited from the prior contour diagnostic;
the 10 cm / energy / range constants come from the fixed existing ToF proxy.
No scientific parameters were chosen from this run's outcomes. Before freezing,
synthetic review caught that whole-region energy is not the proxy's strongest
bin; implementation and tests were corrected before any cohort inference.

Inputs-only `observations.json` contains cached prediction paths/hashes and
public boxes/values. The inference executable has no native truth, NFO masks,
evaluation domain or scene-family access. Predictions were sealed before the
separate evaluator opened truth. Full protocol and per-zone pairs are retained.

## Same original task, physical optical z < 2 m

| Method | Far-small recall | Far-small IoU | Mixed recall | Pure-far FP pixels | Original gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original NFO | 68.36% | 11.13% | 95.00% | 212,988 | reference |
| Frozen native Depth Pro | 61.29% | 23.93% | 81.64% | 344,336 | 1/4 |
| Depth Pro + global scale | **67.84%** | **42.62%** | 88.08% | **19,236** | 2/4 |
| Depth Pro + layered scale | 58.66% | 38.81% | **90.22%** | 21,845 | 2/4 |

Original requirements remain: far-small recall ≥75%, far-small IoU ≥NFO,
mixed recall ≥94.5%, pure-far FP ≤NFO. Both new methods fail the two recall
requirements. Global has 99 fewer far-small TP and 43,755 fewer mixed TP than
NFO, despite its much lower FP. There is no replacement or alert promotion.

“Far-small” retains the old evaluation definition: a zone with a far public
return and at most 20% true near pixels, containing both near and far pixels.
It is not an independently labeled thin-object/rod instance set; examples also
include floor and wall threshold crossings. FP numbers count pixels, not events.

| Additional fixed diagnostic | NFO | Native | Global | Layered |
| --- | ---: | ---: | ---: | ---: |
| Full-image IoU | 61.62% | 70.91% | **87.22%** | 85.66% |
| Outside-ToF FP pixels | 1,003,881 | 397,424 | **145,635** | 205,537 |
| Near-mask boundary F1, tolerance 1 px | 4.61% | 44.43% | **54.44%** | 50.97% |
| Directed ratio-edge F1, all 500 | not defined | **53.20%** | **53.20%** | 46.67% |

Global scale preserves the original relative-depth edges. Layer corrections
change some boundaries and relative ordering; higher near-mask F1 versus bare
Depth Pro does not establish preserved scale-independent structure.

Against global, layered loses 3,838 far-small TP and rescues 2,074 FN: **1,764
net additional misses**. It improves mixed recall by 2.13 percentage points,
but lowers far-small recall by 9.18 points and adds 2,609 pure-far FP. It fails
the predeclared all-four-quantities dominance check. Far-small recall improves
over global in only two of six families; all family counts are retained.

## What the attribution can establish

Of 25,552 valid saved returns, 25,362 received a predicted in-range pair;
190 were unused because no sufficient in-range predicted support existed.
There are 172,549 regions across the 500 images, of which 7,429 receive at
least one return. They cover 14,672,207 of 24,576,000 prediction pixels;
the other 9,903,793 are retained exactly from Depth Pro.

For far-small true near pixels, 14,243/19,213 lie in a region assigned at least
one return. Of the layered method's 7,942 far-small misses, **6,094 occur in
assigned regions**, versus 1,848 in unassigned regions. Thus the errors cannot
all be described as an absence of assigned measurements. Assignment does not
mean correct source identity or a genuine near-surface anchor: a region can
contain pixels on both sides of 2 m, receive a background return, or have a
poor representative/scale. No GT-driven reassignment or oracle correction ran.

The frozen evaluator-only distance-consistency proxy found compatible true
pixels somewhere in 25,360 paired zones, and inside the selected region in
24,529. In 831 zones they existed only elsewhere; two zones had none under the
proxy. Of 3,930 observed returns below 2 m, 116 were assigned to a zone-region
intersection with no true near pixels. Compatibility uses
`abs(z-return) ≤ 3*(0.01+0.02*return)+0.1 m`; it is a loose distance test, **not
proof of which surface generated a return**, and does not validate all other
assignments. It is diagnostic after sealed inference, never an input.

The useful next decision is to keep global scale as the comparison that a
future mechanism must beat. This experiment supplies no justification for
tuning this region recipe on the consumed cohort or automatically buying a
different sensor. A future authorized study would need to separate region
merging/splitting, incorrect attribution and genuinely unmeasured foreground.

## Separate old G5 suspended-bar diagnostic

**NOT_EVALUABLE_NO_FROZEN_TOF.** Read-only inventory of the 18-view source found
RGB, dense native depth and poses, but no saved ToF boxes/values/packets. Dense
native depth remains evaluator-only; generating a new ToF simulation from it
would be a different observation-construction experiment. No such construction
was performed. The previous RGB-only bar result remains 0/1,054 pixels below
2 m with native Depth Pro median 2.875 m versus 1.947 m reference. It is neither
a new fusion failure nor pooled into the 500-frame metrics.

## Validation, resources and evidence

- Four synthetic tests pass: background-only echo cannot move an unanchored
  foreground; cross-zone continuity; one-region equivalence / missing identity;
  bounded chains / singleton retention; strongest-bin selection / no candidates
  are covered within those four tests.
- Independent saved-array verification passes on all **500 × 4 arms × 11
  domains**, using a separate `bincount` confusion calculation. Original NFO
  and native Depth Pro counts reproduce exactly; **350,140 UNKNOWN** pixels
  remain excluded and counted, never converted to far.
- Every output and input digest is checked; global multiplication, per-region
  multiplication and all 9,903,793 unsupported values verify exactly. Saved
  region range bounds and 25,362 anchor/support counts verify independently.
- CPU `TASK_NOT_GPU_SUITABLE`: sequential ordered graph union, 64-zone scalar
  reductions and saved-array audit; no model inference. Calibration/regions:
  **7.606 s** total, **24.453 s** inference wall including compressed I/O,
  **0.632 s** separate Numba warmup; evaluation **21.810 s**. These are desktop
  postprocessing timings, not end-to-end or phone performance.
- No model calls, training updates, original-test reads, parameter search,
  second-return access, background worker or paid allocation.

Durable payload root: `artifacts.local/work/ba-nfo-depthpro-echo-20260919/`.
Keep `protocol.json`, `observations.json`, `manifest.json`, all predictions and
per-zone receipts, `prediction-seal.json`, `results.json`, `frame-results.json`,
`verification.json`, disposition, source audit, execution logs and delivery
receipt. `paired-extremes.png` shows two largest losses and two largest gains
of layered far-small TP versus global; selection/crops are explicitly posthoc
and recorded in `visual-selection.json`. It does not replace all-frame results.

Implementation: [inference](ba_nfo_depthpro_echo.py),
[evaluator](evaluate_ba_nfo_depthpro_echo.py),
[independent verifier](verify_ba_nfo_depthpro_echo.py),
[synthetic tests](test_ba_nfo_depthpro_echo.py),
[renderer](render_ba_nfo_depthpro_echo.py).

Local decision: `NFO_DEPTHPRO_EXISTING_TOF_SCALE_GAIN_LAYER_RECIPE_REJECTED`.
Original NFO is retained; global is a `COMPONENT_OR_CHALLENGER` for descriptive
calibration comparison only, and this layered replacement recipe is a
`NEGATIVE_CONTROL`. The knowledge registration command again fails on the
pre-existing ledger line 303 input-fingerprint mismatch. Global terminal
registration/inheritance remains pending; structured local `disposition.json`
and actual command-failure receipts preserve the scoped decision without
rewriting or bypassing that unrelated ledger. This metadata gap does not alter
the verified experimental result.

All data are consumed synthetic Development, not fresh generalization, actual
ToF hardware, obstacle events, deployment or safety evidence. Predicted regions
are bounded depth patches, not guaranteed semantic surfaces; the single-return
histogram is the existing simplified sensor proxy, not a hardware measurement.
