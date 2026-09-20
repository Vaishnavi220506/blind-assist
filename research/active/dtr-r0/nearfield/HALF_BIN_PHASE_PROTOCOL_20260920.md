# One paired half-bin phase sensitivity diagnostic

2026-09-20 EXPLORE. User authorized one global half-bin shift after the onset
audit located background selection at f0005. Question: how sensitive are the
original single-return proxy and frozen readouts to an arbitrary histogram origin?
This is consumed simulator sensitivity, not a proposed hardware sensor or algorithm.

Use only complete ba-core-workpoint-transfer-20260920:36clips/432frames,
Core78P/210N and Boundary78P/66N. Same native arrays,64 boxes, eligible domain
[0.1,8)m, >=4 hits, inverse-square weights, first-argmax tie break, mean,5% dropout,
noise sigma=0.01+0.02*mean, and output validity range. Baseline phase0 must reproduce
every stored float32 range (including missingness), contributor and readout exactly.

Only shifted bin membership differs. Baseline executes original
`minimum((hits / .1).astype(int),79)`. Shifted executes
`floor((hits - float32(.05)) / float32(.1)).astype(int)`, asserted0..79, with
no clipping/merging. Inputs and arithmetic remain float32 as in the proxy.
Shifted boundaries are0.05+0.1*k; the fixed eligibility domain truncates only first
and last bins ([.1,.15),[7.95,8)). Interior widths remain0.1m. Report winning edge
bins separately. No offset grid, alternate shift, target-conditioned merge or
best-phase selection. Candidate bin IDs themselves are not comparable identities;
compare actual pixel contributor sets, centroids and output ranges.

Pair both phases by identical original identity seed and original conditional
random-call order. Eligible counts do not change, so dropout draws and normal
draw sites remain identical. Assert equal dropout draws and equal standardized
Gaussian noise within1e-14. Sigma remains center-dependent: a changed selected
mean changes noise amplitude under the frozen sensor rule. Report this as paired
full-proxy sensitivity, not a noise-free binning effect. No new RNG seeds.

## Stages and accounting

Freeze code, source hashes and protocol before construction. Construct both source
vectors from native arrays without truth masks; seal the arrays and private pairing
receipts. Predictor receives only64 values/boxes and clip/time identity; no native,
bin-owner, future frame or evaluator labels. Seal predictions before label/owner join.
Preserve both frozen readouts: Calibration T0=.007085703945147101 and strong
T=.4071309640537889, with mandatory definite bypass. Also report the existing
nonrecursive one-frame genuine-strong hold unchanged under each phase. UNKNOWN is
computed from each phase's current observation, never forced into clear space.

For all27648 zone-frames report candidate contributor-set changes (before output
validity), observed-set changes, same/dropout status, range differences and >1m
range changes; separate first/last winning bins. Do not count renumbered bins as
changed contributors. Native source ownership is evaluator-only, with existing
±2cm bounds proxy; report target/background winner changes and actual corridor
contributors gained/lost, not only the six previously examined frames.

For each fixed readout and phase report complete/Core/Boundary counts, precision,
recall,FPR, FP segments/time, INSIDE negative/OUTSIDE, BODY/HEAD/background groups,
all changed positive/negative IDs,12event first times/coverage/gaps including tail,
clip-first times and preexisting alerts. Retain original source denominators and
geometry labels (including f0293's1mm entry). Explicitly inspect f0005/z35 and the
two onset triples; no target-only success conclusion.

No arbitrary pass threshold for being 'over-sensitive': report measured changes
and counterexamples, distinguishing any local repair from full-cohort stability.
Either outcome ends this diagnostic. Do not replace the proxy/baseline, change the
task contract, claim real-sensor behavior, tune thresholds, shift again or start a
successor. Keep original results and consumed authority intact.

CPU array/scalar work is TASK_NOT_GPU_SUITABLE; retained native data are local,
so avoid worker source transfer for this bounded replay. No new capture, training,
GPU job or persistent processes. Evidence under
artifacts.local/work/ba-half-bin-phase-20260920/.
