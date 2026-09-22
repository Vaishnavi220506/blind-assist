# Score-Factor Collapse Audit: overlap-sensitive winner, mixed source transition

2026-09-20. COMPLETE. No new algorithm, changed output or successor.

**The requested dominant-zone counterfactual is overlap-only sensitive in4/5
gap frames; either factor replacement suffices in1/5. No frame is depth-only.**
This is not proof of faulty geometry. Fixed-zone tracks show that the preceding
high-overlap winner has moved to an entirely far support interval (d=0) in those
same four frames. A different zone with nonzero depth fraction and very small
overlap becomes dominant. The two views describe different parts of one source
transition; neither establishes a physical dropout cause.

Keep Calibration144/146/0,24/24 events and original onsets. Zone-Handoff and
RGB/noRGB remain closed in their tested roles. All five diagnostic gap frames,
and their eight flanking frames, are strict negatives. Their suppressed alerts
are not missed true obstacles or evidence that the low factor is wrong.

The [frozen audit protocol](SCORE_FACTOR_COLLAPSE_PROTOCOL_20260920.md) uses exactly
four gap windows /13 frames, with five gap frames. Both frames of the two-frame
gap use the same enclosing alert references. Factors and baseline outputs come
from the sealed original432-frame predictions. Threshold T=.007085703945147101.

## Requested before / gap / after expansion

Here d is the original support interval's fraction inside the0.3..3m depth slab;
o is the depth-conditional full-footprint overlap. Neither is return confidence.

| Frame | Nominal time | Role | Dominant zone | d | o | s=d*o | s/T |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| f0375 | .6s | Before | 43 | .257724 | .994886 | .256406 | 36.1864 |
| f0376 | .8s | Gap | 42 | .248819 | .008536 | .002124 | .2997 |
| f0377 | 1.0s | After | 51 | .363984 | .984984 | .358519 | 50.5975 |
| f0388 | .8s | Before | 50 | .381279 | .021154 | .008066 | 1.1383 |
| f0389 | 1.0s | Gap | 42 | .355706 | .018608 | .006619 | .9342 |
| f0390 | 1.2s | After | 42 | .634929 | .047633 | .030244 | 4.2683 |
| f0266 | .4s | Before | 36 | .035032 | .980170 | .034338 | 4.8461 |
| f0267 | .6s | Gap1 | 37 | .222109 | .006258 | .001390 | .1962 |
| f0268 | .8s | Gap2 | 29 | .278212 | .007471 | .002079 | .2934 |
| f0269 | 1.0s | After | 37 | .536946 | .037216 | .019983 | 2.8202 |
| f0303 | .6s | Before | 35 | .177168 | .992635 | .175863 | 24.8194 |
| f0304 | .8s | Gap | 26 | .225391 | .004330 | .000976 | .1377 |
| f0305 | 1.0s | After | 34 | .577975 | .041557 | .024019 | 3.3898 |

The earlier14%-33% upper-bound statement concerned unrestricted all-zone sums
for the four failed bridging frames. This table instead reports dominant s/T;
the denominators must not be conflated. f0389 was the only bridgeable gap.

## Two factor substitutions, with their directions explicit

For two flanks the median is their arithmetic mean. Define d_ref and o_ref from
the respective dominant factors, even though their zone ids differ. The product
d_ref*o_ref is not necessarily either original flank score or median flank score.

- User `CF_depth = d_gap * o_ref` keeps depth and **replaces overlap**.
- User `CF_overlap = d_ref * o_gap` keeps overlap and **replaces depth**.

These are posthoc arithmetic substitutions, using a future flank, not proposed
causal predictors. A value >=T is only a score crossing, not a valid new alert.

| Gap | d_gap/d_ref | o_gap/o_ref | CF_depth: replace overlap | CF_overlap: replace depth | Threshold sensitivity |
| --- | ---: | ---: | ---: | ---: | --- |
| f0376 | 80.04% | .862% | .246315 >=T | .002653 <T | Overlap replacement only |
| f0389 | 70.01% | 54.10% | .012234 >=T | .009455 >=T | Either replacement suffices |
| f0267 | 77.66% | 1.230% | .112985 >=T | .001790 <T | Overlap replacement only |
| f0268 | 97.28% | 1.469% | .141525 >=T | .002137 <T | Overlap replacement only |
| f0304 | 59.69% | .837% | .116549 >=T | .001635 <T | Overlap replacement only |

Counts: overlap-only4; depth-only0; either1; both-replacements-required0. Both
factors are lower than their separate flank medians in all five cases, so a bare
'both decreased' definition would label everything joint collapse and conceal
the threshold sensitivity. The exact relative and signed-log contributions are
retained instead of selecting an arbitrary collapse percentage cutoff.

## Fixed-zone comparison prevents a false causal label

Tracking every zone dominant anywhere in each window gives11 zone tracks and36
frame/zone slots (two are missing valid returns). Three different patterns matter:

1. **Outgoing winners lose depth-slab support.** Zone43 at f0376 has interval
   [7.331,8.543]m; zone36 at f0267/f0268 has [6.896,8.053]/[7.249,8.451]m;
   zone35 at f0304 has [7.350,8.565]m. All have d=0 and o undefined. These are
   valid far-return intervals, not missing packets or an overlap-only collapse
   of the same zone. They remain far at the following alert flank.
2. **Incoming winners retain partial depth support but little overlap.** The gap
   winners have d=.2221..3557 and o=.00433..01861. For example zone42 in the
   first window evolves d=.0879->.2488->.5821 and o=0->.00854->.04199: this
   zone is gaining score, not itself abruptly losing overlap. The apparent
   dominant-overlap drop comes from switching away from high-overlap zone43.
3. **Some references are unavailable or below threshold.** Fixed-gap-zone CFs
   are not evaluable for f0267,f0268,f0304 because a flank has d=0/o undefined
   or the return is missing. For f0376/zone42, neither replacement crosses T;
   even d_ref*o_ref=.007033949 is just below T. f0389/zone42 remains ambiguous:
   overlap replacement .009231934 and depth replacement .007740598 both cross T.

Thus the same-zone result is: three NOT_EVALUABLE, one reference-not-explanatory,
one either-factor-sufficient. Missing/undefined factors were never replaced with
zero or carried from a neighbor. Original possible/definite flags and every
support interval remain in the full JSON tracks.

## Decision and evidence boundary

The audit does **not** support '4/5 current dominant depth fractions collapse'.
It also does not support '4/5 geometry calculations are wrong': cross-zone
substitution puts another zone's high overlap onto the current low-overlap zone.
d and o are coupled functions of the same depth interval and footprint, not
independent experts. No Product-of-Experts or probability interpretation is
established by this geometric factorization.

The defensible finding is a transition from high-overlap zone support to far
returns while lower-overlap zones supply the remaining partial depth support.
Whether the outgoing near return was lost because of projected object motion,
mixed-surface winning-bin selection, dropout or another proxy effect is not
determined by the factors alone. We have not accessed native ownership/return
lineage in this audit. No physical near-surface failure can be inferred from
these deliberately selected negative frames.

If a next diagnostic is requested, the single unresolved question is the return
lineage of the outgoing high-overlap zones: why their near interval became far.
That question comes before choosing a range-quality or geometry algorithm. It
is recorded here as a question only; no successor or additional analysis ran.

## Validation and delivery

Four synthetic attribution checks pass, covering substitution direction,
inclusive-threshold ambiguity, jointly-required substitutions, reference failure
and missing factors. Source code hashes and original observation/prediction/
evaluation seals pass before and after; all original432 predictions remain
unchanged. Independent scalar review checks the13-frame expansion, five CFs,
fixed-zone status and preserved source hashes without production attribution
imports. The audit uses standard-library CPU arithmetic,0.185s including JSON
and hash checks; this is not device latency. No GPU, model or new data is used.

Protocol SHA256:
`13db4bfae0418c7a80bd3b674359ac616cc7e5f16878237f0177a81b5594ae6e`.
Durable payload: `artifacts.local/work/ba-score-factor-collapse-20260920/` includes
frozen protocol, full factor tracks, all CF values, source/code hashes, result
seal, validation and registry receipts. No persistent task resources or disposable
datasets/models were created.

The dominant-depth-only explanation is a scoped `NEGATIVE_CONTROL`; retain this
attribution evidence and the Calibration baseline. Global registration/inheritance
remains pending the existing ledger303/unknown-terminal blockers, with actual
failure receipts retained and no ledger bypass.

Implementation: [audit](audit_score_factor_collapse.py),
[synthetic attribution checks](test_score_factor_collapse.py).
