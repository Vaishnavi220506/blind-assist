# Zone-Handoff Ceiling: one of four gaps, stop aggregation

2026-09-20. COMPLETE. **The frozen spatial component ceiling bridges only1/4
gaps (1/5 frames). Do not implement component aggregation as the next method.**
The other four gap frames remain below the unchanged threshold even when every
positive possible zone is summed without any adjacency or compatibility limit.
Keep Calibration144/146/0 as the branch baseline; RGB/noRGB remain closed controls.

The [protocol](ZONE_HANDOFF_CEILING_20260920.md) froze one four-neighbor,
positive-depth-overlap, corridor-shared-edge graph before feature computation.
All432 frames are consumed controlled simulator evidence. Baseline source hashes,
22,371 original zone scores/intervals and432 decisions replay exactly. No overlap
formula, depth envelope, threshold, baseline output or model changed.

## Gap-level ceiling

The inclusive threshold remains **0.007085703945147101**. Four gaps contain five
frames; the two-frame gap must be completely filled to count as reconnected.
All are negative frames, so filling one means adding a false alert.

| Gap frame | s1 | s2 | Component sum = all-zone sum | Connected pair? | Threshold reached? |
| --- | ---: | ---: | ---: | --- | --- |
| f0376 | .002123864 | .000191603 | .002315466 | Yes,42/50 | No |
| f0389 | .006619121 | .003457474 | .010076595 | Yes,42/50 | Yes |
| f0267 | .001390056 | .000038919 | .001428975 | Yes,37/29 | No |
| f0268 | .002078609 | 0 | .002078609 | Only zone29 | No |
| f0304 | .000975872 | 0 | .000975872 | Only zone26 | No |

f0267/f0268 form one gap. In three gap frames a second positive zone exists and
passes all three connection conditions; only f0389 has enough summed score.
Two gap frames have no second positive possible zone at all. The failed frames'
fully unconstrained sums reach only13.8%-32.7% of the frozen threshold. Loosening
the graph cannot bridge them under the requested unchanged positive-score sum.

This rejects the proposed max-versus-sum explanation as a sufficient account of
the four fragmenting gaps. It does not prove that zone handoff never matters:
one gap is algebraically bridgeable. Nor does that single case prove two returns
belong to one physical surface; connectivity remains observable compatibility.

## Full432 diagnostic

Counts below use each entire row population. Depth-compatible counts refer to
two available top zones; no missing second zone is silently counted compatible.
Gap5 is a subset of the142 negative-withheld rows, not an extra denominator.

| Population | Frames | Top2 available | Adjacent top2 | Depth-compatible top2 | Corridor-continuous top2 | Same component top2 | Component >= threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 432 | 260 | 219 | 260 | 217 | 252 | 294 |
| True positive frames | 144 | 134 | 117 | 134 | 117 | 132 | 144 |
| Calibration FP frames | 146 | 113 | 91 | 113 | 89 | 108 | 146 |
| Negative-withheld UNKNOWN | 142 | 13 | 11 | 13 | 11 | 12 | 4 |
| Target gap frames | 5 | 3 | 3 | 3 | 3 | 3 | 1 |

There are no certified TN: all288 negative-frame predictions are UNKNOWN, and
all391 original UNKNOWN flags remain unchanged. Indirect graph paths explain why
top2 can share a component without being adjacent. Top2 sum and unconstrained
all-zone sum happen to give the same threshold counts as component sum in every
listed population; this is a cohort observation, not a mathematical equivalence.

| Metric | Frozen Calibration | Hypothetical component sum |
| --- | ---: | ---: |
| TP / FP / FN | 144 / 146 / 0 | 144 / 150 / 0 |
| Precision | 49.66% | 48.98% |
| Recall | 100% | 100% |
| F1 | 66.36% | 65.75% |
| FPR over288 negatives | 50.69% | 52.08% |
| Events / first in-event alert | 24/24 / baseline | 24/24 / identical |
| FP segments | 40 | 39 |
| Sampled FP duration | 29.2s | 30.0s |
| OUTSIDE-layout FP /144 | 87 | 90 |
| OUTSIDE-layout FP segments | 13 | 12 |
| BODY TP/FP/FN /216 | 72/78/0 | 72/82/0 |
| HEAD TP/FP/FN /216 | 72/68/0 | 72/68/0 |

New alerts are f0196, f0208, f0389 and f0424. Only f0389 bridges a target gap;
the other three are collateral FP, each at0.8s in body_large_solid arrangements
(one BOUNDARY, two OUTSIDE). They extend existing FP segments without creating
an isolated new segment. The recovered gap is the OUTSIDE suspended-solid clip
at1.0s. Clip-first alerts are distinct from first in-event timing.

The frozen gate passes TP/onset preservation, <=5 added FP and no new isolated
FP segments. It fails >=3 complete gaps and <=37 total segments. Therefore the
conditional implementation authorization is not triggered; no runtime component
challenger, temporal layer, new cutoff, floor or connectivity variant follows.

## Interpretation, validation and disposition

Nonnegative component sums are >= the original per-zone maximum. At an unchanged
threshold they can only add alerts: they cannot reduce FP below146. TP/event
retention here is structurally expected, not new recognition evidence. Summed
zone-normalized scores are neither probabilities nor conserved physical area;
pairwise depth compatibility also does not establish one globally shared depth.
Pixel IoU is not defined for this scalar alert task.

The proposed paper wording about correcting systematic spatial bias is not
supported by this experiment: the retained Calibration selects an operating
threshold on a full-footprint geometric score; it does not fit sensor extrinsics.
The new result supports neither a physical uncertainty model nor a general
surface-handoff mechanism. Keep these conclusions scoped to nominal simulated
zones, this fixed sum and this consumed cohort.

Six pre-freeze synthetic graph checks pass. The observable feature pass reproduces
the original readout from each saved observation and seals graph features before
joining labels/gap ids. It takes1.360s including hashes, IO, baseline replay and
small graphs on the existing project Python runtime; no GPU kernels are used
(`TASK_NOT_GPU_SUITABLE`), and this is not device latency. Independent code review
and scalar recount cover the graph, FP/event timing, gap completeness and seals.

Protocol SHA256:
`fd9e3ab4e9b0431fc1263e2d0064cdc85cc9a05de49879fc1f8ec98e6afd92f8`.
Feature SHA256:
`9fc004697929cf319b4508888500fafb3b7edd303f1d819f3f5879a1f17d3489`.

Disposition: exact spatial component-sum explanation is `NEGATIVE_CONTROL` for
the requested >=3/4 low-cost gap-bridging role; Calibration remains `RETAINED_CORE`.
Global registration still fails at existing ledger303; terminal inheritance also
cannot be claimed complete. Actual failure logs and local structured disposition
are retained without editing or bypassing the global ledger.

Durable payload: `artifacts.local/work/ba-zone-handoff-ceiling-20260920/` contains
frozen protocol/code identities, feature/result seals, every graph component,
gap table, complete frame metrics, validation and registration receipts. Existing
observations and baseline predictions remain immutable. No persistent processes,
new models, datasets or paid allocations remain; no disposable payload was made.

Implementation: [observable graph](zone_handoff_ceiling.py),
[staged diagnostic runner](run_zone_handoff_ceiling.py),
[synthetic checks](test_zone_handoff_ceiling.py).
