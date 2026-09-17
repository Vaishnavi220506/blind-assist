# Public positive readout: useful calibrated Development gain

**The unchanged 1,537-parameter public ToF head, with pooled scene-held-out
calibration, recovers the eight changed-domain clear misses without adding any
false alert anywhere in the four reporting cohorts.** Changed clear improves
95/27/13 to **103/27/5, F1 82.61% to 86.55%**. No A alert or onset is lost.
Old/MZ146/MZ158 alerts are exactly A. This is a real public-input result under
grouped consumed Development, not a new independent confirmation, single
deployment model, or proof of broad transfer.

The matched negative-frame peak-loss arm also recovers all eight changed clear
misses and one old clear miss, but adds three boundary false frames. Retain it
as a tradeoff; prefer ordinary return BCE plus pooled calibration as this round's
simpler Development challenger. This preference is made after comparing the
reported arms; it is not an independently validated final-method selection.

## Existing data and unchanged component

The [prospective Development brief](PUBLIC_POSITIVE_V2_PROTOCOL_20260917.md)
keeps the public 30->32->16->1 head, return-level native supervision, maximum
valid-return aggregation, frozen A and positive-only OR. No graph, RGB pixel,
DA-V2, native coordinate, actor identity, scene index or label enters inference.
Zero usable returns contribute no positive evidence and preserve A. The new
loss uses evaluator labels only on fitting frames, never as inference inputs.

Reuse the prior 768-frame cache and append complete MZ146288 and MZ158288:
**1,344 frames, 112 paired scene groups**. All are consumed Development. The
1,152 report frames comprise old288, changed288, MZ146288 and MZ158288;
anchor192 is fit-only. Original MZ136 test48 and dev48 remain excluded.
Capture receipts, raw/evaluator hashes, A feature caches and A scores were
authenticated. Recomputed A scores for both additional cohorts match bitwise.
Whole scene geometry/camera signatures have no exact duplicate. Repeated public
tokens occur mostly inside one scene; the only cross-scene token duplicate
contains five all-invalid ToF/Radar frames. No claim of independent generators.

The added cohorts have no clear A misses with sampled corridor witnesses.
They broaden backgrounds/negative coverage, not positive repair opportunities.
Across report cohorts, those opportunities are old1 + changed8 + MZ1460 + MZ1580,
concentrated in just three scene groups. The changed eight frames represent two
episodes, not eight independent successful configurations.

Six outer folds jointly hold scene index k across all four cohorts: 192 report
frames each. The other 960 frames form development, plus anchor192 for fitting.
Three inner folds over the five development scene indices provide one truly
out-of-fit score per development frame. A common threshold maximizes pooled
clear final-OR F1, then fewer FP, then higher threshold. Final refitting uses
all 1,152 fitting frames; its held outer scores are never used for selection.
The inner-to-refit score transfer is evaluated rather than assumed reliable.
Fixed logit0 is separately prespecified this round, not relabelled old evidence.

## Main clear-task results

Each cohort keeps 216/288 frames (75% coverage), 108 positives and108 negatives.
Entries show TP / FP / FN and F1. BCE and peak use identical architecture,
data, seed, order, optimizer, epochs and calibration rule; only the loss differs.

| Cohort | A | BCE + pooled calibration | Peak loss + pooled calibration |
| --- | --- | --- | --- |
| Changed | 95/27/13; 82.61% | **103/27/5; 86.55%** | 103/27/5; 86.55% |
| Old | 107/22/1; 90.30% | 107/22/1; 90.30% | **108/22/0; 90.76%** |
| MZ146 | 106/27/2; 87.97% | 106/27/2; 87.97% | 106/27/2; 87.97% |
| MZ158 | 107/21/1; 90.68% | 107/21/1; 90.68% | 107/21/1; 90.68% |

Changed BCE precision rises 77.87% to79.23%; recall 87.96% to95.37%.
HEAD becomes36/8/0, rod31/13/5, BODY remains36/6/0. These changed clear counts
equal the existing sampled-positive OR reference; this does not turn the public
head into a geometric oracle or imply its return estimates are universally exact.
The remaining five changed clear misses lack the corresponding usable ToF
measurement. The OR component cannot remove A's27 clear FP; negative alert burden
therefore remains25%. No false-positive suppression claim is made.

## Why pooled calibration worked here

The crucial outer0 calibration actually contains **one** supported clear A miss:
`mz170_suspended_head_scene3_in_04`. Its inner out-of-fit logit is
8.038854598999023. Selecting that threshold changes pooled calibration from
352/90/8 to353/90/7, with F1 .8778054863 to .8792029888 and no new FP.
The inactive candidate13.665743827819826 is not selected. The final refit assigns
the eight changed scene0 misses logits8.517..10.664, all above the selected
threshold. This is a threshold learned from another group's actual repair
opportunity, not leakage from outer0 or a finite inactive-threshold overflow.

| Outer report index | Pooled calibration clear A FN | Supported FN | BCE tau |
| --- | ---: | ---: | ---: |
| 0 | 8 | 1 | 8.03885 |
| 1 | 15 | 9 | 6.68049 |
| 2 | 16 | 9 | 7.27415 |
| 3 | 16 | 8 | 6.38422 |
| 4 | 15 | 9 | 5.97790 |
| 5 | 15 | 9 | 6.34092 |

Correction to the sealed brief's sentence about outer0 lacking independent
supported misses: the added MZ146/MZ158 cohorts have none, but old HEAD scene3
has one. The actual data, split code and counts always retained it. The brief
is preserved unchanged; this reporting correction changes no fit or decision.

The preceding run differs in data coverage, partitioning, fitting size and seed.
Thus this round establishes a useful combined Development recipe; it does not
isolate how much of the previous-to-current gain comes from calibration alone.
Its calibration opportunity still hinges on very few physical configurations.

## Fixed zero and negative peak loss

The second arm adds weight1 times mean softplus(max valid logit) on fitting
frames that are A-silent, clear-negative and have a usable return. Anchors,
positive/boundary frames and zero-return frames are excluded from this term.
It keeps the original class-balanced per-return loss; a single sparse positive
return can still trigger. There is no positive frame-label propagation or vote
count requirement, and no loss-weight search.

| Cohort | BCE fixed0 clear | Peak fixed0 clear | BCE fixed0 boundary | Peak fixed0 boundary |
| --- | --- | --- | --- | --- |
| Changed | 103/33/5; 84.43% | 103/32/5; 84.77% | 33/32/3 | 31/30/5 |
| Old | 108/22/0; 90.76% | 108/22/0; 90.76% | 34/23/2 | 34/22/2 |
| MZ146 | 106/27/2; 87.97% | 106/27/2; 87.97% | 31/22/5 | 31/16/5 |
| MZ158 | 107/25/1; 89.17% | 107/21/1; 90.68% | 29/23/7 | 28/18/8 |

At fixed0, peak loss removes5 added clear FP across cohorts without losing clear
TP relative to BCE, and removes14 boundary FP while also losing3 boundary TP.
It helps some erroneous peaks, but does not eliminate spatial confusion. On
changed clear, BCE adds6 rod FP and peak adds5; both false activations remain
concentrated in two rod/far-wall negative episodes. Class-balanced training
means sigmoid0.5 is not an empirically calibrated obstacle probability.

After each arm's own prescribed calibration, the loss tradeoff changes: peak
adds one old clear TP relative to BCE, plus one changed boundary FP and two
MZ146 boundary FP. This is not uniformly superior, and the extra loss is not
needed for the eight-frame changed-domain gain.

## Strict, boundary and event costs

| Cohort | A strict | BCE calibrated strict | Peak calibrated strict | Boundary FP A / BCE / peak |
| --- | --- | --- | --- | ---: |
| Changed | 120/45/24 | **128/45/16** | 128/46/16 | 18 / 18 / 19 |
| Old | 137/28/7 | 137/28/7 | 138/28/6 | 6 / 6 / 6 |
| MZ146 | 137/37/7 | 137/37/7 | 137/39/7 | 10 / 10 / 12 |
| MZ158 | 135/31/9 | 135/31/9 | 135/31/9 | 10 / 10 / 10 |

Changed strict BCE F1 improves77.67% to80.76%, with boundary25/18/11 unchanged.
Core detection remains18/18 in each cohort. Strict event counts remain old30/30,
changed29/30, MZ14629/30 and MZ15830/30. Do not report30/30 for changed strict.
Changed HEAD scene0 first observation alert improves1.25->0s; rod scene0
.75->.25s. No A alert is deleted/delayed. All episodes remain left-censored;
this is faster observed response, not demonstrated advance warning.

Fixed0 remains unsuitable as the preferred general readout: old strict BCE
is142/45/2 (F185.80%) versus A88.67%; peak142/44/2 (86.06%). Changed fixed0
strict BCE136/65/8 (78.84%) and peak134/62/10 (78.82%). Their boundary costs
remain substantial despite clear gains. Both working-point controls are shown.

## Verification, compute and retained evidence

Eleven focused tests pass: public token parity/invariance, missingness/OR,
threshold precision, nested group exclusion and negative-peak gradient routing.
Independent auditing reconstructs every inner and final checkpoint, fit-only
normalization, training orders, calibration candidates, report decisions and
event timing. All24 matched fit seeds/indices/orders are identical between arms.
For calibrated added true alerts, the maximum-scoring return itself has a sampled
corridor witness in8/8 BCE and9/9 peak cases; this is stronger than merely finding
some witness elsewhere in the same frame. Separate public raw-row inference replays1,152 frames per arm:
all scores match exactly, and calibrated/fixed0 decisions agree. No evaluator
information is needed by the inference entry.

Both arms use120epochs,24fits each, final checkpoint only; runtime211.01s BCE
and228.35s peak. CPU placement for the unchanged BCE reuses the prior measurement;
new peak-loss equivalent batch16 forward/backward measures CPU.678ms versus
CUDA1.062ms (`CPU_FASTER_MEASURED`). The public frontend+head replay on this host
measures BCE mean1.48/p501.39/p952.38ms, peak1.78/1.61/2.62ms. Architecture is
identical; these separate-run timings are not a loss-induced inference cost
claim. They exclude A/capture/transport and do not represent phone or full-chain
latency. No persistent worker, GPU allocation or capture process remains.

Retain **BCE + pooled crossfit calibration** as the preferred Development
challenger, with frozen A remaining the base and App unchanged. Preserve the
peak-loss tradeoff, fixed0 controls and original null result. This experiment
does not produce one selected all-data deployment checkpoint; it produces six
outer models per arm with honest out-of-group results. No further run is needed
to complete this round; wider transfer and false-alert suppression remain open.

Artifacts under `artifacts.local/work/corridor-public-positive-v2-20260917/`
retain authenticated1344 inputs,48 checkpoints/optimizer states, exact training
orders, inner/outer logits, calibration curves, source/data/prediction seals,
all cases, independent audits and public replay. Registration still fails at
the existing `experiments/index.jsonl:303 input_fingerprint` mismatch. Intended
component inheritance remains pending; supported-command logs are preserved,
with no ledger edit or bypass.
