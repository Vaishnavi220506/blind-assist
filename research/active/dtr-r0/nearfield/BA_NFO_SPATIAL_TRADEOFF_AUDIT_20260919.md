# Paired spatial tradeoff: 183 old hits lost, 61 misses recovered

The saved joint-fit and late-fusion predictions were compared pixel-by-pixel on
all 32 existing TRAIN frames at the unchanged 2m / 0.081 criteria. No training,
model forward, validation, test or cutoff sweep was performed. There are 126
small-support zones, containing 5,535 known near and 56,520 known far pixels.

The previously reported 122 extra FN are a **net** number: the late branch loses
183 previously correct near pixels and recovers 61 previous FN. All 183 losses
are outside the 16 selected positive zones, although those other zones were also
present in full-image training. Most losses lie between 1.8m and 2m. This is a
conditional localization benefit with a retention cost, not a general failure
to represent the four-pixel selected target and not a uniformly better model.

## Complete small-domain accounting

| Domain | Near pixels | Joint recall | Late recall | Old hits lost | Old misses recovered |
|---|---:|---:|---:|---:|---:|
| All 126 small-support zones | 5,535 | 92.43% | 90.23% | 183 | 61 |
| 16 selected positive zones | 400 | 98.75% | 99.50% | 0 | 3 |
| Other 110 small-support zones | 5,135 | 91.94% | 89.50% | 183 | 58 |
| Near depth <=1.8m | 3,164 | 92.83% | 92.60% | 45 | 38 |
| Near depth >1.8m and <2m | 2,371 | 91.90% | 87.05% | 138 | 23 |

The selected/other rows partition the full small-support domain; the last two
rows are a separate depth partition. The net 122 losses comprise 7 at <=1.8m
and 115 at >1.8m. Do not rewrite the 2m near definition, dismiss the 138 lost
near-boundary pixels as label noise, or omit the 45 lost <=1.8m pixels.

Of the 126 zones, 21 have net recall loss, 13 net gain and 92 equal near-hit
counts. Ten of the 32 frames have net small-support recall loss. Across this
domain the late model removes 4,980 previous FP and adds 1,452 new FP, producing
the earlier net reduction of 3,528. Globally on all known image pixels, it loses
608 old near hits and recovers 2,575, explaining why full recall improves while
the small-support subgroup regresses.

## Original depth-separated stratum

Apply the **previously specified** separation conditions without selecting new
examples: q90 near depth<=1.8m, q10 far depth>=2.2m, gap>=0.5m, and observed
ToF>=2.2m. Of the 126 small-support zones, 30 satisfy these conditions, including
the 16 selected positives. This is a descriptive stratum, not a new promotion
gate or a substitute denominator. Other original selection filters, such as
minimum near count and known fraction, are not applied here.

| Metric on 30 separated zones | Joint | Late |
|---|---:|---:|
| Near denominator | 606 | 606 |
| TP / FN | 551 / 55 | 556 / 50 |
| Recall | 90.92% | 91.75% |
| FP | 1,091 | 845 |

No old near hit is lost in this stratum; five previous misses are recovered,
three in selected zones and two elsewhere. All 183 lost hits lie outside this
stratum. That supports retaining the observed spatial-localization contribution
for separated support, while preserving failures elsewhere. These are the same
seen frames, with related source assets; this is not held-out generalization.

Observed ToF partitions also preserve every pixel. Old-hit losses / rescues are
72/35 for ToF<2m, 44/0 for 2-2.2m, 28/25 for >=2.2m, and 39/1 for missing ToF.
Thus missing measurements are part of the cost, but neither missing data nor
near-boundary depth alone explains every loss.

## Largest losses and supervision context

| Scene / camera / frame / zone | Old hits lost | Rescued | Total near | q90 near | Observed ToF |
|---|---:|---:|---:|---:|---:|
| ai_011_005 /00 /0038 /43 | 38 | 0 | 38 | 1.992m | 2.028m |
| ai_011_005 /00 /0038 /35 | 32 | 0 | 89 | 1.922m | Missing |
| ai_006_002 /00 /0014 /58 | 22 | 0 | 61 | 1.857m | 1.846m |
| ai_016_009 /03 /0073 /37 | 13 | 0 | 25 | 1.973m | 2.366m |

The first listed zone goes from detecting all 38 near pixels to detecting none;
it remains a real loss under the fixed near definition. A high average target
recall does not cover this behavior. The first two zones belong to a frame whose
selected control zone was pure far; the rest of that frame was not all far.

The fixed half-full/half-target loss supervises 1,558,122 full-image known pixels
and 15,620 selected-zone known pixels across these 32 frames. Separately averaged
loss terms give a selected pixel approximately 101 times the explicit per-pixel
loss coefficient of an unselected pixel: across the exact 512 batches, the ratio
is min97.00, median100.77, max103.61. This is coefficient arithmetic, not a claim
that actual gradients have that ratio. The same allocation is used by both
models, so it does **not** establish that supervision caused the architecture's
paired loss. Both selected and other zones are TRAIN, not a train/test split.

## Decision

Retain the late branch's measured selected/separated-support contribution,
original/joint baselines and all losses. Keep the original failed 12/16 versus
14/16 coverage gate. The audit does not promote the late model or authorize
removing boundary-depth or missing-ToF examples. It also does not justify
threshold retuning or another variant of the closed supervision-allocation fit.

The next independent method claim must cover both separated support localization
and retention of the other small-near pixels under the original 2m definition.
The complete 126-zone accounting is a useful regression surface for that future
question. It cannot become a fresh validation set after this audit. No new model,
sample selection, training objective or additional run is part of this delivery.

## Verification and retained evidence

[Audit implementation](audit_ba_nfo_spatial_tradeoff.py) reproduces original full
and small-domain TP/FP/FN/TN exactly for both saved predictions, verifies that
selected/other and every size/ToF/depth-separation/frame-kind partition sum to
the original domain, and retains all per-frame/per-zone changes. An independent
process uses pixel-index set differences to check every one of the 126 zones,
the source/output hashes, depth-separation flags and aggregate losses/rescues.
No new inference or neural compute is needed; CPU array reductions are used.

Evidence is retained at
`artifacts.local/work/ba-nfo-spatial-tradeoff-audit-20260919/`: protocol, complete
result table, largest-loss table, verification, command receipts, disposition
and delivery receipt. All processes exited; `-B` avoids new Python caches.
Global registration/inheritance still encounter the pre-existing ledger line303
fingerprint error; preserve local structured evidence without a ledger bypass.
