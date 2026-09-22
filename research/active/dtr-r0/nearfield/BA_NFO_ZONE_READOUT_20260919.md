# NFO local-ranking diagnosis and one conditional-readout contrast

## Completed result: ranking improves, frozen working point fails

Retain original NFO. The one eight-parameter fit improves conditional ranking
and suppresses background FP, but fails the predeclared joint foreground-recovery
target at the unchanged0.081 cutoff. The fixed-cutoff replacement recipe is a
NEGATIVE_CONTROL; retain its measured ranking signal and the local diagnosis
as evidence, without promoting the readout or running another fit.

| 2m metric | Original NFO | Zone readout |
|---|---:|---:|
|Far-small recall|69.612%|63.596%|
|Far-small TP / FP / FN|9,651 /65,965 /4,213|8,817 /50,119 /5,047|
|Far-small precision|12.763%|14.960%|
|Far-small IoU|12.090%|13.780%|
|All-public-far FP|241,463|169,920|
|Pure-far FP|157,187|106,249|
|Mixed IoU|51.042%|51.906%|
|Mixed recall|94.699%|94.358%|
|Full-image IoU|56.663%|57.630%|
|Validation mixed recall|95.0004%|94.5978%|

The 2m fixed-point effect is suppression:0 rescued TP/1,878 lost TP and
0 added FP/71,543 removed FP across the full image. Far-small accounts for
834 lost TP and15,846 removed FP; pure-far accounts for50,938 removed FP.
Validation recall falls below95%, and small-foreground recovery fails, even
though all FP and IoU guards pass. Do not describe this as successful rescue.

![Conditional ranking comparison](../../../../artifacts.local/work/ba-nfo-zone-readout-20260919/comparison-curves.png)

Unlike Hybrid's essentially unchanged conditional AP, this readout has a modest
ranking gain: AP14.701% to15.857% (+1.156pp); at the original65,965FP budget,
diagnostic recall69.612% to71.401% (+248TP at an attainable point); at>=80%
recall, required FP91,526 to88,463 (3,063 fewer). These curves apply only to the
far-small subgroup, not all public-far or pure-far pixels. Their thresholds
are posthoc diagnostics and have not been adopted. A threshold attaining a
subgroup budget does not prove retention of the wider-background FP guards.

Mean test offsets are-0.419 logits in far-small zones,-0.461 in pure-far zones,
and-0.532 in other observed-far zones; none approaches the2-logit bound.
The trained readout learned differing amounts of suppression, not selective
positive recovery at the fixed operating point. Public summaries contain some
cross-zone ranking signal, but this objective/readout/cutoff combination does
not deliver the intended recovery. Neither the broad possibility of conditional
readout nor the need for a high-resolution branch is settled by this one fit.

One12-epoch/1500-update CUDA run took66.57s, no restart or additional training.
The base model is bit-identical after training. All original full/mixed/small
test counts reproduce at all four distances; gate-out scores are bit-identical
on validation and test. UNKNOWN remains350,140validation/160,365test pixels.
Four focused tests pass. Independent inference loads the saved checkpoint and
only public RGB/values, reproducing the selected frame's full/mixed/small raw
counts, finite outputs and four-distance nesting. Paired arithmetic and curve
endpoints/monotonicity pass. A read-only code review found no material defect.
The curves and original six identity-selected preview images were inspected.

This round is complete. No cutoff recalibration, spatial decoder training,
feature/amplitude/loss sweep, A/A* change or Android promotion follows.

This user-authorized round first locates the error, then executes exactly one
selected contrast. It does not reopen Hybrid or automatically train a spatial
branch after an unsuccessful readout. All data are already consumed synthetic
Hypersim Development with simulated single-return ToF, not fresh confirmation.

## Local diagnosis selects readout, not a new decoder

Use the retained four-model score cache and original exact449-zone2m subgroup.
Reconstruct each zone's pixel indices from hashed prepared inputs; no inference
or label-dependent input is added. Compare same-zone positive/negative scores
with tied-score half-credit AUC. Independently find each zone's first attainable
recall>=80%, with entire equal-score groups included.

Before outcomes, the pragmatic routing rule was written to `protocol.json`:
locally good means AUC>=0.8 and FPR at recall>=0.8 <=0.2. Select readout if more
than half zones are good and aggregate negative-weighted local FPR<=0.2;
otherwise select fine-scale conditioning. These are declared engineering
decision thresholds, not statistical guarantees of sufficiency.

| Frozen arm | Mean zone AUC | Median zone AUC | Good zones /449 | Local threshold TP / FP | Local FPR |
|---|---:|---:|---:|---:|---:|
|Depth|0.7992|0.8982|255|11,266 /55,711|26.852%|
|NFO|0.8567|0.9465|304|11,266 /40,305|19.427%|
|RGB-only|0.6470|0.6918|155|11,266 /93,281|44.960%|
|Hybrid|0.8603|0.9485|315|11,266 /39,603|19.088%|

NFO's median local FPR is8.669%; pair-weighted AUC is0.8500. The result selects
**conditional readout**. It shows substantial existing local order and a
remaining difficult tail, not that all zones have good localization. Integer
rounding makes total local recall81.261%, above80%.

Each local threshold uses truth and is **evaluator-only**. It is an idealized
zone-specific operating-point diagnostic, not an achievable model, common-cut
Pareto bound or online router. In particular it says nothing about whether the
correct offsets can be predicted from public features or kept harmless in
pure-far zones. The next single contrast tests that missing step.

## Frozen one-fit design

Retain the actual trained NFO baseline. Freeze all its parameters and run it in
eval mode. Add one linear7-input readout plus bias, eight trainable parameters,
all initialized to zero. A zone receives a single offset `2*tanh(linear)` added
to every pixel and all four **raw logits**. Shared offsets commute with ordinal
cummax and preserve zone-internal ordering in exact arithmetic. Their effect
is cross-zone alignment, not newly learned boundaries.

Seven features are normalized public range, zone-center x/y, and frozen2m
score mean, standard deviation, maximum and fraction>=0.081. Summaries include
**all pixels of the public zone**, not just evaluator-known or positive pixels.
Fixed natural feature scales are used; no test statistics or GT labels form
features, gates, normalization or model inputs.

The public gate is valid return>=2m for every affected head; this is one
2m-targeted mechanism, not a separate far-return gate per distance head. Apply
it to **all** such zones, including pure-far backgrounds. Use the original
exact192x256 calibrated zone boxes, independent of depth truth. Missing returns
and out-of-gate pixels receive zero offset. The backbone stays frozen, so
gate-out logits/scores remain exactly original before thresholding.

Use original3000train/500validation/500test identities, identical realized
returns, seed190921, batch24,12epochs/1500updates, AdamW0.002/0.0001, cosine
to10%, gradient clip5, last checkpoint only. Reuse the original full-known-image
four-head BCE +0.2 ordinal loss; no depth auxiliary or subgroup reweighting.
Only readout is fitted; this does not claim equal trainable capacity or FLOPs
to a full backbone fit. BCE and ordinal contributions are logged separately.

Both arms use the original NFO cutoff **0.081**. Deliberately do not repeat
candidate-specific global calibration: a second cutoff would introduce another
change and alter gate-out predictions. Original validation mixed recall>=95%
must still be feasible at this frozen cutoff. Infeasibility cannot be rescued
by lowering the cutoff.

The joint pre-outcome target is: validation mixed recall>=95%;2m far-small
recall improves over NFO with FP<=65,965; pure-far and all-public-far FP do not
increase; mixed IoU retained and mixed recall loss<=1pp. Report full, mixed,
small, all public far, pure far and gate-out domains at1/1.5/2/3m, UNKNOWN,
paired changes, scenes, readout offsets and diagnostic score curves. Do not
adopt a test-selected threshold or the per-zone oracle's thresholds.

One fit only. An unsuccessful readout stops this round; no feature, offset
amplitude, loss, cutoff sweep or fallback decoder training follows.

## Evidence and reproduction

- `ba_nfo_local_diagnostic.py`: stored-score local diagnosis.
- `ba_nfo_zone_readout.py`: frozen-backbone fit, evaluation and independent
  public-input inference (`--checkpoint`, `--sample`, `--output`).
- `test_ba_nfo_zone_readout.py`: zero-offset identity, real parameter freezing,
  public gating, constant-offset ordering/nesting, readout-only gradients and
  tied-score AUC.

Local evidence roots:
`artifacts.local/work/ba-nfo-local-diagnostic-20260919/` and
`artifacts.local/work/ba-nfo-zone-readout-20260919/`.
The fit entry refuses an existing protocol to prevent an accidental repeat.

Keep trained checkpoint, local diagnostic scores/results, subgroup curves,
frame/scene/zone records, training and verification receipts, original input
identity/hashes and previews. The owning process exited normally; no background
worker or full-image temporary score cache remains. These payloads are retained
locally under the canonical F:-backed artifact junction.

Global registration remains pending: the actual registration attempt hit the
existing ledger303 fingerprint mismatch, and inheritance rejected the unknown
terminal. Local structured `disposition.json` and both command receipts are
preserved. No ledger bypass or repair was performed.
