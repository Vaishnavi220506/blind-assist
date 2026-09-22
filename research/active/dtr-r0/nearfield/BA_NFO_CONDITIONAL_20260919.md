# Four-model conditional ranking and spatial diagnosis

Decision: `BA_NFO_CONDITIONAL_NO_OVERALL_SEPARABILITY_GAIN`.
Keep the original NFO comparator and the frozen Hybrid recipe as a negative
control. The diagnostic is a retained component. No fifth training arm,
lambda sweep, calibration fit, local branch or alert change was performed.

The result is closest to **B with local curve crossings**, rather than exact
score translation. Hybrid does not dominate NFO, nor show the proposed pattern
of better high-recall ranking. A restricted local spatial mechanism remains a
testable hypothesis, not an established remedy or automatically started run.
This rejects this auxiliary recipe's joint-benefit story; it does not reject
all multi-task learning or prove no information can be extracted.

## Same cohort and original behavior

Reuse the four actual frozen checkpoints from matched depth/NFO, RGB-only and
Hybrid. Infer all 500 original test frames, batch24 and original contiguous RGB
tensor layout, on CUDA RTX5060 Laptop. The 2m conditional domain is finite
public ToF return >=2m and 0 < GT near pixels / known pixels <=20% per zone.
Truth near means depth<2m. This selects 449 zones in190 frames/53 scenes,
13,864 positive and207,474 negative pixels. Full-test160,365 UNKNOWN pixels are
excluded. All four arms exactly reproduce full/mixed/small and conditional
2m TP/FP/FN/TN counts at their original validation cutoffs.

GT occupancy defines an **evaluator-only** subgroup. It is not an observable
online trigger. A future router cannot assume it knows that a zone contains
small near foreground. These are consumed synthetic Hypersim Development
results with simulated ToF, not fresh confirmation, real-sensor or alert gains.

Scores retain the deployed transform: NFO uses cumulative maximum of sigmoid
scores across the four distance heads; depth uses a monotone log-depth score.
They are not established calibrated probabilities. All distinct float32 score
thresholds are evaluated, with complete tied-score groups; no test cutoff is
selected or adopted. IoU-recall is supplied as requested but is algebraically
determined by precision/recall on this common domain, not independent evidence.

## Curves: no overall ranking improvement

![Conditional curves](../../../../artifacts.local/work/ba-nfo-conditional-20260919/conditional-curves.png)

| Arm | Conditional AP | Original recall | Original precision | Original IoU | FP at >=80% recall |
|---|---:|---:|---:|---:|---:|
| Depth RGB+ToF |12.470%|89.635%|8.796%|8.707%|103,310|
| NFO RGB+ToF |14.701%|69.612%|12.763%|12.090%|91,526|
| RGB-only NFO |8.359%|97.887%|6.363%|6.354%|156,052|
| Hybrid NFO+depth |14.606%|79.977%|10.278%|10.020%|96,944|

| Matched recall target | NFO FP | Hybrid FP | Hybrid minus NFO |
|---|---:|---:|---:|
|10%|6,625|6,477|-148|
|30%|19,123|17,577|-1,546|
|50%|34,749|37,196|+2,447|
|69.612%|65,928|70,204|+4,276|
|80%|91,526|96,944|+5,418|
|90%|126,032|128,829|+2,797|
|95%|149,890|149,489|-401|

The curve entry is the first attainable recall >=target; actual recall is
retained in JSON. At69.612%, NFO's minimal-FP threshold has65,928FP; the old
fixed cutoff0.081 has65,965FP with the same TP. These are different points on
a recall plateau, not a reproduction discrepancy.

At NFO's old65,965FP budget, Hybrid achieves67.412% recall versus69.612%.
At96,794FP, NFO achieves81.607% versus Hybrid79.977%. At very low budgets the
curves cross too:100FP recall0.332%/0.137%,1,000FP1.168%/1.255%,5,000FP
7.343%/6.795% (NFO/Hybrid). There is no uniform low-FP advantage.

Scene AP improves in24/53 scenes and worsens in29/53; macro mean delta+0.571pp,
median-0.483pp. Pixel-micro AP delta is-0.095pp. A paired300-replicate scene
cluster bootstrap (seed190919) gives a descriptive95% interval[-1.342,+0.839]pp
for micro AP delta. This shows heterogeneity and uncertainty, not statistical
equivalence or a universal absence of representational information.

## What moved: scores and cutoff both matter

![Score shifts](../../../../artifacts.local/work/ba-nfo-conditional-20260919/score-shifts.png)

| Hybrid minus NFO | Positive | Negative |
|---|---:|---:|
|Mean score delta|+0.009411|+0.010824|
|Median|+0.007397|+0.002763|
|5th percentile|-0.179648|-0.117794|
|95th percentile|+0.197174|+0.133621|
|Fraction increased|55.943%|64.472%|
|Mean transformed-score logit delta|+0.083279|+0.237596|

There is no preferential positive mean increase. However, the spread includes
substantial increases and decreases, inconsistent with a literal constant
translation. Logits here are computed from the deployed ordinal scores,
clipped to[1e-7,1-1e-7]; they are not a separate unmodified network-head audit.
Whole-image positive mean delta is-0.030768, negative+0.001540: do not generalize
the conditional observation into a claim that all image scores shifted near.

The original validation-selected cutoffs also differ: NFO0.081, Hybrid0.051.
The following cross-cutoff table is descriptive only:

| Model score | Cutoff | TP | FP |
|---|---:|---:|---:|
|NFO|0.081|9,651|65,965|
|NFO|0.051|10,717|84,284|
|Hybrid|0.081|10,294|80,647|
|Hybrid|0.051|11,088|96,794|

Holding NFO's cutoff while changing model adds643TP/14,682FP; subsequently
lowering the Hybrid cutoff adds794TP/16,147FP. This exact arithmetic path
accounts for1,437TP/30,829FP, but is not a unique causal attribution (changing
cutoff first yields a different partition). The final near expansion cannot
all be described as a raw-score shift caused by depth supervision.

## Spatial coupling exists, but does not explain all new FP

The exact paired changes reproduce1,711 rescued TP,274 lost TP,36,741 added FP
and5,912 removed FP. Distances below are Euclidean pixels at192x256; queries
are eligible subgroup background pixels. Distances may cross zone boundaries.
The denominator is baseline-TN pixels that could become new FP, not total
area. Distances to GT near use any known near pixel in the full frame; rescue
distances use subgroup rescued TP only.

| Distance to nearest rescued subgroup TP | Added FP | Eligible baseline TN | Flip rate |
|---|---:|---:|---:|
|<=2px|1,854|2,155|86.03%|
|2-4px|2,035|2,999|67.86%|
|4-8px|4,006|7,982|50.19%|
|8-16px|5,895|19,810|29.76%|
|16-32px|3,609|18,148|19.89%|
|>32px or no rescue in frame|19,342|90,415|21.39%|

The last row explicitly includes no-rescue frames:13,022 new FP (35.44% of all
new FP) occur in frames with no rescued subgroup TP;6,320 further new FP lie
>32px away in frames that have rescues. Only3,889/36,741 (10.58%) new FP lie
within4px of a rescue.15,764 (42.91%) occur in zones containing a rescue;
23,719 (64.56%) in frames containing a rescue.23,077 (62.81%) share an
8-connected full-image Hybrid near-prediction component with a subgroup
rescue, but large components can bridge unrelated regions.

Against any GT near foreground, baseline-TN flip rate decreases from33.21%
within2px to18.24% at16-32px; all subgroup negatives are within32px of some
near foreground because the domain itself is small mixed zones. Thus raw
foreground proximity alone is strongly constrained by subgroup selection.

There is local spill around recovered foreground, plus substantial expansion
without local recovery. This supports a mixed spatial/readout error, not the
exclusive diagnosis of blurred boundaries around thin rods. The ground truth
does not label thin-rod semantics, and observational spatial association is
not causal proof of a reconstruction/dilation mechanism.

[Five spatial examples](../../../../artifacts.local/work/ba-nfo-conditional-20260919/spatial-examples.jpg)
show the three largest rescue frames and two largest added-FP frames with no
rescue, with deterministic identity tie breaks. They are deliberately selected
illustrations. Aggregate counts above carry the conclusion.

## Reproduction and delivery

Entry: `ba_nfo_conditional_diagnostic.py`; normal invocation performs frozen
inference and analysis. `--cache-only` reuses the durable subgroup score cache.
Inference/initial analysis took9.15s before plots; see receipt for full time.
Three focused tests pass: atomic ties/monotone transform, empty-rescue and
Euclidean distance, and the precision/recall-to-IoU identity. Exact original
counts, UNKNOWN, paired differences and curve endpoints/monotonicity pass.
Curves, shift histograms and five spatial examples were visually inspected.

Evidence: `artifacts.local/work/ba-nfo-conditional-20260919/`, including all
distinct-score curves, paired subgroup scores/identity mapping, frame/zone
records, original-inference receipt, final cache-analysis receipt, plots,
bootstrap draws and results. The original receipt describes the initial
analysis; `analysis-receipt.json` hashes the final results and updated chart.
No full-frame score cache, model checkpoint or new training output was created.
The inference process exited; no task-owned background process remains.

Global registration failed on the existing ledger303 fingerprint mismatch;
the inheritance tool rejected the unknown terminal. Local `disposition.json`
records COMPONENT_OR_CHALLENGER/COMPONENT and both command receipts are retained.
The ledger was not bypassed or repaired as part of this diagnostic.
