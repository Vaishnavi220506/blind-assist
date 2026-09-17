# Counterfactual corridor intrusion pilot

Status: DEFERRED_BEFORE_TRAINING by the user's later request for existing-data
lateral tolerance re-evaluation. No trained I/R/J model exists. The training
capture was cancelled with verified process release;210partial RGB files lack
raw/evaluator and are not admitted data. No report capture started. The text
below preserves the earlier design, not an active instruction to execute it.
See [completed re-evaluation](TOLERANCE_RESULTS_20260917.md). Existing MZ158288
can replace a new report acquisition in a future disclosed Development run;
it is historically consumed and must not be called fresh confirmation.

## Question and scope

Can background counterfactuals and coarse signed corridor supervision improve
transfer beyond ordinary training on the same expanded public inputs and data?
The two previously reported 288-frame cohorts are consumed Development. S1's
Development gain remains recorded; its online DA-V2 recipe is closed. No CNH,
depth-model rescue or original MZ136 test access is part of this experiment.

## Data and source roles

- Training: original MZ136 TRAIN192, consumed MZ170288 and S1-confirmation288,
  plus new counterfactual TRAIN144: 912 frames.
- Selection: original MZ136 DEV48 and new counterfactual DEV144: 192 frames.
- Report: separate seed181017, 288 frames / 12 target-camera groups / 24
  background configurations. Capture and inference follow model/threshold seal.
- New training seed180017 has 8 independent target-camera groups, four TRAIN
  and four DEV. All lateral and background variants remain in their target group.
  It deliberately varies only lateral target location within an intrusion pair,
  and only background geometry within a same-label counterfactual set.
- All-object native geometry supplies labels. Physical metadata, group identity
  and evaluator bounds are never model inputs. Topology/distance remain coupled
  by negative-scene geometry, so this is not a fully factorial causal study.

## Matched arms

1. Frozen A: original pickle and threshold0.3917890013717321.
2. J: same A features and original single HGB configuration (150 iterations,
   7 leaves, learning rate0.05, min leaf8, L2=1), refit on joint TRAIN only.
3. R: small residual over frozen A logits, all2485 A features, observable local
   RGB patches and public dual-return ToF. Frame BCE only.
4. I: identical model, initialization, batches and updates; frame BCE plus
   0.25 query occupancy BCE, 0.2 normalized signed-margin SmoothL1,
   0.1 lateral pair hinge, 0.1 same-label background score consistency and
   0.05 margin consistency on query pairs with equal clipped geometric targets.
   Background motion can change an empty query's clearance despite unchanged
   frame truth; those unequal targets are not forced to match. This correction
   was made before training or new capture outcomes. Margin is an AABB
   separating-face slack in metres,
   clipped to +/-0.3m for training, over8 BODY/HEAD corridor query volumes.
   It is not dense surface depth or Euclidean clearance.

R/I: seed182017, AdamW learning rate0.0003, weight decay0.0001, 2000 updates,
batch96 with32 uniform frames,16 lateral pairs and16 background pairs. Missing
pair pools fall back to uniform frames. Every100updates evaluate DEV frame
BCE; select lowest BCE, earlier checkpoint on ties. Per-arm threshold maximizes
pooled DEV F1, then minimizes FP and selects higher threshold, exactly the
existing selection helper. No report-based choice, rescue or sweep.
Standardization uses TRAIN only, std floor0.01; A logit clips probability to
[1e-5,1-1e-5]. Native labels may train auxiliary heads but never enter inference.

## Evidence and decisions

Authenticate existing scores for domain diagnosis; show association, not causal
attribution, across physical background/width and public range/return-quality,
corridor-margin and score distributions. Preserve missing observations.
Inspect TRAIN fit and matched DEV selection before report access. An observed
implementation or optimization defect may be diagnosed and corrected with a
recorded revision before sealing; no arbitrary elapsed-time cutoff ends capture.

Seal all four arms before fresh inference. Report paired TP/FP/FN, F1/recall,
events and first reminders, native contributor retention, family and background
strata, query margin error, invocation-free single-frame mean/p50/p95 latency.
Prediction seal precedes report evaluator decode. An85% F1 is an aspiration,
not an acceptance rule to tune toward. Same-generator controlled evidence only.
If joint HGB explains the gain, credit data coverage; if I exceeds matched R,
credit the tested combined objective only. Failure closes this recipe without
automatically starting another architecture or information-source extension.
