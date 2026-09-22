# Inherited path and shared-error model: one precision contrast

2026-09-22, user-authorized EXPLORE within the historical-mechanism batch.
Question: does additional range information become useful task decisions when
the already implemented shared-bias and initial-camera query model is retained?
The earlier angle/quantization diagnostic only counted distinguishable pairs;
it did not execute this fine-quantization classifier. No old failed recipe is
retuned, no hardware is assumed to possess this measurement capability.

Use all 180 consumed analytic single-rectangle scenes from ba-bias-drift, in
exactly three conditions: nominal, range_constant_minus, combined_constant_plus.
The first-camera-relative full-extent query, all 13 x-then-z views / 12 cm travel,
the anchored wall, shape domain, +/-2 mm shared range and +/-1 mm shared pose
bounds, forward witness validation, projection and solver budgets stay fixed.
Baseline is the sealed initial-relative 100 mm result on these same histories.
The sole candidate exports 1 mm quantized bins from the already saved biased
analog simulated ranges. This is hypothetical additional sensor information,
not interpolation of existing coarse bins or a hardware resolution claim.

Private observation-generation records may supply analog ranges. Inference
accepts only nominal camera poses and quantized bins. Generate and seal public
queries before inference; seal every prediction before opening source truth or
joining the old decisions. Query truth follows the actual initial camera, as
in the parent; do not relabel old results or use evaluator identities in inference.

One candidate / one full run, at most 540 histories and 1080 MILP calls, each
using the existing 1 second / 10000 node budget. Deduplicate identical public
histories. No parameter, path, quantizer, bias-bound or solver sweep. Mechanical
repair may preserve failed receipts, but no result-directed scientific retry.
CPU SciPy/HiGHS, TASK_NOT_GPU_SUITABLE; no allocation persists after execution.

Report TP, FP, false OUT, correct OUT, UNKNOWN, precision/recall/FPR, same-case
gains/losses, general and each boundary stratum, numerical failure reasons,
solve time and bin count/measurement cost. There are no temporal alert events
or IoU outputs in this analytic classification experiment. Never report
UNKNOWN as a certified negative or a solver status as a formal proof.

Retain as an information component if nominal correct decisions rise by at
least 10/180 with no new wrong decisions, no more than 2 lost coarse-correct
cases, and at least one extra correct decision in each biased condition.
All true-generating assignments must pass exact forward validation and outer
constraint containment; any observed invalid exclusion prevents a recognition
claim. Otherwise retain only the diagnostic, or record NOT_EVALUABLE if source
or mechanical validity cannot be established. Stop this fixed contrast after
results; do not extrapolate to multi-object/real-sensor or adaptive-policy gains.
