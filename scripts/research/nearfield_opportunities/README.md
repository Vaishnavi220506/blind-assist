# Bounded nearfield opportunity probes

[Candidate recovery](../../../research/active/dtr-r0/nearfield/CANDIDATE_RECOVERY_RESULTS_20260922.md)
captures raw solver vectors and restores94of97rejectedIN witnesses by closed-query
projection plus unchanged full validation. Fixed13view numerical incompleteness
falls64to3;98of101UNKNOWN now have explicit opposing geometries. All classifications
remain unchanged. The initial42witness cache recovers only3fixed-path cases.
This is numerical/ambiguity diagnosis, not recognition gain or relaxed sensing.

[Saved-trace deepening](../../../research/active/dtr-r0/nearfield/OBSERVATION_DEEPENING_RESULTS_20260922.md)
joins13view fixed-path measurements with continuous rectangle constraints:
model-conditional correct decisions4to79/180, without observed wrong decisions,
at11extra views and unchanged12cm motion. A single witness-pair path heuristic
loses14TP; initial openloop planning reproduces every feedback decision.
`run_observation_deepening.py` seals one consumed replay; conditional numerical
exclusion is not a formal certificate or sensor free-space claim. Candidate
validation failures remain UNKNOWN and are distinct from opposing witnesses.

[Five observation mechanisms](../../../research/active/dtr-r0/nearfield/OBSERVATION_MECHANISMS_RESULTS_20260922.md)
share one frozen180scene source. Fixed+X path sampling raises pair separability
35to81/90 at2to13views; strict pure-quantization attribution yields0gains.
Continuous MILP witnesses expose opposite-label ambiguity beyond the old bank.
Executable3view two-step improves24to28TP and11to10wrong commitments, but adds
one new wrong OUT. Reasoned UNKNOWN remains abstention, not alert accuracy.
`run_observation_mechanisms.py` owns the sealed run and per-mechanism costs.

[Boundary-pair information check](../../../research/active/dtr-r0/nearfield/BOUNDARY_SEPARABILITY_RESULTS_20260922.md)
uses `boundary_separability.py` on90 new analytic pairs. Of58 initial aliases,
38have an allowed separating action; the evaluator class-common ceiling is32
versus12for all frozen selectors. Twenty pairs remain aliased across all poses.
Most missed opportunities have no matching old hypothesis. These are information
ceilings only; no new selector, source-aware action or expanded bank is admitted.

The subsequent [frozen +/-1cm transfer](../../../research/active/dtr-r0/nearfield/OPPORTUNITY_TRANSFER_RESULTS_20260921.md)
uses `opportunity_transfer.py` without changing either component or prior. New
348/1304 geometries are correlated derivatives of old layouts, not independent
source confirmation. Shape union adds0TP/2FP over point; positive-priority adds
8TP while retaining all fixed TP, but adds2FP/2false OUT. Both full transfer
criteria fail; preserve the original consumed gains only in their original scope.

These two analytical probes implement the shape and observation-choice ideas
from the 2026-09-21 missed-opportunity review. Their models and observation laws
are deliberately explicit synthetic assumptions. They are not production
inference, hardware emulation, or replacements for Calibration A.

The user-authorized [continuation](../../../research/active/dtr-r0/nearfield/OPPORTUNITY_DEEPENING_RESULTS_20260921.md)
adds a saved-output non-veto union (`shape_support_union.py`), a finite-prior
opportunity audit (`active_view_opportunity_audit.py`), and a separately identified
positive-priority selector. These use consumed synthetic data and leave the
original runs and their dispositions unchanged. They are not fresh confirmation.

| Probe | Public input and decision | Protocol and result |
| --- | --- | --- |
| `shape_hypotheses.py` | Ideal binary silhouette plus 64 zonal radial means; compare one fitting box, all consistent boxes, and a zone-centre point proxy | [Protocol](../../../research/active/dtr-r0/nearfield/SHAPE_HYPOTHESES_PROTOCOL_20260921.md), [result](../../../research/active/dtr-r0/nearfield/SHAPE_HYPOTHESES_RESULTS_20260921.md) |
| `active_view.py` | Eight quantized radial ranges; select one translation from initial finite-prior ambiguity before obtaining the next view | [Protocol](../../../research/active/dtr-r0/nearfield/ACTIVE_VIEW_PROTOCOL_20260921.md), [result](../../../research/active/dtr-r0/nearfield/ACTIVE_VIEW_RESULTS_20260921.md) |

Both runners refuse an existing output directory. Predictions are sealed before
truth joins; active-view choices are additionally sealed before actual second
observations. Every case remains in evaluation, including UNKNOWN and out-of-prior
cases. Negative answers are conditional on the finite prior, not certified free
space. Original observations, source truth, predictions, seals and local
dispositions remain in each artifact tree.

The one authorized scored run of each probe is complete. Results retain only
narrow mechanism components: shape consensus avoids three incorrect commitments
inside its bank but still makes four false OUT decisions outside it; adaptive
view choice gives ten net additional correct decisions over fixed motion, while
positive identification falls by one and all 32 out-of-prior cases stay UNKNOWN.
Neither result is an overall obstacle-recall improvement.

The third probe uses actual UE RGB capture and frozen B/N model inference; its
runner lives beside the existing source and evaluator dependencies:
[appearance_probe.py](../../../research/active/dtr-r0/nearfield/appearance_probe.py).
See the [appearance protocol](../../../research/active/dtr-r0/nearfield/APPEARANCE_PROTOCOL_20260921.md).

Focused tests can run without a new scientific capture or training:

```powershell
python -B -m unittest discover -s scripts/research/nearfield_opportunities -p test_shape_hypotheses.py -v
python -B -m unittest discover -s scripts/research/nearfield_opportunities -p test_active_view.py -v
```

Python needs NumPy for shape hypotheses; active-view uses the standard library.
The five-mechanism suite additionally uses SciPy/HiGHS for continuous witnesses;
its result records the executed Python environment and SciPy version.
Exact executed commands, runtime assumptions, evidence locations and registration
receipts are in the result reports. Global ledger admission is separately blocked
by the existing row-303 fingerprint error; local evidence is retained without a
ledger bypass.
