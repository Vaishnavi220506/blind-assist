# Bounded nearfield opportunity probes

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
Exact executed commands, runtime assumptions, evidence locations and registration
receipts are in the result reports. Global ledger admission is separately blocked
by the existing row-303 fingerprint error; local evidence is retained without a
ledger bypass.
