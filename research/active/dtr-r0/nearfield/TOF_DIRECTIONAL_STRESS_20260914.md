# Equal-weight ToF direction baseline and mixed-support falsifier

## Decision

Freeze `readout(row)` as equal zone weights. The explicit
`equal_weights=False` switch preserves the old 1/0.5 arm for reproduction only;
it is not the default or a promoted component. No weighting gain was found.
The collision pipeline and MZ116 are unchanged.

The earlier demo established left/right/center readout and separation of
**disconnected** bilateral angular support. It did not establish arbitrary
multimodal preservation: this implementation forms four-connected components
without considering range, then emits one centroid direction per component.
Background can connect two near surfaces into a single region. Changing status
weights after grouping cannot split that region. Even two resolved near/far
slots are collapsed to one zone contribution by this representation.

## Fixed question, source and scoring

[Protocol](tof_directional_stress_protocol_20260914.json) and
[runner](run_tof_directional_stress.py) define exactly six families of ten
**hand-constructed synthetic packets**, with no capture, echo physics, training,
parameter fitting or natural data. Five mirrored pairs per family except one
ten-frame dropout sequence. Near surfaces are specified at 1 m, far at 4 m;
merged tuples use a deliberately far-biased nominal 3.6 m. These are authored
inputs and evaluator labels, not sensor measurements. Status is not a calibrated
confidence or a near/far ownership label. Repeats are stress variants, not 60
independent empirical samples or a population-accuracy estimate.

Both arms consume the identical 64-zone observable packet and differ only in
the existing status-weight flag. Evaluator footprints and artificial RGB
direction annotations never enter `readout`. Packets and labels were written
before either arm ran; protocol, code, geometry, input and prediction hashes
are recorded in the retained summary.

Direction correctness is exact agreement between emitted horizontal direction
sets and the evaluator's **near-obstacle** set, including extra background
directions as errors. This deliberately probes a proposed near-obstacle use of
an all-return angular-support readout; it does not retrospectively invalidate
its narrower original contract. UNKNOWN remains in the denominator as an
incorrect near-obstacle answer. False CENTER counts any frame emitting CENTER
without a true central near obstacle. Bilateral merging requires a single
emitted region to contain both true near footprints; losing an unobserved side
alone is a miss, not a merge.

The predeclared retention condition requires strictly more correct frames in
the two near/far mixture families combined, no lower overall correctness and no
increase in either primary error. No thresholds were changed after results.

## Result

Both arms give the same values in every cell below.

| Synthetic family | Correct / 10 | False CENTER frames | Bilateral merge frames |
| --- | ---: | ---: | ---: |
| Near obstacle + far background | 0 | 10 | 0 |
| Sparse reliable side + weak merged side | 0 | 10 | 10 |
| Same-zone pole + wall, resolved or merged | 0 | 10 | 0 |
| One-side alternating no return | 5 | 0 | 0 |
| Two near obstacles, progressively bridged | 4 | 6 | 2 |
| Artificial RGB/ToF direction conflict | 10 | 0 | 0 |
| **Total, each arm** | **19/60 (31.67%)** | **36** | **12** |

There are 30 bilateral-truth frames, so the merge count is 12/30 opportunities;
false CENTER is 36/60 opportunities. Both arms emit some support in all 60
frames, with zero whole-frame UNKNOWN. This does not mean full observation:
the dropout sequence misses the right mode in five frames and changes its
direction set at all nine transitions. It alternates LEFT+RIGHT / LEFT, rather
than directly flipping LEFT / RIGHT. Weak-side connectivity causes ten merges;
the complete far bridge in the bilateral family causes the remaining two.
Incomplete bridges can introduce CENTER without merging both near modes.

The conflict family is only a ToF independence control: RGB is an authored
annotation, not an image, and the readout does not consume it. Its 10/10 result
cannot establish cross-sensor association or which modality a fusion system
should trust. Same-zone merged labels likewise cannot establish physical
recoverability of a suppressed echo; both methods lack depth-layer ownership.

Paired scalar readout took 0.0211 s on CPU (`TASK_NOT_GPU_SUITABLE`), excluding
serialization. This is a logic diagnostic, not sensor or device latency.

## Validation and evidence disposition

Eight focused tests pass, including the equal default with an explicit old arm,
far-bridge bilateral collapse, and distinguishing dropout from merging/UNKNOWN
correctness. The four original consumed Willow frames replay successfully with
both explicit arms; their directions remain LEFT -13.39 degrees, RIGHT +13.39
degrees, CENTER 0 degrees and empty UNKNOWN. The artificial disjoint bilateral
control still produces LEFT+RIGHT.

Intended disposition: equal weighting is the retained additive direction
baseline, while fixed 1/0.5 status weighting is `NEGATIVE_CONTROL` for the tested
mixed-support improvement role. Preserve the original code path and evidence
for reproduction; remove it from default use. A future depth-layer or return
association representation would be a changed information/representation
hypothesis, not another weight sweep. No successor is started by this result.

Structured registration remains **pending**, not completed. The attempted
`register-experiment --id tof-directional-stress-20260914-v1 --status active`
with this protocol failed with the existing error:
`experiments/index.jsonl:303: input_fingerprint does not match input_refs in the checkout or at recorded code_revision`.
The subsequent `set-terminal-inheritance` attempt returned
`unknown terminal id: tof-directional-stress-20260914-v1`. The existing ledger
was not rewritten or bypassed. Local artifacts retain this pending disposition.

## Reproduction

```powershell
python research/active/dtr-r0/nearfield/test_tof_directional_readout.py
python research/active/dtr-r0/nearfield/run_tof_directional_stress.py --output artifacts.local/work/tof-directional-stress-20260914/run-v1
```

Choose a new output directory on replay. Durable local artifacts are in
`artifacts.local/work/tof-directional-stress-20260914/run-v1/`: packets,
evaluator labels, both per-frame predictions and the hashed summary. Original
clean-frame regression is in the sibling `clean-replay/` directory. No persistent
process, worker, paid allocation, downloaded payload or disposable capture tree
was created.
