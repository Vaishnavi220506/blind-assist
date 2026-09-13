# MZ124 paired changed-cell training diagnostic

Status: `TWO_MATCHED_FIXED_FITS_COMPLETE / NO_JOINT_DEVELOPMENT_POINT`.
Authority: `CONSUMED_MZ123_PAIRED_DEVELOPMENT_NO_FRESH_CONFIRMATION`.

The fixed changed-cell ranking recipe did not improve the alert tradeoff.
At the common diagnostic threshold 0.58 it slightly raised occupied-volume IoU,
but reduced alert precision, recall, F1 and paired both-correct answers. Neither
fit has a threshold satisfying the unchanged MZ121 joint helper on the 101-point
development grid. MZ116 remains the baseline; this is not a replacement candidate.

## Question and bounded comparison

The hypothesis was that explicitly training opposite spatial answers within a
matched pair would make the same observation-only OccupancyNet more sensitive
to corridor-relative position. The falsifier was a matched BCE control, with
the same split, initialization, batches, augmentation, optimizer and capacity.
The only candidate change was an additional training-label changed-cell ranking
loss: `0.25 * mean(relu(1 - (logit_a-logit_b)*(label_a-label_b)))` over cells whose
native occupancy labels differ. Unchanged cells keep their original BCE loss.
Pairs with no changed cells have a finite zero ranking loss and gradient.

The user authorized all brainstorming trials. This branch ran exactly two fits,
one seed, 800 steps each, with no outcome-driven restart, sweep or new acquisition.
The 20-minute training cap per fit was not approached.

| Fixed element | Value |
| --- | --- |
| Source | Already consumed MZ123, all four families, 24 complete episodes / 288 frames |
| Train | Pair 0 and pair 1 in every family; 16 episodes / 192 frames |
| Development | Pair 2 in every family; 8 episodes / 96 frames |
| Initialization | Exact original MZ122 `early_initial.pt`; both arms identical |
| Parameters | 11,305 per arm; no architecture or capacity difference |
| Schedule | Seed 124013, 800 batches of four complete aligned frame pairs, replacement sampling |
| Augmentation | Same contrast and offset for each pair's two members; identical schedule in both arms |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Occupancy BCE | Original global training-only positive weighting rule; actual weight 7.089887640449438 |
| Ranking | Candidate only; fixed weight 0.25, logit margin 1, training labels only |
| Threshold policy | Both fixed 0.58 diagnostic and unchanged MZ121 `select_joint`; full 0.00–1.00 curves at 0.01 intervals |

Pair 2 was excluded from these fits but **was already consumed by MZ123 result
judgment**. It does not regain independent confirmation status because it is
untrained in these two fits. Native occupied object-volume labels are supervised
targets, not physical return identity. Pair IDs and families control split,
schedule and evaluation; the model forward path receives RGB plus the existing
explicit observation encoding only. Existing independent sensor support and
UNKNOWN semantics are not changed in MZ116.

## Results on the 96-frame consumed development partition

| Metric | MZ116 on same frames | Matched BCE at 0.58 | BCE + pair ranking at 0.58 |
| --- | ---: | ---: | ---: |
| TP / FP / FN | 45 / 35 / 3 | 48 / 43 / 0 | 35 / 34 / 13 |
| Alert-frame precision | 56.25% | 52.75% | 50.72% |
| Positive-frame recall | 93.75% | 100.00% | 72.92% |
| Frame F1 | 0.7031 | 0.6906 | 0.5983 |
| Positive events detected | 5/5 | 5/5 | 5/5 |
| Maximum detected-event delay | 0.25 s | 0.00 s | 2.75 s |
| Suspended HEAD positive frames hit | 11/12 | 12/12 | 1/12 |
| Suspended HEAD events hit | 1/1 | 1/1 | 1/1 |
| Paired both-alert-answers correct | 10/48 | 5/48 | 3/48 |
| No-alert / inherited UNKNOWN frames | 16/96 | 5/96 | 27/96 |
| False-alert bin duration | 8.75 s | 10.75 s | 8.50 s |
| Occupied-volume IoU | Not available | 0.28598 | 0.29006 |
| Spatial HEAD cell recall | Not available | 42/48 = 87.50% | 25/48 = 52.08% |
| Changed cells with both paired answers correct | Not available | 92/192 | 59/192 |
| MZ121 jointly feasible grid thresholds | Not a learned readout | 0 | 0 |

Event hit alone hides the ranking arm's HEAD failure: its only correct HEAD
frame is at 2.75 seconds, the last sample in that positive event. Frame recall,
event timing and spatial HEAD recall must therefore remain visible alongside
5/5 overall event detection. The no-alert output is not a separately calibrated
abstention class and never establishes that the corridor is clear. Durations
are controlled sampled-bin statistics, not real-use notification frequency.

Compared with the matched BCE control, ranking removes 9 net false-positive
frames but loses 13 net true-positive frames. Compared with MZ116 on the same
96 frames, it removes only 1 net FP while adding 10 net FN. These are net
differences; detailed frame pairing is retained in the saved arrays.

The occupied-volume IoU increase (+0.00407) coexists with a severe HEAD loss.
It does not establish useful obstacle-alert improvement. Since there is no
jointly feasible point for either fit, no alternative development threshold is
selected or presented as a successful rescue. The fixed 0.58 comparison is a
diagnostic working point, not a claim that both new models are equally calibrated.

## Training partition and interpretation

At 0.58, BCE versus ranking on the 192 training frames is respectively
84/67/12 versus 69/61/27 TP/FP/FN. Suspended HEAD recall falls from 12/24 to
1/24; both fits completely miss the pair-0 HEAD positive episode. Paired
both-alert-answers correct falls from 18/96 to 13/96, while occupied-volume IoU
rises from 0.29434 to 0.30037. The problem is therefore not solely an unseen
pair-2 effect: this exact objective/working-point recipe fails on training
examples as well. This is evidence about the fixed small recipe, not a proof
that paired supervision, all thresholds, or all architectures cannot work.

This branch did not isolate representation, calibration and optimization as
separate causes. It does show that adding this one direct changed-cell loss,
without changing observations or capacity, is insufficient for the requested
joint alert improvement. Do not start a successor sweep from these results.

## Execution, checks and retained evidence

The registered secondary worker executed both fits on an actual NVIDIA GeForce
RTX 3060 Laptop GPU using Torch 2.9.1+cu128. The common batch-8 inference probe
measured GPU medians of 4.89 / 5.47 ms versus CPU 68.76 / 50.57 ms. Actual
training times were 18.4108 / 17.7848 seconds; complete branch runtime was
45.4493 seconds. These are task runtime measurements, not deployment latency.

Three focused worker unit tests passed: signed ranking and unchanged-cell
isolation; no-changed-cell finite zero gradient; fixed pair-disjoint split and
matched augmentation/schedule. The controller independently verified all 25
manifest files, reconstructed all 12,960 native occupancy labels, and recomputed
frame, spatial and paired counts at all 404 train/development curve points
without re-running model inference or training. Python compilation passed.

Implementation: [runner](run_mz124_pair_training.py),
[focused tests](test_mz124_pair_training.py). Local evidence root:
`artifacts.local/work/mz124-all-directions-20260913/pair-training-v1/`.
It contains `verification.json`, `verify_return.py`, the frozen transfer bundle,
and `returned-v1/fits-v1/` with schedule, split, input/RGB hashes, prediction
seals, two checkpoints, probabilities, complete curves and summary. The
returned runner result, stdout/stderr, unit tests and process-release receipt
remain under `returned-v1/`.

| Identity | SHA-256 |
| --- | --- |
| Original MZ122 early initialization | `d4c477fd96fabd45ea1753c7461eda5c848ac7b8d29a3e4876e85b782f7490db` |
| Frozen task bundle | `a42bf94c685781dd6b7a44e2b5b807aec7ede4bde0e275f8b3ac5813b39db92f` |
| Fit freeze | `026f99b492f0fbc7070ba157448742476f78849a78382c9a3c4c67ea7de1db93` |
| BCE checkpoint | `b382734cd447ebf313d28a85a3935777af8cfb199301561055b41d8a82fae170` |
| Ranking checkpoint | `73c05c04c4b3d55e9c7f6730d2d80d83e3b40db6095230dc127cc2a403ee461c` |
| Summary | `66ade55786629a882e9374b7aca23cf493dca047ac892bc067796ab9372f751e` |
| Returned ZIP | `3557394d0b9f07c456cf76c7e51053eb5f36a4a73f1e5ec4d623734ba1fcbaaf` |

Worker job `mz124-pair-training-20260913` terminated successfully. The release
receipt verifies zero owned processes and zero scheduled tasks. Durable worker
evidence remains under the worker artifact root's `work/mz124-pair-training-20260913/`;
the full original RGB capture remains at the existing MZ123 worker root. These
are retained research payloads owned by MZ124/MZ123 respectively, with no live
resource needing release. Do not delete them as temporary worker capacity.

Limitations: one seed, small related controlled simulation, hypothetical
sensors, consumed development outcomes, no hardware or natural-use evidence.
Structured MZ124 disposition and ledger integration belong to the root task;
this branch does not alter the existing ledger blocker or claim registration
closure.
