# Branch-off ablation: suppression is useful and also removes valid near support

One fixed-checkpoint ablation is complete on the same 32 TRAIN frames. Removing
the late residual branch restores 102 of the 183 old joint-fit near hits lost
by the complete late-fusion model. The other 81 remain missed. However, removing
the branch raises pure-far FP from 9,496 to 24,688 and reduces selected-target IoU
from 70.07% to 29.05%. The branch is a material false-positive suppressor, with
a near-support retention cost. Simply disabling it is not a useful replacement.

This is an inference-time intervention on **the same final co-trained weights**.
Branch-off is not the original NFO or the independently trained joint-fit model,
nor the model that would have resulted from training with a frozen backbone.
No training, threshold or branch-coefficient sweep, validation or test occurred.

## Same final checkpoint, residual off versus on

All metrics use original 2m labels, 0.081 cutoff and unchanged denominators.

| Metric | Final base, residual off | Complete model, residual on |
|---|---:|---:|
| Selected-target TP / FP / FN | 398 / 970 / 2 | 398 / 168 / 2 |
| Selected-target recall | 99.50% | 99.50% |
| Selected-target IoU | 29.05% | 70.07% |
| Positive zones passing original individual criteria | 7/16 | 12/16 |
| Selected-negative FP | 11 | 0 |
| Full-image recall | 99.62% | 99.50% |
| Full-image IoU | 69.24% | 76.78% |
| Pure-far FP | 24,688 | 9,496 |
| All small-support recall | 92.01% | 90.23% |
| All small-support IoU | 19.79% | 25.15% |
| All small-support TP / FP / FN | 5,093 / 20,198 / 442 | 4,994 / 14,322 / 541 |

Equal selected-target TP totals do not mean identical masks: the branch removes
one selected near hit and recovers another. The four previously missed
high-coverage pixels are all detected by the final base; the branch suppresses
one, leaving three detected in the complete model. Thus the earlier whole-model
recovery cannot be attributed specifically to a direct positive residual effect.

The selected-domain native-empty-near FP count changes from 879 off to 96 on.
This supports a genuine suppressive localization function in the final model,
not merely a change in total alert count. The raw residual can have either sign;
the intervention concerns its aggregate behavior, not a claim that it is always
negative or equivalent to one global cutoff.
All near decisions use the existing four-head cumulative-maximum probabilities;
mask changes do not establish the sign of the raw 2m-head residual by itself.

## Which old hits are lost, and what the branch directly does

Relative to the independently trained joint-fit reference, the complete model
loses 183 old small-support hits and recovers 61 old FN. Within those fixed sets:

- 102/183 lost hits (55.7%) are detected by branch-off: in this final checkpoint,
  the residual directly moves them below the fixed near cutoff.
- 81/183 remain below the cutoff without the residual. Their failure cannot be
  repaired merely by removing the branch. This does not prove that freezing the
  base during training would recover them.
- Of 61 old FN recovered by the complete model, 7 require the residual to cross
  the cutoff; 54 are already detected by the final base.

Across **all** 5,535 small-near pixels, turning the residual on removes 135
base hits and recovers 36 base FN, for a net loss of 99. This comparison differs
from the 183/61 comparison against the old joint-fit model. It removes 6,422
base FP and adds 546, for a net FP reduction of 5,876.

For near depths <=1.8m the direct branch change is 13 hits removed / 27 FN
recovered; for >1.8m and <2m it is 122 removed / 9 recovered. The latter pixels
remain valid near foreground under the original task. Do not redefine the task
to discard them. Across the full known image, the branch removes 553 base near
hits and recovers 153 while reducing FP by a net 46,336.

## Decision and limitations

Retain the complete branch's error-suppression contribution, all baseline
checkpoints, and the existing failed 12/16 versus 14/16 coverage gate. Do not
promote branch-off: its target IoU, pure-far FP and coverage all worsen.
Do not treat this result as proof that a smaller residual coefficient, a different
cutoff, or freezing the base would solve the tradeoff; none was tested.

The useful next training question, if pursued separately, must address retaining
valid near support while preserving the branch's spatial rejection of far pixels.
The present ablation narrows the problem to that interaction; it does not isolate
the training history or establish that all loss originates in either component.
A co-trained base and correction can adapt to each other, so the weak standalone
base mask is not evidence that the extra branch is redundant or intrinsically
harmful. No successor training was started.

## Implementation and verification

[Ablation implementation](ablate_ba_nfo_late_branch.py) loads the final checkpoint
read-only and feeds only RGB and public ToF. A hook captures residual logits;
on every batch, full logits equal base logits plus residual exactly. All 32
complete-model four-head score arrays reproduce saved results bit-for-bit.
The intervention is checked a second way by zeroing the final residual layer
**only in memory**; all 32 outputs then match the standalone final base exactly.
No modified weights are saved.

An independent process checks all source/output hashes and recomputes the lost
and rescued pixel sets and branch-off small-domain confusion counts. All checks
pass; score arrays remain finite and nested. Original and final checkpoint
files are unchanged. CUDA uses the same architectures, shapes and retained
placement evidence as the prior run. Paired base/full forward work took0.70s
on the host RTX5060 Laptop GPU; this excludes the later zero-layer equivalence
check and is not phone latency or a comparative speed claim.

Evidence root: `artifacts.local/work/ba-nfo-late-branch-ablation-20260919/` retains
protocol, branch-off scores, all metrics and transitions, verification and
disposition/delivery receipts. All processes exited and `-B` avoided new bytecode
caches. Global registration/inheritance remain pending at the existing ledger
line303 fingerprint error; actual command failures are retained without bypass.
