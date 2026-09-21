# B/N disagreement: useful screening, no budget-valid binary-only rescue

The requested saved-output diagnostic is complete. B/N agreement contains useful
Core rescue with very little additional error. Boundary positives and false
alerts coexist in every nonempty binary pattern. All seven nontrivial global
retention rules fail the existing Boundary current FP cap on these consumed
layouts; only A alone passes every cost cap. This is not evidence that all fusion
or input-conditioned verification is impossible. No model or cutoff is changed.

## Scope and definitions

[Brief](BRANCH_DISAGREEMENT_PROTOCOL_20260921.md); evidence at
`artifacts.local/work/ba-branch-disagreement-20260921/`; input is the sealed1152
evaluation frames of [136801f7's completed run](DATA_COVERAGE_RESULTS_20260921.md).
These16 layouts are consumed, posthoc Development. The code/brief seal records
execution identity; it is not prospective preregistration or unseen confirmation.
The original protected test stays closed. No RGB/native-depth payload is decoded
or geometry queried; dependency hash verification remains active. Continuous
logits present in saved prediction records do not enter any rule or selection.
No new capture, feature extraction, fit, threshold selection, type router or
verifier is executed.

B here is **B_control**, with the old B weights and its saved new-dev cutoff
6.888704776763916. N keeps7.03014612197876. Legacy B7.6612162590026855 remains a
separate historical working point. All inputs, source seals and relevant member
hashes verify; outputs reconcile exactly with saved A/B_control/N predictions.

First exclude A alerts for the corresponding mode, then partition the remaining
frames by B/N outputs: common, B-only, N-only, neither. The first three rows below
are added TP/FP; neither contains missed positives and unalerted negatives, not
emitted alarms. Full frame IDs including neither are retained in pattern-counts
and group-patterns.json. Current and held tables describe their respective saved
flags; they do not silently mix current A with held B/N.

## Shared and exclusive rescue/error counts

Each cell is **added TP / added FP relative to A in that same mode**.

| Mode | Stratum | Common B/N | B only | N only |
| --- | --- | --- | --- | --- |
| Current | Core |14 /0|1 /31|6 /5|
| Current | Boundary |41 /3|49 /4|45 /5|
| Held | Core |13 /1|0 /40|3 /8|
| Held | Boundary |51 /5|44 /9|44 /9|

Core disagreement is informative: of31 current B-only false positives,29 occur
in BODY and2 in HEAD, while the common group has14 true positives and0 false
positives. But rejecting disagreement also rejects6 current N-only Core true
positives. Agreement is neither a lossless filter nor a universal correctness cue.

The Boundary exchange is explicit even after A removal:

| Mode | Layer | Common | B only | N only |
| --- | --- | --- | --- | --- |
| Current | HEAD |28 /3|1 /0|42 /5|
| Current | BODY |13 /0|48 /4|3 /0|
| Held | HEAD |32 /5|0 /1|41 /9|
| Held | BODY |19 /0|44 /8|3 /0|

HEAD/BODY and Core/Boundary here are evaluator strata, not available route labels
supplied to the algorithm. These tables do not implement a type-aware selector.
Some finer strata have zero observed FP; selecting them after seeing labels would
not establish a public-input rule or its transferability.

## Fixed binary combinations, then unchanged hold

The replay enumerates all eight global subsets of the three nonempty CURRENT
patterns, always retains A, and applies the original nonrecursive one-frame hold
after combination. It does not optimize a mask, train a gate or search cutoffs.
For example AND means A OR (B_current AND N_current), followed by hold.

| Rule | Core held TP/FP/FN | Boundary current TP/FP/FN | Boundary held TP/FP/FN | Boundary events, held | All costs pass |
| --- | --- | --- | --- | --- | --- |
| A alone |159/25/17|8/3/168|11/5/165|3/16|Yes|
| AND / common only |172/26/4|49/6/127|62/10/114|10/16|No|
| B-only current pattern |159/65/17|57/7/119|66/14/110|9/16|No|
| B |172/66/4|98/10/78|106/19/70|12/16|No|
| N-only current pattern |163/33/13|53/8/123|61/14/115|9/16|No|
| N |175/34/1|94/11/82|106/19/70|13/16|No|
| XOR |163/73/13|102/12/74|116/23/60|14/16|No|
| OR |175/74/1|143/15/33|150/28/26|15/16|No|

The limitation follows directly from the current Boundary partition: common,
B-only and N-only each add3,4,5 false-positive frames, respectively, while the
global current budget allows only2. The categories are disjoint. Any nonempty
global subset therefore already fails before hold. A alone passing is a trivial
no-supplement result, not an algorithmic improvement.

The rule class is deliberately narrow: deterministic, global, memoryless
retain/discard based only on the two saved binary outputs, outside A. It excludes
continuous scores, RGB/ToF evidence, time history, input-dependent grouping and
new sensing. It is not a geometric oracle or an upper bound on those methods.

Held-pattern descriptions and a current-pattern rule followed by hold are not
generally interchangeable. This cohort happens to have0 differences between
intersection of held B/N outputs and hold of their current intersection; a
counterexample test with alternating B/N current alerts demonstrates the general
non-equivalence. B-only's descriptive held bucket has44 added Boundary TP, while
the current B-only rule followed by hold has55 added TP (66 total versus A11).
Do not add the descriptive held buckets to manufacture a different algorithm.

## AND's partial benefit and remaining cost

AND improves Core held159/25/17 to172/26/4, detecting16/16 events versus A15/16.
The extra Core event is actual rescue, not merely structural retention. Its Core
current and held costs pass: added FP0/1 versus caps5/11; added segments0/1 versus
caps4/4. This useful precision-oriented agreement signal is retained as diagnostic
evidence, despite failure of the combined task's gate.

AND's Boundary result62/10/114 has35.23% recall,86.11% precision and4.81% FPR,
with10/16 detected events and.4s maximum delay among detected events. Current
added FP3 and segments3 both exceed caps2; held added FP5 exceeds4 and segments3
exceed2. Six Boundary events remain wholly missed. It does not satisfy the full
task, and simply changing hold would not erase its current error-budget failure.

OR obtains150/28/26 Boundary held,15/16 events, but adds23 FP versus cap4 and17
segments versus cap2. Its Core held adds49 FP versus11 and16 segments versus4.
Higher rescue alone does not provide low-cost retention.

All eight rules preserve every original A current/held flag and existing A event
onset, and carry UNKNOWN unchanged. Core is not considered solved: A's17 held
missed positive frames and one missed Core event remain explicit in the baseline.

The user's HEAD=N/BODY=B held family splice is independently reproduced as
**147 TP /27 FP /29 FN**,83.52% recall,22 FP above A versus budget4. This is
posthoc family arithmetic only, not an implemented public-input router or a
general fusion bound. Actual framewise OR differs and is reported separately.

## Decision and delivery

Keep the original A position and B/N/R dispositions. Preserve N's Core/HEAD
improvement and B's BODY coverage, alongside this agreement signal. Do not start
HEAD/BODY expert training, another data-coverage fit or a complex gate on the
basis of these counts. This check answers the binary-pattern question: the
patterns alone do not yield budget-valid Boundary rescue on these layouts.

A later verifier needs an independently justified observation available at
inference that separates positives from false alerts **within** a binary pattern.
The41 current common Boundary positives and3 common false alerts, with their
saved frame IDs, are one concrete diagnostic subset. No such separating input
evidence is demonstrated here. Future methods informed by this analysis require
different, isolated validation data; these16 layouts cannot become unseen again.

Four focused tests pass, covering the complete boolean truth table, A retention,
cross-time agreement versus same-frame consensus, clip-reset hold, and exclusion
of A/interpretation of neither. A separate reviewer independently recomputed
Core/Boundary/HEAD/BODY partitions, AND/OR counts, event detection/delays and
incremental costs from sealed source outputs; all matched. Full metrics and
events for every rule and group are saved; no rule is selected for deployment.
Execution uses CPU for scalar/binary posthoc arithmetic; no model inference or
training process, paid worker or service is retained.

Local structured inheritance retains this diagnostic as COMPONENT, not an
algorithm upgrade. Supported global registration/inheritance remains blocked by
the existing ledger303 input-fingerprint mismatch and unknown terminal; receipts
are retained without editing the global ledger. This metadata gap is not hidden.
