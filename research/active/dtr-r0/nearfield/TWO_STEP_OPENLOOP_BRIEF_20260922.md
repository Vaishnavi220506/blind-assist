# Public two-step open-loop comparator

2026-09-22. EXPLORE, one separately frozen comparator on already consumed
`ba-observation-mechanisms-20260922/run-v1` observations. User authorized further
work on valuable mechanisms. This is a new disclosed Development comparison,
not a retune, replacement, or fresh confirmation of the prior feedback policy.

## Question and mechanism

The prior two-step policy changed both path choice and second-step feedback.
Its 24-to-28 TP gain versus the fixed +X/+X path does not isolate feedback.
The evaluator ceilings were adaptive157 = nonadaptive157, but these are not
implemented policies. Test one public open-loop path selector to isolate the
value of observing a midpoint before selecting the second movement.

Use precisely the unchanged original621-scene bank and its saved forecasts.
Given only initial eight quantized bins, filter compatible prior scenes. For
each of16 ordered paths of two6cm axial moves, partition candidates by the joint
predicted midpoint AND final bins. Count positives in mixed-label partitions,
then total candidates in those partitions. Choose the lexicographic minimum:
mixed positives, mixed total, first-action index, second-action index. Ordering
is +X/-X/+Z/-Z at each step, matching the prior policy's deterministic tie order.

Both actions are selected and sealed before the actual midpoint observation is
read by the executable comparator. Midpoint information still updates posterior
support and remains part of the final observation signature. The difference is
feedback permission, not discarding the midpoint measurement. Empty initial
support selects the deterministic +X/+X path and remains UNKNOWN under the
existing posterior/decision functions; it never implies OUT.

## Budget, controls and interpretation

Every path uses3views and12cm total movement, including a return to origin.
No new true-scene observations, source cohort, hypothesis bank, score/threshold,
quantization, geometry domain or sensor changes. Reuse saved public forecasts and
recorded views only. The callable signature is
`two_step_openloop.choose(initial, forecasts, labels)` and returns
`first_action`, `second_action`, `costs`, `candidates`. Root owns execution,
posterior scoring and sealing. The implementation author does not pre-score it.

Compare to the unchanged public feedback policy at equal3view/12cm cost and to
fixed +X/+X. Report all TP/FP/falseOUT/correct-negative/UNKNOWN counts and exact
gained/lost IDs. Also report same/different selected paths and observation-class
consistency. Any useful difference must survive comparison of both benefit and
wrong commitments; aggregate error reductions do not hide newly wrong cases.

Feedback contribution is supported only if feedback adds TP over open-loop,
retains all its TP, introduces no new FP/falseOUT IDs and does not reduce correct
decisive cases. Otherwise report the exact trade or equivalence. If open-loop
matches or improves feedback, the old gain is compatible with better path
planning without demonstrated feedback benefit on this finite cohort.
This is not a claim that feedback is universally unnecessary.

One fixed comparator evaluation, no subsequent objective/path-order/bank tuning.
Hand fixtures cover public-input isolation, identical input determinism,
midpoint retention, all16paths/budget, empty support, and a constructed case where
feedback can beat open-loop. Retain prior sealed evidence, scoped failure audit,
and all costs. Small scalar work uses CPU / TASK_NOT_GPU_SUITABLE.
