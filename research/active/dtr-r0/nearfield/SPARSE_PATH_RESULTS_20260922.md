# Sparse path sampling: bounded native SkyDiscover result

2026-09-22 EXPLORE / Development. Terminal `ba-sparse-path-20260922`.

Neither selected policy retained 90% of the full 13-view information on held
geometry. Native SkyDiscover did not outperform the ordinary independent-candidate
control or the simple uniform schedule. Keep this as a component diagnostic;
do not promote a sampling policy or expand this consumed pilot.

## Question and evidence

The [frozen protocol](SPARSE_PATH_PROTOCOL_20260922.md) asks whether the same
12cm +X movement can use 3/4 observations instead of 13. Each observation has
eight rays and 10cm radial bins. Policies choose all sample positions from the
initial observation, before movement; they are not feedback during movement.
The motion remains 12cm. Readouts fall from 104 rays to 24/32 rays per scene,
which is not a measured latency, energy, or motion saving.

Search used 180 previously consumed scenes (90 pairs). After sealing both
selected policies, one held evaluation used 192 new, exactly nonoverlapping
scenes (96 pairs) from the same analytic generator. There is no sensor noise or
pose error. This remains Development, not device or independent-source evidence.

“Resolved” means that all scenes sharing an identical public observation
signature have the same truth in the evaluated finite cohort. This is an
information diagnostic, not measured classifier accuracy, alert recall or
continuous-domain identifiability. Positions are part of the signature.

## Counts

Each cell is **resolved scenes / separated designated pairs**. Development
denominators are 180/90; held denominators are 192/96.

| Method | Dev 3 views | Dev 4 views | Held 3 views | Held 4 views |
|---|---:|---:|---:|---:|
| Uniform | 108 / 67 | 95 / 61 | 104 / 77 | 131 / 81 |
| Greedy / exact global | 108 / 67 | 128 / 72 | 104 / 77 | 122 / 81 |
| Exact initial-signature lookup | 148 / 79 | 156 / 81 | 104 / 77 | 122 / 81 |
| Direct independent candidates | 128 / 71 | 140 / 74 | 107 / 76 | 112 / 78 |
| Native SkyDiscover top-k | 123 / 72 | 137 / 76 | 101 / 77 | 99 / 77 |

Full13 resolves 156/180 development scenes and separates 81/90 pairs;
held resolves 171/192 and separates 92/96 pairs. Endpoint-only2 resolves
30/180 and 66/192 respectively (35/90 and 66/96 separated pairs).

| Held 4-view method | Resolved positive / negative | Retention of full13 | Lost vs full13 |
|---|---:|---:|---:|
| Uniform | 66 / 65 | 76.6% | 40 |
| Greedy / global / lookup | 62 / 60 | 71.3% | 49 |
| Direct | 60 / 52 | 65.5% | 59 |
| SkyDiscover | 52 / 47 | 57.9% | 72 |

The frozen 90% criterion requires at least **154 of 171** resolved scenes.
Neither selected model policy meets it. Sky also fails the incremental-value
criterion: it is below both direct and the strongest declared simple baseline
in resolved scenes and separated pairs.

Grouping development and held together leaves each method's **held slice**
unchanged (4-view 131/122/122/112/99). Joint totals must not mask failed transfer:
direct's joint total252 exceeds uniform226, but its held slice112 trails131.
Only24/192 held scenes have an initial signature seen in development; the lookup
uses its exact-global fallback on168. Its development156 does not transfer.

Three- and four-view schedules are independently selected and are not nested.
Thus Sky's 4-view99 below 3-view101 does not mean adding an observation loses
information. All endpoint/sparse/full subset checks passed. Individual positive,
negative, resolved, lost and pair IDs and initial-alias denominators are retained
in the machine-readable result.

## What actually ran

Both arms ran through SkyDiscover's native Runner and databases at local source
revision `269c9e6`, using `gpt-5.6-sol` medium, the same exact-global seed,
candidate grammar, prompt and development class-target hints. Direct uses native
`best_of_n` with seed parent fixed and no archive context. Sky uses native `topk`
with feedback and changing best parents. Its first context includes the seed
again. This tests a small top-k loop, not AdaEvolve, EvoX, SkySynth or every
ordinary Codex workflow.

| Arm | Model calls | Evaluations including seed | Input tokens | Output tokens | Arm elapsed seconds |
|---|---:|---:|---:|---:|---:|
| Direct | 3 | 4 | 55,787 | 3,471 | 148.712 |
| Sky | 3 | 4 | 56,511 | 4,934 | 201.905 |

There were no retries, discarded calls or model tools. These costs exclude root
setup and review. Consumer-local wrappers record calls and persistent budgets,
disable retries and the native extra test rescore, and checkpoint each iteration.
Generated literal policies are AST-parsed, never executed. The evaluator canary
passed before dispatch. Scalar replay was local (`TASK_NOT_GPU_SUITABLE`); no
remote allocation or SkyDiscover repository edit was made.

## Validation, disposition and retained evidence

Six focused core tests passed. The independent audit passed all36
split/method/budget comparisons, all generated-candidate parent provenance,
selected hashes, frozen inputs, actual call budgets and tool restrictions;
103 original source evidence files remained unchanged. Held evaluation was not
rerun after selection or inspection.

Canonical retained artifact root:
`artifacts.local/work/ba-sparse-path-20260922/run-v1`.
It contains `manifest.json`, `selection_seal.json`, `held_start.json`,
`result.json`, `completion_seal.json`, per-arm selected policies, native
checkpoints, prompts, usage/event receipts, held choices, and
`independent_audit_receipt.json`. Audit and administrative receipts are separate
from the scored completion seal. No task-owned running resources remain.

Local disposition is `COMPONENT_OR_CHALLENGER` / `COMPONENT`, without default
promotion. Supported registration remains blocked by the pre-existing
`experiments/index.jsonl:303: input_fingerprint does not match input_refs`.
The supported inheritance command consequently reports unknown terminal ID.
Both failure receipts and `local-disposition.json` are retained; global ledger
and inheritance metadata were not bypassed or repaired.

This negative applies to the frozen policy class and these two three-call
searches. It does not prove that four observations are impossible, that richer
adaptive sensing cannot help, or that SkyDiscover is generally ineffective.
It does show that development fitness gains alone were insufficient here, and
provides no measured reason to adopt this search output over uniform sampling.
