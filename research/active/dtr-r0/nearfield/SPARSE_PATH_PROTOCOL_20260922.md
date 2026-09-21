# Sparse path sampling with native SkyDiscover

2026-09-22 EXPLORE, user-authorized pilot. Retain the 12cm +X path, 8 rays,
10cm radial bins and known poses. Ask whether 13-view information can be retained
with 3/4 views. This is a new observation-budget question, not a retry of prior
classifier, threshold, quantization or two-step recipes.

## Inputs and budget fixed before generation

- Search uses all 180 already-consumed scenes / 90 pairs in sealed
  `ba-observation-mechanisms-20260922/run-v1`. No fresh claim on those data.
- Held geometry is generated only after both selected policies are sealed:
  sides ±1, widths .09/.15/.25/.43m, depths 1.10/1.70/2.20/2.70m,
  margins .009/.021/.033m, with opposite offsets per pair: 96 pairs/192 scenes.
  Retain thickness .04m and wall 4.2m. Check exact nonoverlap against the search
  source. New geometry from the SAME analytic generator is Development, not
  independent-source or device confirmation. No noise or pose error is modeled.
- Fixed schedules always include 0 and 12cm; choose 1 or 2 interior centimetres.
  Enumerate all 11/55 choices. Uniform, greedy, exact-global and exact
  initial-signature lookup (unseen signature falls back to exact-global) are
  fitted using Development only. Exact lookup is a stronger, less compact baseline.
- Search candidate: separate policies for budgets3/4, each at most3 ordered
  `feature <= threshold` rules and one default. Feature uses only initial8bins:
  min/max/left_min/right_min/spread/argmin/near_count(bins<=30). All positions
  selected before movement. This is initial-conditioned, not mid-path feedback.
  No scene IDs, pair identities, truth, raw ranges or future views enter policy.
  Parse literal POLICY through AST; never execute generated Python.
- Ordinary Codex control: native `best_of_n`, frozen seed parent for all three
  proposals, zero archive context. SkyDiscover: native `topk`, best parent plus
  at most one other candidate, three sequential generate/evaluate iterations.
  Both use the same strong exact-global seed, prompt and development class-target
  table. Direct is an independent-candidate control, not every possible ordinary
  Codex iterative workflow. This does not test AdaEvolve, EvoX or full SkySynth.
- Same `gpt-5.6-sol`, medium reasoning; each arm at most3 calls,120000 input+output
  tokens,4 candidate evaluations including seed,240seconds per call. Provider and
  evaluator retries disabled. No tool use by generators. Record actual tokens;
  overshoot response is discarded and counted. Root setup/review cost is separate.
- Local scalar replay: TASK_NOT_GPU_SUITABLE. No remote allocation. Native framework
  remains read-only; consumer owns all outputs under canonical `artifacts.local`.

## Selection and comparison

Select by Development resolved-scene count summed across budgets3/4, then
designated separated pairs /1000. Tie retains incumbent. The finite-cohort
purity calculation groups ALL scenes with identical public signatures; include
positions in signatures. It is an information diagnostic, not alert accuracy.
Provide the same Development per-initial-class exact schedule hints to both arms.
Held counts never feed prompts, selection, retry or stopping.

Report each budget separately: resolved positive/negative/total and IDs, pair
separation, initial alias denominator, retention/loss versus full13, endpoint
retention, observations/rays and12cm motion. Also group Development+held together
and report held purity in that larger universe; held-only purity can hide
opposite-label aliases from other cohorts. No continuous-domain uniqueness claim.

Compression is promising if a selected policy retains at least90% of held full13
resolved scenes at4views. Sky-specific incremental value additionally requires
at least4 more held resolved scenes than BOTH the direct selected candidate and
the strongest declared simple baseline at4views, without lower pair separation;
joint-cohort held purity must not reverse the gain. Otherwise keep measured
partial effects without promoting the search framework. This single pilot cannot
establish statistical superiority. Do not select a new policy after held scoring.

## Recovery and delivery

Run a zero-model-call native evaluator canary before paid generation. Checkpoints
and external-call dispatch/finish journals persist after every accepted iteration.
No automatic resume/retry after interruption: uncertainty consumes its slot;
retain any completed checkpoint as partial evidence. Maximum interrupted unit is
one240second generation per arm. Never reset budgets or overwrite terminals.
Mechanical evaluator/adapter failures may be repaired before generation; after
generation preserve receipts and distinguish them from scientific negatives.

Stop after both three-call arms and one held evaluation. No extra training,
generator sweep, policy-class expansion, additional cohort or Android changes.
Record results, supported registration/inheritance receipts, narrow checks and
resource release. Existing ledger failures must remain intact.
