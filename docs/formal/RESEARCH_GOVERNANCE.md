# Formal research governance

Use this document for protected final/blind access, claim-critical numbers and
formal decision-changing inheritance. Exploration/source debugging uses the
[research workflow](../../research/WORKFLOW.md); `AGENTS.md` owns common integrity rules.

## Scope of historical constraints

A failure constrains its method version, responsibility, evidence domain and
criteria. Before excluding a successor, explain the actual discrepancy and why
that scope applies. New information, representation, composition or criteria can
justify a new hypothesis without erasing old results. Historical successor lists
and shared labels do not ban mechanism families. Missing evaluability is an
evidence gap, never a method falsification. Mechanical recovery follows the run's
retry contract; changing future criteria cannot retroactively pass an old run.

## Freeze before access

Freeze the minimum surface needed for the claim:

- cohort/source and implementation/model identities;
- observable inputs and evaluator-only truth boundary;
- metric, denominator, thresholds and missing-data behavior;
- retries, interruption, checkpoints and `in_doubt` semantics;
- claim ceiling and stop condition.

Hash bytes whose change could alter the conclusion; resolve inconsistencies before
opening protected outcomes. Independent evidence protects against adaptive selection
in synthetic and real evaluation. Choose separation to match the stated claim.
Before inferential collection, assess power/precision using independent units,
plausible effects and dependence; frames/assertions are not independent by default.
Report small-category counts and uncertainty. Small counts can support an explicit
engineering stop rule, not unsupported category-level inference.

## Formal inheritance

Execution labels (`ACTIVE`, `GATE_NOT_MET`, `NOT_EVALUABLE`, `CLOSED`) do not decide
method usefulness. At a terminal changing mainline, baseline or governed reuse
authority, assign a role scoped as `method version x responsibility x evidence domain`:

- `RETAINED_CORE`: the surface remains in the main algorithm. Retain its evidence
  ceiling/failure signature; this does not imply independent gate success.
- `COMPONENT_OR_CHALLENGER`: record bounded `COMPONENT` or full-system `CHALLENGER`.
  A composition freezes interface/parent baseline and tests contribution on new
  evidence; it does not inherit the old component's claim authority.
- `NEGATIVE_CONTROL`: retain the named failure signature as a fixed falsifier;
  do not retune it to make a successor win. A clean control alone validates no claim.
- `DEAD_FOR_THIS_ROLE`: a severe test falsified the core hypothesis for that role.
  Threshold tuning, fusion or relabeling cannot restore it. A genuinely different
  information source/representation/responsibility needs a new hypothesis citing the old boundary.

Roles can differ by responsibility. `UNKNOWN`/`NOT_EVALUABLE` may leave a working
role unset; no evaluable algorithm test means no `DEAD_FOR_THIS_ROLE` verdict.
Before entry into the formal ledger, classify the surface actually inherited.
Do not backfill roles from `CLOSED` or gate labels; classify when next used in a
formal baseline, composition or successor decision.

Each owning current/ledger entry records:

- `terminal_status`, `evidence_verdict`, `inheritance_role` and component `inheritance_mode`;
- `role_scope`, retained surface/failure signature, evidence anchor and forbidden reuse;
- a decision-changing revisit trigger.

Update `research/knowledge/decision/inheritance.json` through
`python tools/knowledge.py set-terminal-inheritance`. Context tools consume the role:
`DEAD_FOR_THIS_ROLE` blocks the same role, related negative controls are required,
retained core supplies baselines, and components/challengers retain their recorded mode.
A role change needs new versioned evidence or an explicitly different responsibility;
relabeling selected outcomes restores no confirmation authority.

## After protected access

These rules govern the protected run and claim; separately labeled Development
can diagnose old outcomes without overwriting sealed outputs or regaining independence.

- Do not tune, resample, fuse or rerun a consumed protected arm after seeing outcomes.
- Preserve failures, partial coverage and denominators.
- An interrupted external call is `in_doubt` unless the provider proves it was not consumed.
- A protected successor needs a versioned protocol and evidence adequate to its claim.
- Record terminal/inheritance in the owning current/ledger and stop at the registered condition.

Use [the protocol template](RESEARCH_PROTOCOL_TEMPLATE.md) when entering formal mode.
