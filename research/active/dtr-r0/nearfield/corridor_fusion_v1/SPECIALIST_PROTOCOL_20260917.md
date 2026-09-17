# S1: one cheap conditional invocation gate for frozen C veto

User-directed EXPLORE, 2026-09-17. A is the research balanced baseline in this
scope; no App/default modification. Preserve old E1/E1-I outcomes. This changes
C's responsibility, not its weights, input adapter, PCA, or threshold.

Question: can an observable cheap gate retain A's high recall while exploiting
C's conditional false-warning discrimination? Family-specific rod success and
the137/17/7 family oracle are consumed hypotheses, not deployed evidence.

- Freeze A threshold=.3917890013717321 and C threshold=.728388090293355,
  models/checkpoint/PCA/feature definitions from their completed runs.
- Gate source: existing consumed MZ146288, independent of A/C fitting192.
  Whole scene groups0–3 are gate TRAIN192, groups4–5 gate DEV96. Fit/selection
  can use labels there. MZ170288 remains report-only for this run, historically
  consumed and the source of this hypothesis. No new capture, original test48,
  protected cohort, A/C retraining, layer/PCA/threshold sweep or E2.
- Cheap gate inputs: A score; selected existing public geometry summaries of
  proposal width, lateral/height overlap, edge response, mask/seed counts;
  dual-return separation/counts, MERGED rate, valid-return distance/sigma/signal
  statistics and input masks. No family, scene ID, target identity, evaluator
  geometry, C score, token feature or predicted depth enters invocation gating.
- One DecisionTreeClassifier: max_depth3,min_samples_leaf6,random_state178017,
  no class balancing or parameter search. Fit on A-alert TRAIN rows. Target1
  means C would reject an A false alert; otherwise0. Harmful disagreement rows
  (true alert that C would reject) receive weight8, all other rows weight1.
  This learns observable expected veto usefulness, not a simulated family label.
- Select only invocation probability threshold on gate DEV: minimize final FP
  subject to at most1lost A TP and no previously detected A event lost or delayed
  more than.25s; ties fewer lost TP, fewer invocations, then higher threshold.
  Include a no-invocation fallback. Gate/C selection sealed before report scoring.
  These are new design choices, not established optimal hyperparameters.
- Online: run A; if A nonalert return UNKNOWN. If cheap gate below threshold,
  keep A. Otherwise compute C once; veto only if C below its fixed threshold.
  Missing/nonfinite C keeps A. C cannot add alerts. No RGB-only override of raw
  sensor data; any native-supported warning removed is explicitly audited.
- Report A, C global, conditional A/C, and diagnostic gate-only suppression;
  the latter tests whether C contributes beyond a cheap veto classifier.
  Compute rod-family oracle only after prediction seal, clearly privileged.
  Report frames/families/pressure, events/onset/release, paired decisions,
  native support, precision/recall/F1, invocation fraction, average/p50/p95/max
  complete warm latency over all288report frames, and invoked/noninvoked costs.
  Cost includes actual conditional token execution, never all-frame cached
  features. Keep model resident; cold model load reported separately. No phone
  claim. Binary policies have no directly comparable continuous PR ranking.
- Development success is useful FP/F1 improvement with recall/event tradeoff
  disclosed; 90%F1 and30/30events are directions, not proof of generalization.
  Failure closes this gate recipe, with no gate tuning on report. Then assess
  richer ToF/CNH input availability separately; no unapproved hardware change
  or immediate large CNH experiment. One run,3600seconds; mechanical recovery
  from sealed caches only. Finish evidence, docs, push and process release.
