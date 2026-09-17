# Existing-return tri-state evidence probe

EXPLORE; evaluator-only diagnostic on the same consumed 288 changed-domain
frames. No new capture, fit, threshold selection, model changes or public-input
method claim. Reuse frozen A decisions and the independently audited returned
native points/faces. The prior fixed-A support-field intervention was inert;
this probe changes the decision readout instead of A's feature vector.

Question: how much error can existing returned spatial evidence resolve when
zero returns never count as clear? Preserve the original nominal corridor
x=[0.2,3.6], y=[-0.3,0.3], z=[0.4,2.05] metres. Report the adopted 5 cm clear
subset (216/288), strict 288 and boundary 72 separately.

Freeze these rules before computing this probe's outcomes:

- P: at least one completed usable return's exact native face intersects the
  corridor. Use the union of faces, never its enclosing AABB. Positive evidence
  takes precedence even if another slot is unresolved.
- N_candidate: received packet, at least one usable return, every usable slot
  fully resolved, and no P. This proves only that all returned surfaces are
  outside, not that the corridor was observed exhaustively. No minimum count,
  coverage threshold or actor/case selection will be fitted. Treat the resulting
  veto as a deliberately permissive diagnostic of the negative-evidence idea.
- U: otherwise. Missing packets, received packets without usable returns, and
  unresolved support without P are UNKNOWN. Every arm falls back to A on U.

Repeat once with exact linked sampled points replacing full faces. A missing
sampled point likewise cannot certify free space. This is the existing
sampled/full contrast, not an additional model search.

Four arms per representation: A; A OR P; A AND NOT N_candidate;
(A OR P) AND NOT N_candidate. Readout inputs are A's frozen decision, packet
presence, usable/resolved slot flags and native points/faces only. Evaluation
labels, family, error identity and unreturned rays never select P/N/U or a veto.
Seal per-frame states and decisions before joining prior evaluation/diagnosis.
The cohort and its previous oracle outcomes are already consumed; sealing does
not create independence or prospective confirmation authority.

Report TP/FP/FN, precision/recall/F1, family transitions, gained/lost A true
frames, UNKNOWN fallback retention, core event detection/first alert, negative
alert burden and strict/boundary controls. Inspect any vetoed true frame to
distinguish observed outside support from an unobserved corridor obstacle.
Preserve all contributors; decisions do not modify geometry. Use one independent
reconstruction check and focused UNKNOWN/union/positive-priority tests.

If useful, retain an evaluator-only opportunity for a later public-feature
spatial evidence readout. If veto harms a true frame, disclose it even when F1
improves; an outside-return state is not automatically strong negative evidence.
No result authorizes a training run, sensor change, new collection, or App update.
Small scalar CPU computation; record backend and elapsed analysis time without
claiming deployable latency. Finish report and scoped delivery after this probe.
