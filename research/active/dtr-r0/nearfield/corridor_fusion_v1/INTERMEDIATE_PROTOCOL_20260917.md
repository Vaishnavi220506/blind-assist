# E1-I: one frozen intermediate representation diagnostic

The user's continuation authorizes the one diagnostic proposed after E1.
EXPLORE only, all data remains consumed controlled Development. No E2 launch.

Hypothesis: scalar relative-depth summary statistics discarded local information
that remains in DA-V2 early/late patch tokens. This tests information retained
by a different frozen representation, not an independently supervised boundary
head or proof that the information represents a physical boundary.

- Reuse unchanged E1 TRAIN192/dev48/report288 identities and base2485features.
  Original test48 stays excluded. Preserve E1 A/B models and scores as references;
  rerunning their identical three-candidate search is unnecessary.
- Same official Small checkpoint, float32, native640x360, actual518x924 input.
  Read normalized patch tokens after encoder blocks2and11 (zero-based), with
  no decoder forward. Token grids are37x66. Freeze all encoder parameters.
- Sample a2x2 layout in each of the same8projected BODY/HEAD range-plane query
  windows, plus a4x4 whole-image layout:48positioned vectors per layer.
  Bilinear sampling uses native-normalized coordinates with align_corners=False;
  out-of-FOV positions get zero vectors and explicit missing masks. No truth,
  family, identity, native geometry, label, or old alert enters these features.
- Fit8PCA channels per layer on valid TRAIN tokens only, without labels. Concatenate
  all48positions in fixed order:2x48x8+48mask=816new features. Retaining spatial
  positions reduces pooling, but patch resolution/PCA can still lose thin edges.
  PCA is part of the new adapter, not a claim of unchanged model input/output.
- C=base2485+816token features. Same three HGB configurations, seed177017,
  dev F1/AP/config-order tie-break, and dev-recall>=.95 secondary threshold as E1.
  Freeze chosen C before report-label evaluation. No tuning after report access.
- Compare A/B/C against frozen MZ129/MZ145/four-expert working points, with full
  frame, family/pressure, event onset/release, false occupancy, ranking/paired,
  native support and complete inference cost. Nonalert remains UNKNOWN.
- A useful gain could justify a separately scoped E2 decision; no gain pauses
  this DA-V2 online-feature branch. One representation, three prespecified head
  candidates, no capture/seed/threshold/PCA-dimension sweep or automatic successor.
- Output hashes, PCA/head models, per-frame scores, caches, summary and representative
  video under ignored artifacts. At most3600seconds, one task-owned local process,
  resumable authenticated per-frame tokens, failure/completion receipts; release GPU
  on exit. Runtime checks repair mechanics without altering the experiment budget.
