# S1 acquisition recovery: complete the unchanged confirmation

The user corrected the self-imposed1200second capture cutoff. That value came
from a reused engineering helper, not the user's requested experiment budget.
The root agent incorrectly treated it as immutable scientific methodology and
stopped useful work despite healthy capture progress. This correction authorizes
completion of the original request without another approval step.

The failed first capture and its diagnostics remain intact. Its RGB-only output
cannot resume the lost in-memory public/evaluator records; perform a complete
capture of the **identical288frames and24configurations**, with the original
spec SHA256 `9a3efcded95b7b42c630eb8ede9962bd1ecd6528be83ceb71740f93dbc661025`
and model recipe SHA256 `ccb106baca7427c6b2573d4fc46a2cc28c10036bba60be7427ea55810722352c`.

No model, gate, threshold, PCA, scene geometry, texture, seed, render settings,
sensor simulation or scoring criterion changes. New capture payload is isolated
under `artifacts.local/work/corridor-depth-confirmation-recovery-20260917/`;
evaluation uses the originally frozen runner and original recipe directory.
No partial scores from the first attempt were available or used for selection.

Engineering change: adapt only the portable launch and process wait to accept
`timeout=None`. Preserve process-tree tracking and `finally` cleanup. Monitor
frame growth, compiler/process activity and error receipts. Investigate actual
stalls or errors; healthy progress is not stopped solely by elapsed wall time.
The old design freeze still records the original allowance as historical data;
the new bundle's `recovery.json`, launcher record and this amendment explicitly
supersede that engineering allowance and the earlier no-retry disposition.

Original source/model identities remain the scientific freeze. The failed
attempt is not erased, promoted or treated as algorithm-negative evidence.
Finish admission, sealed prediction, requested metrics and scoped delivery.
