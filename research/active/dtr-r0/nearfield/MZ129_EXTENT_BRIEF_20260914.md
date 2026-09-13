# MZ129: consumed corridor-extent correction

EXPLORE. User authorized continuing the MZ128 false-positive diagnosis with a
bounded correction. Hypothesis: overly broad interval enclosures and inherited
visual boxes contribute to off-corridor alerts; correcting their geometry can
reduce nuisance while retaining near-obstacle support.

Use exactly the existing 288 MZ123 frames, frozen MZ125 boxes/allocations and
MZ128 binary-four readout. This is reused controlled Development evidence, not
fresh validation. No capture, training, parameter sweep, source expansion or
integration of the separate 60-packet directional component.

Freeze four arms before scoring:

- Baseline: reproduce every MZ128 binary-four output.
- ToF: apply the existing MZ124 depth-six interval subdivision to each MZ125
  localized ROI with its unchanged range, pitch and yaw bounds. A return loses
  possible support only when all subdivided enclosures are disjoint. Budget
  exhaustion remains possible. Keep all records, the binary weights, certain
  coarse override, original Radar and guards. MERGED retains its full .02–4 m
  range; absent evidence never becomes CLEAR.
- Radar: recompute the frozen MZ111 plane/range-filter branch using MZ125 boxes.
  Reconstruct the complete original MZ113 flow carry from the original images
  and boxes, including support hidden by the original current-Radar OR. First
  require exact reproduction of all original common-Radar bits. Preserve that
  carry, original and corrected-box resolution guards, and explicit current raw
  center-in-corridor support independently. Report added raw protection as a
  coupled change; this is not a box-only causal contrast.
- Combined: use both corrections with identical protections.

MZ124 already found only three inherited-branch FP exclusions on whole-zone
MZ116, with increased fragmentation; preserving its broad raw-Radar uncertainty
branch erased the gain. That negative result stays intact. The changed input
here is MZ125 localized ROIs and MZ128 binary alert-use, plus corrected Radar
boxes. This experiment inherits MZ128's working geometry; it does not certify
physical clearance or test MZ124's distinct broad ±12-degree raw-null policy.

Persist observations-only predictions, input/code hashes and this brief before
the scoring stage reads labels. Report TP/FP/FN, precision, family counts,
individual event timing, false segments/duration, changed frames, and native
attribution of excluded ToF support. Accept a scoped candidate only if FP fall,
all 139 baseline TP remain, all 15 events remain, and no event alert is later.
Report nuisance fragmentation even if the frame-count criterion passes. Null
or tradeoff results remain diagnostics; MZ116/default application is unchanged.

One fixed replay ends the experiment. Mechanical failures may be repaired with
attempt logs; no outcome-dependent tuning. Small scalar geometry uses CPU
(TASK_NOT_GPU_SUITABLE); frozen optical-flow reconstruction uses the existing
CPU implementation, with separately reported runtime. Artifacts and all
temporary material stay under artifacts.local/work/mz129-extent-correction-20260914.
