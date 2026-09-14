# MZ135: fixed-hardware causal temporal correspondence diagnostic

Frozen before this run's outputs. User steering prioritizes past 3–5 frames of
the same RGB + 8×8 ToF + Radar + IMU, without training or extra hardware.
This run uses current plus four past frames, never future frames. All 288 MZ123
frames are consumed Development; MZ129 is the frame-output baseline. MZ134's
single-frame weak-source model is a negative comparator, not a new baseline.

Question: do observed short-window pixel tracks supply source-specific spatial
constraints that can withdraw off-corridor false support without assuming that
every beam sees the same reflector? MZ119 established a concrete failure:
coherently moving anchors can masquerade as camera translation and corrupt
background depth. Therefore this run never converts IMU or optical flow to metric
camera translation and never labels a fit score a probability.

One frozen method uses forward/backward-checked LK tracks (256 features, 0.5px
closure, 21px window, three pyramid levels, seven-pixel feature spacing).
For each current VALID return, past tracks inside the current native zone link
to every prior VALID return whose native zone contains the earlier feature.
Range compatibility allows the two 3-sigma measurement intervals plus an explicit
2m/s relative-range nuisance bound over elapsed time. This is only a hypothesis
filter, not a calibrated speed limit or identity certificate.

For each reference, translate all matched prior zone rectangles by the measured
feature flow with a fixed 2px pad; union alternatives by their bounding box, then
intersect across at least two past references and the current saved ROI. Retain
the current ROI's outside-image portion. This produces a **conditional local-flow
envelope**, explicitly assuming a shared reflecting surface, locally translational
flow across its footprint, and completeness of the tracked explanations.
It is the experimental candidate, not a deployment claim. Conflicting intersections
fall back; MERGED and untracked returns stay unresolved. No thresholds change
after scoring.

Wrong feature-to-ToF association, untracked/current-only reflectors within a
matched beam, and nonuniform surface flow remain explicit model risks. They do
not automatically veto the candidate: positive temporal associations gain new
decision authority and may experimentally withdraw prior support. Raw returns
remain stored, and unmatched returns retain their current support. The unchanged
MZ129 per-return readout is a conservative control, not an unconditional union
with the candidate. MZ129 independent
Radar/raw/carry/guards remain the controlled unchanged branch in this ToF test;
report ToF-only so they cannot hide temporal effects. This is not a finished
joint Radar-motion model; missing Doppler remains missing.

Record all frame/family metrics, BODY/rod support reach, lost TP, every event's
first alert, UNKNOWN, false segments, and native contributor retention for both
control and candidate envelopes. Report actual tracks, five-frame coverage,
IMU-compensated pixel residuals and cross-zone track movement. Native source IDs
may audit proposed links only after predictions are sealed. Their mismatch rate
does not establish identity for matching actors or pixels.

Falsifiers: stationary observations cannot manufacture a finer bound; coherent
object motion and camera motion can have identical relative observations; a
roundtrip-consistent image track can belong to a different surface than a zone's
returned ToF component. These disclose the limits of the conditional candidate;
they are not blanket gates preventing a measured experiment.
Candidate retention requires FP improvement with no lost TP, no later event
alert and no newly discarded native corridor contributor. Report measured gains
or failures and the exact assumptions, not an impossibility theorem or fabricated
posterior improvement. No numerical probability is estimated.

Output only under `artifacts.local/work/mz135-temporal-dealias-20260914/`.
No training, parameter search, source expansion, default change or successor.
