"""Canonical frozen Core alert readout; observation-only, causal and stateful.

This engineering adapter does not select a threshold or change source geometry.
Call reset() before an explicit replay seek. A clip change or non-0.2s interval
also resets continuity automatically. Held alerts do not create current support.
"""
from pathlib import Path
import math
import sys

import numpy as np

NEARFIELD = Path(__file__).resolve().parent.parent
if str(NEARFIELD) not in sys.path:
    sys.path.insert(0, str(NEARFIELD))

from tof_corridor_calibration import score_frame, decide
from tof_fov45_core import boxes45

CALIBRATION_THRESHOLD = .007085703945147101
STRONG_THRESHOLD = .4071309640537889
SAMPLE_INTERVAL_S = .2
_BOXES = boxes45()


class CoreAlertPolicy:
    """One stream of the frozen strong-plus-one-genuine-strong hold policy.

    step(boxes=..., values=..., clip_id=..., frame_id=..., time_s=...) returns
    score, raw support, 64 zone descriptions and an observation-only decision.
    There is no truth, family, layout, threshold or model argument.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = None

    def step(self, *, boxes, values, clip_id, frame_id, time_s):
        if not isinstance(clip_id, str) or not isinstance(frame_id, str):
            raise ValueError('Clip and frame identities must be strings')
        time_s = float(time_s)
        if not math.isfinite(time_s):
            raise ValueError('A finite nominal timestamp is required')
        boxes = np.asarray(boxes)
        values = np.asarray(values, dtype=np.float32)
        if boxes.shape != (64, 4) or not np.array_equal(boxes, _BOXES):
            raise ValueError('Only the frozen original 64-zone geometry is supported')
        if values.shape != (64,):
            raise ValueError('Exactly 64 original zone values are required')
        scored = score_frame(boxes, values)
        calibration = decide(scored, CALIBRATION_THRESHOLD)
        strong = decide(scored, STRONG_THRESHOLD)
        previous = self._previous
        continuous = bool(previous is not None and previous['clip_id'] == clip_id
                          and previous['frame_id'] != frame_id
                          and abs(time_s-previous['time_s']-SAMPLE_INTERVAL_S) < 1e-6)
        previous_strong = bool(continuous and previous['strong'])
        decision = dict(alert=bool(strong['alert'] or previous_strong),
            strong=strong['alert'], calibration=calibration['alert'],
            held_only=bool(previous_strong and not strong['alert']),
            unknown=strong['unknown'], previous_strong=previous_strong,
            previous_frame_id=previous['frame_id'] if continuous else None)
        anchors = {a['zone']: a for a in scored['anchors']}
        factors = {s['zone']: s for s in scored['zone_scores']}
        zones = []
        for zone in range(64):
            anchor, factor = anchors.get(zone), factors.get(zone)
            zones.append(dict(zone=zone,
                interval=anchor['interval_m'] if anchor else None,
                possible=anchor['possible'] if anchor else False,
                definite=anchor['definite'] if anchor else False,
                joint=factor['joint'] if factor else 0.,
                depth=factor['depth'] if factor else 0.,
                angular_given_depth=factor['angular_given_depth'] if factor else None))
        # Store genuine strong only: a held frame never refreshes persistence.
        self._previous = dict(clip_id=clip_id, frame_id=frame_id, time_s=time_s,
                              strong=strong['alert'])
        return dict(score=scored['score'], raw=scored['baseline'], zones=zones,
                    decision=decision)
