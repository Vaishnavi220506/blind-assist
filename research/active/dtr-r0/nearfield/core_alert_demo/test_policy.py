"""Causal edge checks; full saved-cohort parity is performed by build_data.py."""
import unittest
from unittest.mock import patch
import numpy as np
from policy import CoreAlertPolicy, STRONG_THRESHOLD, _BOXES


def scored(score, raw=True, definite=False):
    return dict(score=score, anchors=[], zone_scores=[], baseline=dict(
        alert=raw, definite_zones=int(definite), possible_zones=int(raw),
        valid_zones=64, ambiguous=raw and not definite))


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = CoreAlertPolicy()

    def step(self, sample, frame, time, clip='clip'):
        with patch('policy.score_frame', return_value=sample):
            return self.policy.step(boxes=_BOXES, values=np.ones(64,np.float32),
                clip_id=clip, frame_id=frame, time_s=time)['decision']

    def test_inclusive_threshold_and_one_frame_nonrecursive_hold(self):
        self.assertTrue(self.step(scored(STRONG_THRESHOLD),'a',0)['strong'])
        held = self.step(scored(0,raw=False),'b',.2)
        self.assertTrue(held['alert'] and held['held_only'] and held['unknown'])
        self.assertEqual(held['previous_frame_id'],'a')
        self.assertFalse(self.step(scored(0,raw=False),'c',.4)['alert'])

    def test_clip_gap_seek_and_explicit_reset(self):
        for reason in ('clip','gap','seek','reset'):
            self.policy.reset()
            self.step(scored(.9),'a',1)
            if reason == 'reset':
                self.policy.reset()
            decision = self.step(scored(0,raw=False),'b',
                .8 if reason=='seek' else 1.4 if reason=='gap' else 1.2,
                'another' if reason=='clip' else 'clip')
            self.assertFalse(decision['alert'],reason)
            self.assertIsNone(decision['previous_frame_id'],reason)

    def test_definite_bypass_and_current_unknown(self):
        decision=self.step(scored(0,raw=True,definite=True),'a',0)
        self.assertTrue(decision['strong'] and decision['calibration'])
        self.assertFalse(decision['unknown'])
        decision=self.step(scored(0,raw=False),'b',.2)
        self.assertTrue(decision['held_only'] and decision['unknown'])

    def test_score_without_raw_support_does_not_alert(self):
        self.assertFalse(self.step(scored(.9,raw=False),'a',0)['alert'])

    def test_geometry_and_value_count_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.step(boxes=_BOXES,values=[1]*63,clip_id='x',frame_id='a',time_s=0)
        wrong=_BOXES.copy();wrong[0,0]+=1
        with self.assertRaises(ValueError):
            self.policy.step(boxes=wrong,values=[1]*64,clip_id='x',frame_id='a',time_s=0)


if __name__=='__main__':
    unittest.main()
