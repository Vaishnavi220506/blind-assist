import unittest
from unittest.mock import patch
import numpy as np
from mz125_observable_correction import foreground, predict_frame, allocation_alert


class CorrectionTests(unittest.TestCase):
    def test_constant_has_no_positive_foreground(self):
        boxes, mask=foreground(np.full((90,160,3),90,np.uint8))
        self.assertEqual(boxes,[]);self.assertEqual(int(mask.sum()),0)

    def test_bright_clipped_object_recovers_on_smooth_background(self):
        x=np.linspace(-1,1,160);base=(70-8*x*x)[None,:]+np.zeros((90,1))
        image=np.repeat(base[...,None],3,axis=2);image[10:,110:140]+=55
        boxes,_=foreground(image.astype(np.uint8))
        self.assertTrue(any(b[0]<=111 and b[2]>=139 and b[1]<=11 and b[3]>=89 for b in boxes))
        shifted,_=foreground((image+20).astype(np.uint8))
        self.assertEqual(boxes,shifted)

    def test_missing_new_box_retains_old_proposals_and_radar(self):
        cached=dict(proposals=[[10,10,40,40]],common_radar=True,guard_events=[],integrated_yaw_deg=0.)
        with patch('mz125_observable_correction.foreground',return_value=([],np.zeros((10,10),np.uint8))),\
             patch('mz125_observable_correction.tof.allocate',return_value=[]) as allocate:
            result,_=predict_frame({},cached,np.zeros((10,10,3),np.uint8))
        self.assertEqual(allocate.call_args.args[1]['proposals'],cached['proposals'])
        self.assertTrue(result['candidate']);self.assertTrue(result['fallback'])

    def test_certain_coarse_support_survives_localized_disjoint(self):
        e=dict(coarse_xyz=[(1.,1.1),(-.1,.1),(1.,1.1)],localized_xyz=[(8.,9.),(0.,1.),(1.,2.)])
        self.assertTrue(allocation_alert(dict(common_radar=False,guard_events=[]),[e]))


if __name__=='__main__':unittest.main()
