"""Rectification geometry and explicit loss of fine-edge availability."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from mz136_boundary_geometry import camera_to_body
from mz136_rectified_edges import peak_subpixel, rectification, refine, transform


class RectifiedEdgesTests(unittest.TestCase):
    def row(self):
        return dict(id='public-frame', camera_pitch_deg=-3.,
                    camera_in_body_m=[0., 0., 1.7], tof_packet_received=False,
                    tof_zones=[], rgb_intrinsics=dict(width=640, height=360,
                    fx=457., fy=457., cx=320., cy=180.))

    def test_world_vertical_and_projective_roundtrip(self):
        row=self.row(); yaw=7.
        body=np.array([[2., .4, -.8], [2., .4, 0.], [2., .4, .3]])
        camera=body@camera_to_body(row, yaw)
        pixels=np.c_[320.+457.*camera[:,1]/camera[:,0],
                     180.-457.*camera[:,2]/camera[:,0]]
        H=rectification(row, yaw); aligned=transform(pixels, H)
        np.testing.assert_allclose(aligned[:,0], 320.+457.*.4/2., atol=1e-10)
        np.testing.assert_allclose(transform(aligned, np.linalg.inv(H)), pixels, atol=1e-10)

    def test_signed_subpixel_peak_and_unavailable_neighbor(self):
        x=np.arange(7); peak=9.-(x-3.25)**2
        self.assertAlmostEqual(peak_subpixel(peak, 3, 1), 3.25)
        self.assertAlmostEqual(peak_subpixel(-peak, 3, -1), 3.25)
        peak[2]=np.nan
        self.assertEqual(peak_subpixel(peak, 3, 1), 3.)

    def test_missing_seed_is_unavailable_and_nonmutating(self):
        row=self.row(); before=copy.deepcopy(row)
        result=refine(row, np.zeros((360,640,3), np.uint8), 0.)
        self.assertEqual(result['state'], 'NO_OBSERVABLE_PROPOSAL')
        self.assertIsNone(result['candidate'])
        self.assertEqual(row, before)

    def test_failed_refinement_retains_coarse_seed(self):
        seed=dict(box=[340,70,350,280], zones=[1], authority='COARSE_HYPOTHESIS')
        with patch('mz136_rectified_edges.proposals', return_value=([seed], [])):
            result=refine(self.row(), np.zeros((360,640,3), np.uint8), 0.)
        self.assertEqual(result['state'], 'INSUFFICIENT_SIGNED_EDGE_SUPPORT')
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['seed'], seed)


if __name__=='__main__':
    unittest.main()
