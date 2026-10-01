import unittest
import numpy as np
import cnh_corridor_labels as C
import cnh_proposal_attribution_scenes as S
from cnh_track_a_readout import BOXES


class LabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.scenes=S.make_scenes(1000)

    def test_six_query_order_and_central_compatibility(self):
        np.testing.assert_array_equal(C.QUERY_BOXES,BOXES)
        for s in self.scenes:
            y=C.labels_for_all(s['boxes'],s['travel'])
            self.assertEqual(y.shape,(16,6))
            np.testing.assert_array_equal(y[:,[2,3]],s['labels'])
            yopen=C.labels_for_all(s['boxes'],s['travel'],boundary='interior')
            np.testing.assert_array_equal(yopen[:,[2,3]],s['labels'])

    def test_boundary_mask_only_excludes_target_height_group(self):
        m=C.central_margin_masks([0,1,0,1,0],[-.015,.045,.12,-.05,.050001])
        np.testing.assert_array_equal(m['target_near_band'],[[1,0],[0,1],[0,0],[0,1],[0,0]])
        self.assertTrue(m['strict'].all())
        np.testing.assert_array_equal(m['outside_band'],~m['target_near_band'])
        self.assertEqual(C.penetration_stratum(np.array([-.049,-.05,-.149,-.15,-.30,.015,0])).tolist(),
            ['inside_lt5cm','inside_5to15cm','inside_5to15cm','inside_15to30cm','inside_15to30cm','outside','touch'])

    def test_face_contact_is_an_explicit_label_choice(self):
        b=[dict(lo=[.35,.42,1.],hi=[.55,.7,1.2],rho=.5)]
        pose=np.eye(4)[None]
        closed=C.labels_for_all(b,pose,boundary='closed')
        interior=C.labels_for_all(b,pose,boundary='interior')
        self.assertEqual(closed[0,4],1)
        self.assertEqual(interior[0,4],0)
        self.assertEqual(interior[0,5],1)
        np.testing.assert_array_equal(closed[:,[2,3]],interior[:,[2,3]])

    def test_empty_and_invalid_inputs(self):
        np.testing.assert_array_equal(C.labels_for_all([],np.eye(4)[None]),np.zeros((1,6)))
        with self.assertRaises(ValueError):C.central_margin_masks([0,2],[.1,.2])
        with self.assertRaises(ValueError):C.labels_for_all([],np.eye(4))

    def test_turning_pure_contact_can_change_intermediate_central_label(self):
        s=S.make_scenes(1004)[7]
        closed=C.labels_for_all(s['boxes'],s['travel'],boundary='closed')
        interior=C.labels_for_all(s['boxes'],s['travel'])
        np.testing.assert_array_equal(closed[:,[2,3]],s['labels'])
        self.assertTrue((interior<=closed).all())
        np.testing.assert_array_equal(interior[-1,[2,3]],s['labels'][-1])
        diff=closed[:,2]-interior[:,2]
        np.testing.assert_array_equal(np.flatnonzero(diff),np.arange(8,15))


if __name__=='__main__':unittest.main()
