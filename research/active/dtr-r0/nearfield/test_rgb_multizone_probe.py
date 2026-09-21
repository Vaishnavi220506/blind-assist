import unittest
import numpy as np
from rgb_multizone_probe import connected_pair, extent_relation, probe
from tof_fov45_core import boxes45


class MultiZoneTest(unittest.TestCase):
    def test_diagonal_and_row_wrap_are_not_adjacent(self):
        self.assertFalse(connected_pair({7,8}))
        self.assertFalse(connected_pair({10,19}))
        self.assertTrue(connected_pair({10,18}))

    def test_complete_interval_changes_lateral_claim(self):
        self.assertEqual(extent_relation(250,270,[2.,3.]),'OUTSIDE')
        self.assertEqual(extent_relation(250,270,[1.,3.]),'CROSSING')

    def test_uniform_clipped_region_cannot_anchor(self):
        zones=[dict(valid=True,interval_m=[1.8,2.2],depth_state='CONTAINED',possible=True,horizontal_relation='CROSSING') for _ in range(64)]
        result,_=probe(np.full((360,640,3),128,np.uint8),boxes45(),zones)
        self.assertEqual(sum(c['eligible'] for c in result['components']),0)
        self.assertTrue(all(q['relation']=='UNKNOWN' for q in result['queries']))


if __name__=='__main__':unittest.main()
