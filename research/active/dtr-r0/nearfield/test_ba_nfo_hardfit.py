import unittest
import numpy as np
from ba_nfo_hardfit import zone_info


class TestCaseSelection(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2)
        self.rgb=rng.integers(0,256,size=(20,25,3),dtype=np.uint8)

    def test_separated_support_and_threshold_slice(self):
        d=np.full((20,25),3.,np.float32);d[:2,:10]=1.5
        self.assertTrue(zone_info(self.rgb,d,3.)['positive'])
        d[:2,:10]=1.95
        self.assertFalse(zone_info(self.rgb,d,3.)['positive'])
        d[:2,:10]=1.5
        self.assertFalse(zone_info(self.rgb,d,np.nan)['positive'])

    def test_unknown_and_textured_negative(self):
        d=np.full((20,25),3.,np.float32)
        self.assertTrue(zone_info(self.rgb,d,3.)['negative'])
        self.assertFalse(zone_info(np.zeros_like(self.rgb),d,3.)['negative'])
        d[:3]=np.nan
        self.assertFalse(zone_info(self.rgb,d,3.)['negative'])
        d[:]=np.nan
        info=zone_info(self.rgb,d,3.)
        self.assertFalse(info['negative']);self.assertFalse(info['positive'])


if __name__=='__main__':unittest.main()
