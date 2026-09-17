import copy
import unittest
from confirmation_source import source,check_source


class SourceChecks(unittest.TestCase):
    def test_design_is_deterministic_and_complete(self):
        a=source();self.assertEqual(a,source())
        self.assertEqual(check_source(a)['positive_frames'],144)

    def test_background_cannot_be_exempted_from_truth(self):
        spec=source()
        spec['frames'][0]['objects'][1].update(center_m=[2.,0.,1.],size_m=[.1,2.,2.])
        with self.assertRaises(AssertionError):check_source(spec)

    def test_nonlateral_pair_change_is_rejected(self):
        spec=source();spec['frames'][0]['objects'][0]['tof_reflectance_proxy']=.12345
        with self.assertRaises(AssertionError):check_source(spec)


if __name__=='__main__':unittest.main()
