import copy
import unittest
import mz155_active_sampling as source


class SourceContrast(unittest.TestCase):
    def test_stationary_matched_design(self):
        spec=source.source()
        self.assertEqual(source.check_source(spec)['episodes'],48)
        self.assertEqual(spec,source.source())
        self.assertEqual(source.inherited.SEED,136014)

    def test_unpaired_geometry_rejected(self):
        spec=source.source()
        row=next(f for f in spec['frames'] if f['observation_arm']=='scan')
        row['objects'][0]['center_m'][0]+=.01
        with self.assertRaises(AssertionError):source.check_source(spec)


if __name__=='__main__':unittest.main()
