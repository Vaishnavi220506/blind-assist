import copy
import unittest
from mz136_paired_source import source, check_source


class SourceTests(unittest.TestCase):
    def test_deterministic_group_partition_and_lateral_pairs(self):
        s=source();self.assertEqual(s,source());a=check_source(s)
        self.assertEqual(a['split_frames'],dict(train=192,dev=48,test=48))
        self.assertEqual((a['positive_frames'],a['negative_frames']),(144,144))

    def test_mismatched_pair_appearance_rejected(self):
        s=source();s['frames'][6]['objects'][0]['texture_seed']+=1
        with self.assertRaises(AssertionError):check_source(s)

    def test_context_intrusion_cannot_be_excluded_from_labels(self):
        s=source();s['frames'][0]['objects'][1]['center_m']=[2.,0.,1.]
        with self.assertRaises(AssertionError):check_source(s)

    def test_pair_variant_split_leak_rejected(self):
        s=source()
        for f in s['frames'][6:12]:f['split']='test'
        with self.assertRaises(AssertionError):check_source(s)


if __name__=='__main__':unittest.main()
