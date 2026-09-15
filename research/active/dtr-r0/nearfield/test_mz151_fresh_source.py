import copy
import unittest
import mz136_paired_source as inherited
from mz151_fresh_source import at_seed,check_source,geometry_signatures,source

class SourceTests(unittest.TestCase):
    def test_fixed_source_is_disjoint_and_restores_generator(self):
        original=inherited.SEED;spec=source()
        self.assertEqual(inherited.SEED,original)
        self.assertEqual(spec,source())
        audit=check_source(spec)
        self.assertEqual(audit['frames'],288)
        self.assertEqual(audit['positive_frames'],144)
        self.assertEqual(audit['split_frames'],{'confirmation':288})
        self.assertEqual(len({f['scene_group'] for f in spec['frames']}),24)
        for seed in (136014,146016):
            self.assertFalse(geometry_signatures(spec)&geometry_signatures(at_seed(seed)))
        self.assertEqual(inherited.SEED,original)

    def test_partition_and_pair_geometry_changes_are_rejected(self):
        spec=source();bad=copy.deepcopy(spec);bad['frames'][0]['split']='test'
        with self.assertRaises(AssertionError):check_source(bad)
        bad=copy.deepcopy(spec);bad['frames'][0]['objects'][0]['center_m'][0]+=1
        with self.assertRaises(AssertionError):check_source(bad)

if __name__=='__main__':unittest.main()
