"""Source-only paired intervention checks; no scientific observations consumed."""
import copy
import unittest
from local_transfer_spec import specification, check_spec
from query_occupancy_spec import specification as old_spec


class SourceTests(unittest.TestCase):
    def test_counts_new_geometry_and_background_only(self):
        spec=specification()
        result=check_spec(spec,[old_spec()])
        self.assertEqual(result['appearance_pairs'],288)
        self.assertEqual(result['geometry_triples'],192)
        self.assertEqual(result['counts']['base'],dict(frames=288,positives=128))
        self.assertEqual(spec,specification())

    def test_target_appearance_intervention_rejected(self):
        spec=specification()
        for c in spec['cases'][:12]:
            c['objects'][0]['material']='/Game/StreetLab/Materials/Terracotta'
        with self.assertRaises(AssertionError):check_spec(spec)

    def test_camera_or_noise_mismatch_rejected(self):
        for key in ('camera','sensor_noise_key'):
            spec=specification()
            if key=='camera':spec['cases'][0]['camera']['y']+=.001
            else:spec['cases'][0][key]+='unpaired'
            with self.assertRaises(AssertionError):check_spec(spec)

    def test_old_geometry_and_wrong_truth_rejected(self):
        spec=specification()
        with self.assertRaises(AssertionError):check_spec(spec,[copy.deepcopy(spec)])
        spec['cases'][0]['layout_relation']='BOUNDARY'
        spec['cases'][0]['frame_in_clip']=3
        with self.assertRaises(AssertionError):check_spec(spec)


if __name__=='__main__':unittest.main()
