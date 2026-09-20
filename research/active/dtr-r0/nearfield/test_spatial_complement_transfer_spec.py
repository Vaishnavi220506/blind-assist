"""New-cohort count/disjointness checks; no saved labels or model outcomes."""
import copy
import unittest

from core_transfer_spec import specification as old_core
from core_workpoint_transfer_spec import specification as old_workpoint
from core_hold_validation_spec import specification as old_hold
from full_event_spec_20260920 import specification as old_full
from spatial_bce_spec import specification as old_bce
from spatial_complement_transfer_spec import specification, check_spec
from launch_spatial_complement_transfer import validate_cases, CAPTURE_BUDGET_SECONDS


class TransferSourceTests(unittest.TestCase):
    def test_new_groups_counts_ranges_and_disjointness(self):
        spec = specification()
        receipt = check_spec(spec, [old_core(), old_workpoint(), old_hold(), old_full(), old_bce()])
        self.assertEqual(receipt['groups'], 16)
        self.assertEqual(receipt['frames'], 1152)
        self.assertEqual(receipt['compared_old_groups'], 40)
        self.assertEqual(receipt['compared_old_cohorts'], 5)
        self.assertEqual(spec, specification())
        validate_cases(spec)
        self.assertEqual(CAPTURE_BUDGET_SECONDS, 1800)

    def test_old_reuse_pair_mutation_and_partition_change_rejected(self):
        spec = specification()
        with self.assertRaises(AssertionError):
            check_spec(spec, [spec])
        changed = copy.deepcopy(spec)
        changed['cases'][24]['objects'][0]['size_m'][0] += .001
        with self.assertRaises(AssertionError):
            check_spec(changed, [])
        changed = copy.deepcopy(spec)
        changed['cases'][0]['split'] = 'train'
        with self.assertRaises(AssertionError):
            check_spec(changed, [])

    def test_capture_mechanics_only_count_adaptation(self):
        from pathlib import Path
        root = Path(__file__).parent
        expected = (root/'spatial_bce_capture.py').read_text().replace('2880', '1152').replace('120 clips', '48 clips').replace('for c in cases})==120', 'for c in cases})==48')
        self.assertEqual((root/'spatial_complement_transfer_capture.py').read_text(), expected)


if __name__ == '__main__':
    unittest.main()
