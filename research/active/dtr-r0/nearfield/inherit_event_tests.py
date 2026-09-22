"""Analytic decoder controls: no UE data or algorithm-effect claims."""
import unittest

from inherit_event_core import Support, SupportInheritance, THRESHOLD


def sup(zone=27, value=2., lo=1.8, hi=2.2, joint=.3, possible=True):
    return Support(zone, value, lo, hi, possible, False, joint)


class InheritanceTests(unittest.TestCase):
    def run_sequence(self, rows):
        model = SupportInheritance()
        return [model.step(str(i), clip, index, time, current, supports)
                for i, (clip, index, time, current, supports) in enumerate(rows)]

    def test_birth_threshold_equal_and_current_preserved(self):
        row = self.run_sequence([('c', 0, 0, True, [sup(joint=THRESHOLD)])])[0]
        self.assertTrue(all(row['flags'].values()))

    def test_nonrecursive_inheritance_and_unchanged_old_hold(self):
        rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                 ('c', 1, 200_000_000, False, [sup()]),
                                 ('c', 2, 400_000_000, False, [sup()])])
        self.assertEqual([r['flags']['support_inherit'] for r in rows], [True, True, False])
        self.assertEqual(rows[1]['seed_id'], '0')
        self.assertIsNone(rows[2]['seed_id'])

    def test_missing_then_reappearance_cannot_refresh_seed(self):
        rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                 ('c', 1, 200_000_000, False, []),
                                 ('c', 2, 400_000_000, False, [sup()])])
        self.assertEqual([r['flags']['support_inherit'] for r in rows], [True, False, False])

    def test_second_target_different_zone_requires_new_birth(self):
        rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                 ('c', 1, 200_000_000, False, [sup(zone=28)]),
                                 ('c', 2, 400_000_000, True, [sup(zone=28, joint=.9)])])
        self.assertEqual([r['flags']['support_inherit'] for r in rows], [True, False, True])

    def test_same_zone_indistinguishable_second_target_is_not_resolved(self):
        # Explicit limitation: public returns cannot distinguish such a substitution.
        rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                 ('c', 1, 200_000_000, False, [sup()])])
        self.assertTrue(rows[1]['inherited'])

    def test_interval_disjoint_or_current_nominal_exit_releases(self):
        for later in (sup(value=2.6, lo=2.4, hi=2.8),
                      sup(value=3.01, lo=1.9, hi=3.3), sup(possible=False)):
            rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                     ('c', 1, 200_000_000, False, [later])])
            self.assertFalse(rows[1]['inherited'])

    def test_expiry_and_clip_reset(self):
        for clip, time in (('c', 200_000_001), ('new', 200_000_000)):
            rows = self.run_sequence([('c', 0, 0, True, [sup(joint=.8)]),
                                     (clip, 1, time, False, [sup()])])
            self.assertFalse(rows[1]['inherited'])

    def test_nonmonotonic_clock_rejected(self):
        with self.assertRaises(ValueError):
            self.run_sequence([('c', 0, 20, True, [sup(joint=.8)]),
                               ('c', 1, 20, False, [sup()])])


if __name__ == '__main__':
    unittest.main()
