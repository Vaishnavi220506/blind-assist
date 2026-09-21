import unittest
from run_conditional_hold import decode


class ConditionalHoldTest(unittest.TestCase):
    def test_weak_gap_and_release(self):
        self.assertEqual(decode([True, False, True, False], [.75, .42, .71, .06], .4),
                         [True, True, True, False])

    def test_recursive_tail_and_equality(self):
        self.assertEqual(decode([True, False, False, False], [.8, .4, .4, .39], .4),
                         [True, True, True, False])

    def test_no_start_and_no_cross_clip_state(self):
        self.assertEqual(decode([False, False], [.6, .6], .4), [False, False])
        self.assertEqual(decode([True], [.8], .4), [True])
        self.assertEqual(decode([False], [.6], .4), [False])

    def test_current_trigger_preserved(self):
        self.assertEqual(decode([True, True, False], [.8, -.9, -.8], .4), [True, True, False])


if __name__ == '__main__':
    unittest.main()
