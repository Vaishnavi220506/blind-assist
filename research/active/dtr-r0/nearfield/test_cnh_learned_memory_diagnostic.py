"""Focused causal-history and calibration-boundary checks for the diagnostic."""
import unittest
import numpy as np
from cnh_learned_memory_diagnostic import rank_score, fit_fusion
from cnh_learned_memory_fusion import causal_ewma


class DiagnosticTests(unittest.TestCase):
    def test_rank_ties_and_outside_reference_are_finite(self):
        score = rank_score(np.array([-100, 1, 1, 2, 100]), [1, 1, 2, 3])
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue((np.diff(score) >= 0).all())
        self.assertEqual(score[1], score[2])

    def test_audit_never_enters_calibration_reference(self):
        def unit(value):
            return dict(main=np.ones(2, bool), y=np.zeros((2, 6)),
                        NN=np.full((2, 6), value), M=np.full((2, 6), value), S2=np.full((2, 6), value))
        units = {0: unit(1), 32: unit(1e9)}
        refs = fit_fusion(units, [0])
        for arms in refs.values():
            for reference in arms.values():
                np.testing.assert_array_equal(reference, np.ones(6))

    def test_future_change_does_not_change_past(self):
        x = np.arange(72).reshape(12, 6).astype(float)
        config, frame = np.zeros(12), np.arange(12)
        for window in (None, 5):
            first = causal_ewma(x, config, frame, window=window)
            changed = x.copy()
            changed[8:] += 1e8
            second = causal_ewma(changed, config, frame, window=window)
            np.testing.assert_array_equal(first[:8], second[:8])

    def test_finite_window_and_sequence_reset(self):
        x = np.zeros((12, 6))
        x[0] = 1000
        config = np.repeat([0, 1], 6)
        frame = np.tile(np.arange(6), 2)
        y = causal_ewma(x, config, frame, window=5)
        np.testing.assert_array_equal(y[5:], np.zeros((7, 6)))
        # Interleaving trajectories must not mix their histories.
        order = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
        mixed = causal_ewma(x[order], config[order], frame[order], window=5)
        np.testing.assert_array_equal(mixed, y[order])


if __name__ == '__main__':
    unittest.main()
