"""Decision-critical checks: atomic thresholds, silence rejection, causal hold."""
import unittest
import numpy as np
from run_spatial_bce import held, select_threshold, threshold_record, metrics, ARMS


def fixture():
    meta = []
    for name in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
        for i in range(4):
            meta.append(dict(id=f'{name}{i}', index=len(meta), clip_id=name, frame_in_clip=i,
                time_s=.2*i, layout_relation=name, layer='HEAD', background='test',
                phase='approach' if i < 2 else 'depart'))
    truth = np.array([False, True, True, False] * 2 + [False] * 4)
    baseline = np.array([False, True, True, False] + [False] * 8)
    return meta, truth, baseline


class DecisionTests(unittest.TestCase):
    def test_hold_does_not_recur_or_cross_clips(self):
        meta, _, _ = fixture()
        np.testing.assert_array_equal(held([True] + [False]*11, meta), [True, True] + [False]*10)
        values = np.zeros(12, bool); values[3] = True
        self.assertFalse(held(values, meta)[4])

    def test_silence_is_not_admissible(self):
        meta, truth, a = fixture()
        r = threshold_record(1, np.zeros(12), truth, meta, a)
        self.assertFalse(r['admissible'])
        self.assertFalse(r['checks']['retain_Core_current'])

    def test_atomic_ties_cannot_be_split_for_better_scores(self):
        meta, truth, a = fixture()
        selected, records = select_threshold(np.zeros(12), truth, meta, a)
        self.assertIsNone(selected)
        self.assertEqual(len(records), 2)

    def test_boundary_gain_cannot_hide_held_exit_fp(self):
        meta, truth, a = fixture()
        logits = np.where(truth, 2., -2.)
        # B could gain Boundary current TPs without raw FP, but same hold adds an
        # exit FP where the silent Boundary baseline has no FP budget.
        selected, _ = select_threshold(logits, truth, meta, a)
        self.assertIsNone(selected)
        r = threshold_record(2, logits, truth, meta, a)
        self.assertFalse(r['checks']['Boundary_hold_FP'])

    def test_admissible_gain_and_unknown_preserved(self):
        meta, truth, a = fixture()
        logits = np.where(a, 2., -2.)
        logits[5] = 2.  # Boundary first positive; hold ends on final positive.
        selected, _ = select_threshold(logits, truth, meta, a)
        self.assertIsNotNone(selected)
        self.assertEqual(selected['summary']['Boundary_current']['B_TP'], 1)
        rows = []
        flags = dict(A_current=a, B_current=logits >= 2, A_hold=held(a, meta), B_hold=held(logits >= 2, meta))
        for i, r in enumerate(meta):
            rows.append(dict(**r, truth=bool(truth[i]), flags={k: bool(v[i]) for k, v in flags.items()},
                current_unknown={k: True for k in ARMS}))
        result = metrics(rows)
        self.assertEqual(result['Core']['arms']['B_current']['frames']['TN'], 0)
        self.assertEqual(result['Boundary']['arms']['B_current']['frames']['TP'], 1)
        self.assertEqual(result['Boundary']['arms']['B_hold']['frames']['TP'], 2)


if __name__ == '__main__':
    unittest.main()
