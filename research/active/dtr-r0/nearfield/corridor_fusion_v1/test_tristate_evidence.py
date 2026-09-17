import unittest
from tristate_evidence import classify_returns, decisions


def record(*slots):
    return dict(slots=list(slots), usable_slots=sum(s['usable'] for s in slots))


def slot(*faces, completed=True):
    return dict(usable=True, completed=completed, faces=list(faces),
                points=[[f[0][0], f[1][0], f[2][0]] for f in faces])


class EvidenceTests(unittest.TestCase):
    def test_missingness_preserves_independent_alert(self):
        for received in [False, True]:
            state, _ = classify_returns(record(), received, 'full_faces')
            self.assertEqual(state, 'UNKNOWN')
            self.assertTrue(all(decisions(True, state).values()))
            self.assertFalse(any(decisions(False, state).values()))

    def test_unresolved_negative_is_unknown(self):
        r = record(slot([[1, 1], [.5, .6], [1, 2]]), slot(completed=False))
        self.assertEqual(classify_returns(r, True, 'full_faces')[0], 'UNKNOWN')

    def test_positive_takes_precedence(self):
        r = record(slot([[1, 1], [0, .1], [1, 2]]), slot(completed=False))
        state, _ = classify_returns(r, True, 'full_faces')
        self.assertEqual(state, 'POSITIVE')
        self.assertTrue(decisions(False, state)['A_POSITIVE_AND_VETO'])

    def test_disjoint_surfaces_do_not_fill_corridor_gap(self):
        r = record(slot([[1, 1], [-1, -.5], [1, 2]],
                        [[1, 1], [.5, 1], [1, 2]]))
        self.assertEqual(classify_returns(r, True, 'full_faces')[0], 'OUTSIDE_ONLY')

    def test_sampled_and_unsampled_surface_are_distinct(self):
        r = record(slot([[1, 1], [-1, 1], [1, 2]]))
        self.assertEqual(classify_returns(r, True, 'sampled_points')[0], 'OUTSIDE_ONLY')
        self.assertEqual(classify_returns(r, True, 'full_faces')[0], 'POSITIVE')


if __name__ == '__main__':
    unittest.main()
