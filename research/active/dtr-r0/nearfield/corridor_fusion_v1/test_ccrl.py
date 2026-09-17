import unittest
import copy
import numpy as np
import torch
from ccrl import CorridorResidual, objective, pairing, local_pairs


class CounterfactualTests(unittest.TestCase):
    def test_zero_initial_residual_preserves_baseline(self):
        model = CorridorResidual()
        base = torch.tensor([-2., 3.])
        self.assertTrue(torch.equal(model(torch.randn(2, 2485), base), base))

    def test_ranking_gradient_has_correct_direction(self):
        z = torch.zeros(2, requires_grad=True)
        y = torch.tensor([1., 0.])
        pairs, empty = torch.tensor([[0, 1]]), torch.empty((0, 2), dtype=torch.long)
        loss = objective(z, y, pairs, empty, True)
        loss.backward()
        self.assertLess(float(z.grad[0]), 0)
        self.assertGreater(float(z.grad[1]), 0)

    def test_pair_authentication_and_exact_invariance(self):
        frame = dict(id='a', episode='in', pair_member='in', camera={'x': 0},
                     time_s=0., sensor_seed=1, objects=[dict(center_m=[2., 0., 1.], size_m=[.1, .1, 2.]), {'wall': 6}])
        out = copy.deepcopy(frame)
        out.update(id='b', episode='out', pair_member='out')
        out['objects'][0]['center_m'][1] = .6
        meta = [dict(id='a', group='g', time_s=0., truth=True), dict(id='b', group='g', time_s=0., truth=False)]
        pairs, inv, rejected = pairing(meta, {'a': frame, 'b': out})
        self.assertEqual(pairs.tolist(), [[0, 1]])
        self.assertEqual(len(inv), 0)
        self.assertFalse(rejected)
        out['sensor_seed'] = 2
        self.assertEqual(len(pairing(meta, {'a': frame, 'b': out})[0]), 0)

    def test_fold_does_not_keep_cross_boundary_pair(self):
        p = np.array([[0, 1], [2, 3]])
        self.assertEqual(local_pairs(p, [0, 2, 3]).tolist(), [[1, 2]])


if __name__ == '__main__':
    unittest.main()
