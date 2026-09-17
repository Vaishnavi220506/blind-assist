import unittest
import numpy as np
import torch
from public_positive_v2 import nested_folds, fitting_weights, negative_peak_loss


class DevelopmentTests(unittest.TestCase):
    def test_nested_group_exclusion(self):
        meta = [dict(cohort=c, scene_index=s, group=f'{c}/{s}')
                for c in ['anchor', 'old', 'new', 'mz146', 'mz158'] for s in range(6)
                for _ in range(2)]
        seen = []
        for f in nested_folds(meta):
            fi, ri = set(f['fit']), set(f['report'])
            self.assertFalse(fi & ri)
            cal_seen = []
            for inner in f['inner']:
                ti, ci = set(inner['fit']), set(inner['calibration'])
                self.assertEqual(ti | ci, fi)
                self.assertFalse(ti & ci)
                self.assertFalse({meta[i]['group'] for i in ti} & {meta[i]['group'] for i in ci})
                cal_seen += inner['calibration']
            self.assertEqual(sorted(cal_seen), [i for i in f['fit'] if meta[i]['cohort'] != 'anchor'])
            seen += f['report']
        self.assertEqual(sorted(seen), [i for i,m in enumerate(meta) if m['cohort'] != 'anchor'])

    def test_peak_gradient_masks_and_sparse_return(self):
        z = torch.tensor([[1., 9., 2.], [3., 0., 0.], [8., 8., 8.],
                          [5., 4., 3.], [6., 5., 4.]], requires_grad=True)
        valid = torch.tensor([[True, False, True], [True, False, False],
                              [False, False, False], [True, True, True], [True, True, True]])
        loss = negative_peak_loss(z, valid, torch.tensor([.5, .5, 0., 0., 0.]))
        loss.backward()
        self.assertEqual(torch.nonzero(z.grad).tolist(), [[0, 2], [1, 0]])
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_weights_exclude_nonnegative_and_report(self):
        meta = [dict(A=a, truth=y, stratum=s) for a,y,s in
                [(False,False,'negative'), (False,True,'positive'),
                 (True,False,'negative'), (None,False,'negative'),
                 (False,False,'boundary'), (False,False,'negative'),
                 (False,False,'negative')]]
        valid = np.ones((7,2), bool); valid[5] = False
        target = np.zeros((7,2), np.float32); target[1,0] = 1
        w,p = fitting_weights(meta, list(range(6)), valid, target, valid)
        np.testing.assert_allclose(w.sum(), 1., atol=1e-6)
        np.testing.assert_array_equal(p, [1,0,0,0,0,0,0])
        self.assertFalse(w[6].any())


if __name__ == '__main__':
    unittest.main()
