import unittest
import numpy as np
from diagnose_ba_depth_support import transitions


class DiagnosticTests(unittest.TestCase):
    def test_net_losses_are_not_new_losses(self):
        gt=np.ones((1,4))
        raw=np.array([[1.,3.,3.,1.]])
        pred=np.array([[3.,1.,3.,1.]])
        m=transitions(gt,raw,pred,np.ones((1,4),bool))
        self.assertEqual([int(m[k].sum()) for k in ('new_fn','rescued_fn','shared_fn')],[1,1,1])

    def test_quantile_can_pass_while_correct_surface_is_missing(self):
        # A compatible patch elsewhere fulfills existence without recovering the true patch.
        pred=np.full((10,10),4.);pred[:2,:]=1.4
        gt=np.full((10,10),4.);gt[8:,:]=1.4
        raw=np.full((10,10),1.4)
        m=transitions(gt,raw,pred,np.ones_like(gt,bool))
        self.assertEqual(m['new_fn'].sum(),20)
        self.assertLessEqual(np.quantile(pred,.1),1.5)
        # The one-sided constraint also accepts a wrongly much closer surface.
        self.assertLessEqual(np.quantile(np.full((10,10),.1),.1),1.5)


if __name__=='__main__':unittest.main()
