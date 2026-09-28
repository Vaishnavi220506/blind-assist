import unittest
import numpy as np
import torch
from cnh_corridor_retrain import validate_arrays,training_batches,frozen_module,predict

class CorridorTrainingTests(unittest.TestCase):
    def fixture(self):
        return dict(z4=np.zeros((4,8,8,16),np.float16),z1=np.zeros((4,8,8,16),np.float16),
                    support=np.ones((4,6,8,8,16),bool),labels=np.zeros((4,6),np.int8),unit=np.array(1),
                    split=np.array('train'),query_frame=np.array('travel'),scene=np.zeros(4,int),frame=np.arange(4),train_mask=np.array([False,False,False,True]))
    def test_train_contract(self):self.assertEqual(validate_arrays(self.fixture()),4)
    def test_reject_calibration_and_wrong_frame(self):
        for key,value in [('split',np.array('calib')),('split',np.array('audit')),('query_frame',np.array('head'))]:
            d=self.fixture();d[key]=value
            with self.assertRaises(ValueError):validate_arrays(d)
    def test_no_boundary_exclusion_or_extra_warmup(self):
        d=self.fixture();d['train_mask'][:]=False
        with self.assertRaises(ValueError):validate_arrays(d)
    def test_duplicate_sample_rejected(self):
        d=self.fixture();d['frame'][1]=0
        with self.assertRaises(ValueError):validate_arrays(d)
    def test_accumulation_exact_weights(self):
        groups=list(training_batches(np.arange(11),8,3))
        self.assertEqual([sum(len(ids) for ids,w in g) for g in groups],[8,3])
        self.assertTrue(all(abs(sum(w for ids,w in g)-1)<1e-12 for g in groups))
    def test_gradients_match_effective_batch(self):
        torch.manual_seed(0);m=torch.nn.Linear(3,2).double();x=torch.randn(11,3,dtype=torch.float64);y=torch.rand(11,2,dtype=torch.float64)
        torch.nn.functional.binary_cross_entropy_with_logits(m(x),y).backward();expected=[p.grad.clone() for p in m.parameters()];m.zero_grad()
        for ids,w in next(training_batches(np.arange(11),11,4)):(torch.nn.functional.binary_cross_entropy_with_logits(m(x[ids]),y[ids])*w).backward()
        for p,grad in zip(m.parameters(),expected):torch.testing.assert_close(p.grad,grad,rtol=1e-12,atol=1e-12)
    def test_architecture_prediction_unmodified(self):
        torch.set_num_threads(2);L=frozen_module();model=L.Readout().cpu().eval();d=self.fixture()
        result=predict([model],d['z4'],d['z1'],d['support'],batch_size=2)
        with torch.no_grad():expected=model(torch.zeros((4,2,8,8,16)),torch.tensor(d['support'])).numpy()
        np.testing.assert_allclose(result,expected,rtol=1e-5,atol=1e-5)
        self.assertEqual(result.shape,(4,6))

if __name__=='__main__':unittest.main()
