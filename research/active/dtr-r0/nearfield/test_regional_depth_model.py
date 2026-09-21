import unittest
import torch
import regional_depth_model as m


class RegionalMeasurementTests(unittest.TestCase):
    def test_region_sees_off_center_surface_point_does_not(self):
        logits = torch.full((1,81,64,64), -20.)
        logits[:,60] = 20.  # Far background.
        logits[:,10,0:3,0:3] = 60.  # Near return away from the center sample.
        x = torch.zeros(1,7,64,64)
        x[:,3,:8,:8] = 1.05/8; x[:,4,:8,:8] = 1
        self.assertLess(m.measurement_loss(logits,x,'REGION').item(),
                        m.measurement_loss(logits,x,'POINT').item())

    def test_missing_return_is_not_negative_depth_supervision(self):
        logits = torch.randn(1,81,64,64, requires_grad=True)
        x = torch.zeros(1,7,64,64)
        value = m.measurement_loss(logits,x,'REGION')
        self.assertEqual(value.item(),0)
        value.backward()
        self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_uniform_surface_has_equal_operators(self):
        logits = torch.randn(1,81,1,1).expand(1,81,64,64)
        self.assertTrue(torch.allclose(m.zone_distribution(logits,'REGION'),
                                       m.zone_distribution(logits,'POINT'),atol=1e-7))

    def test_ineligible_near_bin_cannot_win_a_return(self):
        logits = torch.zeros(1,81,64,64)
        changed = logits.clone(); changed[:,0] = 4
        x = torch.zeros(1,7,64,64); x[:,3] = 1.05/8; x[:,4] = 1
        for arm in m.ARMS:
            self.assertTrue(torch.allclose(m.measurement_loss(logits,x,arm),
                                           m.measurement_loss(changed,x,arm),atol=1e-6))

    def test_corridor_geometry_and_normalized_distribution(self):
        logits = torch.full((1,81,64,64),-30.); logits[:,10] = 30.
        x = torch.zeros(1,7,64,64)
        x[:,5,:,32:] = 1.0  # X=1.05m, outside despite same near depth.
        p = m.corridor_probability(logits,x)
        self.assertTrue((p[:,:,:32] > .99).all())
        self.assertTrue((p[:,:,32:] < .01).all())
        self.assertTrue(torch.allclose(m.zone_distribution(logits,'REGION').sum(1),torch.ones(1,8,8)))

    def test_training_has_finite_gradients_in_both_arms(self):
        torch.set_num_threads(2)
        x = torch.rand(2,7,64,64); x[:,4] = 1
        classes = torch.full((2,64,64),10,dtype=torch.long)
        classes[:,0] = -100
        for arm in m.ARMS:
            h = m.Head(); value,_ = m.loss(h,x,classes,torch.tensor([0.,1.]),arm)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))


if __name__ == '__main__':
    unittest.main()
