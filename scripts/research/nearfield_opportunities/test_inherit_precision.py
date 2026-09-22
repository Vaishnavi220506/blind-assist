import unittest

import initial_relative_inference as original
import inherit_precision as candidate
from test_shared_bias_inference import observations, residual, witness


class PrecisionTests(unittest.TestCase):
    def test_private_constants_and_coarse_constraint_parity(self):
        model = candidate.model_for(.1)
        obs = observations(witness(), original.PATH)
        for label in ('IN', 'OUT'):
            self.assertEqual(model._constraints(obs, label).rows,
                             original._constraints(obs, label).rows)
        self.assertEqual(original.shared.base.RANGE_STEP, .1)
        fine = candidate.model_for(.001)
        self.assertEqual(fine.shared.base.RANGE_STEP, .001)
        self.assertEqual(original.shared.base.RANGE_STEP, .1)
        self.assertEqual(model.shared.base.RANGE_STEP, .1)

    def test_quantization_no_truth_and_fine_forward_containment(self):
        for w in (witness(), witness(x=.3805, px=.001, pz=.001, b=.002),
                  witness(x=-.65, b=-.002), witness(z=3.0204, pz=.001)):
            coarse = observations(w, original.PATH)
            views = original.shared.validation_detail(w, coarse, 'IN')['replay']
            obs = candidate.public_views(views, .001)
            model = candidate.model_for(.001)
            label = model.relative_label(w)
            self.assertTrue(model.validate_witness(w, obs, label))
            self.assertLessEqual(residual(model._constraints(obs, label), w), 1e-10)
            self.assertTrue(all(set(row) == {'camera', 'bins'} for row in obs))
            with self.assertRaises(ValueError):
                model.infer([dict(obs[0], truth=True), *obs[1:]])

    def test_illegal_precision_or_ranges_rejected(self):
        with self.assertRaises(ValueError): candidate.model_for(.01)
        with self.assertRaises(ValueError):
            candidate.public_views([dict(camera=[0, 0], biased_ranges=[float('nan')]*8)], .001)

    def test_fine_and_coarse_bins_use_same_analog_values(self):
        w = witness()
        old = observations(w, original.PATH)
        views = original.shared.validation_detail(w, old, 'IN')['replay']
        self.assertEqual(candidate.public_views(views, .1), old)
        fine = candidate.public_views(views, .001)
        self.assertNotEqual(fine, old)
        self.assertEqual([v['camera'] for v in fine], [v['camera'] for v in old])


if __name__ == '__main__': unittest.main()
