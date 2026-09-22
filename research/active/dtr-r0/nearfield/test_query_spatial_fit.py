"""Synthetic tests for cohort admission, area loss and fit-decision metrics."""
import math
import unittest

import numpy as np
import torch

from run_query_spatial_fit import select_cohort, fit_loss, cohort_feasibility, binary_counts, FRAMES


class SpatialFitTests(unittest.TestCase):
    def test_cohort_uses_only_training_identity_rule(self):
        identities = []
        for split in ('evaluation', 'dev', 'train'):
            for family in 'abcd':
                for group in ('01', '00'):
                    for relation in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
                        for frame in range(12):
                            identities.append(dict(index=len(identities), split=split, type_id=family,
                                base_group_id=family+group, layout_relation=relation, frame_in_clip=frame))
        selected, groups = select_cohort(identities)
        self.assertEqual(len(selected), 84)
        self.assertEqual(set(groups.values()), {'a00', 'b00', 'c00', 'd00'})
        self.assertTrue(all(m['split'] == 'train' and m['frame_in_clip'] in FRAMES for m in selected))

    def outputs_labels(self, fg, coverage=1., valid=True):
        logits = torch.zeros((1, 1, 1, len(fg)), requires_grad=True)
        output = dict(mask_logits=logits, distance_logits=torch.zeros(1, 1, 7, requires_grad=True))
        labels = dict(mask=torch.tensor(fg).reshape(1, 1, 1, -1),
                      coverage=torch.full((1, 1, len(fg)), coverage),
                      classes=torch.tensor([[6]]), valid=torch.tensor([[valid]]))
        return output, labels

    def test_empty_query_background_gradient(self):
        output, labels = self.outputs_labels([0., 0.])
        loss = fit_loss(output, labels)
        self.assertAlmostEqual(float(loss.detach()), math.log(7)+math.log(2), places=6)
        loss.backward()
        self.assertTrue(bool((output['mask_logits'].grad > 0).all()))

    def test_fractional_area_loss(self):
        output, labels = self.outputs_labels([.25, 0.], coverage=.5)
        # At p=.5 BCE is ln2. Dice = 1-(.25+1)/(.5+.25+1).
        expected = math.log(7)+math.log(2)+1-1.25/1.75
        self.assertAlmostEqual(float(fit_loss(output, labels).detach()), expected, places=6)

    def test_all_invalid_has_zero_loss_and_gradient(self):
        output, labels = self.outputs_labels([.25, 0.], valid=False)
        loss = fit_loss(output, labels)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(output['mask_logits'].grad.abs().sum()), 0.)

    def test_fixed_order_ceiling_and_fractional_iou(self):
        classes = np.full((2, 6), 6)
        classes[0, 0], classes[1, 1] = 0, 0
        valid = np.zeros((2, 6), bool)
        valid[:, :2] = True
        fg = np.zeros((2, 6, 1, 2), np.float32)
        fg[0, 0, 0, 0] = 1.
        fg[1, 1, 0, 0] = .25
        result = cohort_feasibility(dict(classes=classes, valid=valid, mask=fg,
                                        coverage=np.ones((2, 1, 2), np.float32)))
        self.assertEqual(result['fixed_query_order_max_win_rate']['all'], .5)
        self.assertEqual(result['binary_grid_oracle_iou_mean'], .625)
        self.assertTrue(result['evaluable'])

    def test_probability_cutoff_includes_ties_and_validity(self):
        result = binary_counts(np.array([.5, .5, .49, .9]), np.array([1, 0, 1, 0], bool),
                               np.array([1, 1, 1, 0], bool))
        self.assertEqual((result['TP'], result['FP'], result['FN']), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
