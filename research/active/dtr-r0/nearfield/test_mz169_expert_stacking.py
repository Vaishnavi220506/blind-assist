"""Small synthetic-only checks; no scientific scores or persisted models."""
import pickle
import unittest

import numpy as np

from mz169_expert_stacking import fit_model, mean_score, predict_score, transform_scores


def synthetic():
    scores = np.array([
        [.05,.10,.20,.15], [.12,.25,.15,.20], [.20,.18,.35,.10],
        [.35,.40,.25,.45], [.60,.65,.75,.55], [.75,.90,.65,.70],
        [.80,.75,.90,.85], [.95,.85,.80,.90],
    ], dtype=np.float64)
    return scores, np.array([0,0,0,0,1,1,1,1])


class ExpertStackingTests(unittest.TestCase):
    def test_clipping_signed_control_and_input_immutability(self):
        values = np.array([[0.,1.,.5,.25], [1e-10,1.-1e-10,.75,.1]])
        before = values.copy()
        transformed = transform_scores(values)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.dtype, np.float64)
        self.assertAlmostEqual(transformed[0,0], np.log(1e-6)-np.log1p(-1e-6))
        self.assertEqual(transformed[0,0], transformed[1,0])
        self.assertEqual(transformed[0,1], transformed[1,1])
        self.assertEqual(transformed[0,2], 0.)
        np.testing.assert_array_equal(mean_score(values), transformed.mean(axis=1))
        self.assertLess(mean_score(np.full((1,4),.2))[0], 0.)
        np.testing.assert_array_equal(values, before)

    def test_reject_malformed_scores_and_labels(self):
        bad = [[], np.zeros((0,4)), np.zeros(4), np.zeros((2,3)), np.zeros((1,4,1)),
               [[0.,.5,1.,np.nan]], [[0.,.5,1.,np.inf]], [[-.001,.1,.2,.3]],
               [[.1,.2,.3,1.001]], [['bad',.2,.3,.4]]]
        for values in bad:
            with self.subTest(values=str(values)):
                with self.assertRaises(ValueError): transform_scores(values)
        scores, target = synthetic()
        for labels in (target[:-1], target[:,None], np.full(8,2), np.zeros(8), [0,0,0,0,1,1,1,np.nan]):
            with self.subTest(labels=str(labels)):
                with self.assertRaises(ValueError): fit_model(scores,labels)

    def test_training_statistics_use_only_supplied_rows(self):
        scores, target = synthetic()
        before = scores.copy(); labels_before = target.copy()
        model = fit_model(scores,target)
        scaler = model.named_steps['scaler']
        expected = transform_scores(scores)
        np.testing.assert_allclose(scaler.mean_,expected.mean(axis=0),rtol=0,atol=1e-14)
        np.testing.assert_allclose(scaler.var_,expected.var(axis=0),rtol=0,atol=1e-14)
        self.assertEqual(int(scaler.n_samples_seen_),len(scores))
        snapshot = pickle.dumps(model)
        # Extreme, separate query rows must not influence fitted statistics.
        held = np.array([[0.,1.,0.,1.],[1.,0.,1.,0.]])
        result = predict_score(model,held)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result>=0)&(result<=1)).all())
        self.assertEqual(pickle.dumps(model),snapshot)
        np.testing.assert_array_equal(scores,before)
        np.testing.assert_array_equal(target,labels_before)
        classifier = model.named_steps['classifier']
        self.assertEqual((classifier.C,classifier.solver,classifier.max_iter,classifier.random_state,classifier.class_weight),
                         (1.,'lbfgs',1000,169016,None))

    def test_serialization_parity_and_row_permutation_equivariance(self):
        scores,target = synthetic();model=fit_model(scores,target)
        result=predict_score(model,scores)
        loaded=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(predict_score(loaded,scores),result)
        order=np.array([7,2,0,5,3,6,1,4])
        np.testing.assert_array_equal(predict_score(model,scores[order]),result[order])
        np.testing.assert_array_equal(mean_score(scores[order]),mean_score(scores)[order])
        refit=fit_model(scores[order],target[order])
        np.testing.assert_allclose(predict_score(refit,scores),result,rtol=0,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
