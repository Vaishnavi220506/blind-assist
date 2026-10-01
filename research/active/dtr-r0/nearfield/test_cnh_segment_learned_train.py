"""CPU synthetic tests of input parity, pooling inheritance and split isolation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import cnh_learned_readout as L
import cnh_segment_learned_train as T


class LearnedSegmentTrainTests(unittest.TestCase):
    def test_original_feature_math_and_support_are_exact(self):
        artifact = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-segment-learned-dev-20260928'
        artifact.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='train-test-', dir=artifact) as td:
            path = Path(td)/'unit000.npz'
            rng = np.random.default_rng(18)
            support = rng.random((2, 6, 8, 8, 16)) > .5
            np.savez(path, z4=rng.normal(size=(2, 8, 8, 16)).astype(np.float16),
                     z1=rng.normal(size=(2, 8, 8, 16)).astype(np.float16),
                     sup=np.packbits(support, axis=-1), labels=np.zeros((2, 6)),
                     main=np.array([False, True]), witness=np.ones((2, 6)),
                     strata=np.full((2, 6), 'tiny'), config=np.zeros(2), frame=np.arange(2), split=np.array('train'))
            original, new = L.load(td)[0], T.load_unit(path)
            np.testing.assert_array_equal(original['x'], new['x'])
            np.testing.assert_array_equal(original['sup'], new['sup'])
            np.testing.assert_array_equal(original['y'], new['labels'])

    def test_third_channel_raw_fraction_broadcast_without_changing_tof(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(2, 2, 8, 8, 16)).astype(np.float32)
        mask = np.zeros((2, 8, 8), np.float16)
        mask[0, 2, 3] = .25
        with patch.object(L, 'DEV', torch.device('cpu')):
            tensor = T.tensor_input(x, mask)
        self.assertEqual(tuple(tensor.shape), (2, 3, 8, 8, 16))
        np.testing.assert_array_equal(tensor[:, :2].numpy(), x)
        np.testing.assert_array_equal(tensor[0, 2, 2, 3].numpy(), np.full(16, .25))
        self.assertEqual(float(tensor[1, 2].sum()), 0.)

    def test_architecture_inherits_exact_query_pooling_and_mask_is_active(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        original = L.Readout().cpu()
        torch.manual_seed(7)
        new = T.Readout().cpu()
        self.assertIs(T.Readout.forward, L.Readout.forward)
        self.assertEqual(tuple(new.body[0].weight.shape), (16, 3, 3, 3, 3))
        for key, value in original.state_dict().items():
            if not key.startswith('body.0.'):
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        x = np.zeros((1, 2, 8, 8, 16), np.float32)
        sup = np.ones((1, 6, 8, 8, 16), bool)
        with patch.object(L, 'DEV', torch.device('cpu')):
            empty = T.predict(new, x, sup, np.zeros((1, 8, 8), np.float32))
            full = T.predict(new, x, sup, np.ones((1, 8, 8), np.float32))
        self.assertEqual(empty.shape, (1, 6))
        self.assertGreater(float(np.max(np.abs(empty-full))), 1e-7)

    def test_train_calib_loaders_never_open_audit(self):
        features, masks = Path('synthetic-features'), Path('synthetic-masks')
        seen_features, seen_masks = [], []
        def unit(path):
            u = int(path.stem[4:]); seen_features.append(u)
            return dict(x=np.zeros((2, 2, 8, 8, 16), np.float32), sup=np.ones((2, 6, 8, 8, 16), bool),
                labels=np.zeros((2, 6)), main=np.array([False, True]), config=np.zeros(2), frame=np.arange(2),
                split=np.array('train' if u < 96 else 'calib'))
        class MaskFile(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        def mask(path):
            seen_masks.append(int(path.stem[4:]))
            return MaskFile(config=np.zeros(2), frame=np.arange(2), IDEAL=np.zeros((2, 8, 8)),
                            MIXED=np.ones((2, 8, 8)), AUG=np.zeros((20, 2, 8, 8)))
        with patch.object(T, 'load_unit', side_effect=unit), patch.object(T.np, 'load', side_effect=mask):
            train = T.load_training(features, masks)
            calib = T.load_calib(features, masks)
        self.assertEqual(seen_features, list(range(128)))
        self.assertEqual(seen_masks, list(range(128)))
        self.assertEqual(train['x'].shape[0], 96)
        self.assertEqual(train['AUG'].shape[:2], (20, 96))
        self.assertEqual(len(calib), 32)
        self.assertNotIn('SHIFT05', calib[0])


if __name__ == '__main__':
    unittest.main()
