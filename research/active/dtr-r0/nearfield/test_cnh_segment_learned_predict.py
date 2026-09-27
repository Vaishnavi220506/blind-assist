"""CPU-only freeze/attachment fixtures; never reads experiment predictions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import cnh_segment_learned_predict as P


class LearnedSegmentPredictTests(unittest.TestCase):
    def setUp(self):
        artifact = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-segment-learned-dev-20260928'
        artifact.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='predict-test-', dir=artifact)
        self.root = Path(self.temp.name)
        self.models, self.tof = self.root/'models', self.root/'tof'
        self.models.mkdir(); self.tof.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def fixtures(self):
        for mode in ('IDEAL', 'AUG'):
            for seed in range(3):
                stem = f'{mode}_seed{seed}'
                path = self.models/f'{stem}.pt'
                path.write_bytes(stem.encode())
                (self.models/f'{stem}.json').write_text(json.dumps(dict(status='complete', completed_epochs=20,
                                                                     model_sha256=P.sha(path))), encoding='utf-8')
        for seed in range(3):
            (self.tof/f'model_seed{seed}.pt').write_bytes(f'tof{seed}'.encode())

    def test_freeze_requires_six_complete_jobs(self):
        self.fixtures()
        path = self.models/'AUG_seed2.json'
        row = json.loads(path.read_text()); row['status'] = 'running'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'All six'):
            P.freeze(self.models, self.tof)
        self.assertFalse((self.models/'FROZEN.json').exists())
        row.update(status='complete', completed_epochs=19); path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'All six'):
            P.freeze(self.models, self.tof)

    def test_freeze_hash_check_and_idempotent_manifest(self):
        self.fixtures()
        frozen = P.freeze(self.models, self.tof)
        self.assertEqual(len(frozen['models']), 9)
        self.assertEqual(P.freeze(self.models, self.tof), frozen)
        (self.models/'IDEAL_seed0.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            P.freeze(self.models, self.tof)

    def test_m1_alignment_rejects_wrong_rows_without_rewrite(self):
        pred, source = self.root/'pred', self.root/'m1/scores'
        pred.mkdir(); source.mkdir(parents=True)
        arrays = dict(config=np.array([1, 1]), frame=np.array([3, 4]), S2=np.ones((2, 6)))
        target = pred/'unit128.npz'
        np.savez(target, **arrays)
        before = target.read_bytes()
        bad = dict(arrays, frame=np.array([4, 3]))
        bad.update({f'M1__{c}': np.zeros((2, 6)) for c in P.CONDITIONS})
        np.savez(source/'unit128.npz', **bad)
        with self.assertRaisesRegex(ValueError, 'alignment'):
            P.attach_m1(pred, source.parent)
        self.assertEqual(target.read_bytes(), before)
        bad['frame'] = arrays['frame']; np.savez(source/'unit128.npz', **bad)
        P.attach_m1(pred, source.parent)
        with np.load(target) as f:
            for c in P.CONDITIONS:
                np.testing.assert_array_equal(f[f'M1__{c}'], np.zeros((2, 6)))
            np.testing.assert_array_equal(f['S2'], arrays['S2'])

    def test_validator_rejects_actual_checkpoint_directory_substitution(self):
        self.fixtures()
        frozen = P.freeze(self.models, self.tof)
        self.assertEqual(P.validate_frozen(self.models, self.tof), frozen)
        alternate = self.root/'other-tof'; alternate.mkdir()
        for seed in range(3):
            # Identical bytes in a different requested checkpoint set still reject.
            (alternate/f'model_seed{seed}.pt').write_bytes((self.tof/f'model_seed{seed}.pt').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Actual nine'):
            P.validate_frozen(self.models, alternate)
        (self.tof/'model_seed0.pt').write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'Frozen model changed'):
            P.validate_frozen(self.models, self.tof)

    def test_validator_rechecks_receipts_and_frozen_code(self):
        self.fixtures(); P.freeze(self.models, self.tof)
        receipt = self.models/'AUG_seed1.json'
        row = json.loads(receipt.read_text()); row['status'] = 'failed'; receipt.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'Incomplete training'):
            P.validate_frozen(self.models, self.tof)
        row['status'] = 'complete'; receipt.write_text(json.dumps(row))
        manifest = self.models/'FROZEN.json'
        frozen = json.loads(manifest.read_text()); frozen['code']['cnh_segment_learned_predict.py'] = 'bad'
        manifest.write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            P.validate_frozen(self.models, self.tof)

    def test_original_predict_holder_y_alias_is_valid_and_nonmutating(self):
        class TinyNet(torch.nn.Module):
            def forward(self, x, sup):
                return x[:, 0, 0, 0, 0, None].expand(-1, 6)
        labels = np.zeros((3, 6), np.float32)
        d = dict(x=np.ones((3, 2, 8, 8, 16), np.float32), sup=np.ones((3, 6, 8, 8, 16), bool),
                 labels=labels, main=np.ones(3, bool))
        holder = {128: dict(d, y=d['labels'])}
        with patch.object(P.L, 'DEV', torch.device('cpu')):
            P.L.predict(TinyNet(), holder, [128])
        np.testing.assert_array_equal(holder[128]['NN'], np.ones((3, 6)))
        np.testing.assert_array_equal(labels, np.zeros((3, 6)))
        self.assertNotIn('NN', d)


if __name__ == '__main__':
    unittest.main()
