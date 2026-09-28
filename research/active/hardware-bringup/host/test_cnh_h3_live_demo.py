"""Focused engineering contracts; no real-sensor accuracy assertions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cnh_h3_live_demo as demo

TEMP_ROOT = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-h3-live-demo-20260928/tests'
TEMP_ROOT.mkdir(parents=True,exist_ok=True)


class DemoTests(unittest.TestCase):
    def test_frozen_smoothing_exact_and_causal(self):
        from cnh_learned_memory_fusion import causal_ewma
        scores = np.random.default_rng(9).normal(size=(17,6)).astype(np.float32)
        wanted = causal_ewma(scores, np.zeros(17), np.arange(17),alpha=.5,window=5)
        actual = np.array([demo.smooth_logits(scores[:i+1]) for i in range(17)])
        np.testing.assert_array_equal(wanted,actual)

    def test_invalid_registration_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            path = Path(td)/'registration.json'
            path.write_text(json.dumps(dict(verified=False,evidence='nominal',zone_polygons_normalized=np.zeros((64,4,2)).tolist())))
            with self.assertRaises(ValueError): demo.load_registration(path)

    def test_background_temporal_covariance_and_gap(self):
        rng = np.random.default_rng(3)
        x = np.cumsum(rng.integers(-5,6,size=(84,64,16)),axis=0)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            path = Path(td)/'background.jsonl'
            with path.open('w') as f:
                for i,h in enumerate(x):
                    sensor = dict(type='cnh_frame',seq=i+(i>=40),ms=i*200,bins=16,rows=8,cols=8,
                        distance_mm=[100]*64,target_status=[5]*64,nb_target=[1]*64,
                        hist_raw=h.tolist(),hist_scaler=np.zeros((64,16),int).tolist(),ambient_raw=[0]*64,ambient_scaler=[0]*64)
                    f.write(json.dumps(dict(sensor=sensor,host_received_monotonic_ns=i*200_000_000))+'\n')
            mean,var,receipt = demo.noise_model(path)
        self.assertEqual(receipt['sum_windows'],[84,82,80,78])
        self.assertGreater(float(np.median(var[3]/var[0])),8)
        np.testing.assert_allclose(mean,x.mean(0).reshape(8,8,16))

    def test_sampler_no_future_or_duplicates(self):
        records = [dict(sensor=dict(seq=i,ms=t),host_received_monotonic_ns=t*1_000_000) for i,t in enumerate([0,90,190,310,410,590,710])]
        sampled = list(demo.causal_sample(records))
        self.assertEqual([r['sensor']['ms'] for r in sampled],[0,190,310,590])
        self.assertTrue(all(r['host_received_monotonic_ns'] <= r['sample_tick_ns'] for r in sampled))

    def test_sampler_same_receipt_chunk_keeps_latest(self):
        records = [dict(sensor=dict(seq=i,ms=i*100),host_received_monotonic_ns=t*1_000_000) for i,t in enumerate([0,0,210])]
        sampled = list(demo.causal_sample(records))
        self.assertEqual([r['sensor']['seq'] for r in sampled],[1])


if __name__ == '__main__': unittest.main()
