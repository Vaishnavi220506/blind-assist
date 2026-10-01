"""Train oracle opt-in and exact observation identity checks (no data cohorts)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import cnh_track_a_v13_sensor as sensor
from cnh_segment_learned_prepare import compare_observations


class PrepareTests(unittest.TestCase):
    def test_exact_check_rejects_small_difference_dtype_and_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.npz', Path(tmp)/'b.npz'
            values = dict(rate=np.array(5), hist=np.array([1.], np.float32))
            np.savez(a, **values); np.savez(b, **values)
            self.assertEqual(compare_observations(a, b)['hist'], [1])
            for altered in (dict(values, hist=np.array([1.000001], np.float32)),
                            dict(values, hist=np.array([1.], np.float64)),
                            dict(values, extra=np.array(1))):
                np.savez(b, **altered)
                with self.assertRaises(AssertionError):
                    compare_observations(a, b)

    def test_train_oracle_default_off_and_explicit_on(self):
        obs = dict(hist=np.zeros((3, 1, 8, 8, 16)), ambient=np.zeros((3, 1, 8, 8)),
                   T_Q_tof=np.eye(4)[None])
        seen = []
        def render(config, mount, params, rate, keep):
            seen.append((rate, keep))
            return obs, dict(object_id=np.zeros((1, 8, 8, 256))) if keep else None
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp)/'geometry'; unit = geo/'unit00'; unit.mkdir(parents=True)
            (unit/'unit00.json').write_text(json.dumps(dict(split='train', configs=[dict(config=0)])))
            with patch.object(sensor, 'reference_parameters', return_value=({}, {})), \
                 patch.object(sensor, 'render_config', side_effect=render), \
                 patch.object(sensor, 'FORCE_RATE', None):
                sensor.run(geo, Path(tmp)/'default', 0, -10)
                sensor.run(geo, Path(tmp)/'explicit', 0, -10, oracle_train=True)
            self.assertEqual(seen, [(5, False), (5, True)])
            self.assertFalse((Path(tmp)/'default/unit00-mount-10-oracle.npz').exists())
            self.assertTrue((Path(tmp)/'explicit/unit00-mount-10-oracle.npz').exists())


if __name__ == '__main__':
    unittest.main()
