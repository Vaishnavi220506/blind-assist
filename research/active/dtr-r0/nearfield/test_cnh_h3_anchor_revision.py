"""Focused numerical checks for the correction, not scientific validation."""
import ast
from pathlib import Path
import unittest

import numpy as np

import cnh_h3_anchor_revision as a


class AnchorMathTests(unittest.TestCase):
    def test_recover_exact_forward_moments(self):
        unit = a.forward_planes([.635], rho=.5)[0, a.CENTRE]
        g, c, ambient = 1400., .08, 1.2
        mean = c*g*unit
        var = c*c*(g*unit+16*ambient)
        fit = a.fit_moments(unit, mean, var)
        self.assertTrue(fit['identified'])
        for name, expected in [('signal_counts', g), ('unit_scale_c', c), ('ambient_counts', ambient)]:
            self.assertAlmostEqual(fit[name]/expected, 1., places=10)
        self.assertLess(fit['loss'], 1e-20)

    def test_reference_formula_agrees_with_existing_function(self):
        # Execute only the existing trusted reference function, avoiding unrelated
        # geometry's scipy dependency. Deterministic seeds suffice for its check.
        path = Path(a.__file__).with_name('cnh_track_a_v13_sensor.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'reference_parameters')
        namespace = dict(np=np, replace=a.replace, SensorParameters=a.SensorParameters,
                         synthesize_response=a.synthesize_response, RAW_BIN_M=a.RAW_BIN_M,
                         LEVELS=(3, 6, 12), seed_for=lambda *args, **kwargs: 1234)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        params, _ = namespace['reference_parameters']()
        for level, p in params.items():
            self.assertAlmostEqual(a.reference_snr(p.signal_counts, p.ambient_counts), level, places=7)

    def test_stable_windows_reject_gaps_and_multitarget(self):
        d = dict(H=np.ones((16, 64, 16)), D=np.full((16, 64), 300.),
                 S=np.full((16, 64), 5), N=np.ones((16, 64)),
                 seq=np.arange(16), ms=np.arange(16)*196)
        data = {s: {k: v.copy() for k, v in d.items()} for s in a.SEGMENTS[1:3]}
        self.assertEqual(len(a.stable_windows(data)), 16)
        data[a.SEGMENTS[1]]['seq'][4:] += 1
        data[a.SEGMENTS[2]]['N'][:, a.CENTRE] = 2
        self.assertEqual(len(a.stable_windows(data)), 4)


if __name__ == '__main__':
    unittest.main()
