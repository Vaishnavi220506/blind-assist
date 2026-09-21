import unittest
import json
from pathlib import Path
import numpy as np
from cnh_components import fit_two, classify, marker_windows, zone_result


class ComponentsTests(unittest.TestCase):
    def test_exact_mixture_and_scaled_single_baseline(self):
        b, f = np.array([0., 1., 4.]), np.array([5., 1., 0.])
        fit = fit_two(.4*b+.6*f, b, f)
        self.assertAlmostEqual(fit['a'], .4)
        self.assertAlmostEqual(fit['b'], .6)
        self.assertLess(fit['error'], 1e-12)
        self.assertGreater(fit['gain'], .1)
        pure = fit_two(3*b, b, f)
        self.assertAlmostEqual(pure['best_single_error'], 0)
        self.assertAlmostEqual(pure['gain'], 0)

    def test_negative_unconstrained_coefficient_uses_boundary(self):
        fit = fit_two([2., -1.], [1., 0.], [0., 1.])
        self.assertEqual((fit['a'], fit['b']), (2., 0.))
        with self.assertRaises(ValueError):
            fit_two([0., 0.], [1., 0.], [0., 1.])

    def test_dual_requires_gain_and_both_components(self):
        p = {'maximum_relative_rmse': .25, 'minimum_single_template_error_gain': .1}
        self.assertTrue(classify({'a': .5, 'b': .5, 'error': .1, 'gain': .2}, .1, .1, p))
        self.assertFalse(classify({'a': .5, 'b': .1, 'error': .1, 'gain': .2}, .1, .1, p))
        self.assertFalse(classify({'a': .5, 'b': .5, 'error': .1, 'gain': .01}, .1, .1, p))

    def test_marker_contract_rejects_missing_or_overlap(self):
        s = {'schema': 'hardware-bringup.cnh-components.v1', 'start_host_monotonic_ns': 0,
             'segments': [{'phase': p, 'start_host_monotonic_ns': i*10_000_000_000, 'end_host_monotonic_ns': i*10_000_000_000+6_000_000_000}
                          for i,p in enumerate(('background', 'foreground', 'mixture', 'return'))]}
        self.assertEqual(marker_windows(s)['background'], (1_000_000_000, 5_000_000_000))
        s['segments'][1]['start_host_monotonic_ns'] = 1_000_000_000
        s['segments'][1]['end_host_monotonic_ns'] = 7_000_000_000
        with self.assertRaises(ValueError):
            marker_windows(s)

    def test_zone_controls_keep_unknown_mixture_and_reject_false_return(self):
        p = json.loads((Path(__file__).parent.parent/'cnh-components-protocol.json').read_text())
        b, f = np.zeros(24), np.zeros(24)
        b[7], f[1] = 10, 20
        waves = {'background': b, 'foreground': f, 'mixture': .5*b+.5*f, 'return': b}
        phases = {name: [{'h': np.tile(wave, (16, 1)), 'block': i//5, 'seq': i,
                         'valid': [name != 'mixture']*16,
                         'sensor': {'distance_mm': [200 if name == 'foreground' else 900]*16}}
                        for i in range(20)] for name, wave in waves.items()}
        result = zone_result(0, phases, p)
        self.assertEqual(result['verdict'], 'DUAL_TEMPLATE_COMPATIBILITY_SUPPORTED')
        self.assertEqual(result['scalar']['mixture']['unknown'], 20)
        self.assertEqual(len(result['mixture']['frames']), 20)
        phases['return'] = phases['mixture']
        failed = zone_result(0, phases, p)
        self.assertEqual(failed['verdict'], 'NOT_SUPPORTED')
        self.assertEqual((result['a_floor'], result['b_floor']), (failed['a_floor'], failed['b_floor']))


if __name__ == '__main__':
    unittest.main()
