"""Source-design admission tests; no capture or predictor files are accessed."""
import json
import unittest

import mz117_mixture_source as source


class MixtureSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = source.source()

    def test_segment_first_hit_parallel_and_range_limit(self):
        hit = source.segment_aabb_fraction
        self.assertAlmostEqual(hit([0, 0, 1], [4, 0, 1], [3.3, 0, 1], [.2, .1, 2]), .8)
        self.assertIsNone(hit([0, 0, 1], [4, 0, 1], [3.3, .2, 1], [.2, .1, 2]))
        self.assertIsNone(hit([0, 0, 1], [4, 0, 1], [4.2, 0, 1], [.2, .1, 2]))
        self.assertIsNone(hit([0, 0, 1], [4, 0, 1], [-1, 0, 1], [.2, .1, 2]))

    def test_trace_occlusion_uses_first_actor_not_target_order(self):
        near = source.AnalyticActor(dict(center_m=[2, 0, 1], size_m=[.2, .1, 2]))
        far = source.AnalyticActor(dict(center_m=[3.7, 0, 1], size_m=[.2, 2, 2]))
        result = source.analytic_trace([far, near], source.Vector(0, 0, 100), source.Vector(400, 0, 100)).to_tuple()
        self.assertIs(result[10], near.static_mesh_component)
        self.assertAlmostEqual(result[5].x, 190.)

    def test_all_six_have_actual_real_actor_mixtures_and_retained_misses(self):
        audit = self.spec['analytic_mixture_precheck']; counts = audit['counts']
        episodes = self.spec['designated_mixture_episodes']
        self.assertEqual(len(episodes), 6)
        self.assertEqual([audit['episode_counts'][e]['returned_real_mixture_frames'] for e in episodes], [11, 2, 3, 3, 11, 5])
        self.assertEqual(counts['returned_real_mixture_frames'], 35)
        self.assertEqual(counts['returned_real_mixture_targets'], 286)
        self.assertEqual(counts['near_rod_sampled_frames'], 39)
        self.assertEqual(counts['near_rod_unsampled_frames'], 33)
        self.assertGreater(counts['weak_near_far_biased_mean_targets'], 0)
        for frame in audit['frames']:
            if not frame['packet_received']: self.assertEqual(frame['returned_real_mixtures'], [])
            for target in frame['returned_real_mixtures']:
                self.assertEqual(target['status'], 'SIM_MERGED')
                self.assertEqual(target['real_actor_ids'], [frame['episode']+'/shape0', frame['episode']+'/shape1'])
                self.assertLess(target['near_weighted_slant_mean_m'], target['pre_noise_slant_mean_m'])
                self.assertLess(target['pre_noise_slant_mean_m'], target['far_weighted_slant_mean_m'])

    def test_labels_and_global_context_exclusion(self):
        source.check_source(self.spec)
        self.assertEqual(self.spec['source_design_aabb_frame_counts'], dict(positive=145, negative=95))
        self.assertEqual(self.spec['source_design_transition_counts'], dict(enter=3, exit=3))
        self.assertEqual(self.spec['background'], dict(center_m=[12.7, 0., 1.75], size_m=[.14, 18.6, 8.5], texture=False, tof_reflectance_proxy=.50))
        self.assertEqual(self.spec['floor'], dict(center_m=[4., 0., -.05], size_m=[24., 20., .1], tof_reflectance_proxy=.30))
        for frame in self.spec['frames']:
            self.assertFalse(source.intersects(frame, self.spec['background']))
            self.assertFalse(source.intersects(frame, self.spec['floor']))
            self.assertTrue(all(o['name'].startswith('shape') for o in frame['objects']))

    def test_texture_independence_and_determinism(self):
        altered = source.source(source.TEXTURE_SEED+1)
        self.assertEqual(self.spec['analytic_mixture_precheck'], altered['analytic_mixture_precheck'])
        self.assertTrue(any(a['texture_seed'] != b['texture_seed'] for f, g in zip(self.spec['frames'], altered['frames']) for a, b in zip(f['objects'], g['objects'])))
        self.assertEqual(json.dumps(self.spec, sort_keys=True), json.dumps(source.source(), sort_keys=True))


if __name__ == '__main__': unittest.main()
