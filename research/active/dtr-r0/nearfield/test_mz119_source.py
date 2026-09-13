"""Focused source-only checks; no rendered RGB, outcomes or capture launched."""
import json
import math
import unittest

import mz117_mixture_source as analytic
import mz119_parallax_source as source


class ParallaxSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.spec = source.source()

    def test_source_cardinality_labels_and_actual_mixture_admission(self):
        source.check_source(self.spec)
        counts = self.spec['analytic_mixture_precheck']['counts']
        self.assertEqual(counts['returned_real_mixture_frames'], 35)
        self.assertEqual(counts['returned_real_mixture_targets'], 198)
        self.assertEqual(counts['near_rod_unsampled_frames'], 33)
        self.assertEqual(counts['weak_near_far_biased_mean_targets'], 96)

    def test_lateral_rotation_and_low_translation_controls(self):
        self.assertEqual(self.spec['camera_motion_episode_counts'], source.CAMERA_COHORTS)
        for audit in self.spec['source_audit']:
            frames = [f for f in self.spec['frames'] if f['episode'] == audit['episode']]
            positions = audit['camera_positions_m']; mode = audit['camera_motion_mode']
            if mode == 'lateral':
                self.assertGreater(max(abs(p[1]) for p in positions), .115)
                self.assertGreater(max(math.dist(a, b) for a, b in zip(positions, positions[1:])), .03)
                self.assertTrue(any(a[1] < b[1] for a, b in zip(positions, positions[1:])))
                self.assertTrue(any(a[1] > b[1] for a, b in zip(positions, positions[1:])))
            if mode in ('rotation_only', 'stationary'): self.assertEqual(positions, [[0., 0., 1.7]]*12)
            if mode in ('rotation_only', 'low_translation_rotation'):
                self.assertGreater(max(abs(f['camera']['yaw']) for f in frames), 3.)
            for f in frames: self.assertEqual(f['body_origin_m'], [f['camera']['x'], f['camera']['y'], 0.])

    def test_source_trajectory_never_enters_unchanged_raw_schema(self):
        # Source-only stand-in, not actual captured native evidence.
        state = {}; contexts = [analytic.AnalyticActor(self.spec[k]) for k in ('background', 'floor')]
        state['mz115_environment_reflectance'] = [(a.static_mesh_component, a.obj['tof_reflectance_proxy']) for a in contexts]
        for frame in self.spec['frames'][:3]:
            actors = [analytic.AnalyticActor(o) for o in frame['objects']]
            raw, _, _ = analytic.zonal.sensors(analytic.ANALYTIC_NATIVE, actors+contexts, frame, actors, state)
            self.assertEqual(raw['camera_in_body_m'], [0., 0., 1.7])
            self.assertEqual(raw['camera_pitch_deg'], -3.)
            self.assertEqual(raw['delta_pitch'], 0.)
            self.assertTrue({'camera', 'body_origin_m', 'wearer_speed', 'source_audit', 'objects'}.isdisjoint(raw))
            self.assertFalse(any('translation' in k or 'native' in k for k in raw))

    def test_fresh_geometry_texture_independence_and_determinism(self):
        previous = analytic.source()
        signatures = {tuple((tuple(o['start_m']), tuple(o['size_m'])) for o in a['objects'])
                      for a in previous['source_audit'] if a['objects']}
        for audit in self.spec['source_audit']:
            if audit['objects']:
                self.assertNotIn(tuple((tuple(o['start_m']), tuple(o['size_m'])) for o in audit['objects']), signatures)
        old_textures = {o['texture_seed'] for a in previous['source_audit'] for o in a['objects']}
        new_textures = {o['texture_seed'] for a in self.spec['source_audit'] for o in a['objects']}
        self.assertTrue(old_textures.isdisjoint(new_textures))
        changed_texture = source.source(source.TEXTURE_SEED+1)
        self.assertEqual(self.spec['analytic_mixture_precheck'], changed_texture['analytic_mixture_precheck'])
        self.assertNotEqual(self.spec['frames'][0]['objects'][0]['texture_seed'], changed_texture['frames'][0]['objects'][0]['texture_seed'])
        self.assertEqual(json.dumps(self.spec, sort_keys=True), json.dumps(source.source(), sort_keys=True))


if __name__ == '__main__': unittest.main()
