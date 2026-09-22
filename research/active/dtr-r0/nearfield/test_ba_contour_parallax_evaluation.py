import unittest
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluate_ba_contour_parallax as evaluation
from evaluate_ba_contour_parallax import evaluate_line, interval_near, native_support, points_world, replay_old_clip


POSE = dict(x=0., y=0., z=1.82, pitch=0., yaw=0., roll=0.)


class ContourParallaxGeometryTest(unittest.TestCase):
    def test_camera_axes_and_yaw(self):
        np.testing.assert_allclose(points_world([[319.5,179.5]], [2.], POSE), [[2.,0.,1.82]])
        rotated = dict(POSE, yaw=90.)
        np.testing.assert_allclose(points_world([[319.5,179.5]], [2.], rotated), [[0.,2.,1.82]], atol=1e-12)

    def test_full_heading_interval_cannot_use_near_midpoint(self):
        near, _, _ = interval_near([[319.5,179.5]]*4, [[1.,3.],[1.,3.01],[.08,1.],[np.nan,np.nan]], POSE)
        self.assertEqual(near.tolist(), [True,False,False,False])

    def test_pitch_compensates_heading_and_whole_height_interval(self):
        pose = dict(POSE, pitch=-10.)
        near, heading, height = interval_near([[319.5,179.5]], [[2.,3.]], pose)
        self.assertTrue(near[0])
        np.testing.assert_allclose(heading[0], np.array([2.,3.])*np.cos(np.deg2rad(10.)))
        near, _, height = interval_near([[319.5,179.5]], [[1.,7.]], pose)
        self.assertFalse(near[0])
        self.assertLess(height.min(), .65)

    def test_height_entire_interval_not_just_point_estimate(self):
        focal = 640/(2*np.tan(np.deg2rad(50.)))
        pixel = [[319.5,179.5-focal*.1]]
        near, _, height = interval_near(pixel, [[.5,2.]], POSE)
        self.assertFalse(near[0])
        self.assertLess(height.min(), 1.85)
        self.assertGreater(height.max(), 1.85)

    def test_target_tolerance_nearest_pixel_and_reference_unknown(self):
        native = np.zeros((360,640), np.float32)
        native[180,320] = 2.
        world = points_world([[320,180]],[2.],POSE)[0]
        obj = dict(name='bar',center_m=world.tolist(),size_m=[.002,.002,.002])
        support = native_support([[319.5,179.5],[640.,0.],[0.,0.]],native,POSE,[obj])
        self.assertEqual(support['valid'].tolist(), [True,False,False])
        self.assertEqual(support['target'].tolist(), [True,False,False])
        obj['center_m'][0] += .007
        self.assertFalse(native_support([[319.5,179.5]],native,POSE,[obj])['target'][0])

    def test_recovery_requires13_native_consistent_target_supports(self):
        pixels = [[float(x),180.] for x in range(300,317)]
        line = dict(id=0,endpoints=[pixels[0],pixels[-1]],points=pixels,depth_m=2.,interval_m=[1.9,2.1],
                    accepted=True,point_supported=[True]*17)
        native = np.full((360,640), 2., np.float32)
        obj = dict(name='bar',center_m=[2.,0.,1.82],size_m=[.08,1.2,.08])
        score = evaluate_line(line,native,POSE,[obj])
        self.assertTrue(score['target_recovered'])
        native[180,300:305] = 4.
        score = evaluate_line(line,native,POSE,[obj])
        self.assertEqual(score['target_consistent_supported_points'],12)
        self.assertFalse(score['target_recovered'])
        self.assertEqual(score['non_target_false_near_points'],5)

    def test_rejected_unknown_line_keeps_candidates_and_reference(self):
        pixels = [[float(x),180.] for x in range(300,317)]
        line = dict(id=0,endpoints=[pixels[0],pixels[-1]],points=pixels,depth_m=None,interval_m=[None,None],
                    accepted=False,point_supported=[False]*17)
        score = evaluate_line(line,np.full((360,640),6.,np.float32),POSE,[])
        self.assertFalse(score['near'])
        self.assertEqual(score['candidate_points'],17)
        self.assertEqual(score['non_target_candidate_points'],17)

    def test_old_replay_keeps_best_height_semantics_separate(self):
        pixel = [[320,153]]
        world = points_world(pixel,[1.],POSE)[0]
        obj = dict(name='bar',center_m=world.tolist(),size_m=[.08,1.2,.08])
        old = dict(clip_id='old',dense_path='unused.npy',dense_sha256='test',sparse=dict(
            points=pixel,depth_m=[1.],accepted=[True],interval_m=[[.5,2.]]))
        native = np.ones((360,640),np.float32)
        with patch.object(evaluation,'sha',return_value='test'), patch.object(evaluation.np,'load',return_value=native):
            replay = replay_old_clip(old,native,POSE,[obj])
        self.assertEqual(replay['groups']['target']['body_head_near'],1)
        self.assertFalse(interval_near(pixel,[[.5,2.]],POSE)[0][0])

    def test_exact_runner_repair_cannot_authorize_matcher_or_postscore_change(self):
        name = 'run_ba_contour_parallax.py'
        protocol = {'code_hashes':{name:'old','ba_contour_parallax.py':'old'}}
        repair = dict(file=name,original_sha256='old',repaired_sha256='new',kind='RESULT_SERIALIZATION',
            scientific_settings_unchanged=True,completed_predictions_before_repair=0,model_calls_before_repair=0,matcher_calls_before_repair=1)
        with patch.object(evaluation,'sha',return_value='new'), patch.object(evaluation,'read',return_value={'repairs':[repair]}):
            self.assertEqual(evaluation.frozen_code_hash(Path('.'),protocol,name),'new')
            with self.assertRaisesRegex(ValueError,'scientific'):
                evaluation.frozen_code_hash(Path('.'),protocol,'ba_contour_parallax.py')
            repair['completed_predictions_before_repair'] = 1
            with self.assertRaisesRegex(ValueError,'repair receipt'):
                evaluation.frozen_code_hash(Path('.'),protocol,name)

    def test_historical_schema_omission_is_not_zero_and_enriched_count_is_required(self):
        replayed = dict(clip_id='old',all_candidate_count=20,sparse_target_body_head_recovered=False,
            native_target_pixels=5,dense_target_near_pixels=0,
            groups={'target':{'candidates':5,'false_near':0,'false_body_head_near':0},
                    'non_target':{'candidates':15,'false_near':3,'false_body_head_near':2}})
        original = copy.deepcopy(replayed)
        for group in original['groups'].values():
            del group['false_body_head_near']
        evaluation.validate_old_replay(replayed,original,enriched=False)
        evaluation.validate_old_replay(replayed,copy.deepcopy(replayed),enriched=True)
        with self.assertRaisesRegex(ValueError,'Missing required old baseline count'):
            evaluation.validate_old_replay(replayed,original,enriched=True)
        original['groups']['non_target']['false_body_head_near'] = 0
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            evaluation.validate_old_replay(replayed,original,enriched=False)
        del original['groups']['target']['false_near']
        with self.assertRaisesRegex(ValueError,'Missing required old baseline count'):
            evaluation.validate_old_replay(replayed,original,enriched=False)


if __name__ == '__main__':
    unittest.main()
