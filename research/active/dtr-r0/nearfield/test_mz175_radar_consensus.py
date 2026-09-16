"""Synthetic public-contract checks; no source records or models are loaded."""
import copy
import unittest
from unittest.mock import patch

import mz175_radar_consensus as consensus
from mz111_spatial_evidence import current_returns


def public_row(**overrides):
    row = dict(id='synthetic', imu_valid=True, radar_packet_received=True,
               radar_range_m=[2., None, None, None],
               radar_angle=[10., None, None, None],
               radar_velocity=[None] * 4, radar_valid=[True, False, False, False],
               camera_pitch_deg=0., camera_in_body_m=[0., 0., 1.7],
               rgb_intrinsics=dict(width=640, height=360, fx=450., fy=450.,
                                   cx=320., cy=180.))
    row.update(overrides)
    return row


def corrected(boxes):
    return dict(proposals=copy.deepcopy(boxes), integrated_yaw_deg=0.)


def return_record(slot, candidates, proposal=None):
    return dict(slot=slot, range_m=2., baseline=False,
                candidates=candidates, proposal=proposal)


class RadarConsensusTests(unittest.TestCase):
    def assert_branches(self, result, any_support, all_support, baseline=False):
        self.assertEqual(result['additional'], {'any': any_support, 'all': all_support})
        for arm, support in [('any', any_support), ('all', all_support)]:
            flag = baseline or support
            self.assertEqual(result['arms'][arm]['candidate'], flag)
            self.assertEqual(result['arms'][arm]['state'], 'ALERT' if flag else 'UNKNOWN')
        self.assertEqual(result['baseline'], baseline)
        self.assertTrue(result['full_baseline_retained'])

    def test_missing_radar_or_imu_adds_nothing_and_retains_incumbent(self):
        boxes = corrected([[300, 130, 320, 210], [330, 120, 355, 220]])
        for old in [False, True]:
            for row in [public_row(radar_packet_received=False),
                        public_row(radar_valid=[False] * 4)]:
                with self.subTest(old=old, row=row):
                    result = consensus.predict_frame(row, boxes, {'candidate': old})
                    self.assert_branches(result, False, False, old)
                    self.assertEqual(result['evidence'], [])
            # Invalid IMU must not even request candidate geometry.
            with patch.object(consensus, 'current_returns', side_effect=AssertionError('IMU fallback')):
                result = consensus.predict_frame(public_row(imu_valid=False), boxes,
                                                 {'candidate': old})
            self.assert_branches(result, False, False, old)
            self.assertFalse(result['imu_available'])

    def test_empty_and_singleton_ambiguous_sets_do_not_trigger(self):
        cases = [(public_row(), corrected([])),
                 (public_row(radar_range_m=[2., 2.2, None, None],
                             radar_angle=[10., 10., None, None],
                             radar_valid=[True, True, False, False]),
                  corrected([[300, 130, 350, 210]]))]
        for row, geometry in cases:
            # Two returns compete for one proposal: this really is ambiguous,
            # despite each return having a singleton candidate set.
            returns = current_returns(row, geometry)
            self.assertTrue(all(ret['proposal'] is None for ret in returns))
            self.assertTrue(all(len(ret['candidates']) < 2 for ret in returns))
            with patch.object(consensus, 'plane_inside', side_effect=AssertionError('small candidate set')):
                result = consensus.predict_frame(row, geometry, {'candidate': False})
            self.assert_branches(result, False, False)
            self.assertEqual(result['evidence'], [])

    def test_unique_association_is_skipped_even_if_raw_plane_would_alert(self):
        row = public_row()
        geometry = corrected([[300, 130, 350, 210]])
        returns = current_returns(row, geometry)
        self.assertEqual(returns[0]['proposal'], 0)
        self.assertTrue(consensus.plane_inside(geometry['proposals'][0], 2., row, 0.))
        with patch.object(consensus, 'plane_inside', side_effect=AssertionError('raw unique reassociation')):
            result = consensus.predict_frame(row, geometry, {'candidate': False})
        self.assert_branches(result, False, False)
        self.assertEqual(result['evidence'], [])

    def test_any_and_all_reduce_within_each_return_then_across_returns(self):
        geometry = corrected([[0, 0, 1, 1], [2, 0, 3, 1],
                              [4, 0, 5, 1], [6, 0, 7, 1]])
        decisions = {0: True, 2: False, 4: True, 6: False}
        mixed = return_record(0, [0, 1])
        unanimous = return_record(1, [0, 2])
        for records, expected in [([mixed], (True, False)),
                                  ([mixed, unanimous], (True, True)),
                                  ([return_record(0, [1, 3])], (False, False))]:
            # Mock only the Boolean surface decision to isolate quantifiers.
            with patch.object(consensus, 'current_returns', return_value=records), \
                    patch.object(consensus, 'plane_inside',
                                 side_effect=lambda box, *_: decisions[box[0]]):
                result = consensus.predict_frame(public_row(), geometry, {'candidate': False})
            self.assert_branches(result, *expected)
            self.assertEqual(len(result['evidence']), len(records))
            self.assertEqual([len(e['hypotheses']) for e in result['evidence']],
                             [len(ret['candidates']) for ret in records])

    def test_real_geometry_distinguishes_mixed_and_unanimous_hypotheses(self):
        row = public_row()
        # At horizontal range 2 m the first box overlaps |y|<=.3 m;
        # the right box lies wholly beyond y=.3 m. Both satisfy the
        # unchanged 12-degree Radar angular eligibility around 10 degrees.
        inside = [300, 130, 350, 210]
        outside = [440, 130, 470, 220]
        geometry = corrected([inside, outside])
        raw = current_returns(row, geometry)[0]
        self.assertIsNone(raw['proposal'])
        self.assertEqual(raw['candidates'], [0, 1])
        self.assertFalse(raw['baseline'])
        result = consensus.predict_frame(row, geometry, {'candidate': False})
        self.assert_branches(result, True, False)
        self.assertEqual([h['support'] for h in result['evidence'][0]['hypotheses']],
                         [True, False])
        inside_pair = corrected([[300, 130, 320, 210], [330, 120, 355, 220]])
        self.assert_branches(consensus.predict_frame(row, inside_pair, {'candidate': False}),
                             True, True)

    def test_candidate_order_and_private_metadata_do_not_change_decisions_or_inputs(self):
        row = public_row()
        geometry = corrected([[300, 130, 350, 210], [440, 130, 470, 220]])
        baseline = {'candidate': False}
        original = copy.deepcopy((row, geometry, baseline))
        forward = consensus.predict_frame(row, geometry, baseline)
        reversed_geometry = corrected(list(reversed(geometry['proposals'])))
        tagged = dict(row, truth=True, family='irrelevant', actor_id='hidden',
                      native_yaw_deg=999., observation_arm='irrelevant')
        reverse = consensus.predict_frame(tagged, reversed_geometry, baseline)
        self.assertEqual(forward['arms'], reverse['arms'])
        self.assertEqual(forward['additional'], reverse['additional'])
        normalize = lambda output: sorted((tuple(h['box']), h['support'])
                                          for h in output['evidence'][0]['hypotheses'])
        self.assertEqual(normalize(forward), normalize(reverse))
        self.assertEqual((row, geometry, baseline), original)


if __name__ == '__main__':
    unittest.main()
