"""Synthetic timing/denominator checks; no experiment outcomes used."""
import json
import random
import unittest

from full_event_metrics_20260920 import ARMS, evaluate


def clip(truth, alerts, clip_id='a', layout='INSIDE', unknown=False):
    return [dict(id=f'{clip_id}-{i}', clip_id=clip_id, time_s=i * .2,
                 truth=bool(t), layout_relation=layout, layer='HEAD', background='wall',
                 phase='approach' if i < 2 else 'depart' if i >= len(truth) - 2 else 'dwell',
                 flags={arm: bool(a) for arm in ARMS}, current_unknown={arm: unknown for arm in ARMS})
            for i, (t, a) in enumerate(zip(truth, alerts))]


def arm_result(rows):
    return evaluate(rows)['overall']['arms']['strongest_hold']


class FullEventMetricsTest(unittest.TestCase):
    def test_episode_starts_separate_preexisting_internal_and_postexit_alerts(self):
        rows = clip([0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0])
        rows += clip([0, 0, 0], [1, 0, 1], 'outside', 'OUTSIDE')
        result = arm_result(rows)
        event = result['events'][0]
        self.assertTrue(event['preexisting_alert_at_entry'])
        self.assertEqual(event['in_event_new_alert_episode_count'], 1)
        self.assertEqual(event['in_event_new_alert_episode_start_times_s'], [3 * .2])
        self.assertEqual(event['postexit_new_alert_episode_count'], 1)
        self.assertEqual(result['clips'][0]['alert_episode_count'], 3)
        self.assertEqual(result['clips'][0]['alert_episode_start_times_s'], [0, 3 * .2, 1.0])
        self.assertEqual(result['clips'][1]['alert_episode_count'], 2)
        self.assertEqual(result['alert_episode_count'], 5)

    def test_preentry_false_alert_does_not_erase_onset_delay(self):
        result = arm_result(clip([0, 1, 1, 0], [1, 0, 1, 0]))
        event = result['events'][0]
        self.assertEqual(result['frames']['FP'], 1)
        self.assertEqual(event['first_in_event_alert_delay_s'], .2)
        self.assertEqual(event['whole_clip_first_alert_relative_to_entry_s'], -.2)
        self.assertEqual(event['initial_silent_frames'], 1)
        self.assertEqual(event['positive_coverage'], .5)

    def test_silence_partition_and_internal_interruptions(self):
        result = arm_result(clip([0] + [1] * 8 + [0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]))
        event = result['events'][0]
        self.assertEqual(event['initial_silent_frames'], 1)
        self.assertEqual(event['internal_interruption_count'], 2)
        self.assertEqual(event['internal_silent_frames'], 3)
        self.assertAlmostEqual(event['internal_silent_sampled_s'], .6)
        self.assertEqual(event['terminal_silent_frames'], 1)
        self.assertEqual(event['total_silent_frames'], 5)

    def test_immediate_clear_and_one_sample_hold_tail(self):
        for alerts, tail in (([0, 1, 0, 0], 0), ([0, 1, 1, 0], 1)):
            event = arm_result(clip([0, 1, 0, 0], alerts))['events'][0]
            self.assertEqual(event['postexit_carryover_alert_frames'], tail)
            self.assertEqual(event['postexit_carryover_sampled_s'], tail * .2)
            self.assertEqual(event['first_silent_relative_to_exit_s'], tail * .2)
            self.assertFalse(event['release_right_censored'])

    def test_later_reactivation_is_not_tail(self):
        event = arm_result(clip([0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1]))['events'][0]
        self.assertEqual(event['postexit_carryover_sampled_s'], .2)
        self.assertEqual(event['postexit_false_alert_frames'], 3)
        self.assertEqual(event['postexit_false_alert_episode_count'], 2)
        self.assertEqual(event['postexit_new_alert_episode_count'], 1)
        self.assertEqual(event['postexit_new_alert_episode_start_times_s'], [.8])

    def test_new_alert_after_terminal_silence_is_not_carryover(self):
        event = arm_result(clip([0, 1, 1, 0, 0], [0, 1, 0, 1, 0]))['events'][0]
        self.assertEqual(event['postexit_carryover_alert_frames'], 0)
        self.assertEqual(event['postexit_leading_alert_frames'], 1)
        self.assertEqual(event['postexit_new_alert_episode_count'], 1)

    def test_release_censorship_and_absent_exit(self):
        event = arm_result(clip([0, 1, 0, 0], [0, 1, 1, 1]))['events'][0]
        self.assertTrue(event['release_right_censored'])
        self.assertIsNone(event['first_silent_relative_to_exit_s'])
        self.assertEqual(event['postexit_carryover_sampled_s'], .4)
        event = arm_result(clip([0, 1], [0, 1]))['events'][0]
        self.assertFalse(event['exit_observed'])
        self.assertTrue(event['release_right_censored'])
        self.assertIsNone(event['postexit_carryover_sampled_s'])

    def test_missed_event_has_nonoverlapping_silence_and_unknown_not_tn(self):
        result = arm_result(clip([0, 1, 1, 0], [0, 0, 0, 0], unknown=True))
        event = result['events'][0]
        self.assertFalse(event['detected'])
        self.assertIsNone(event['first_in_event_alert_delay_s'])
        self.assertEqual(event['initial_silent_frames'], 2)
        self.assertEqual(event['terminal_silent_frames'], 0)
        self.assertEqual(result['frames']['FN'], 2)
        self.assertEqual(result['frames']['TN'], 0)
        self.assertEqual(result['frames']['abstained_negative'], 2)
        self.assertEqual(result['frames']['current_unknown'], 4)
        self.assertEqual(result['frames']['FPR'], 0)

    def test_outside_frames_in_core_denominator_without_fake_event(self):
        rows = clip([0, 1, 0], [0, 1, 0]) + clip([0, 0, 0], [1, 0, 0], 'b', 'OUTSIDE')
        rows += clip([0, 1, 0], [1, 0, 0], 'c', 'BOUNDARY')
        result = evaluate(rows)
        core = result['Core']['arms']['strongest_hold']
        self.assertEqual(core['frames']['negative_frames'], 5)
        self.assertEqual(core['frames']['FPR'], .2)
        self.assertEqual(core['event_count'], 1)
        self.assertIsNone(core['clips'][1]['event'])
        self.assertEqual(result['Boundary']['arms']['strongest_hold']['frames']['FN'], 1)

    def test_sorting_paired_flags_and_json_serialization(self):
        rows = clip([0, 1, 1, 0], [0, 1, 0, 1])
        rows[2]['flags']['closest_hold'] = True
        expected = evaluate(rows)
        random.Random(42).shuffle(rows)
        self.assertEqual(evaluate(rows), expected)
        json.dumps(expected, allow_nan=False)
        self.assertEqual(expected['overall']['arms']['closest_hold']['frames']['FN'], 0)
        self.assertEqual(expected['overall']['arms']['strongest_hold']['frames']['FN'], 1)

    def test_no_denominator_in_empty_group_and_invalid_timeline_rejected(self):
        self.assertIsNone(evaluate([])['overall']['arms']['strongest_hold']['frames']['FPR'])
        rows = clip([0, 1, 0], [0, 1, 0])
        for bad in (rows + rows, rows[1:], clip([1, 0, 1], [1, 0, 1]),
                    clip([0, 1, 0], [0, 1, 0], layout='OUTSIDE')):
            with self.assertRaises(ValueError):
                evaluate(bad)


if __name__ == '__main__':
    unittest.main()
