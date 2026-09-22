import unittest

from core_transfer_metrics import ARMS, evaluate


def row(index, truth=False, alerts=(), clip='c', relation=None, layout='INSIDE', unknown=True):
    relation = relation or ('INSIDE' if truth else 'OUTSIDE')
    return dict(id=f'{clip}-{index}', clip_id=clip, frame_in_clip=index, time_s=.2 * index,
        truth=truth, boundary=relation == 'BOUNDARY', relation=relation, layout_relation=layout,
        layer='BODY', type_id='wide', background='a',
        predictions={arm: dict(alert=arm in alerts, unknown=unknown, ambiguous=arm in alerts) for arm in ARMS})


def zone(frame_id='c-0', native=0):
    return dict(sample_id=frame_id + '-z0', frame_id=frame_id,
        truth_relation='OUTSIDE', rgb_relation='OUTSIDE', no_rgb_relation='CROSSING',
        rgb_suppress=True, no_rgb_suppress=False, native_corridor_contributors=native)


class TransferMetricsTest(unittest.TestCase):
    def test_pre_entry_false_alert_removal_is_not_delay(self):
        rows = [row(0, alerts=('raw', 'calibrated', 'no_rgb')), row(1, True, ARMS), row(2, True, ARMS)]
        rows.append(row(0, alerts=('raw', 'calibrated', 'no_rgb'), clip='outside', layout='OUTSIDE'))
        result = evaluate(rows, [zone()])
        self.assertTrue(result['primary_gate']['passed'])
        event = result['metrics']['arms']['rgb']['events'][0]
        self.assertEqual(event['first_alert_relative_to_entry_s'], 0)
        self.assertEqual(result['primary_gate']['retention']['raw']['detected_events_delayed'], [])

    def test_fragmentation_is_reported(self):
        rows = [row(i, alerts=ARMS if i != 1 else ('raw', 'calibrated', 'no_rgb')) for i in range(3)]
        result = evaluate(rows, [zone()])
        self.assertEqual(result['outside_only']['rgb']['FP'], 2)
        self.assertEqual(result['outside_only']['rgb']['false_alert_segment_count'], 2)
        self.assertEqual(result['outside_only']['raw']['false_alert_segment_count'], 1)
        self.assertAlmostEqual(result['outside_only']['rgb']['false_alert_sampled_duration_s'], .4)
        self.assertFalse(result['primary_gate']['passed'])

    def test_relation_mask_does_not_merge_segments(self):
        rows = [row(0, alerts=ARMS), row(1, True, ARMS, relation='BOUNDARY'), row(2, alerts=ARMS)]
        result = evaluate(rows, [zone()])
        self.assertEqual(result['outside_only']['rgb']['false_alert_segment_count'], 2)
        self.assertEqual(result['outside_only']['rgb']['frames'], 2)

    def test_lost_event_and_delayed_onset_fail(self):
        for last_alerts in ((), ARMS):
            rows = [row(0, alerts=('raw', 'calibrated', 'no_rgb')),
                    row(1, True, ('raw', 'calibrated', 'no_rgb')), row(2, True, last_alerts)]
            result = evaluate(rows, [zone()])
            retention = result['primary_gate']['retention']['raw']
            self.assertFalse(result['primary_gate']['passed'])
            self.assertEqual(len(retention['positive_alert_frames_lost']), 1)
            self.assertTrue(retention['detected_events_delayed'] if last_alerts else retention['detected_events_lost'])

    def test_unknown_is_not_true_negative(self):
        result = evaluate([row(0)], [])
        metrics = result['metrics']['arms']['rgb']['frames']['all_known']
        self.assertEqual(metrics['TN'], 0)
        self.assertEqual(metrics['abstained_negative'], 1)
        self.assertEqual(metrics['false_alert_rate_known_negative'], 0)
        self.assertIsNone(metrics['precision'])
        self.assertIsNone(metrics['recall'])
        self.assertFalse(result['primary_gate']['passed'])

    def test_empty_and_absent_zone_opportunities_never_pass(self):
        result = evaluate([], [])
        self.assertFalse(result['primary_gate']['passed'])
        self.assertIsNone(result['zone_attribution']['arms']['rgb']['outside_recall'])
        rows = [row(0, alerts=('raw', 'calibrated', 'no_rgb')), row(1, True, ARMS)]
        self.assertFalse(evaluate(rows, [])['primary_gate']['passed'])

    def test_native_contributor_suppression_fails(self):
        rows = [row(0, alerts=('raw', 'calibrated', 'no_rgb')), row(1, True, ARMS)]
        result = evaluate(rows, [zone(native=2)])
        self.assertFalse(result['primary_gate']['passed'])
        self.assertEqual(result['zone_attribution']['arms']['rgb']['native_corridor_contributions_suppressed'], 2)

    def test_wide_intruding_object_uses_truth_not_layout(self):
        rows = [row(0, True, ARMS, layout='OUTSIDE')]
        result = evaluate(rows, [])
        self.assertEqual(result['metrics']['arms']['rgb']['frames']['all_known']['TP'], 1)
        self.assertEqual(result['outside_only']['rgb']['frames'], 0)
        self.assertEqual(result['core_events']['arms']['rgb']['event_count'], 0)

    def test_boundary_contact_separate_from_core(self):
        rows = [row(0, True, ARMS, layout='BOUNDARY', relation='BOUNDARY')]
        result = evaluate(rows, [])
        self.assertEqual(result['boundary_events']['arms']['rgb']['event_count'], 1)
        self.assertEqual(result['core_events']['arms']['rgb']['event_count'], 0)

    def test_inconsistent_identity_and_relation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([row(0), row(0)], [])
        with self.assertRaises(ValueError):
            evaluate([row(0, True, relation='OUTSIDE')], [])
        with self.assertRaises(ValueError):
            evaluate([row(0)], [zone(), zone()])


if __name__ == '__main__':
    unittest.main()
