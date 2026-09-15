import copy
import json
import unittest

from mz155_active_sampling import source
from mz155_sampling_audit import audit_sampling, ROD


def fixture(packet_loss=False):
    spec = source(); rows = []; es = []
    previous = {}
    for f in spec['frames']:
        step = round(f['time_s']/.25)
        rod_hit = f['family'] == ROD and (step == 0 or (f['observation_arm'] == 'scan' and step == 1))
        late_hit = f['family'] != ROD and step == 5
        hit = rod_hit or late_hit
        packet = not (packet_loss and f['family'] == ROD and f['observation_arm'] == 'scan' and step == 1)
        targets = [dict(status='SIM_VALID', distance_m=3., signal_strength_proxy=10., range_noise_sigma_m=.04),
                   dict(status='SIM_VALID', distance_m=2., signal_strength_proxy=1., range_noise_sigma_m=.04)] if hit and packet else []
        zones = [dict(zone_id=z, theta_bounds_deg=[-22.5+(z%8)*5.625, -22.5+(z%8+1)*5.625],
            phi_bounds_deg=[22.5-(z//8+1)*5.625, 22.5-(z//8)*5.625],
            target_count=len(targets) if z == 0 else 0, targets=copy.deepcopy(targets) if z == 0 else []) for z in range(64)]
        yaw = f['camera']['yaw']; old_yaw = previous.get(f['episode']); previous[f['episode']] = yaw
        delta = 0. if old_yaw is None else yaw-old_yaw+.05
        rows.append(dict(id=f['id'], episode_id=f['episode'], time_s=f['time_s'],
            camera_in_body_m=[0., 0., 1.7], camera_pitch_deg=-3., rgb_path=f['id']+'.png',
            tof_packet_received=packet, tof_zones=zones, radar_packet_received=True,
            radar_valid=[True, False, False, False], radar_range_m=[2., None, None, None],
            radar_angle=[0., None, None, None], delta_yaw=delta, imu_valid=True))
        native = [dict(zone_id=z, packet_received=packet, private_rays=[], returned_lineage=[]) for z in range(64)]
        if hit:
            native[0]['private_rays'] = [dict(subray=0, actor_id=f['episode']+'/shape0', range_m=2.),
                                         dict(subray=1, actor_id=f['episode']+'/env0', range_m=3.)]
        if targets:
            native[0]['returned_lineage'] = [dict(target_index=0, hit_indices=[1]), dict(target_index=1, hit_indices=[0])]
        es.append(dict(id=f['id'], episode_id=f['episode'], time_s=f['time_s'], family=f['family'],
            camera=copy.deepcopy(f['camera']), body_origin_m=copy.deepcopy(f['body_origin_m']),
            native_bounds=[dict(name=o['name'], center_m=list(o['center_m']), extent_m=[s/2 for s in o['size_m']]) for o in f['objects']],
            zonal_tof_native=native))
    return spec, rows, es


class SamplingAuditTests(unittest.TestCase):
    def test_gate_and_second_slot_evidence_with_causal_prefixes(self):
        spec, rows, es = fixture()
        before = json.dumps((spec, rows, es), sort_keys=True)
        result = audit_sampling(spec, rows, es)
        self.assertTrue(result['component_gate']['passed'])
        self.assertEqual(result['component_gate']['rod_geometric_frame_gain'], 6)
        self.assertEqual(result['component_gate']['rod_valid_return_frame_gain'], 6)
        rod = result['by_arm']['scan']['families'][ROD]['all']['signals']
        self.assertEqual(rod['valid_target_return']['current_frames'], 12)
        self.assertEqual(rod['strongest_target_return']['current_frames'], 0)
        late = next(e for e in result['episodes'] if e['family'] != ROD)
        self.assertEqual(late['prefix']['valid_target_return'], [False]*5+[True])
        self.assertEqual(late['first_time_s']['valid_target_return'], 1.25)
        self.assertTrue(result['first_frame_public_sensor_parity']['all_equal'])
        self.assertEqual(result['by_arm']['passive']['all']['counts']['public_radar_valid_slots'], 144)
        self.assertLess(result['paired_stochastic_differences']['maximum_abs_imu_residual_difference_deg'], 1e-12)
        self.assertEqual(before, json.dumps((spec, rows, es), sort_keys=True))
        json.dumps(result, allow_nan=False)

    def test_geometric_gain_does_not_hide_packet_loss(self):
        result = audit_sampling(*fixture(packet_loss=True))
        self.assertFalse(result['component_gate']['passed'])
        self.assertEqual(result['component_gate']['rod_geometric_frame_gain'], 6)
        self.assertEqual(result['component_gate']['rod_valid_return_frame_gain'], 0)
        self.assertEqual(result['paired_stochastic_differences']['tof_packet_disagreements'], 6)

    def test_native_geometry_and_return_correspondence_rejected(self):
        spec, rows, es = fixture()
        es[0]['native_bounds'][0]['center_m'][0] += .01
        with self.assertRaisesRegex(ValueError, 'Native center/extent'):
            audit_sampling(spec, rows, es)
        spec, rows, es = fixture()
        e = next(e for e in es if e['zonal_tof_native'][0]['returned_lineage'])
        e['zonal_tof_native'][0]['returned_lineage'][0]['hit_indices'] = [99]
        with self.assertRaisesRegex(ValueError, 'lineage hit indices'):
            audit_sampling(spec, rows, es)

    def test_lost_ever_observed_case_remains_explicit(self):
        spec, rows, es = fixture()
        episode = next(f['episode'] for f in spec['frames'] if f['family'] == ROD and f['observation_arm'] == 'scan')
        for row in rows:
            if row['episode_id'] == episode:
                row['tof_zones'][0]['targets'] = []
                row['tof_zones'][0]['target_count'] = 0
        for e in es:
            if e['episode_id'] == episode:
                e['zonal_tof_native'][0]['returned_lineage'] = []
        result = audit_sampling(spec, rows, es)
        gate = result['component_gate']
        self.assertFalse(gate['conditions']['no_lost_passive_rod_ever_observed_cases'])
        self.assertFalse(gate['conditions']['no_family_valid_return_ever_count_reduction'])
        self.assertEqual(len(gate['lost_passive_rod_valid_return_cases']), 1)
        self.assertEqual(gate['lost_passive_rod_geometric_cases'], [])
        self.assertEqual(gate['family_valid_return_ever_count_changes'][ROD], -1)

    def test_first_frame_parity_reports_changed_field(self):
        spec, rows, es = fixture()
        scan = next(r for r in rows if r['episode_id'].endswith('_scan') and r['time_s'] == 0.)
        scan['delta_yaw'] = .2
        result = audit_sampling(spec, rows, es)
        self.assertFalse(result['first_frame_public_sensor_parity']['all_equal'])
        self.assertEqual(result['first_frame_public_sensor_parity']['equal_pairs'], 23)
        self.assertEqual(next(p for p in result['first_frame_public_sensor_parity']['pairs'] if not p['equal'])['differing_fields'], ['delta_yaw'])


if __name__ == '__main__':
    unittest.main()
