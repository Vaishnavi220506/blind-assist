import copy
import unittest

from test_tof_directional_readout import frame
from tof_depth_connectivity import depth_edge, depth_readout, near_view
from tof_directional_readout import readout


class DepthConnectivityTest(unittest.TestCase):
    def test_far_bridge_cut_preserves_two_near_modes_and_far_context(self):
        row = frame(list(range(25, 31)))
        for i in range(26, 30):
            row['tof_zones'][i]['targets'][0]['distance_m'] = 4.
        before = copy.deepcopy(row)
        for params in ({'abs_gap_m': .3}, {'abs_gap_m': .15, 'rel_gap': .1}):
            result = depth_readout(row, **params)
            self.assertEqual([r['horizontal'] for r in result['regions']], ['LEFT','CENTER','RIGHT'])
            view = near_view(result, 2.)
            self.assertEqual([r['horizontal'] for r in view['regions']], ['LEFT','RIGHT'])
            self.assertEqual(view['deferred_far_regions'][0]['zone_ids'], list(range(26,30)))
        self.assertEqual(row, before)
        self.assertEqual([r['horizontal'] for r in readout(row)['regions']], ['CENTER'])

    def test_merged_nominal_far_is_not_known_depth_or_discarded(self):
        row = frame([25, 26], 'SIM_MERGED')
        for i in (25, 26):
            row['tof_zones'][i]['targets'][0]['distance_m'] = 3.6
        result = depth_readout(row, abs_gap_m=.3)
        self.assertEqual(result['graph']['accepted_edges'], [])
        self.assertEqual(result['graph']['node_depths_m'], {25:None, 26:None})
        self.assertEqual(len(near_view(result, 2.)['regions']), 2)
        self.assertTrue(all(r['distance_state']=='UNKNOWN_MERGED_ONLY' for r in result['regions']))

    def test_multiple_slots_use_nearest_valid_without_decomposing_zone(self):
        row = frame([25,26])
        row['tof_zones'][25]['targets'] = [dict(status='SIM_VALID',distance_m=4.),dict(status='SIM_VALID',distance_m=1.)]
        row['tof_zones'][26]['targets'][0]['distance_m'] = 1.
        a = depth_readout(row, abs_gap_m=.3)
        self.assertEqual(a['graph']['node_depths_m'][25], 1.)
        self.assertEqual(len(a['regions']),1)
        row['tof_zones'][25]['targets'].reverse()
        b = depth_readout(row, abs_gap_m=.3)
        self.assertEqual(a['regions'], b['regions'])

    def test_strict_gate_and_relative_range_scaling(self):
        self.assertFalse(depth_edge(1.,1.25,.25,0.))
        self.assertTrue(depth_edge(1.,1.249,.25,0.))
        self.assertFalse(depth_edge(1.,1.4,.15,.1))
        self.assertTrue(depth_edge(4.,4.4,.15,.1))
        self.assertFalse(depth_edge(None,1.,.3,0.))

    def test_missing_packet_and_invalid_return_preserve_unknown(self):
        row = frame([25]); row['tof_packet_received'] = False
        self.assertEqual(depth_readout(row,abs_gap_m=.3)['state'],'UNKNOWN_PACKET_MISSING')
        row['tof_packet_received'] = True
        row['tof_zones'][25]['targets'][0]['distance_m'] = float('nan')
        self.assertEqual(depth_readout(row,abs_gap_m=.3)['state'],'UNKNOWN_NO_RETURNS')

    def test_same_depth_shape_and_nonadjacent_columns_do_not_change(self):
        for ids in ([24,25,26], [24,32,31,39], [31,32]):
            row = frame(ids)
            self.assertEqual(depth_readout(row,abs_gap_m=.3)['regions'], readout(row)['regions'])

    def test_common_selector_threshold_far_and_unresolved_regions(self):
        row = frame([25])
        self.assertEqual(len(near_view(readout(row), 2.)['regions']), 1)
        row['tof_zones'][25]['targets'][0]['distance_m'] = 4.
        result = near_view(readout(row), 2.)
        self.assertEqual(result['regions'], [])
        self.assertEqual(result['state'], 'NO_NEAR_VALID_SUPPORT')
        self.assertEqual(len(result['deferred_far_regions']), 1)
        row['tof_zones'][25]['targets'].append(dict(status='SIM_MERGED',distance_m=3.6))
        self.assertEqual(len(near_view(readout(row), 2.)['regions']), 1)

    def test_bridge_success_cannot_be_created_by_deletion_or_unknown(self):
        from run_tof_depth_connectivity import bridge_opportunity, bridge_success
        row = frame([25,26])
        row['tof_zones'][26]['targets'][0]['distance_m'] = 4.
        truth = dict(near_footprints={'LEFT':[25]})
        base = readout(row)
        opportunity = bridge_opportunity(base,truth,2.)
        self.assertTrue(opportunity['eligible'])
        self.assertFalse(bridge_success(base,near_view(base,2.),truth,opportunity))
        raw = depth_readout(row,abs_gap_m=.3)
        selected = near_view(raw,2.)
        self.assertTrue(bridge_success(raw,selected,truth,opportunity))
        # Dropping the far component is display selection, never a graph cut.
        self.assertFalse(bridge_success(selected,selected,truth,opportunity))
        unknown = dict(raw,state='UNKNOWN_NO_RETURNS',regions=[])
        self.assertFalse(bridge_success(unknown,unknown,truth,opportunity))
        row['tof_zones'][25]['targets'] = [dict(status='SIM_MERGED',distance_m=3.6)]
        self.assertFalse(bridge_opportunity(readout(row),truth,2.)['eligible'])


if __name__ == '__main__':
    unittest.main()
