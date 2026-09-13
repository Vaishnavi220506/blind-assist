import copy
import unittest
from mz115_zonal_tof import zone_geometry
from tof_directional_readout import readout


def frame(ids, status='SIM_VALID'):
    zones=[]
    for i in range(64):
        g,_=zone_geometry(i)
        zones.append(dict(g, targets=[dict(status=status,distance_m=2.)] if i in ids else []))
    return dict(tof_packet_received=True,tof_zones=zones)


class DirectionalReadoutTest(unittest.TestCase):
    def test_bilateral_regions_do_not_average_to_center(self):
        result=readout(frame([24,32,31,39]))
        self.assertEqual([r['horizontal'] for r in result['regions']],['LEFT','RIGHT'])

    def test_merged_only_preserves_direction_without_metric_distance(self):
        r=readout(frame([24,32],'SIM_MERGED'))['regions'][0]
        self.assertEqual(r['horizontal'],'LEFT')
        self.assertIsNone(r['valid_slant_median_m'])

    def test_missing_packet_and_no_returns_are_unknown(self):
        f=frame([27]);f['tof_packet_received']=False
        self.assertEqual(readout(f)['state'],'UNKNOWN_PACKET_MISSING')
        self.assertEqual(readout(frame([]))['state'],'UNKNOWN_NO_RETURNS')

    def test_duplicate_targets_do_not_amplify_a_zone(self):
        f=frame([27,28]);g=copy.deepcopy(f)
        g['tof_zones'][27]['targets']*=2
        self.assertEqual(readout(f)['regions'],readout(g)['regions'])

    def test_single_zone_kept_and_vertical_orientation(self):
        r=readout(frame([3]))['regions'][0]
        self.assertEqual((r['horizontal'],r['vertical']),('CENTER','UPPER'))
        self.assertEqual(readout(frame([59]))['regions'][0]['vertical'],'LOWER')


if __name__=='__main__':unittest.main()
