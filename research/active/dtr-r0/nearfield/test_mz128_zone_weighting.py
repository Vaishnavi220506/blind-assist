import unittest
from mz128_zone_weighting import readout,column_weights,SOFT


def evidence(zone):
    return dict(zone_id=zone,localized_xyz=[(.5,1.),(-.1,.1),(1.,1.2)],coarse_xyz=[(.1,2.),(-1.,1.),(0.,3.)])


class WeightTests(unittest.TestCase):
    def test_duplicates_in_zone_do_not_multiply_votes(self):
        p=dict(common_radar=False,guard_events=[],integrated_yaw_deg=0.,spatial_evidence=[evidence(1),evidence(1)])
        self.assertEqual(readout(p,{1:.5})['score'],.5)
        self.assertFalse(readout(p,{1:.5})['candidate'])
        p['spatial_evidence'].append(evidence(2))
        self.assertTrue(readout(p,{1:.5,2:.5})['candidate'])

    def test_radar_and_certain_support_survive_low_weight(self):
        p=dict(common_radar=True,guard_events=[],integrated_yaw_deg=0.,spatial_evidence=[])
        self.assertTrue(readout(p,{})['candidate'])
        p['common_radar']=False;e=evidence(1);e['coarse_xyz']=e['localized_xyz'];p['spatial_evidence']=[e]
        self.assertTrue(readout(p,{1:0.})['candidate'])

    def test_symmetric_and_vertical_invariant(self):
        row=dict(tof_zones=[dict(zone_id=r*8+c,theta_bounds_deg=[c,c+1]) for r in range(8) for c in range(8)])
        w=column_weights(row,SOFT)
        self.assertEqual([w[i] for i in range(8)],[w[i] for i in range(7,-1,-1)])
        self.assertTrue(all(w[r*8+c]==w[c] for r in range(8) for c in range(8)))


if __name__=='__main__':unittest.main()
