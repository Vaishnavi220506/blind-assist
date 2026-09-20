import unittest

from zone_handoff_ceiling import adjacent, depth_overlap, features, shared_corridor_depth


def boxes():
    return [[40+14*r, 84+11*c, 54+14*r, 95+11*c] for r in range(8) for c in range(8)]


class HandoffTest(unittest.TestCase):
    def test_topology(self):
        self.assertTrue(adjacent(27, 28))
        self.assertTrue(adjacent(27, 35))
        self.assertFalse(adjacent(27, 36))
        self.assertFalse(adjacent(7, 8))

    def test_depth_touch_is_not_overlap(self):
        self.assertFalse(depth_overlap([1, 2], [2, 3]))
        self.assertTrue(depth_overlap([1, 2.1], [2, 3]))

    def test_shared_edge_inside_and_outside_corridor(self):
        intervals={27:[1, 2],28:[1.5, 2.5],24:[2, 3],25:[2, 3]}
        self.assertEqual(shared_corridor_depth(27,28,boxes(),intervals),[1.5,2])
        self.assertIsNone(shared_corridor_depth(24,25,boxes(),intervals))

    def test_pair_and_unrelated_weak_zones(self):
        anchors=[dict(zone=z,possible=True,interval_m=[1,2]) for z in (27,28,63)]
        scores=[dict(zone=z,joint=s) for z,s in ((27,.004),(28,.0035),(63,.002))]
        f=features(boxes(),anchors,scores)
        self.assertAlmostEqual(f['S_component'],.0075)
        self.assertAlmostEqual(f['S_all'],.0095)
        self.assertTrue(f['top2']['same_component'])
        self.assertEqual(len(f['components']),2)

    def test_no_valid_support_and_zero_score(self):
        f=features(boxes(),[],[])
        self.assertEqual((f['S_component'],f['S_all'],f['positive_zone_count']),(0,0,0))
        f=features(boxes(),[dict(zone=27,possible=True,interval_m=[1,2])],[dict(zone=27,joint=0)])
        self.assertEqual(f['components'],[])

    def test_depth_disjoint_blocks_connected_sum(self):
        a=[dict(zone=27,possible=True,interval_m=[1,1.5]),dict(zone=28,possible=True,interval_m=[2,2.5])]
        s=[dict(zone=27,joint=.004),dict(zone=28,joint=.0035)]
        f=features(boxes(),a,s)
        self.assertEqual(f['S_component'],.004)
        self.assertFalse(f['top2']['same_component'])


if __name__ == '__main__':
    unittest.main()
