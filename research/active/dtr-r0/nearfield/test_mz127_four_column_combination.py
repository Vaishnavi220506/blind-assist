import unittest
from unittest.mock import patch
from mz127_four_column_combination import four_columns,predict


class FourColumnTests(unittest.TestCase):
    def test_whole_middle_four_bins_keep_every_vertical_row(self):
        zones=[dict(zone_id=r*8+c,theta_bounds_deg=[(c-4)*5.625,(c-3)*5.625],
            phi_bounds_deg=[r,r+1]) for r in range(8) for c in range(8)]
        kept=four_columns(dict(tof_zones=list(reversed(zones))))
        self.assertEqual({z['zone_id'] for z in kept},{r*8+c for r in range(8) for c in (2,3,4,5)})
        self.assertTrue(all(z in zones for z in kept))

    def test_combination_reallocates_using_selected_zones_and_supplied_boxes(self):
        cached=dict(common_radar=True,guard_events=[],integrated_yaw_deg=0.,proposals=[])
        row=dict(id='frame',tof_zones=[dict(zone_id=0)])
        boxes=[[2,3,4,5]]
        with patch('mz127_four_column_combination.four_columns',return_value=[]),patch('mz127_four_column_combination.tof.allocate',return_value=[]) as allocator:
            result=predict(row,cached,boxes,True)
        self.assertEqual(allocator.call_args.args[0]['tof_zones'],[])
        self.assertEqual(allocator.call_args.args[1]['proposals'],boxes)
        self.assertTrue(result['candidate'])


if __name__=='__main__':unittest.main()
