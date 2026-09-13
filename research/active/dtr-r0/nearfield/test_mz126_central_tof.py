import unittest
from mz126_central_tof import selected_zones, alert


class SelectionTests(unittest.TestCase):
    def test_full_footprint_not_center_and_vertical_unrestricted(self):
        row=dict(rgb_intrinsics=dict(width=640,cx=320,cy=180,fx=457.,fy=457.),tof_zones=[
            dict(zone_id=0,theta_bounds_deg=[0,5.625],phi_bounds_deg=[16.875,22.5]),
            dict(zone_id=1,theta_bounds_deg=[5.625,11.25],phi_bounds_deg=[-22.5,-16.875]),
            dict(zone_id=2,theta_bounds_deg=[-22.5,-16.875],phi_bounds_deg=[0,5.625])])
        self.assertEqual([z['zone_id'] for z in selected_zones(row,.25)],[0])
        self.assertEqual([z['zone_id'] for z in selected_zones(row,.5)],[0,1])
        self.assertEqual(len(selected_zones(row,1.)),3)

    def test_no_tof_does_not_delete_radar_or_claim_clearance(self):
        self.assertTrue(alert(dict(common_radar=True,guard_events=[]),[]))
        self.assertFalse(alert(dict(common_radar=False,guard_events=[]),[]))


if __name__=='__main__': unittest.main()
