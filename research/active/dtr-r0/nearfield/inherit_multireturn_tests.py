"""Synthetic tests only; no pilot inputs or outcome access."""
import unittest
from unittest.mock import patch
import numpy as np
import inherit_multireturn_20260922 as m


class InheritanceTests(unittest.TestCase):
    def ids(self,n=3):
        return [dict(index=i,id=f'x{i}',clip_id='c',frame_in_clip=i,time_s=.2*i) for i in range(n)]

    def test_unknown_and_nonrecursive_hold(self):
        ranges=np.full((3,64,2),np.nan,np.float32); ranges[0,35,0]=1.
        rows=m.prediction_rows(self.ids(),m.boxes45(),ranges)
        self.assertEqual([r['flags']['strongest_hold'] for r in rows],[True,True,False])
        self.assertTrue(rows[1]['predictions']['strongest_hold']['unknown'])
        self.assertFalse(rows[1]['flags']['strongest'])

    def test_closest_can_lose_support_and_two_keeps_it(self):
        ranges=np.full((1,64,2),np.nan,np.float32); ranges[0,35]=[1.,.1]
        row=m.prediction_rows(self.ids(1),m.boxes45(),ranges)[0]
        self.assertTrue(row['flags']['strongest']); self.assertFalse(row['flags']['closest_exported'])
        self.assertTrue(row['flags']['two_returns'])

    def test_prediction_does_not_read_files_or_native(self):
        ranges=np.full((3,64,2),np.nan,np.float32); ranges[0,35]=[5.,1.]
        with (patch.object(m,'read',side_effect=AssertionError('No file inputs')),
              patch.object(m,'simulate_two',side_effect=AssertionError('No depth construction'))):
            rows=m.prediction_rows(self.ids(),m.boxes45(),ranges)
        self.assertEqual([r['flags']['two_returns_hold'] for r in rows],[True,True,False])
        self.assertEqual(rows[0]['triggers']['two_returns'],[dict(zone=35,slot=1)])

    def test_hold_does_not_cross_clips(self):
        ranges=np.full((2,64,2),np.nan,np.float32); ranges[0,35,0]=1.
        identities=self.ids(2); identities[1].update(clip_id='new',frame_in_clip=0,time_s=0.)
        rows=m.prediction_rows(identities,m.boxes45(),ranges)
        self.assertFalse(rows[1]['flags']['strongest_hold'])

    def test_native_contributor_not_outside_corridor(self):
        native=np.ones((360,640),np.float32)
        obj=dict(name='target',render_bounds_center_m=[1,0,1.82],render_bounds_extent_m=[.1,2,2])
        mask=m.contributor_mask(native,dict(x=0,y=0,z=1.82),[obj]).reshape(192,256)
        self.assertTrue(mask[96,128]); self.assertFalse(mask[96,0])
        obj['render_bounds_center_m'][1]=5
        self.assertFalse(m.contributor_mask(native,dict(x=0,y=0,z=1.82),[obj]).any())


if __name__=='__main__':
    unittest.main()
