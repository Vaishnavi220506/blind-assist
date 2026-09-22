"""Focused acceptance checks, using synthetic public inputs only."""
import unittest
import numpy as np

from inherit_spatial_model import extract, query_bands, canonical_tof, RAW_SIZE, FEATURE_SIZE
from inherit_spatial_run import select_threshold, metric_rows, report
from tof_fov45_core import boxes45


class InheritedSpatialContract(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0,256,(3,180,320),dtype=np.uint8)
        tof = np.zeros((64,6),np.float32)
        tof[:,0] = 1.5/8
        tof[:,1] = 1
        tof[:,2:] = boxes45()/[192,256,192,256]
        return image,tof

    def test_matched_common_no_mutation_and_missing_local(self):
        image,tof = self.inputs()
        before = tof.copy()
        values = extract(image,tof)
        self.assertEqual(values.shape,(6,2,FEATURE_SIZE))
        np.testing.assert_array_equal(values[:,0,:RAW_SIZE],values[:,1,:RAW_SIZE])
        self.assertTrue((values[:,0,RAW_SIZE:]==0).all())
        self.assertTrue((values[:,1,RAW_SIZE:]!=0).any())
        np.testing.assert_array_equal(tof,before)
        tof[:,1] = 0
        missing = extract(image,tof)
        self.assertTrue((missing[:,:,RAW_SIZE:]==0).all())
        self.assertTrue((missing[:,:,4:14]!=0).any())

    def test_general_query_interval_partition(self):
        ax=np.array([0.,.2,.2,-.4,0.]); ay=np.array([.4,.4,.4,.4,0.])
        low=np.array([1.2,1.2,1.4,1.2,1.2]); high=np.array([1.8,1.8,1.5,1.8,1.8])
        masks=query_bands(ax,ay,low,high,[-.3,.3,.42,.9])
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks),np.ones(5))
        np.testing.assert_array_equal(masks[0],[True,False,True,False,False])
        np.testing.assert_array_equal(masks[1],[False,True,False,False,False])

    def test_invalid_range_cannot_supply_band(self):
        _,tof=self.inputs()
        tof[0,0]=np.nan
        z,valid,_,lo,hi=canonical_tof(tof)
        self.assertFalse(valid[0]); self.assertEqual(z[0],0)
        self.assertEqual(lo[0],0); self.assertEqual(hi[0],0)

    def test_atomic_ties_disabled_and_OR_retention(self):
        # Two equally scored negative rows exhaust the dev +1 FP budget.
        metas=[dict(clip_id=str(i),baseline=dict(alert=i==0)) for i in range(4)]
        labels=dict(classes=np.array([[0]*6,[0]*6,[6]*6,[6]*6]),valid=np.ones((4,6),bool))
        selection=select_threshold(np.ones(4)*.5,labels,metas)
        self.assertGreater(selection['selection']['threshold'],.5)
        self.assertEqual(selection['selection']['TP'],1)
        self.assertEqual(selection['selection']['FP'],0)
        self.assertEqual(len(selection['curve']),2)


if __name__=='__main__':
    unittest.main()
