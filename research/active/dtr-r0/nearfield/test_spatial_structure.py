import unittest
import numpy as np
from spatial_structure_model import ordered_grid,build_inputs,arm_inputs
from tof_fov45_core import boxes45


class SpatialStructureTest(unittest.TestCase):
    def test_lossless_order(self):
        source=np.arange(216*64,dtype=np.float32).reshape(1,216,8,8)
        grid=ordered_grid(source)
        for feature in range(24):
            for row in range(24):
                for col in range(24):
                    self.assertEqual(grid[0,feature,row,col],source[0,feature*9+(row%3)*3+col%3,row//3,col//3])

    def test_query_geometry_and_ablation(self):
        boxes=boxes45(); f=np.zeros((1,216,8,8),np.float32)
        values=np.full((1,64),2.,np.float32)
        x=build_inputs(f,values,boxes)
        focal=640/(2*np.tan(np.deg2rad(50)))
        radius=.1+3*(.01+.02*2)
        for row,col in [(0,0),(7,11),(12,12),(23,23)]:
            box=boxes[(row//3)*8+col//3]
            px=(box[1]+(box[3]-box[1])*(col%3+.5)/3)*640/256
            py=(box[0]+(box[2]-box[0])*(row%3+.5)/3)*360/192
            want=[]
            for z in (2-radius,2+radius):
                xx,yy=(px-320)/focal*z,(py-180)/focal*z
                want.extend([xx+.3,.3-xx,yy+.2,.9-yy])
            np.testing.assert_allclose(x[0,28:,row,col],np.clip(want,-4,4)/4,atol=2e-7,rtol=0)
        u=arm_inputs(x,'U'); g=arm_inputs(x,'G')
        np.testing.assert_array_equal(u[:,:28],g[:,:28]); self.assertTrue((u[:,28:]==0).all())
        self.assertTrue(np.any(g[:,28:]!=0)); np.testing.assert_array_equal(x,g)

    def test_missing_and_no_rgb_average(self):
        f=np.arange(216*64,dtype=np.float32).reshape(1,216,8,8)
        x=build_inputs(f,np.full((1,64),np.nan),boxes45())
        self.assertTrue((x[:,24:26]==0).all()); self.assertTrue((x[:,28:]==0).all())
        np.testing.assert_array_equal(x[:,:24],ordered_grid(f))


if __name__=='__main__': unittest.main()
