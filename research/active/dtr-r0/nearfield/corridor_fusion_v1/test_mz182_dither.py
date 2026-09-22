import unittest
from mz182_dither_evaluate import union,intersect,feasible,readout

class TestDither(unittest.TestCase):
    def test_nonconvex(self):
        self.assertEqual(intersect([[0,1],[3,4]],[[.5,3.5]]),[[.5,1],[3,3.5]])
        self.assertEqual(union([[0,1],[1,2],[3,4]]),[[0,2],[3,4]])
    def test_horizontal_geometry(self):
        self.assertTrue(feasible([[-2,2]],[1,2]))
        self.assertFalse(feasible([[30,40]],[1,2]))
        self.assertFalse(feasible([[-2,2]],[5,6]))
    def test_unknown_does_not_remove_baseline(self):
        p=dict(angles=[[-2,2]],radius=[1,2])
        self.assertTrue(readout([p,dict(angles=[],radius=None)])['alert'])
        self.assertTrue(readout([p,dict(angles=[[5,6]],radius=[1,2])])['alert'])

if __name__=='__main__':unittest.main()
