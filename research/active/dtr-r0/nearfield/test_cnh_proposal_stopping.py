import unittest
import numpy as np
from cnh_proposal_stopping import contact,collision_distance,summarize

class StoppingChecks(unittest.TestCase):
    def test_cane_cannot_touch_overhead(self):
        tri=np.array([[[-1,-1.5,1],[1,-1.5,1],[0,-1.5,2]]])
        self.assertFalse(contact(np.array([[0,-.8,0]]),np.array([[0,.8,.6]]),tri,1.).any())
    def test_cane_length_contact(self):
        tri=np.array([[[-1,-1,1],[1,-1,1],[0,1,1]]])
        org=np.array([[0,0,0]]); d=np.array([[0,0,1]])
        self.assertTrue(contact(org,d,tri,1.2)[0]); self.assertFalse(contact(org,d,tri,.8)[0])
    def test_reaction_monotonic_and_degenerate(self):
        distance=np.array([1.,2.,np.inf,0.]); speed=np.ones(4); first=np.zeros(4)
        early=summarize(first,distance,speed,.3); late=summarize(first,distance,speed,1.5)
        self.assertLessEqual(early['collisions'],late['collisions'])
        self.assertEqual(early['unnecessary_stops'],1); self.assertEqual(early['eligible_n'],3)
        never=summarize(np.full(4,np.inf),distance,speed,.3)
        self.assertEqual(never['collisions'],2); self.assertEqual(never['commands'],0)
    def test_lateral_object_not_collision(self):
        obj={'triangles_world':[[[1,-1,1],[2,-1,1],[1,0,2]]]}
        self.assertTrue(np.isinf(collision_distance([obj],1.7,0,1)))
    def test_supporting_floor_not_collision(self):
        obj={'triangles_world':[[[-5,0,-5],[5,0,-5],[0,0,5]]]}
        self.assertTrue(np.isinf(collision_distance([obj],1.7,0,1)))
    def test_forward_contact_distance(self):
        obj={'triangles_world':[[[-.1,-1,1],[.1,-1,1],[0,0,1.1]]]}
        self.assertAlmostEqual(collision_distance([obj],1.7,0,1),.85)
    def test_initial_overlap(self):
        obj={'triangles_world':[[[-.1,-1,-.1],[.1,-1,-.1],[0,0,.1]]]}
        self.assertEqual(collision_distance([obj],1.7,0,1),0)

if __name__=='__main__': unittest.main()
