import unittest

import numpy as np

from mz135_temporal_dealias import conditional_tiles, falsifiers, causal_tracks


class TemporalTests(unittest.TestCase):
    def test_falsifiers(self):
        result=falsifiers()
        self.assertIn('COUNTEREXAMPLE',result['wrong_association']['status'])

    def test_all_alternatives_union_and_offimage_retention(self):
        intr=dict(width=100,height=100);roi=[80.,80.,120.,120.]
        boxes=[[[80,80,90,90],[100,100,120,120]],[[80,80,120,120]]]
        tiles,_=conditional_tiles(roi,boxes,intr)
        self.assertEqual(tiles,[roi])
        tiles,_=conditional_tiles(roi,[[[80,80,90,90]],[[80,80,90,90]]],intr)
        self.assertTrue(any(b[0]<=110<=b[2] and b[1]<=110<=b[3] for b in tiles))

    def test_conflict_and_short_window_fallback(self):
        intr=dict(width=640,height=360);roi=[100.,100.,200.,200.]
        self.assertEqual(conditional_tiles(roi,[[[100,100,150,150]]],intr)[0],[roi])
        self.assertEqual(conditional_tiles(roi,[[[0,0,20,20]],[[0,0,20,20]]],intr)[0],[roi])

    def test_causal_window_episode_reset(self):
        rows=[dict(episode_id='a' if i<7 else 'b') for i in range(9)]
        observed=[]
        for entry,history in causal_tracks(rows,[{}]*9,lambda _:np.zeros((32,32),np.uint8)):
            observed.append(len(history)+1)
        self.assertEqual(observed,[1,2,3,4,5,5,5,1,2])


if __name__=='__main__':unittest.main()
