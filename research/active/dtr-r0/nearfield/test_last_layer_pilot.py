"""Mechanics that can otherwise invalidate a low-FP layout experiment."""
import unittest
import numpy as np
from run_last_layer_pilot import cutoff, hold, row_weights, role_for, xauc


class LastLayerTests(unittest.TestCase):
    def test_cutoff_accounts_for_exit_hold_and_atomic_ties(self):
        rows=[dict(id=str(i),clip_id='a',time_s=i*.2) for i in range(4)]
        y=np.array([False,True,True,False]);a=np.zeros(4,bool)
        scores=np.array([-2.,5.,8.,-1.])
        op=cutoff(scores,a,y,rows)
        self.assertGreater(op['threshold'],8.)
        self.assertEqual(op['forbidden_ids'],['0','2','3'])
        # Current-only cutoff admits the last true frame and creates exit FP.
        current_only=a|(scores>=0)
        self.assertTrue(hold(current_only,rows)[3])
        self.assertFalse(hold(a|(scores>=op['threshold']),rows)[3])

    def test_no_cross_clip_hold_and_no_zero_floor(self):
        rows=[dict(id='a',clip_id='a',time_s=0.),dict(id='b',clip_id='b',time_s=0.)]
        scores=np.array([-2.,-4.]);a=np.array([False,False]);y=np.array([True,False])
        op=cutoff(scores,a,y,rows)
        self.assertLess(op['threshold'],0)
        np.testing.assert_array_equal(hold(scores>=op['threshold'],rows),[True,False])

    def test_equal_mass_per_nonempty_layout_label_cell(self):
        rows=[dict(base_group_id=g) for g in ['a','a','a','b','b','b','b']]
        y=np.array([1,0,0,1,1,1,0],bool)
        w=row_weights(rows,y,True)
        self.assertAlmostEqual(w.sum(),1.)
        for g in ('a','b'):
            for label in (False,True):
                self.assertAlmostEqual(sum(v for r,t,v in zip(rows,y,w) if r['base_group_id']==g and t==label),.25)

    def test_matched_ordering_does_not_imply_cross_layout_order(self):
        rows=[]
        for g,p,n in [('a',10.,9.),('b',2.,1.)]:
            for relation,value,truth in [('INSIDE',p,True),('BOUNDARY',p,True),('OUTSIDE',n,False)]:
                rows.append(dict(base_group_id=g,frame_in_clip=0,layout_relation=relation,truth=truth,
                                 scores={k:value for k in ('original','uniform','balanced')}))
        r=xauc(rows)['original']
        self.assertEqual(r['primary_matched_ordered'],2)
        self.assertEqual(r['matrix'],[[1.,1.],[0.,1.]])
        self.assertEqual(r['off_diagonal_equal_layout_mean'],.5)

    def test_fixed_roles_and_reject_original_test(self):
        self.assertEqual(role_for(dict(base_group_id='x_g01'),'transfer'),'selection')
        self.assertEqual(role_for(dict(base_group_id='x_g02'),'transfer'),'evaluation')
        with self.assertRaises(AssertionError): role_for(dict(split='test'),'dev')


if __name__=='__main__': unittest.main()
