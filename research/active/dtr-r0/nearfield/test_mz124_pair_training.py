import unittest
import numpy as np
import torch
from run_mz124_pair_training import changed_cell_rank, pair_partition, paired_schedule, STEPS


class PairTrainingTests(unittest.TestCase):
    def test_ranking_uses_signed_changed_cells_only(self):
        labels = torch.tensor([[1.,0.,1.],[0.,1.,1.]])
        logits = torch.tensor([[2.,-2.,-99.],[0.,0.,99.]],requires_grad=True)
        self.assertEqual(changed_cell_rank(logits,labels).item(),0.)
        reversed_logits = -logits.detach(); reversed_logits.requires_grad_()
        loss = changed_cell_rank(reversed_logits,labels)
        self.assertEqual(loss.item(),3.)
        loss.backward()
        self.assertTrue(torch.equal(reversed_logits.grad[:,2],torch.zeros(2)))
        swapped = changed_cell_rank(reversed_logits.flip(0),labels.flip(0))
        self.assertEqual(loss.item(),swapped.item())

    def test_unchanged_pairs_have_finite_zero_gradient(self):
        logits=torch.randn(8,45,requires_grad=True); labels=torch.ones(8,45)
        loss=changed_cell_rank(logits,labels); loss.backward()
        self.assertEqual(loss.item(),0.); self.assertEqual(logits.grad.abs().sum().item(),0.)

    def test_split_and_schedule_never_break_pairs_or_use_dev(self):
        rows=[]; pairs=[]
        for family in ('a','b','c','d'):
            for k in range(3):
                pid=f'{family}_pair{k}'; episodes=[pid+'_in',pid+'_out']
                pairs.append(dict(pair_id=pid,category=family,episodes=episodes))
                for ep in episodes:
                    rows.extend(dict(episode_id=ep,time_s=t*.25) for t in range(12))
        tr,dv,pairing=pair_partition(rows,dict(pairs=pairs))
        schedule,aug=paired_schedule(pairing)
        self.assertEqual(schedule.shape,(STEPS,8)); self.assertTrue(set(schedule.ravel())<=set(tr))
        self.assertFalse(set(schedule.ravel())&set(dv))
        allowed={(p['a'],p['b']) for p in pairing if p['split']=='train'}
        self.assertTrue(all(tuple(p) in allowed for p in schedule.reshape(-1,2)))
        np.testing.assert_array_equal(aug[:,:,0::2],aug[:,:,1::2])
        np.testing.assert_array_equal(schedule,paired_schedule(pairing)[0])


if __name__=='__main__': unittest.main()
