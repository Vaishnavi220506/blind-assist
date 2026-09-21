import copy
import json
from pathlib import Path
import unittest
import numpy as np
from cnh_late_echo import shift_zero, fit_reference, score, evaluate_zone, execution_review_good


class LateEchoTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((Path(__file__).parent.parent/'cnh-late-echo-protocol.json').read_text())
        self.b = np.zeros(24)
        self.f = np.zeros(24)
        self.b[0], self.b[7], self.f[1] = 1., 9., 100.
        self.f[5:10] = [.1, .3, .8, .2, -.1]

    def reference(self):
        return fit_reference([self.b]*10,[self.f]*10,[self.f]*5,[self.b]*5,7,self.p)

    def phases(self):
        waves = {'background':self.b,'foreground':self.f,'mixture':.5*self.b+.5*self.f,'return':self.b}
        return {name:[{'h':np.tile(wave,(16,1)),'block':i//5,'seq':i,'valid':[name!='mixture']*16}
                      for i in range(20)] for name,wave in waves.items()}

    def test_amplitude_and_all_allowed_shift_tails_are_rejected(self):
        ref = self.reference()
        for offset in self.p['template_shifts']:
            for amplitude in (.2,1.,3.):
                result = score(amplitude*shift_zero(self.f,offset),ref,self.p)
                self.assertFalse(result['late'])
                self.assertLessEqual(result['score'],1e-12)
        self.assertTrue(score(.5*self.f+.5*self.b,ref,self.p)['dual'])
        self.assertTrue(score(self.b,ref,self.p)['late'])
        self.assertFalse(score(self.b,ref,self.p)['dual'])

    def test_zero_padding_never_wraps(self):
        wave=np.arange(24.)
        self.assertEqual(shift_zero(wave,1)[0],0)
        self.assertEqual(shift_zero(wave,-1)[-1],0)
        np.testing.assert_array_equal(shift_zero(wave,-1)[:-1],wave[1:])

    def test_unknown_mixture_retained_and_evaluation_cannot_change_reference(self):
        old={'zone':0,'eligible':True,'verdict':'NOT_SUPPORTED','reasons':[]}
        phases=self.phases()
        result=evaluate_zone(old,phases,self.p)
        self.assertEqual(result['verdict'],'EXPLORATORY_TAIL_EXCESS')
        self.assertEqual(result['phases']['mixture']['unknown'],20)
        changed=copy.deepcopy(phases)
        for row in changed['return']:
            row['h']=np.tile(.5*self.f+.5*self.b,(16,1))
        failed=evaluate_zone(old,changed,self.p)
        self.assertEqual(failed['verdict'],'NOT_SUPPORTED')
        self.assertFalse(failed['gates']['return_dual'])
        self.assertEqual(result['reference'],failed['reference'])
        for row in changed['foreground']:
            if row['block']==3:
                row['h']=np.tile(self.f+self.b,(16,1))
        failed=evaluate_zone(old,changed,self.p)
        self.assertFalse(failed['gates']['foreground_negative'])
        self.assertEqual(result['reference'],failed['reference'])

    def test_reference_quality_and_zero_normalization_are_not_negative(self):
        old={'zone':0,'eligible':False,'verdict':'NOT_EVALUABLE','reasons':['invalid pure reference']}
        self.assertEqual(evaluate_zone(old,self.phases(),self.p)['verdict'],'NOT_EVALUABLE')
        with self.assertRaises(ValueError):
            fit_reference([self.b],[np.zeros(24)],[self.f],[self.b],7,self.p)
        with self.assertRaises(ValueError):
            fit_reference([self.b],[self.f],[self.f],[self.b],4,self.p)

    def test_execution_review_cannot_be_bypassed_by_numeric_success(self):
        keys = ('fixed_rig_background','foreground_full_image','mixture_partial_image',
                'approximately_same_foreground_distance','return_clear')
        old = {'rgb_review':dict.fromkeys(keys, True),
               'camera_samples':{n:[{'path':'image.jpg'}]*3 for n in self.phases()}}
        old['rgb_review']['run_id']='sample'
        self.assertTrue(execution_review_good(old,'sample'))
        self.assertFalse(execution_review_good(old,'wrong-run'))
        old['rgb_review']['return_clear']=False
        self.assertFalse(execution_review_good(old,'sample'))
        old['rgb_review']['return_clear']=True
        old['camera_samples']['return']=[]
        self.assertFalse(execution_review_good(old,'sample'))


if __name__=='__main__':
    unittest.main()
