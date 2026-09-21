import unittest
from cnh_incremental_audit import scalar_relation, consecutive_runs, summarize


class IncrementalTests(unittest.TestCase):
    def test_invalid_and_nearzero_do_not_become_foreground(self):
        self.assertEqual(scalar_relation(0,False,40,1000),('UNKNOWN',None))
        self.assertEqual(scalar_relation(2,True,40,1000),('SUSPECT_NEAR_ZERO',None))
        self.assertEqual(scalar_relation(30,True,None,1000),('NOT_COMPARABLE',None))
        self.assertEqual(scalar_relation(30,True,50,40),('NOT_COMPARABLE',None))
        self.assertEqual(scalar_relation(520,True,40,1000),('TIE',.5))
        self.assertEqual(scalar_relation(1000,True,40,1000),('BACKGROUND_SIDE',1))
        self.assertLess(scalar_relation(30,True,40,1000)[1],0)

    def test_runs_never_bridge_phase_missing_sequence_or_negative(self):
        rows=[{'seq':seq,'phase':phase,'elapsed_s':t,'hit':hit} for seq,phase,t,hit in
              [(1,'mixture',0,True),(2,'mixture',.2,True),(4,'mixture',.6,True),
               (5,'mixture',.8,False),(6,'mixture',1.,True),(7,'return',8.,True)]]
        runs=consecutive_runs(rows,lambda r:r['hit'])
        self.assertEqual([r['count'] for r in runs],[2,1,1,1])
        self.assertEqual([r['observed_span_s'] for r in runs],[.2,0,0,0])

    def test_unscored_unknown_and_incremental_roles_are_separate(self):
        dual={'dual':True,'late':True,'near_present':True,'amplitude_extrapolation':True}
        rows=[{'scalar_category':c,'cnh':v} for c,v in [('FOREGROUND_SIDE',dual),
              ('BACKGROUND_SIDE',dual),('UNKNOWN',dual),('FOREGROUND_SIDE',None)]]
        result=summarize(rows)
        self.assertEqual(result['region_frames'],4)
        self.assertEqual(result['scored_region_frames'],3)
        self.assertEqual(result['foreground_scalar_with_late'],1)
        self.assertEqual(result['background_scalar_with_near'],1)
        self.assertEqual(result['unknown_with_dual'],1)
        self.assertIn({'scalar':'FOREGROUND_SIDE','cnh':'UNSCORED','count':1},result['cross_table'])


if __name__=='__main__':
    unittest.main()
