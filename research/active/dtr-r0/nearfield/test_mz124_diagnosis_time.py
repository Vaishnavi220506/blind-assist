import unittest
import ast
import itertools
from pathlib import Path
import numpy as np
from mz124_diagnosis_time import causal, evaluate, CELLS, CORE


class CausalTests(unittest.TestCase):
    def test_exact_incumbent_grid_contract(self):
        tree=ast.parse(Path(__file__).with_name('mz120_occupancy.py').read_text())
        names={'DISTANCE_EDGES','SIDE_EDGES','HEIGHT_EDGES','CELLS','CORE'}
        selected=[node for node in tree.body if isinstance(node,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id in names for t in node.targets)]
        namespace={'np':np,'itertools':itertools}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'grid-contract','exec'),namespace)
        np.testing.assert_array_equal(CELLS,namespace['CELLS'])
        np.testing.assert_array_equal(CORE,namespace['CORE'])

    def rows(self, n, episode='a'):
        return [dict(episode_id=episode,time_s=i*.25) for i in range(n)]

    def test_hold_uses_input_and_resets(self):
        rows=self.rows(3)+self.rows(2,'b')
        self.assertEqual(causal(rows,[True,False,False,False,True],'hold1').tolist(),[True,True,False,False,True])

    def test_majority_startup_and_episode_reset(self):
        self.assertEqual(causal(self.rows(3)+self.rows(2,'b'),[True]*5,'two_of_three').tolist(),[False,True,True,False,True])

    def test_future_changes_cannot_change_prefix(self):
        rows=self.rows(6)
        for mode in ('instant','hold1','two_of_three'):
            a=causal(rows,[False,True,False,False,False,False],mode)
            b=causal(rows,[False,True,False,True,True,True],mode)
            np.testing.assert_array_equal(a[:3],b[:3])

    def test_false_segments_delay_and_missed_event_are_not_hidden(self):
        rows=self.rows(5)+self.rows(3,'b')
        gt=np.array([False,True,True,False,False,True,True,True])
        pred=np.array([True,False,True,True,False,False,False,False])
        m=evaluate(rows,gt,pred)
        self.assertEqual((m['TP'],m['FP'],m['FN']),(1,2,4))
        self.assertEqual((m['detected_events'],m['positive_events']),(1,2))
        self.assertEqual((m['false_segment_count'],m['false_duration_s']),(2,.5))
        self.assertEqual(m['maximum_detected_delay_s'],.25)
        self.assertIsNone(m['all_events_detected_delay_bound_s'])
        self.assertIsNone(m['events'][1]['delay_s'])
        self.assertEqual(m['events'][1]['missed_delay_lower_bound_s'],.75)


if __name__ == '__main__': unittest.main()
