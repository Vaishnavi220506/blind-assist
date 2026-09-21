import unittest
from unittest.mock import patch

import active_view as av
import two_step_observation as step


class TwoStepTests(unittest.TestCase):
    def test_budget_and_pose_closure(self):
        self.assertEqual(len(step.tree_poses()),13)
        for a in step.STEPS:
            for b in step.STEPS:
                self.assertAlmostEqual(sum(abs(x) for x in a)+sum(abs(x) for x in b),.12)
                self.assertIn(step.add(a,b),step.tree_poses())

    def test_decisions_never_turn_empty_support_into_negative(self):
        self.assertEqual(step.decision([], [True,False]),'UNKNOWN')
        self.assertEqual(step.decision([0,1], [True,False]),'UNKNOWN')
        self.assertEqual(step.decision([0], [True,False]),'INTERSECTS')

    def test_public_choice_without_true_scene_or_future_observe(self):
        forecasts={p:((1,),(2,)) for p in step.tree_poses()}
        forecasts[av.ORIGIN]=((0,),(0,))
        with patch.object(av,'observe',side_effect=AssertionError('future')), \
             patch.object(av,'intersects_query',side_effect=AssertionError('truth')):
            choice=step.first_choice((0,),forecasts,(True,False))
        self.assertEqual(choice['action'],0)
        self.assertEqual(step.update(choice['candidates'],(1,),step.STEPS[0],forecasts),(0,))

    def test_second_action_can_depend_on_observed_partition(self):
        labels=(True,False)
        forecasts={p:((0,),(0,)) for p in step.tree_poses()}
        forecasts[step.add(step.STEPS[0],step.STEPS[2])]=((1,),(2,))
        self.assertEqual(step.second_choice([0,1],step.STEPS[0],forecasts,labels)['action'],2)
        self.assertEqual(step.second_choice([],step.STEPS[0],forecasts,labels)['action'],0)


if __name__=='__main__':
    unittest.main()
