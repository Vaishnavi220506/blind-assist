"""Sampled exit scoring cannot hide false support behind later positive events."""
import unittest
from run_mz113_dynamic_flow import exit_metrics


class ExitMetrics(unittest.TestCase):
    def test_exit_count_reset_and_multiple_runs(self):
        rows=[dict(episode_id='a' if i<7 else 'b',time_s=i*.25) for i in range(9)]
        truth=[True,False,False,False,True,False,False,False,False]
        preds=[dict(candidate=p) for p in [True,True,True,False,True,True,False,True,True]]
        result=exit_metrics(rows,truth,preds)
        self.assertEqual(result['exit_transitions'],2)
        self.assertEqual(result['post_exit_false_frames'],3)
        self.assertEqual(result['longest_post_exit_false_run_s'],.5)


if __name__=='__main__':unittest.main()
