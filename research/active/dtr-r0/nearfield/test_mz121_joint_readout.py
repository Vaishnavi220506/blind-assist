"""Regression for the MZ120 spatial-gate-after-selection defect."""
import copy
import unittest
from mz121_joint_readout import select_joint, joint_checks


class JointReadoutTests(unittest.TestCase):
    def baseline(self):
        return dict(metrics=dict(TP=82,FN=2,FP=40),event_hits={'head/0':True},
                    events=dict(max_detected_delay_s=.25,false_alert_bin_duration_s=10))

    def row(self,t,fp,rod):
        return dict(threshold=t,metrics=dict(TP=84,FN=0,FP=fp),event_hits={'head/0':True},
            events=dict(max_detected_delay_s=0,false_alert_bin_duration_s=fp*.25),
            spatial=dict(HEAD=dict(TP=142,FN=0),rod_true_cells=dict(TP=rod,FN=126-rod)))

    def test_lower_fp_point_cannot_win_by_failing_spatial_gate(self):
        chosen,feasible=select_joint([self.row(.71,14,102),self.row(.6,16,114)],self.baseline())
        self.assertEqual(chosen['threshold'],.6);self.assertEqual(len(feasible),1)

    def test_no_joint_point_does_not_fall_back_to_unsafe_winner(self):
        chosen,feasible=select_joint([self.row(.71,14,102)],self.baseline())
        self.assertIsNone(chosen);self.assertEqual(feasible,[])

    def test_missing_critical_class_is_not_perfect_recall(self):
        row=self.row(.6,16,114);row['spatial']['HEAD']=dict(TP=0,FN=0)
        self.assertFalse(joint_checks(row,self.baseline())['head_recall'])

    def test_event_loss_and_delay_are_inside_selection(self):
        row=self.row(.6,16,114);late=copy.deepcopy(row);late['threshold']=.7
        late['events']['max_detected_delay_s']=1.25
        row['event_hits']['head/0']=False
        chosen,_=select_joint([row,late],self.baseline());self.assertIsNone(chosen)


if __name__=='__main__':unittest.main()
