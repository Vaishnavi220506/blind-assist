import copy
import unittest

from run_candidate_recovery import projected_result


def result(inside=None,outside=None,status_in=0,status_out=0):
    return dict(witnesses={'IN':inside,'OUT':outside},
        solver_metadata=[dict(requested_label=k,solver_status=s,exclusion_supported=s==2)
                         for k,s in [('IN',status_in),('OUT',status_out)]])


class ReadoutTests(unittest.TestCase):
    def test_existing_witness_preserved_without_mutating_input(self):
        baseline=result({'original':True},None,status_out=2)
        saved=copy.deepcopy(baseline)
        output=projected_result(baseline,[dict(label='IN',projected_valid=True,projected_candidate={'replacement':True})])
        self.assertEqual(output['witnesses']['IN'],{'original':True})
        self.assertEqual(output['decision'],'IN_MODEL_CONDITIONAL')
        self.assertEqual(baseline,saved)

    def test_recovered_opposing_witness_does_not_create_label(self):
        output=projected_result(result(outside={'out':True}),[dict(label='IN',projected_valid=True,projected_candidate={'in':True})])
        self.assertEqual(output['decision'],'UNKNOWN')
        self.assertEqual(output['valid_witness_count'],2)

    def test_projection_requires_both_validity_and_original_status(self):
        attempt=dict(label='IN',projected_valid=False,projected_candidate={'in':True})
        self.assertEqual(projected_result(result(status_out=2),[attempt])['decision'],'UNKNOWN')
        attempt['projected_valid']=True
        self.assertEqual(projected_result(result(status_in=1,status_out=2),[attempt])['decision'],'UNKNOWN')
        self.assertEqual(projected_result(result(status_out=2),[attempt])['decision'],'IN_MODEL_CONDITIONAL')

    def test_constructive_contradiction_blocks_numerical_label(self):
        output=projected_result(result(inside={'in':True},outside={'out':True},status_out=2),[])
        self.assertEqual(output['decision'],'UNKNOWN')
        self.assertEqual(output['recovery_conflicts'],['OUT'])


if __name__=='__main__':
    unittest.main()
