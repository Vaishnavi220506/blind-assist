import unittest
from audit_score_factor_collapse import attribution


def factors(d,o):
    return dict(d=d,o=o,s=d*o if o is not None else 0)


class AttributionTest(unittest.TestCase):
    def test_replacement_names_and_directions(self):
        flank=factors(.5,.5)
        self.assertEqual(attribution(flank,factors(.05,.5),flank,.1)['category'],'DEPTH_REPLACEMENT_ONLY')
        self.assertEqual(attribution(flank,factors(.5,.05),flank,.1)['category'],'OVERLAP_REPLACEMENT_ONLY')

    def test_ambiguous_both_sufficient_not_forced_cause(self):
        a=attribution(factors(.5,.5),factors(.2,.2),factors(.5,.5),.1)
        self.assertEqual(a['category'],'EITHER_REPLACEMENT_SUFFICIENT')

    def test_both_required_and_reference_failure(self):
        self.assertEqual(attribution(factors(.5,.5),factors(.1,.1),factors(.5,.5),.1)['category'],'BOTH_REPLACEMENTS_REQUIRED')
        self.assertEqual(attribution(factors(.1,.1),factors(.05,.05),factors(.1,.1),.1)['category'],'REFERENCE_NOT_EXPLANATORY')

    def test_missing_or_undefined_overlap_is_not_zero(self):
        for x in (None,factors(0,None)):
            self.assertEqual(attribution(x,factors(.1,.1),factors(.5,.5),.1)['status'],'NOT_EVALUABLE_UNDEFINED_FACTOR')


if __name__=='__main__':
    unittest.main()
