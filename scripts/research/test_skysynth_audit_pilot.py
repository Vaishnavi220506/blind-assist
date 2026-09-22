"""Focused checks for the experiment's evidence gate, not model performance."""
import unittest

from skysynth_audit_pilot import check, function, probe


class EvidenceGateTest(unittest.TestCase):
    def test_missing_field_and_empty_assertions_fail(self):
        self.assertFalse(check(lambda *a, **k: {}, probe([], 'missing', 0)))
        self.assertFalse(check(lambda *a, **k: {}, dict(rows=[], dt_s=.2, expectation='return', assertions=[])))

    def test_only_expected_value_error_is_accepted(self):
        def bad(*args, **kwargs):
            raise ValueError('invalid')
        self.assertTrue(check(bad, probe([])))
        self.assertFalse(check(bad, probe([], 'frames', 0)))
        self.assertFalse(check(lambda *a, **k: {}, probe([])))

    def test_boolean_is_not_a_numeric_count(self):
        self.assertFalse(check(lambda *a, **k: {'n': True}, probe([], 'n', 1)))

    def test_nested_list_and_float_tolerance(self):
        fn = lambda *a, **k: {'events': [{'delay': .1 + .2}]}
        self.assertTrue(check(fn, probe([], 'events.0.delay', .3)))
        self.assertFalse(check(fn, probe([], 'events.0.delay', .2)))

    def test_metric_import_has_separate_namespace(self):
        fn = function('def evaluate_rows(rows, **kwargs):\n    return {"n": len(rows)}\n')
        self.assertTrue(check(fn, probe([], 'n', 0)))


if __name__ == '__main__':
    unittest.main()
