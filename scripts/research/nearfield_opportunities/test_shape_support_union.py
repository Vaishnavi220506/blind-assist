"""Tests of the only proposed rule, including harmful added support accounting."""
import unittest

from shape_support_union import union_state, summarize, contrast


class ShapeUnionTests(unittest.TestCase):
    def test_point_support_cannot_be_vetoed_and_unknown_is_preserved(self):
        for state in ("IN", "OUT_MODEL_PRIOR", "UNKNOWN"):
            self.assertEqual(union_state("IN", state), "IN")
        self.assertEqual(union_state("UNKNOWN", "IN"), "IN")
        self.assertEqual(union_state("UNKNOWN", "UNKNOWN"), "UNKNOWN")
        self.assertEqual(union_state("UNKNOWN", "OUT_MODEL_PRIOR"), "OUT_MODEL_PRIOR")

    def test_false_shape_positive_is_counted_not_called_safe_union(self):
        rows = [dict(case_id="a", truth_inside=False, point_proxy="UNKNOWN", consensus="IN", union="IN"),
                dict(case_id="b", truth_inside=True, point_proxy="IN", consensus="OUT_MODEL_PRIOR", union="IN"),
                dict(case_id="c", truth_inside=True, point_proxy="UNKNOWN", consensus="UNKNOWN", union="UNKNOWN")]
        delta = contrast(rows, "point_proxy")
        self.assertEqual((delta["added_TP"], delta["added_FP"], delta["lost_IN"]), (0, 1, 0))
        result = summarize(rows, "union")
        self.assertEqual((result["TP"], result["FP"], result["FN_alert"]), (1, 1, 1))
        self.assertEqual((result["false_OUT"], result["UNKNOWN_positive"]), (0, 1))

    def test_unexpected_point_clear_state_is_rejected(self):
        with self.assertRaises(ValueError):
            union_state("OUT_MODEL_PRIOR", "IN")


if __name__ == "__main__":
    unittest.main()
