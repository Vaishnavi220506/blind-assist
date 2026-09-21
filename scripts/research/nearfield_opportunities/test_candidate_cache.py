"""Four bounded fixtures; no old/fresh cohort scoring and no solver calls."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import active_view as av
import candidate_cache as cache


def geometry(x=0., width=.2):
    return dict(wall_z=4.2, boxes=[dict(x=x, z=1.2, width=width, thickness=.04)])


def public(value):
    scene = av.Scene(tuple(av.Box(**b) for b in value["boxes"]))
    return [dict(camera=[0., 0.], bins=list(av.observe(scene, (0., 0.))))]


def result(decision="UNKNOWN", inside=None, outside=None, excluded=None):
    return dict(decision=decision, status="UNKNOWN" if decision == "UNKNOWN" else "MODEL_CONDITIONAL",
        reason="original", witnesses=dict(IN=inside, OUT=outside), authority="NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY",
        solver_metadata=[dict(requested_label=l, solver_status=2 if l == excluded else 0,
                              exclusion_supported=l == excluded) for l in ("IN", "OUT")],
        valid_witness_count=int(inside is not None)+int(outside is not None), opposing_witness_analysis=None)


def matched(inside=None, outside=None):
    return dict(witnesses=dict(IN=inside, OUT=outside), matches={l:dict(label=l, geometry=g, provenancequery="query_000")
        if g is not None else None for l,g in (("IN",inside),("OUT",outside))},
        checks=dict(IN=1, OUT=1, total=2), elapsed_seconds=.001)


class CacheTests(unittest.TestCase):
    def test_preserves_original_conditional_and_original_valid_geometry(self):
        original = result("IN_MODEL_CONDITIONAL", inside=geometry(), excluded="OUT")
        snapshot = deepcopy(original)
        recovered = cache.reconcile(original, matched(inside=geometry(.01)))
        self.assertEqual(recovered["decision"], original["decision"])
        self.assertEqual(recovered["witnesses"], original["witnesses"])
        self.assertEqual(recovered["solver_metadata"], original["solver_metadata"])
        self.assertEqual(original, snapshot)

    def test_cache_never_upgrades_unknown_even_if_matching_side_found(self):
        original = result(excluded="OUT")
        recovered = cache.reconcile(original, matched(inside=geometry()))
        self.assertEqual(recovered["decision"], "UNKNOWN")
        self.assertEqual(recovered["valid_witness_count"], 1)
        original = result(outside=geometry(.6))
        recovered = cache.reconcile(original, matched(inside=geometry()))
        self.assertEqual(recovered["decision"], "UNKNOWN")
        self.assertEqual(recovered["reason"], "OPPOSING_WITNESSES_VALIDATED_WITH_INITIAL_PUBLIC_CACHE")

    def test_valid_opposite_witness_downgrades_numerical_exclusion(self):
        original = result("OUT_MODEL_CONDITIONAL", outside=geometry(.6), excluded="IN")
        recovered = cache.reconcile(original, matched(inside=geometry()))
        self.assertEqual((recovered["decision"], recovered["status"]), ("UNKNOWN", "UNKNOWN"))
        self.assertEqual(recovered["cache_recovery"]["exclusion_conflict_labels"], ["IN"])
        self.assertEqual(recovered["solver_metadata"], original["solver_metadata"])

    def test_pool_initial_source_purity_dedup_and_public_only_validation(self):
        value = geometry()
        observations = public(value)
        def fake_read(path):
            if path.name == "witness-query-mapping.json":
                return [dict(id="unused_a", queries=dict(initial="query_001", fixed_endpoint="query_999")),
                        dict(id="unused_b", queries=dict(initial="query_000", fixed_endpoint="query_999"))]
            if path.name not in ("witness-query_000.json", "witness-query_001.json"):
                raise AssertionError("Read future or source file")
            return dict(query_id=path.name[8:-5], observations=observations,
                        result=dict(witnesses=dict(IN=value, OUT=None)))
        with patch.object(cache.sensor, "verify_seal", return_value={}), \
             patch.object(cache, "_read", side_effect=fake_read), \
             patch.object(cache.model, "milp", side_effect=AssertionError("No solver")):
            pool = cache.build_initial_pool("unused_fixture_folder")
            self.assertEqual(len(pool), 1)
            self.assertEqual(pool[0]["provenancequery"], "query_000")
            match = cache.match(observations, pool)
            self.assertEqual(match["witnesses"]["IN"], value)
            self.assertEqual(match["checks"]["total"], 1)
            with self.assertRaises(ValueError):
                cache.match([{**observations[0], "truth": True}], pool)


if __name__ == "__main__":
    unittest.main()
