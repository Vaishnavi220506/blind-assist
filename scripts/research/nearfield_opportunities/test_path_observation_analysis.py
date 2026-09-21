"""Hand-built mechanism/accounting fixtures; never score the fresh cohort."""
import unittest
from unittest.mock import patch

import active_view as av
import path_observation_analysis as pa


def fixture_views(intermediate=False):
    return {p: dict(camera=p, bins=[11 if intermediate and i == 6 else 10]*8,
                    raw_radial_m=[1.04 if intermediate else 1.01]*8)
            for i, p in enumerate(pa.path_poses("plus_x"))}


class PathObservationTests(unittest.TestCase):
    def test_public_signature_and_transitions_need_only_bins(self):
        poses = pa.path_poses("plus_x")
        views = {p: {"bins": [4 if i < 6 else 5]*8} for i, p in enumerate(poses)}
        self.assertEqual(len(pa.public_signature(views, poses)), 13)
        events = pa.bin_transitions(views, poses)[0]["events"]
        self.assertEqual(events, [dict(path_interval_m=[.05, .06], from_bin=4, to_bin=5)])

    def test_endpoint_alias_can_hide_path_difference_and_purity_conflict(self):
        poses = pa.path_poses("plus_x")
        a, b = fixture_views(), fixture_views(True)
        self.assertEqual(pa.public_signature(a, (poses[0], poses[-1])),
                         pa.public_signature(b, (poses[0], poses[-1])))
        self.assertNotEqual(pa.public_signature(a, poses), pa.public_signature(b, poses))
        # The designated a/b pair separates, but another opposite scene c remains.
        signatures = {"a": pa.public_signature(a, poses), "b": pa.public_signature(b, poses),
                      "c": pa.public_signature(a, poses)}
        self.assertEqual(pa.purity(signatures, {"a": True, "b": False, "c": False})["resolved_ids"], ["b"])

    def test_strict_quantization_requires_stable_common_exact_face(self):
        poses = pa.path_poses("plus_x")
        a, b = fixture_views(), fixture_views(True)
        face = (("box", 0, ("x_min",)),)
        with patch.object(pa, "first_surface", return_value=face):
            witnesses = pa.quantization_witnesses(None, None, a, b, poses)
        self.assertEqual(len(witnesses), 8)
        def switching(scene, camera, ray):
            return (("wall",),) if camera == poses[6] else face
        with patch.object(pa, "first_surface", side_effect=switching):
            self.assertEqual(pa.quantization_witnesses(None, None, a, b, poses), [])
        with patch.object(pa, "first_surface", return_value=(("box", 0, ("x_min", "z_min")),)):
            self.assertEqual(pa.quantization_witnesses(None, None, a, b, poses), [])

    def test_no_raw_endpoint_difference_means_no_quantization_attribution(self):
        poses = pa.path_poses("plus_x")
        a, b = fixture_views(), fixture_views(True)
        for endpoint in (poses[0], poses[-1]):
            b[endpoint]["raw_radial_m"] = list(a[endpoint]["raw_radial_m"])
        with patch.object(pa, "first_surface", side_effect=AssertionError("no need for source")):
            self.assertEqual(pa.quantization_witnesses(None, None, a, b, poses), [])

    def test_class_ceiling_cannot_choose_by_hidden_pair(self):
        truth = {"a": True, "b": False, "c": True, "d": False}
        pairs = [dict(id="ab", members=["a", "b"]), dict(id="cd", members=["c", "d"])]
        initial = dict.fromkeys(truth, ((0,),))
        by_spoke = {"x": {"a": (1,), "b": (2,), "c": (3,), "d": (3,)},
                    "z": {"a": (3,), "b": (3,), "c": (1,), "d": (2,)}}
        result = pa.oracle_ceilings(pairs, initial, by_spoke, truth)
        self.assertEqual(result["pair_oracle_separated"], 2)
        self.assertEqual(result["class_common_spoke_separated"], 1)
        self.assertEqual(result["mixed_scene_common_spoke_purity_ceiling"], 2)
        self.assertEqual(result["mixed_scene_per_case_purity_ceiling"], 4)

    def test_surface_identity_distinguishes_wall_and_front(self):
        self.assertEqual(pa.first_surface(av.Scene(()), (0., 0.), 4), (("wall",),))
        self.assertEqual(pa.first_surface(av.Scene((av.Box(0., 1., .5),)), (0., 0.), 4),
                         (("box", 0, ("z_min",)),))

    def test_full_analysis_accounting_on_hand_fixture(self):
        sources = [dict(id=i, scene=dict(boxes=[dict(x=x, z=1., width=.1, thickness=.04)], wall_z=4.2))
                   for i, x in (("a", 0.), ("b", .7))]
        poses = sorted({p for name in pa.SPOKES for p in pa.path_poses(name)})
        observations = [dict(id=i, views=[dict(camera=p, bins=[11 if i == "b" and p == (.06, 0.) else 10]*8,
            raw_radial_m=[1.01]*8) for p in poses]) for i in ("a", "b")]
        result = pa.analyze([dict(id="ab", members=["a", "b"])], sources, observations)
        self.assertEqual(result["denominators"]["initially_aliased_pairs"], 1)
        arm = result["fixed_spokes"]["plus_x"]
        self.assertEqual(arm["endpoint"]["separated_pairs"], 0)
        self.assertEqual(arm["sampled"]["separated_pairs"], 1)
        self.assertEqual(arm["cost"]["additional_views"], 11)
        self.assertEqual(result["primary"]["path_decision"], "PASS_PAIR_AND_COHORT_INFORMATION_GAIN")
        self.assertEqual(result["primary"]["quantization_decision"], "NO_STRICT_CROSSING_GAIN")


if __name__ == "__main__":
    unittest.main()
