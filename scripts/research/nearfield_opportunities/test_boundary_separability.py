"""Diagnostic accounting and authority tests, no scored source observations."""
import unittest
from unittest.mock import patch

import active_view as av
import boundary_separability as bs


class BoundaryTests(unittest.TestCase):
    def test_source_has_unique_strict_boundary_pairs(self):
        pairs, scenes = bs.source()
        self.assertEqual((len(pairs), len(scenes)), (90, 180))
        self.assertEqual(len({bs.geometry_key(r["scene"]) for r in scenes}), 180)
        by_id = {r["id"]: bs.pos.source_scene(r) for r in scenes}
        for pair in pairs:
            self.assertEqual([av.intersects_query(by_id[i]) for i in pair["members"]], [True, False])

    def test_choices_have_no_future_or_source_authority(self):
        bank = av.ForecastBank(((1,), (1,)), (((1,), (2,)),)*4, (True, False))
        with patch.object(av, "observe", side_effect=AssertionError("future")), \
             patch.object(bs.pos, "source_scene", side_effect=AssertionError("source")), \
             patch.object(av, "intersects_query", side_effect=AssertionError("truth")):
            rows = bs.choose([dict(id="a", bins=[1]), dict(id="b", bins=[1])], bank)
        self.assertEqual(rows[0], {**rows[1], "id": "a"})

    def test_pair_oracle_cannot_choose_per_hidden_id_in_common_class(self):
        rows = [dict(id=str(i), initial_equal=True, initial_bins=[1], separating_actions=[i],
            status="ACTION_SEPARABLE", initial_no_match=False,
            policies={p:dict(pair_separated=i == 0) for p in bs.POLICIES}) for i in range(2)]
        metrics = bs.summarize(rows)["metrics"]
        self.assertEqual(metrics["pair_oracle_separated"], 2)
        self.assertEqual(metrics["class_common_oracle_separated"], 1)
        self.assertEqual(metrics["universal_action_classes"], 0)

    def test_pair_separation_is_not_all_opposite_scene_resolution(self):
        scenes = [dict(id=str(i), scene=dict(boxes=[dict(x=x,z=1.,width=.1,thickness=.04)],wall_z=4.2))
                  for i,x in enumerate([0., .6, .7])]
        # Scene0 versus1 separate; scene0 versus2 never separate.
        observations = [dict(id=str(i), views=[dict(bins=[0])]+[dict(bins=[1 if i==1 else 2])]*4) for i in range(3)]
        ceiling = bs.cohort_ceiling(scenes, observations)
        self.assertEqual(ceiling["initial_mixed_scenes"], 3)
        self.assertEqual(ceiling["mixed_common_action_ceiling"], 1)
        self.assertEqual(ceiling["mixed_per_case_action_ceiling"], 1)


if __name__ == "__main__":
    unittest.main()
