import copy
import unittest
from intrusion_source import source,check_source,original,inherited


class IntrusionSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.train=source('train');cls.report=source('report')

    def test_declared_splits_and_size(self):
        self.assertEqual(check_source(self.train)['split_frames'],dict(train=144,dev=144))
        self.assertEqual(check_source(self.report)['split_frames'],dict(confirmation=288))

    def test_no_target_or_background_geometry_reuse_across_sources(self):
        for field in ('target_signature','background_geometry_signature'):
            a={g[field] for g in self.train['scene_groups']};b={g[field] for g in self.report['scene_groups']}
            self.assertFalse(a&b)
        self.assertFalse({f['id'] for f in self.train['frames']}&{f['id'] for f in self.report['frames']})

    def test_background_intervention_cannot_change_target_or_camera(self):
        s=copy.deepcopy(self.train);s['frames'][12]['camera']['yaw']+=.1
        with self.assertRaises(AssertionError):check_source(s)

    def test_all_objects_participate_in_truth(self):
        s=copy.deepcopy(self.train);s['frames'][0]['objects'][1]['center_m']=[1.,0.,1.]
        with self.assertRaises(AssertionError):check_source(s)

    def test_target_groups_never_cross_train_dev(self):
        membership={}
        for f in self.train['frames']:membership.setdefault(f['target_group'],set()).add(f['split'])
        self.assertEqual(len(membership),8);self.assertTrue(all(len(v)==1 for v in membership.values()))

    def test_new_geometry_distinct_from_prior_designs_without_outcomes(self):
        from confirmation_source import source as confirmation_source
        def target_geometry(spec):
            return {inherited.digest(dict(camera=f['camera'],center=f['objects'][0]['center_m'],
                size=f['objects'][0]['size_m'])) for f in spec['frames']}
        def background_geometry(spec):
            return {inherited.digest([{k:o[k] for k in ('center_m','size_m')} for o in f['objects'][1:]])
                    for f in spec['frames']}
        old=[original(seed) for seed in (136014,146016,158016,170016,176016)]+[confirmation_source()]
        for new in (self.train,self.report):
            for prior in old:
                self.assertFalse(target_geometry(new)&target_geometry(prior))
                self.assertFalse(background_geometry(new)&background_geometry(prior))


if __name__=='__main__':unittest.main()
