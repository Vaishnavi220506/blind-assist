"""CPU-only synthetic coverage, authority, registration and bank checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cnh_segment_learned_masks as M
from cnh_route_sensor import angular_rays, FOV_DEG


class LearnedMaskTests(unittest.TestCase):
    pool = [np.array([[0, 0]]), np.array([[0, 0], [0, 1]])]

    def oid(self):
        oid = np.zeros((12, 8, 8, 256), int)
        oid[:, 3, 3, 34:38] = 7
        oid[:, 4, 4, 72:75] = 9
        return oid

    def objects(self):
        return [dict(id=0, category='BACKGROUND'), dict(id=7, category='tiny'), dict(id=9, category='wide')]

    def test_raster_matches_physical_ray_coordinates_and_exact_fraction(self):
        rays, _ = angular_rays(16)
        edge = np.tan(np.deg2rad(FOV_DEG/2))
        xy = np.floor((rays[..., :2]/rays[..., 2:]+edge)/(2*edge)*128).astype(int)
        raw = np.arange(8*8*256).reshape(8, 8, 256)
        image = M.raster(raw)
        np.testing.assert_array_equal(image[xy[..., 1], xy[..., 0]], raw)
        mask = self.oid()[0] == 7
        cov = M.coverage(M.raster(mask))
        self.assertEqual(cov[3, 3], 4/256)
        self.assertEqual(float(cov.sum()), 4/256)

    def test_union_overlap_never_adds_coverage(self):
        a = np.zeros((128, 128), bool)
        a[32:48, 32:48] = True
        b = a.copy()
        b[32:48, 48:56] = True
        cov = M.union_coverage([dict(mask=a), dict(mask=b)])
        self.assertEqual(cov[2, 2], 1.)
        self.assertEqual(cov[2, 3], .5)
        self.assertEqual(float(cov.sum()), 1.5)
        self.assertLessEqual(cov.max(), 1.)

    def test_shift_zero_padded_without_wrap(self):
        m = np.zeros((128, 128), bool)
        m[0, 127] = True
        self.assertFalse(M.shift_mask(m, 1, 0).any())
        self.assertFalse(M.shift_mask(m, 0, -1).any())
        moved = M.shift_mask(m, -8, 8)
        self.assertEqual(np.argwhere(moved).tolist(), [[8, 119]])
        self.assertFalse(M.shift_mask(m, 128, 0).any())

    def test_eval_registration_persistent_and_replay_deterministic(self):
        first = M.eval_candidates(self.oid(), self.objects(), 128, 5, 'synthetic', self.pool)
        again = M.eval_candidates(self.oid(), self.objects(), 128, 5, 'synthetic', self.pool)
        offsets = []
        for t in range(12):
            for condition in M.CONDITIONS:
                self.assertEqual(len(first[condition][t]), len(again[condition][t]))
                for a, b in zip(first[condition][t], again[condition][t]):
                    self.assertEqual(a['id'], b['id'])
                    np.testing.assert_array_equal(a['mask'], b['mask'])
            for source, moved in zip(first['IDEAL'][t], first['SHIFT05'][t]):
                offsets.append(np.argwhere(moved['mask']).mean(0)-np.argwhere(source['mask']).mean(0))
        np.testing.assert_array_equal(np.asarray(offsets), np.broadcast_to(offsets[0], (len(offsets), 2)))
        self.assertLessEqual(abs(np.linalg.norm(offsets[0])-8), .71)

    def test_nonbackground_geometry_only_not_semantics_or_range(self):
        altered = [dict(id=0, category='BACKGROUND', distance=.01, positive=True),
                   dict(id=7, category='not_a_target', distance=500., positive=False),
                   dict(id=9, category='arbitrary', range=-1, label=999)]
        a = M.eval_candidates(self.oid(), self.objects(), 128, 5, 'synthetic', self.pool)
        b = M.eval_candidates(self.oid(), altered, 128, 5, 'synthetic', self.pool)
        for condition in M.CONDITIONS:
            for af, bf in zip(a[condition], b[condition]):
                np.testing.assert_array_equal(M.union_coverage(af), M.union_coverage(bf))
        self.assertEqual({i for i, _ in M.base_frames(self.oid(), self.objects())[0]}, {7, 9})

    def test_twenty_training_banks_materialize_varied_and_deterministic(self):
        artifact = Path(__file__).resolve().parents[4]/'artifacts.local'/'work'/'cnh-segment-learned-dev-20260928'
        artifact.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mask-test-', dir=artifact) as td:
            root = Path(td)
            for path in ('geometry/unit00', 'oracle', 'features', 'out1', 'out2'):
                (root/path).mkdir(parents=True)
            geometry = dict(configs=[dict(config=5, objects=self.objects())])
            (root/'geometry/unit00/unit00.json').write_text(json.dumps(geometry), encoding='utf-8')
            (root/'pool.json').write_text(json.dumps([p.tolist() for p in self.pool]), encoding='utf-8')
            np.savez(root/'oracle/unit00-mount-10-oracle.npz', object_id=self.oid())
            np.savez(root/'features/unit000.npz', config=np.full(12, 5), frame=np.arange(12), split=np.array('train'))
            def job(out):
                return (root/'geometry', [root/'oracle'], root/'features', root/out, root/'pool.json', 'synthetic', 0, True)
            M.build_unit(job('out1'))
            M.build_unit(job('out2'))
            with np.load(root/'out1/unit000.npz') as a, np.load(root/'out2/unit000.npz') as b:
                self.assertEqual(a['AUG'].shape, (20, 12, 8, 8))
                np.testing.assert_array_equal(a['AUG'], b['AUG'])
                self.assertGreaterEqual(a['AUG'].min(), 0)
                self.assertLessEqual(a['AUG'].max(), 1)
                self.assertEqual(len({bank.tobytes() for bank in a['AUG']}), 20)
                np.testing.assert_array_equal(a['IDEAL'][0], M.union_coverage([dict(mask=m) for _, m in M.base_frames(self.oid(), self.objects())[0]]))


if __name__ == '__main__':
    unittest.main()
