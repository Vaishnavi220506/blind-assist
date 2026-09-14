"""Focused interface/FOV regressions; no candidate outcome inspection."""
import json
import unittest

import numpy as np
from mz132_contour_adapter import anonymous_components, outside_image, mask_assign, predict
from mz129_extent_correction import ROOT, SOURCE, CORRECTION
from mz126_central_tof import read, serial, tof


class ContourContract(unittest.TestCase):
    def test_anonymity_and_complete_scene(self):
        labels = np.zeros((20,30), np.uint8)
        labels[:7,:] = 7  # Broad background must not be filtered for size.
        labels[10:13,4:6] = 2
        labels[16:19,4:6] = 2  # Disconnected visible pieces retain one candidate.
        changed = np.zeros_like(labels)
        changed[labels == 7] = 1
        changed[labels == 2] = 99
        a, b = anonymous_components(labels), anonymous_components(changed)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)
        self.assertEqual(sum(c['pixels'] for c in a), np.count_nonzero(labels))
        self.assertEqual(set(a[0]), {'box','tiles','padded_tiles','pixels'})

    def test_fov_partition(self):
        zone = [-4.,-2.,12.,11.]
        outside = outside_image(zone,10,8)
        self.assertAlmostEqual(sum(tof.area(t) for t in outside)+80, tof.area(zone))
        for i,a in enumerate(outside):
            self.assertIsNone(tof.intersection(a,[0,0,10,8]))
            for b in outside[i+1:]:
                self.assertIsNone(tof.intersection(a,b))
        self.assertEqual(outside_image([1,1,3,3],10,8), [])
        self.assertEqual(outside_image([1,9,3,11],10,8), [[1,9,3,11]])

    def test_rectangular_mask_matcher_equivalence_and_ambiguity(self):
        def evidence():
            return [dict(status='SIM_VALID',forward_depth=2.,zone_id=i,target_slot=0,
                         signal=1.,range_m=2.,zone_box=[i*10.,0.,(i+1)*10.,10.],
                         proposal=None) for i in range(2)]
        boxes = [[0.,0.,20.,10.]]
        old = evidence(); new = evidence()
        tof.assign_groups(old, boxes)
        components = [dict(box=b,tiles=[b],padded_tiles=[[-2,-2,22,12]]) for b in boxes]
        mask_assign(new, components)
        self.assertEqual([e['proposal'] for e in old], [e['proposal'] for e in new])
        self.assertEqual([e['roi'] for e in old], [e['tiles'][0] for e in new])
        ambiguous = evidence()
        mask_assign(ambiguous, components*2)
        self.assertEqual([e['proposal'] for e in ambiguous], [None,None])

    def test_missing_contours_preserve_all_288_mz129_outputs(self):
        rows = [json.loads(s) for s in (SOURCE/'capture-v1/raw.jsonl').read_text().splitlines()]
        cache = read(CORRECTION/'predictions.json')
        path = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
        radar = read(path/'radar-v1/predictions.json')
        baseline = read(path/'replay-v1r2/predictions.json')['radar']
        for mode in (False, True):
            predictions, details = predict(rows, cache, radar, [[] for _ in rows], mode)
            self.assertEqual([p['candidate'] for p in predictions], [p['candidate'] for p in baseline])
            self.assertEqual([p['score'] for p in predictions], [p['score'] for p in baseline])
            for old, current in zip(cache,details):
                self.assertEqual([e['roi'] for e in old['spatial_evidence']],
                                 [e['tiles'][0] for e in current['tof']['returns']])


if __name__ == '__main__':
    unittest.main()
