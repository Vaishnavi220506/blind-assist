"""Synthetic saved-output arithmetic only; no project data or model execution."""
import copy
import json
import unittest

import numpy as np

from mz161_dense_task import ray_query
from mz166_query_task import evaluate_queries, ARMS, OFFSETS, ZERO
from test_evaluate_mz161_dense_task import fixture as central_fixture


def fixture():
    rows, es, labels, oldmaps, oldpreds, baseline, spec, metadata = central_fixture()
    maps = {'central': oldmaps['frame'], 'queries': oldmaps['dense']}
    predictions = {'central': oldpreds['frame'], 'queries': oldpreds['dense']}
    prior = copy.deepcopy(predictions['queries'])
    families = list(dict.fromkeys(metadata[r['id']]['family'] for r in rows))
    indices = [i for f in families for i in
        [j for j, r in enumerate(rows) if metadata[r['id']]['partition'] == 'fit' and metadata[r['id']]['family'] == f][:6]]
    targets = np.zeros((24,7,2,2), bool); frames = np.zeros((24,7), bool)
    qm = {a: np.full((24,7,2,2), -2., np.float32) for a in ARMS}
    qp = {a: [] for a in ARMS}
    for local, index in enumerate(indices):
        targets[local,ZERO] = labels[index]['target']; frames[local,ZERO] = True
        for a in ARMS:
            qm[a][local,ZERO] = maps[a][index]
            if a == 'central':
                qm[a][local] = maps[a][index]
            pp = []
            for q, offset in enumerate(OFFSETS):
                shifted = copy.deepcopy(rows[index]); shifted['camera_in_body_m'][1] -= offset
                mask = ray_query(shifted,0.)[1][2] > 0
                pm = float(np.where(mask,qm[a][local,q],-30.).max())
                tm = 2. if a == 'central' and q != ZERO else -3.
                pp.append(dict(frame_logit=max(pm,tm),pixel_max=pm,token_max=tm))
            qp[a].append(pp)
    return [rows,es,labels,maps,predictions,baseline,spec,metadata,prior,qm,qp,indices,targets,frames]


class QueryTaskTests(unittest.TestCase):
    def test_exact_changed_pair_arithmetic_aliases_and_separate_gates(self):
        args = fixture(); predictions_before = copy.deepcopy(args[4])
        result = evaluate_queries(*args); s=result['summary']; d=result['query_diagnostic']
        self.assertTrue(s['gates']['overall_pass']); self.assertTrue(d['strong_query_evidence'])
        self.assertEqual(set(s['partitions']['heldout']), {'baseline','central','queries'})
        self.assertEqual(s['partitions']['heldout']['queries']['metrics']['TP'],24)
        self.assertEqual(s['partitions']['heldout']['queries']['vs_mz165_pretrained']['prior_true_frames'],24)
        self.assertEqual(d['all']['frame_label_changed_pairs'],144)
        self.assertEqual(d['all']['pixel_label_changed_pairs'],144)
        self.assertEqual(d['all']['arms']['central']['frame_both_correct_rate'],0.)
        self.assertEqual(d['all']['arms']['central']['pixel_both_correct_rate'],0.)
        self.assertEqual(d['all']['arms']['queries']['frame_both_correct_rate'],1.)
        self.assertEqual(d['all']['arms']['queries']['pixel_both_correct_rate'],1.)
        self.assertTrue(all(v['frame_label_changed_pairs']==36 for v in d['families'].values()))
        self.assertEqual(args[4],predictions_before)
        json.dumps(result,allow_nan=False)

    def test_mz165_true_retention_is_additional_to_incumbent_and_control(self):
        args=fixture(); rows,es,labels,maps,preds,base,spec,metadata=args[:8]
        i=next(i for i,r in enumerate(rows) if metadata[r['id']]['partition']=='heldout' and r['time_s']==.75)
        base[i]=False
        es[i]['zonal_tof_native'][0]['returned_lineage']=[]
        for a in ARMS:
            maps[a][i]=-2.;preds[a][i].update(frame_logit=-2.,pixel_max=-2.)
        result=evaluate_queries(*args);g=result['summary']['gates']; checks=g['heldout_alert_checks']
        self.assertTrue(checks['all_baseline_true_frames_retained'])
        self.assertTrue(checks['all_control_true_frames_retained'])
        self.assertTrue(checks['all_baseline_events_without_extra_delay'])
        self.assertTrue(checks['zero_native_supported_nonalerts'])
        self.assertFalse(checks['all_mz165_pretrained_true_frames_retained'])
        self.assertFalse(g['overall_pass'])
        self.assertEqual(result['summary']['partitions']['heldout']['queries']['vs_mz165_pretrained']['lost_true'],[rows[i]['id']])
        self.assertTrue(result['query_diagnostic']['strong_query_evidence'])

    def test_pixel_weighting_unknown_exclusion_and_empty_frame_denominator(self):
        args=fixture()
        # Keep frame labels unchanged even though visible first-hit labels change:
        # volume intrusion and visible pixels are separate authorities.
        args[13][:]=True
        for arm in ARMS:
            for predictions in args[10][arm]:
                for q,p in enumerate(predictions):
                    if q!=ZERO:p['token_max']=2.;p['frame_logit']=max(p['pixel_max'],2.)
        # A second known pixel changes on just one pair, making pooled rate
        # 144/145 rather than an average of per-image or per-family rates.
        args[12][0,0,0,1]=True
        result=evaluate_queries(*args);d=result['query_diagnostic']
        self.assertEqual(d['all']['frame_label_changed_pairs'],0)
        self.assertIsNone(d['all']['arms']['queries']['frame_both_correct_rate'])
        self.assertEqual(d['all']['pixel_label_changed_pairs'],145)
        self.assertEqual(d['all']['arms']['queries']['pixel_both_correct_rate'],144/145)
        self.assertFalse(d['strong_query_evidence'])
        self.assertTrue(result['summary']['gates']['overall_pass'])
        args[12][0,0,1,1]=True # This pixel has no reference in the fixture.
        with self.assertRaisesRegex(ValueError,'unchanged known'):
            evaluate_queries(*args)

    def test_fixed_selection_prior_order_and_query_record_validation(self):
        args=fixture(); args[11][0]=args[11][1]
        with self.assertRaisesRegex(ValueError,'distinct'):
            evaluate_queries(*args)
        args=fixture();args[8][0],args[8][1]=args[8][1],args[8][0]
        with self.assertRaisesRegex(ValueError,'prior prediction IDs'):
            evaluate_queries(*args)
        args=fixture();args[10]['queries'][0][0]['pixel_max']=99.
        with self.assertRaisesRegex(ValueError,'masked map'):
            evaluate_queries(*args)


if __name__=='__main__':
    unittest.main()
