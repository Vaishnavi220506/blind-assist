"""Two synthetic reducer checks; no real records, predictor or fitting."""
import copy
import json
import unittest

import numpy as np

from evaluate_mz161_dense_task import evaluate, FAMILIES


def fixture():
    rows=[]; es=[]; labels=[]; metadata={}; pairs=[]; maps={a:[] for a in ('frame', 'dense')}; predictions={a:[] for a in maps}
    for family in FAMILIES:
        for group in range(4):
            scene=f'{family}-{group}'; episodes=[scene+'-in', scene+'-out']; pairs.append(dict(episodes=episodes))
            for positive, episode in zip((True, False), episodes):
                for step in range(6):
                    identity=episode+f'-{step}'
                    row=dict(id=identity, episode_id=episode, time_s=step*.25, imu_valid=True, delta_yaw=0.,
                        camera_pitch_deg=0., camera_in_body_m=[0., 0., 1.],
                        rgb_intrinsics=dict(width=2,height=2,fx=5.,fy=5.,cx=0.,cy=0.),
                        tof_zones=[dict(zone_id=0,targets=[{}])])
                    center=[1., 0. if positive else 1., 1.]
                    e=dict(id=identity,family=family,body_origin_m=[0.,0.,0.],
                        native_bounds=[dict(name='shape0',center_m=center,extent_m=[.1,.1,.1])],
                        zonal_tof_native=[dict(zone_id=0,returned_lineage=[dict(target_index=0,hit_indices=[0])],
                            private_rays=[dict(subray=0,hit_point_m=center)])])
                    target=np.zeros((2,2),bool); target[0,0]=positive
                    visible=np.zeros((2,2),bool); visible[0,0]=True
                    # One pixel has no saved reference and cannot count as a negative.
                    known=np.ones((2,2),bool); known[1,1]=False
                    labels.append(dict(target=target,known=known,target_visible=visible,target_corridor=target&visible))
                    rows.append(row);es.append(e)
                    metadata[identity]=dict(partition='heldout' if group==3 else 'fit', family=family,scene_group=scene)
                    for arm in maps:
                        logits=np.full((2,2),-2.,np.float32)
                        if positive: logits[0,0]=2.
                        if arm=='frame' and not positive and step<3: logits[0,1]=2.
                        pm=float(logits.max()); maps[arm].append(logits)
                        predictions[arm].append(dict(id=identity,frame_logit=pm,pixel_max=pm,token_max=-3.,winner='pixel'))
    maps={a:np.stack(v) for a,v in maps.items()}
    return rows,es,labels,maps,predictions,np.ones(192,bool),dict(pairs=pairs),metadata


class DenseReducerTests(unittest.TestCase):
    def test_fixed_gate_metrics_pairs_and_unknown_denominators(self):
        args=fixture();result=evaluate(*args);summary=result['summary']; held=summary['partitions']['heldout']
        self.assertTrue(summary['gates']['overall_pass'])
        self.assertEqual(held['baseline']['metrics']['FP'],24)
        self.assertEqual(held['frame']['metrics']['FP'],12)
        self.assertEqual(held['dense']['metrics']['FP'],0)
        self.assertEqual(held['dense']['metrics']['TP'],24)
        self.assertEqual(held['dense']['pairs']['both_correct'],24)
        self.assertEqual(held['dense']['events']['positive_segments'],4)
        spatial=summary['spatial']['heldout']['dense']['all']
        self.assertEqual(spatial['unknown_pixels'],48)
        self.assertEqual(spatial['known_pixels'],144)
        self.assertEqual(spatial['target_risk_columns'],24)
        self.assertEqual(spatial['target_risk_half_column_recall'],1.)
        self.assertEqual(held['dense']['native']['totals']['corridor_contributor_samples'],24)
        self.assertEqual(held['dense']['branch_decisions']['pixel_only'],24)
        json.dumps(result,allow_nan=False)

    def test_lost_native_true_frame_delay_and_spatial_na_fail(self):
        args=list(fixture()); rows,es,labels,maps,predictions,baseline,spec,metadata=args
        ix=next(i for i,r in enumerate(rows) if metadata[r['id']]['partition']=='heldout' and metadata[r['id']]['family']=='suspended_head')
        maps['dense'][ix]=-2.
        predictions['dense'][ix].update(frame_logit=-2.,pixel_max=-2.)
        # A separate family has no target risky-column denominator: retain NA.
        for i,r in enumerate(rows):
            if metadata[r['id']]['partition']=='heldout' and metadata[r['id']]['family']=='near_rod_farwall':
                labels[i]['target_corridor'][:]=False; labels[i]['target_visible'][:]=False
        before=copy.deepcopy(predictions)
        result=evaluate(*args);summary=result['summary'];gates=summary['gates']; dense=summary['partitions']['heldout']['dense']
        self.assertFalse(gates['overall_pass'])
        self.assertFalse(gates['heldout_alert_checks']['zero_native_supported_nonalerts'])
        self.assertFalse(gates['heldout_alert_checks']['all_baseline_events_without_extra_delay'])
        self.assertEqual(dense['vs_baseline']['lost_true'],[rows[ix]['id']])
        self.assertEqual(dense['vs_control']['lost_true'],[rows[ix]['id']])
        self.assertEqual(dense['vs_baseline']['event_deltas'][0]['extra_delay_s'],.25)
        self.assertIsNone(gates['heldout_family_spatial_checks']['near_rod_farwall']['target_risk_half_column_recall_at_least_90pct'])
        self.assertNotIn('near_rod_farwall',gates['target_risk_column_applicable_families'])
        self.assertIsNone(gates['all_applicable_target_risk_columns_pass'])
        self.assertFalse(gates['heldout_spatial_pass'])
        self.assertEqual(predictions,before)
        json.dumps(result,allow_nan=False)


if __name__=='__main__': unittest.main()
