"""Synthetic global-assignment and causal optical prior checks; no captures."""
import copy
import itertools
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import mz114_joint_association as model


BOX=[300.,150.,340.,210.]
OTHER=[300.,50.,340.,110.]


def texture():
    image=np.zeros((360,640),np.uint8)
    rng=np.random.default_rng(114013)
    image[150:210,300:340]=rng.integers(25,255,(60,40),dtype=np.uint8)
    return cv2.GaussianBlur(image,(3,3),.6)


def frame(t,ranges=(),boxes=None,velocities=None,angles=None,episode='episode'):
    ranges=list(ranges);angles=[0.]*len(ranges) if angles is None else list(angles)
    velocities=[None]*len(ranges) if velocities is None else list(velocities)
    baseline=any(.2<=r*math.cos(math.radians(a))<=3.6 and abs(r*math.sin(math.radians(a)))<=.3
                 for r,a in zip(ranges,angles))
    row=dict(id=f'{episode}_{t}',episode_id=episode,time_s=t,imu_valid=True,camera_pitch_deg=0.,
             camera_in_body_m=[0.,0.,1.7],rgb_intrinsics=dict(cx=320.,cy=180.,fx=320.,fy=320.),
             radar_packet_received=True,radar_range_m=ranges,radar_angle=angles,
             radar_valid=[True]*len(ranges),radar_velocity=velocities)
    pred=dict(id=row['id'],baseline=baseline,candidate=baseline,tof_support=False,integrated_yaw_deg=0.,
              proposals=[list(BOX)] if boxes is None else copy.deepcopy(boxes))
    return row,pred


def run(frames,images=None,use_temporal=True):
    rows=[r for r,_ in frames];preds=[p for _,p in frames]
    images=[texture()]*len(frames) if images is None else images
    image_by_id={r['id']:im for r,im in zip(rows,images)}
    return model.predict(rows,preds,lambda r:image_by_id[r['id']],use_temporal=use_temporal)


class JointAssociation(unittest.TestCase):
    def test_exact_injective_assignment_and_explicit_null(self):
        out=model.assignments([{0:0.,1:1.},{0:0.,1:1.}])
        self.assertEqual(out['best_cost'],1.)
        self.assertEqual(out['resolved'],[None,None])
        self.assertEqual({tuple(a['proposals']) for a in out['retained']},{(0,1),(1,0)})
        contested=model.assignments([{0:0.},{0:0.}])
        self.assertEqual({tuple(a['proposals']) for a in contested['retained']},{(0,None),(None,0)})
        self.assertEqual(contested['resolved'],[None,None])
        self.assertEqual(model.assignments([{}])['retained'],[dict(proposals=[None],cost=2.)])

    def test_margin_boundary_and_null_can_prevent_resolution(self):
        self.assertEqual(model.assignments([{0:1.5}])['resolved'],[None])
        self.assertEqual(model.assignments([{0:1.499}])['resolved'],[0])
        self.assertEqual(model.assignments([{0:0.,1:.5}])['resolved'],[None])
        self.assertEqual(model.assignments([{0:0.,1:.501}])['resolved'],[0])

    def test_pruning_matches_exhaustive_oracle(self):
        rng=np.random.default_rng(114)
        for _ in range(20):
            costs=[{j:float(rng.integers(0,9))*.25 for j in range(3)} for _ in range(3)]
            exhaustive=[]
            for choices in itertools.product((None,0,1,2),repeat=3):
                nonnull=[j for j in choices if j is not None]
                if len(set(nonnull))!=len(nonnull):continue
                exhaustive.append((sum(2. if j is None else c[j] for j,c in zip(choices,costs)),choices))
            best=min(c for c,_ in exhaustive)
            expected={(c,a) for c,a in exhaustive if c<=best+.5}
            actual=model.assignments(costs)
            self.assertEqual(actual['best_cost'],best)
            self.assertEqual({(a['cost'],tuple(a['proposals'])) for a in actual['retained']},expected)

    def test_bearing_uses_interval_distance_and_unchanged_envelope(self):
        row,pred=frame(0.,[2.],angles=[0.])
        ret=model.current_returns(row,pred,pred['proposals'],0.)[0]
        self.assertEqual(ret['pair_costs'][0]['bearing_cost'],0.)
        edge=math.degrees(math.atan((BOX[2]-320)/320))
        row['radar_angle']=[edge+6.]
        ret=model.current_returns(row,pred,pred['proposals'],0.)[0]
        self.assertAlmostEqual(ret['pair_costs'][0]['bearing_cost'],1.)
        row['radar_angle']=[edge+12.001]
        self.assertEqual(model.current_returns(row,pred,pred['proposals'],0.)[0]['candidates'],[])

    def test_current_global_resolves_competing_envelopes_by_joint_cost(self):
        to_pixel=lambda a:320+320*math.tan(math.radians(a))
        boxes=[[to_pixel(-10),150.,to_pixel(-5),210.],[to_pixel(5),150.,to_pixel(10),210.]]
        out=run([frame(0.,[2.,2.8],boxes=boxes,angles=[-6.5,6.5])],use_temporal=False)[0]
        self.assertEqual([r['candidates'] for r in out['spatial_evidence']],[[0,1],[0,1]])
        self.assertEqual([r['proposal'] for r in out['spatial_evidence']],[0,1])

    def test_optical_prior_resolves_ambiguity_missing_velocity_holds_range(self):
        frames=[frame(0.,[1.5]),frame(.25,[1.5,3.],boxes=[BOX,OTHER])]
        prior=run(frames)
        current=run(frames,use_temporal=False)
        self.assertEqual([r['proposal'] for r in current[1]['spatial_evidence']],[None,None])
        self.assertEqual([r['proposal'] for r in prior[1]['spatial_evidence']],[0,1])
        p=prior[1]['diagnostics']['prior_matches'][0]
        self.assertEqual(p['velocity_state'],'UNKNOWN')
        self.assertEqual(p['range_assumption'],'HOLD_RANGE_VELOCITY_UNKNOWN')
        self.assertEqual(p['predicted_range_m'],1.5)
        self.assertEqual(prior[1]['spatial_evidence'][1]['pair_costs'][0]['temporal_cost'],9.)
        self.assertEqual(prior[1]['spatial_evidence'][1]['pair_costs'][1]['temporal_cost'],0.)

    def test_no_texture_or_nonreciprocal_optical_match_adds_zero_cost(self):
        frames=[frame(0.,[1.5]),frame(.25,[1.5,3.],boxes=[BOX,OTHER])]
        zeros=np.zeros((360,640),np.uint8)
        out=run(frames,[zeros,zeros])
        self.assertEqual(out[1]['diagnostics']['prior_matches'],[])
        self.assertTrue(all(p['temporal_cost']==0. for r in out[1]['spatial_evidence'] for p in r['pair_costs']))
        frames[1][1]['proposals']=[BOX,[302.,152.,342.,212.]]
        out=run(frames)
        self.assertEqual(out[1]['diagnostics']['prior_matches'],[])
        self.assertEqual([r['proposal'] for r in out[1]['spatial_evidence']],[None,None])

    def test_current_measurement_seeds_and_visual_updates_never_refresh_range_age(self):
        frames=[frame(0.,[1.5]),frame(.25),frame(.5),frame(.75,[1.5,3.],boxes=[BOX,OTHER])]
        out=run(frames)
        for index in (1,2):
            p=out[index]['diagnostics']['prior_matches'][0]
            self.assertEqual(p['measurement_index'],0)
            self.assertEqual(p['range_time_s'],0.)
            self.assertEqual(out[index]['diagnostics']['seeded'],0)
        self.assertEqual(out[3]['diagnostics']['prior_matches'],[])
        self.assertEqual([r['proposal'] for r in out[3]['spatial_evidence']],[None,None])
        # A real, resolved new range may refresh its own measurement timestamp.
        frames=[frame(0.,[1.5]),frame(.25,[1.6]),frame(.5)]
        self.assertEqual(run(frames)[2]['diagnostics']['prior_matches'][0]['range_time_s'],.25)

    def test_measured_velocity_changes_prior_but_not_current_range_geometry(self):
        frames=[frame(0.,[2.],velocities=[-1.]),frame(.25,[1.75,3.],boxes=[BOX,OTHER])]
        out=run(frames)
        p=out[1]['diagnostics']['prior_matches'][0]
        self.assertEqual(p['predicted_range_m'],1.75)
        self.assertEqual(p['range_assumption'],'CONSTANT_MEASURED_RADIAL_VELOCITY')
        self.assertEqual(out[1]['spatial_evidence'][0]['range_m'],1.75)
        self.assertEqual(out[1]['spatial_evidence'][0]['evidence_age_s'],0.)

    def test_missing_rgb_bad_time_imu_and_episode_clear_history(self):
        for mode in ('rgb','time','imu','episode'):
            frames=[frame(0.,[1.5]),frame(.25,[1.5,3.],boxes=[BOX,OTHER])]
            images=[texture(),texture()]
            if mode=='rgb':images[1]=None
            elif mode=='time':frames[1][0]['time_s']=0.
            elif mode=='imu':frames[1][0]['imu_valid']=False
            else:frames[1][0]['episode_id']='other'
            out=run(frames,images)
            self.assertEqual(out[1]['diagnostics']['prior_matches'],[],mode)
            self.assertEqual([r['proposal'] for r in out[1]['spatial_evidence']],[None,None],mode)

    def test_independent_radar_unknown_height_and_tof_survive(self):
        out=run([frame(0.,[2.,3.],boxes=[BOX,OTHER])])[0]
        self.assertTrue(out['candidate'])
        self.assertTrue(all(r['height_state']=='HEIGHT_UNKNOWN' and r['support'] for r in out['spatial_evidence']))
        single=frame(0.)
        single[1]['tof_support']=True
        single[0]['imu_valid']=False
        self.assertTrue(run([single],[None])[0]['candidate'])

    def test_invalid_imu_finite_yaw_cannot_create_new_radar_support(self):
        single=frame(0.,[2.],angles=[20.])
        single[0]['imu_valid']=False
        single[1]['integrated_yaw_deg']=-20.
        single[1]['baseline']=False;single[1]['candidate']=False
        out=run([single])[0]
        self.assertFalse(out['candidate'])
        self.assertFalse(out['spatial_evidence'][0]['support'])
        self.assertTrue(out['diagnostics']['baseline_fallback'])

    def test_prefix_causality_input_immutability_and_finite_cost_validation(self):
        frames=[frame(0.,[1.5]),frame(.25,[1.5,3.],boxes=[BOX,OTHER])]
        before=copy.deepcopy(frames)
        prefix=run(frames)
        self.assertEqual(prefix,run(frames+[frame(.5,[2.])])[:2])
        self.assertEqual(frames,before)
        with self.assertRaises(ValueError):model.assignments([{}]*5)
        with self.assertRaises(ValueError):model.assignments([{0:float('nan')}])


if __name__=='__main__':unittest.main()
