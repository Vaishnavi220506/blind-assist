"""Auxiliary broad RGB regions; independent frozen Radar frontend is unchanged."""
import copy
import math
import mz115_spatial_allocation as zonal
import mz116_four_sensor as incumbent
import mz117_surface_intervals as fitted
import mz118_interval_planes as interval
import mz118_surface_regions as regions
import mz111_spatial_evidence as radar
import mz113_flow_persistence as flow


def predict(rows,image_loader):
    base=incumbent.predict(rows,image_loader);primary=base['3.6']['resolution_guard']
    auxiliary=[]
    for row,pred in zip(rows,primary):
        p=copy.deepcopy(pred);p['legacy_proposals']=p['proposals'];p['proposals']=regions.proposals(image_loader(row))
        p['proposal_namespaces']={'guard_events':'legacy_proposals','valid_tof':'legacy_proposals','plane_proposal':'proposals',
            'auxiliary_contract':'legacy prefix retained; additional regions only for surface branch'}
        auxiliary.append(p)
    outputs={'expanded_fit':[fitted.refine(r,p) for r,p in zip(rows,auxiliary)],
             'interval_lp':[interval.refine(r,p) for r,p in zip(rows,auxiliary)]}
    nominal=[dict(proposals=p['proposals'],integrated_yaw_deg=p['integrated_yaw_deg'],tof_support=False,
        candidate=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6),baseline=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6))
        for r,p in zip(rows,primary)]
    empty=[zonal.legacy_empty_tof(r) for r in rows];current=radar.predict(empty,nominal,surface='plane',filter_range=True)
    persisted=flow.predict(empty,nominal,image_loader);curves={}
    for distance,arms in base.items():
        d=float(distance);out={k:arms[k] for k in ('baseline','nominal','resolution_guard')}
        for k,v in outputs.items():out[k]=copy.deepcopy(v)
        for index,(row,n,cur,old,guard,original) in enumerate(zip(rows,nominal,current,persisted,arms['resolution_guard'],primary)):
            yaw=n['integrated_yaw_deg'];raw=zonal.raw_radar(row,yaw,d);common=False
            for ret in cur['spatial_evidence']:
                if ret['proposal'] is not None:common|=zonal.possible(zonal.radar_xyz(ret['box'],ret['range_m'],row,yaw),d)
                else:
                    r=ret['range_m'];a=math.radians(row['radar_angle'][ret['slot']]+yaw)
                    common|=.2<=r*math.cos(a)<=d and abs(r*math.sin(a))<=.3
            common|=not raw and any(zonal.possible(zonal.radar_xyz(n['proposals'][p['proposal']],p['range_m'],row,yaw,False),d) for p in old['diagnostics']['propagated'])
            common|=bool(guard['guard_events'])
            original_tof=any(zonal.certain(e['coarse_xyz'],d) or zonal.possible(e['localized_xyz'],d) for e in original['spatial_evidence'])
            assert bool(common or original_tof)==guard['candidate']
            if d==3.6:assert common==bool(original['common_radar'] or guard['guard_events'])
            for arm in outputs:
                value=out[arm][index]
                tof=any(zonal.certain(e['coarse_xyz'],d) or zonal.possible(e['localized_xyz'],d) for e in value['spatial_evidence'])
                assert [e for e in original['spatial_evidence'] if e['status']=='SIM_VALID']==[e for e in value['spatial_evidence'] if e['status']=='SIM_VALID']
                value.update(candidate=bool(common or tof),common_radar=bool(common),guard_events=guard['guard_events'],
                    guard_added=guard['guard_added'],candidate_state='ALERT' if common or tof else 'UNKNOWN')
        curves[distance]=out
    return curves
