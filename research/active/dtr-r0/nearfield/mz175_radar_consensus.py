"""Conditional task unanimity over ambiguous observable Radar/RGB proposals.

This is a statistical proxy, not a geometric certificate: unseen objects,
ghosts and null associations are not exhausted by the visible proposal set.
"""
from mz111_spatial_evidence import current_returns, plane_inside


METHOD = dict(schema='MZ175_CONDITIONAL_RADAR_TASK_CONSENSUS_V1',
    eligibility='UNCHANGED_MZ111_ANGULAR_MARGIN_12_DEGREES',
    unresolved_only=True, minimum_candidates=2, range='CURRENT_RAW_UNFILTERED',
    geometry='UNCHANGED_MZ111_FRONTOPARALLEL_CORNER_AABB_OVERLAP_PROXY',
    primary='ALL_VISIBLE_ELIGIBLE_PROPOSALS_SUPPORT', control='ANY_SUPPORT',
    final='FULL_MZ129_OR_ADDITIONAL_SUPPORT', null_hypothesis='UNRESOLVED_NOT_EXCLUDED',
    no_alert='UNKNOWN_NOT_CLEAR')


def predict_frame(row, corrected, baseline):
    before = repr((row, corrected, baseline))
    records = []
    if row['imu_valid']:
        for ret in current_returns(row, corrected):
            if ret['proposal'] is not None or len(ret['candidates']) < METHOD['minimum_candidates']:
                continue
            hypotheses = [dict(proposal=j, box=list(corrected['proposals'][j]),
                support=bool(plane_inside(corrected['proposals'][j], ret['range_m'], row,
                                         corrected['integrated_yaw_deg'])))
                for j in ret['candidates']]
            support = [h['support'] for h in hypotheses]
            records.append(dict(slot=ret['slot'], range_m=ret['range_m'],
                raw_center_support=ret['baseline'], hypotheses=hypotheses,
                any_support=any(support), all_support=bool(support) and all(support)))
    branches = {arm: any(r[arm+'_support'] for r in records) for arm in ('any', 'all')}
    old = bool(baseline['candidate'])
    result = dict(id=row['id'], baseline=old, additional=branches,
        arms={arm:dict(candidate=bool(old or support),
                      state='ALERT' if old or support else 'UNKNOWN')
              for arm,support in branches.items()}, evidence=records,
        imu_available=bool(row['imu_valid']),
        full_baseline_retained=True, association_authority='CONDITIONAL_VISIBLE_PROXY_ONLY')
    assert before == repr((row, corrected, baseline)), 'Input mutation'
    return result
