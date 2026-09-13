"""Preserve independently supported Radar when a visual ROI has no robust interior.

The inherited working box error is two pixels. This additive guard does not
delete tiny proposals, change associations, seed tracks, or infer clearance.
"""
import copy
import math
from mz109_interval_extent import BOX_ERROR_PX


def unresolved_extent(box):
    return (len(box)!=4 or not all(math.isfinite(x) for x in box) or
            box[2]-box[0]<=2*BOX_ERROR_PX or box[3]-box[1]<=2*BOX_ERROR_PX)


def protect(row, prediction, spatial, yaw, distance=3.6):
    result=copy.deepcopy(prediction);events=[]
    for ret in spatial['spatial_evidence']:
        if ret['proposal'] is None or not unresolved_extent(ret['box']):continue
        k=ret['slot'];r=row['radar_range_m'][k];a=row['radar_angle'][k]
        if not row['radar_packet_received'] or not row['radar_valid'][k] or r is None or a is None or not math.isfinite(r+a):continue
        angle=math.radians(a+yaw)
        independent=.2<=r*math.cos(angle)<=distance and abs(r*math.sin(angle))<=.3
        if independent:
            events.append(dict(slot=k,proposal=ret['proposal'],box=ret['box'],height_state='HEIGHT_UNKNOWN',
                reason='BOX_HAS_NO_INTERIOR_AFTER_WORKING_ERROR_EROSION',sources=['RADAR','IMU'],evidence_age_s=0.))
    result.update(candidate=bool(prediction['candidate'] or events),guard_events=events,
                  guard_added=bool(events and not prediction['candidate']))
    result['candidate_state']='ALERT' if result['candidate'] else 'UNKNOWN'
    return result
