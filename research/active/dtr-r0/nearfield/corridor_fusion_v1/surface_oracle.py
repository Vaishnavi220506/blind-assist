"""EVALUATOR ONLY: complete faces attached to actually returned public slots."""
import numpy as np
from run_e1 import ROOT
from mz138_support_ceiling import linked_hits,surface_key,contains
from run_mz138_surface_extent import native_face_pieces

SUPPORT_NAMES=('support_x_lo','support_x_hi','support_y_lo','support_y_hi','support_z_lo','support_z_hi')

def columns(names,zone,slot):
    prefix=f'zone{zone:02d}.slot{slot}.'
    return [names.index(prefix+k) for k in SUPPORT_NAMES]

def intersects(piece,halfwidth=.3):
    return all(hi>=a and lo<=b for (lo,hi),(a,b) in zip(piece,[(.2,3.6),(-halfwidth,halfwidth),(.4,2.05)]))

def hull(pieces):
    arr=np.asarray(pieces,float)
    return np.stack([arr[:,:,0].min(0),arr[:,:,1].max(0)],-1)

def intervene(row,e,base,names):
    sampled=base.copy();full=base.copy();native={z['zone_id']:z for z in e['zonal_tof_native']}
    bounds={o['name']:o for o in e['native_bounds']};slots=[];allowed=[]
    for zone in row['tof_zones']:
        zid=zone['zone_id']
        for slot,target in enumerate(zone['targets']):
            prefix=f'zone{zid:02d}.slot{slot}.'
            usable=bool(base[names.index(prefix+'usable')]);ix=columns(names,zid,slot)
            record=dict(zone=zid,slot=slot,status=target['status'],usable=usable,
                original_support=base[ix].reshape(3,2).tolist(),completed=False)
            if not usable:record['fallback_reason']='UNUSABLE_PUBLIC_SLOT';slots.append(record);continue
            allowed.extend(ix);item=dict(zone_id=zid,target_slot=slot);hits=linked_hits(item,native)
            keys=[surface_key(h,bounds) for h in hits]
            pieces=native_face_pieces(dict(surface_keys=keys),e)
            record.update(hits=len(hits),surface_keys=keys,actors=sorted({h['actor_id'].rsplit('/',1)[-1] for h in hits}))
            if pieces is None:
                record['fallback_reason']='NO_RETURNED_LINEAGE' if not hits else 'UNKNOWN_OR_AMBIGUOUS_FACE'
            else:
                pts=np.array([np.asarray(h['hit_point_m'])-e['body_origin_m'] for h in hits])
                sp=np.stack([pts.min(0),pts.max(0)],-1);fp=hull(pieces)
                sampled[ix]=sp.flatten();full[ix]=fp.flatten()
                actual=[bool(intersects([[v,v] for v in p])) for p in pts]
                retained=[any(contains(p,pt) for p in pieces) for pt in pts]
                # Face inference tolerance is1e-5m; containment uses same
                # admitted tolerance rather than requiring exact floating equality.
                retained=[v or any(all(lo-1e-5<=x<=hi+1e-5 for (lo,hi),x in zip(p,pt)) for p in pieces) for v,pt in zip(retained,pts)]
                record.update(completed=True,faces=pieces,sampled_support=sp.tolist(),full_hull=fp.tolist(),
                    sampled_point_possible=any(actual),sampled_hull_possible=intersects(sp),
                    face_union_possible=any(intersects(p) for p in pieces),full_hull_possible=intersects(fp),
                    contributors=len(pts),corridor_contributors=sum(actual),contributors_excluded=sum(not v for v in retained),
                    corridor_contributors_excluded=sum(a and not r for a,r in zip(actual,retained)),
                    points=pts.tolist())
            slots.append(record)
    mask=np.ones(len(base),bool);mask[allowed]=False
    assert np.array_equal(base[mask],sampled[mask]) and np.array_equal(base[mask],full[mask])
    done=[r for r in slots if r['completed']]
    return sampled,full,dict(id=row['id'],slots=slots,usable_slots=sum(r['usable'] for r in slots),
        completed_slots=len(done),fallback_slots=sum(r['usable'] and not r['completed'] for r in slots),
        sampled_point_reachable=any(r['sampled_point_possible'] for r in done),
        full_face_reachable=any(r['face_union_possible'] for r in done),
        hull_bridges=sum(r['full_hull_possible']!=r['face_union_possible'] for r in done),
        contributors_excluded=sum(r['contributors_excluded'] for r in done),
        corridor_contributors_excluded=sum(r['corridor_contributors_excluded'] for r in done),
        non_support_bitwise_unchanged=True)
