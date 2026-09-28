"""Evaluator-only six-query surface labels and final central-query strata.

No labels, margins or target-group identities may enter sensor/render/readout
features. Channel order matches frozen Readout: left HEAD/BODY, centre
HEAD/BODY, right HEAD/BODY. Truth includes occluded physical surfaces, exactly
as the existing scene helper; it is not a visible-return/danger mask.
"""
import numpy as np
import cnh_proposal_attribution_scenes as S

QUERY_BOXES = np.array([([x-.3,y0,.3],[x+.3,y1,3.])
    for x in (-.3,0.,.3) for y0,y1 in ((-.2,.42),(.42,.9))])
CENTRAL_INDICES = (2,3)


def labels_for_all(boxes, travel, *, boundary='interior'):
    """Return [frames,6] labels; default requires query-interior surface area.

    Tangent contact with positive surface area is positive under the retained
    clip-triangle definition. Randomized current margins never equal zero;
    boundary='closed' reproduces the old central helper. The frozen new training
    default 'interior' requires surface area inside the query (1e-8m numerical
    shrink, not a 5cm uncertainty band). Changing this is a frozen label-definition
    choice, not a post-result bugfix. It removes pure face tangency.
    """
    poses=np.asarray(travel,float)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4):raise ValueError('poses must have shape [N,4,4]')
    if boundary not in ('closed','interior'):raise ValueError('unknown boundary convention')
    eps=1e-8 if boundary=='interior' else 0.
    triangles=np.concatenate([S.box_mesh(b['lo'],b['hi']) for b in boxes]) if boxes else np.empty((0,3,3))
    result=np.zeros((len(poses),6),dtype=np.int8)
    for i,pose in enumerate(poses):
        local=(triangles-pose[:3,3])@pose[:3,:3]
        for q,(lo,hi) in enumerate(QUERY_BOXES):
            result[i,q]=int(len(S.clip_triangles(local,lo+eps,hi-eps))>0)
    return result


def central_margin_masks(target_group, margin, band=.05):
    """Final-frame central query masks, [scenes,2]; never broadcast over time.

    scene margin describes ONLY target object's inner lateral face relative to
    final central x=+-0.30m. It says nothing about the other height group, a
    lateral query, an earlier turned pose, or competing surfaces. All examples
    stay in strict scoring; the +/-band sensitivity excludes only the matching
    target group. This is target-margin sensitivity, not whole-scene clearance.
    """
    g=np.asarray(target_group)
    m=np.asarray(margin,float)
    if g.ndim!=1 or m.shape!=g.shape or not np.isin(g,[0,1]).all() or not np.isfinite(m).all():
        raise ValueError('finite scene margins and one HEAD(0)/BODY(1) target group required')
    if not np.isfinite(band) or band<0:raise ValueError('band must be nonnegative')
    owns=g[:,None]==np.arange(2)[None,:]
    near=owns&(abs(m[:,None])<=band)
    return dict(strict=np.ones_like(near),outside_band=~near,target_near_band=near,
        target_inside=owns&(m[:,None]<0),target_outside=owns&(m[:,None]>0))


def penetration_stratum(margin):
    """Metadata stratum, not a replacement for geometric labels.

    Negative margin = target's near lateral face penetrates central corridor;
    this is not a requirement that the entire target box lies inside.
    """
    m=np.asarray(margin,float)
    if not np.isfinite(m).all():raise ValueError('finite margins required')
    p=-m
    return np.select([(p>0)&(p<.05),(p>=.05)&(p<.15),(p>=.15)&(p<=.30),p<0,p==0],
        ['inside_lt5cm','inside_5to15cm','inside_15to30cm','outside','touch'],default='inside_other')
