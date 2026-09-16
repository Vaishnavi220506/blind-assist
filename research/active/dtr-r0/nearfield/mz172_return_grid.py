"""Public full-zone visual sampling grid; sensor validity is unchanged.

Each zone supplies optional RGB context. Samples outside RGB are retained in
the grid and masked separately. No native geometry, actor, label, or packet
status affects the sampling coordinates or visibility mask.
"""
import math
import numpy as np
from mz115_spatial_allocation import zone_box


METHOD=dict(schema='MZ172_PUBLIC_FULL_ZONE_GRID_V1',native_rgb_shape=[360,640],
    samples_per_axis=8,zone_order='ASCENDING_ZONE_ID_0_TO_63',
    sample_order='ROW_MAJOR_Y_THEN_X',grid_sample_align_corners=True,
    coordinates='FULL_CONTINUOUS_PUBLIC_ZONE_BOX_CELL_CENTERS',
    outside='KEEP_UNCLAMPED_GRID_AND_SEPARATE_VISIBLE_MASK',
    authority='OPTIONAL_VISUAL_CONTEXT_ONLY_NO_SENSOR_VALIDITY_CHANGE')


def make_zone_grid(row):
    """Return grid float32[64,64,2], visible bool[64,64], and JSON audit.

    Last dimension is normalized (x,y) for torch grid_sample with
    align_corners=True. Each zone's 64 points are an 8x8 row-major lattice
    spanning its full continuous projected box, including off-image portions.
    """
    intr=row['rgb_intrinsics'];width=intr['width'];height=intr['height']
    if width!=640 or height!=360:raise ValueError('Original640x360 public RGB intrinsics required')
    for key in ('fx','fy','cx','cy'):
        if not isinstance(intr[key],(int,float,np.integer,np.floating)) or not math.isfinite(float(intr[key])):
            raise ValueError('Finite public intrinsics required')
    if intr['fx']<=0 or intr['fy']<=0:raise ValueError('Positive public focal lengths required')
    zones=row['tof_zones']
    if (len(zones)!=64 or any(not isinstance(z.get('zone_id'),(int,np.integer)) or isinstance(z['zone_id'],bool) for z in zones)
            or sorted(z['zone_id'] for z in zones)!=list(range(64))):
        raise ValueError('Exactly64 unique integer zone IDs required')
    grid=np.empty((64,64,2),np.float32);visible=np.empty((64,64),bool)
    boxes=[];counts=[];fraction=(np.arange(8,dtype=np.float64)+.5)/8.
    for zone in sorted(zones,key=lambda z:z['zone_id']):
        for key in ('theta_bounds_deg','phi_bounds_deg'):
            bounds=zone.get(key)
            if not isinstance(bounds,(list,tuple,np.ndarray)) or len(bounds)!=2:
                raise ValueError('Two finite ascending angular bounds required')
            if not all(isinstance(v,(int,float,np.integer,np.floating)) and math.isfinite(float(v)) for v in bounds):
                raise ValueError('Finite angular bounds required')
            if not -90<float(bounds[0])<float(bounds[1])<90:
                raise ValueError('Nonempty pinhole angular interval within(-90,90) required')
        box=np.asarray(zone_box(zone,intr),np.float64)
        if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Finite positive-area projected zone box required')
        x=box[0]+fraction*(box[2]-box[0]);y=box[1]+fraction*(box[3]-box[1])
        xx,yy=np.meshgrid(x,y,indexing='xy');pixels=np.stack((xx.ravel(),yy.ravel()),axis=-1)
        norm=2*pixels/np.array([width-1,height-1])-1
        if not np.isfinite(norm).all() or np.max(np.abs(norm))>np.finfo(np.float32).max:
            raise ValueError('Sampling coordinates must remain finite float32')
        zid=zone['zone_id'];grid[zid]=norm
        visible[zid]=(pixels[:,0]>=0)&(pixels[:,0]<=width-1)&(pixels[:,1]>=0)&(pixels[:,1]<=height-1)
        boxes.append(box.tolist());counts.append(int(visible[zid].sum()))
    return dict(grid=grid,visible=visible,audit=dict(**METHOD,boxes=boxes,visible_samples_per_zone=counts,
        fully_visible_zones=sum(n==64 for n in counts),partly_visible_zones=sum(0<n<64 for n in counts),
        zero_visible_zones=sum(n==0 for n in counts),visible_samples=sum(counts),total_samples=4096,
        sensor_validity_read=False,sensor_slots_dropped=0))
