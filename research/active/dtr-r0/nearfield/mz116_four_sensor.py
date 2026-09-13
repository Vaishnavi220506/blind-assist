"""Runnable four-sensor Development challenger with explicit resolution guard.

The MZ115 arms remain available unchanged. resolution_guard is the new arm;
it uses current Radar evidence and observable RGB boxes only.
"""
import mz111_spatial_evidence as radar
import mz115_spatial_allocation as zonal
from mz116_radar_resolution_guard import protect


def predict(rows,image_loader):
    curves=zonal.predict(rows,image_loader)
    primary=curves['3.6']['allocation']
    nominal=[dict(proposals=p['proposals'],integrated_yaw_deg=p['integrated_yaw_deg'],tof_support=False,
        candidate=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6),baseline=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6))
        for r,p in zip(rows,primary)]
    current=radar.predict([zonal.legacy_empty_tof(r) for r in rows],nominal,surface='plane',filter_range=True)
    for distance,arms in curves.items():
        arms['resolution_guard']=[protect(r,p,s,n['integrated_yaw_deg'],float(distance))
            for r,p,s,n in zip(rows,arms['allocation'],current,nominal)]
    return curves
