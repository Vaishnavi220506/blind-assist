"""Independent scalar recomputation of saved spatial association observations."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cnh_rgb_spatial_association import ROOT, CACHE, BASE, OUT, read, save, self_check, select_spatial, coarse_observations
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_geometry import camera_geometry


def main():
    out=OUT/'run-v2'
    ledger=read(out/'case-ledger.json')
    old={r['id']:r for r in read(BASE/'case-ledger.json')}
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    mode_equal=0
    for row in ledger:
        obs=read(out/'observations'/(hashlib.sha256(row['id'].encode()).hexdigest()[:16]+'.json'))
        geom=camera_geometry(ROOT,manifest[row['frame_id']],cameras[row['frame_id']])
        zones=zone_map(cameras[row['frame_id']])
        with np.load(CACHE/'predictions/native'/f'{row["frame_id"]}.npz') as handle:
            dp=handle['native_depth']
        p=row['prediction']['foreground_depth_m']
        rgb=[]
        factors=[]
        for z in range(64):
            mask=zones==z
            vals=dp[mask]
            valid=vals[np.isfinite(vals)&(vals>0)]
            rgb.append(float(np.count_nonzero((valid>=p/1.1)&(valid<p*1.1))/len(valid)) if len(valid) else 0.)
            factors.append(float(np.median(geom['optical_z_per_radial'][mask])) if mask.any() else 0.)
        np.testing.assert_array_equal(rgb,obs['selection']['rgb_support'])
        for i,mode in enumerate(obs['modes']):
            support=[]
            for z,counts in enumerate(obs['counts']):
                candidate_z=mode['radial_m']*obs['selection']['edge_optical_factor']
                numerator=sum(n for k,n in enumerate(counts) if candidate_z/1.1 <= factors[z]*(k+.5)*.05 < candidate_z*1.1)
                support.append(numerator/sum(counts) if sum(counts)>0 else 0.)
            np.testing.assert_allclose(support,obs['selection']['mode_support'][i],atol=1e-15)
            cells=obs['selection']['neighborhood']
            a=[rgb[z] for z in cells];b=[support[z] for z in cells]
            den=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
            score=sum(x*y for x,y in zip(a,b))/den if den else -1.
            assert abs(score-obs['selection']['scores']['spatial'][i])<1e-12
        selected=max(range(len(obs['modes'])),key=lambda i:obs['selection']['scores']['spatial'][i])
        assert selected==obs['selection']['selected']['spatial']
        mode_equal+=selected==old[row['id']]['selected_mode_indices']['oracle_best_mode']
    # Same footprint at both modes: cosine cannot identify one. The declared
    # fallback remains measurable but is explicitly not identification evidence.
    zones=np.repeat(np.arange(64),100).reshape(80,80)
    raw=np.tile(np.r_[np.full(50,2.025),np.full(50,4.025)],64).reshape(80,80)
    dp=raw*2
    coarse=coarse_observations(raw,zones)
    prediction=dict(status='OK',edge_pixel=[33,60.5],foreground_depth_m=4.05)
    tied=select_spatial(dp,zones,np.ones_like(dp),coarse,27,prediction)
    assert tied['evidence']['spatial']=='INDISTINGUISHABLE_MODES'
    # Unknown extraneous evaluator fields on prediction cannot influence result.
    dirty=select_spatial(dp,zones,np.ones_like(dp),coarse,27,{**prediction,'gt_clearance_m':999,'instance_id':666,'semantic':777})
    assert dirty==tied
    # Permuting reference pixels inside each cell preserves all aggregate inputs.
    permuted=raw.ravel().reshape(64,100)[:,::-1].reshape(raw.shape)
    assert coarse_observations(permuted,zones)==coarse
    failures=[]
    for r in ledger:
        if abs(r['errors_m']['spatial'])>.02:
            base=old[r['id']]
            failures.append(dict(id=r['id'],scene=r['scene'],error_cm=r['errors_m']['spatial']*100,
                selected_mode=r['selection']['selected']['spatial'],
                oracle_mode=base['selected_mode_indices']['oracle_best_mode'],
                target_majority=r['attribution']['spatial']['target_majority'],
                target_fraction=base['cell_target_fraction'],
                best_mode_error_cm=base['errors_m']['oracle_best_mode']*100))
    save(out/'independent-verification.json',dict(status='PASS',cases=15,
        checks=['RGB support independently rebuilt from native cache and camera only',
                '15x modes support independently rebuilt with scalar histogram loop',
                'cosine and strongest-order selection independently rebuilt',
                'equal spatial footprints explicit ambiguity',
                'unused evaluator field mutation cannot alter output',
                'within-cell pixel reference permutation leaves sensor summaries invariant'],
        selected_matches_oracle_mode=mode_equal,failures=failures,
        mathematical_fixture=self_check()))
    print(json.dumps(dict(status='PASS',selected_matches_oracle_mode=mode_equal,failures=failures),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
