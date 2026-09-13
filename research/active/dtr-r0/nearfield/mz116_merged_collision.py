"""Constructed ToF-only observability counterexample, never an obstacle predictor."""
import math
from mz115_zonal_tof import measure_zone,zone_geometry
from mz107_rgb_association import ray


class NoNoise:
    def gauss(self,mean,sigma):return 0.


def witness():
    # Equal zeroth/first signal moments; all reflectances obey the frozen model.
    total=.04;mean=(.022*3.+.018*3.1)/total
    far=(total*mean-.002*2.55)/.038
    configurations=[[(3.,.022/4)]*4+[(3.1,.018/5)]*5,
                    [(2.55,.002)]+[(far,.038/8)]*8]
    # Row4,column5: horizontal5.625..11.25deg, vertical-5.625..0deg.
    results=[];geometry,angles=zone_geometry(37)
    for config in configurations:
        hits=[dict(range_m=r,reflectance_proxy=9*max(r,.2)**2*w) for r,w in config]
        public,private=measure_zone(hits,NoNoise())
        points=[(ray(a['theta_deg'],a['phi_deg'],-3.,0.)*h['range_m']+[0,0,1.65]).tolist() for a,h in zip(angles,hits)]
        hazardous=[.2<=p[0]<=3.6 and abs(p[1])<=.3 and .4<=p[2]<=2.05 for p in points]
        results.append(dict(public=public,private_hits=hits,points=points,hazardous_samples=hazardous,
                            merged_span_m=private['components'][0]['private_hit_span_m']))
    assert len(results[0]['public'])==len(results[1]['public'])==1
    for key in results[0]['public'][0]:
        a,b=results[0]['public'][0][key],results[1]['public'][0][key]
        assert abs(a-b)<1e-12 if isinstance(a,(int,float)) else a==b,key
    assert not any(results[0]['hazardous_samples']) and any(results[1]['hazardous_samples'])
    return dict(status='TOF_TUPLE_COLLISION_CONFIRMED',zone=geometry,cases=results,
        public_numeric_tolerance=1e-12,scope='One ToF zone only; no identical RGB/Radar/history claim',
        limitation='Same mean, strength, noise and merged status cannot by themselves certify absence of a nearer contributor.')


if __name__=='__main__':
    import json
    print(json.dumps(witness(),indent=2))
