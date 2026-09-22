"""Native UE collision-only ceiling source. Run by UE Python, no renderer."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys


def scenes():
    result=[]
    for x in (1.4,2.2,3.0):
        for width in (.1,.4):
            for side in (-1,1):
                for inside in (True,False):
                    result.append(dict(id=f's{len(result):02d}',center=[x,side*((.25 if inside else .35)+width/2),1.1],
                                       size=[.1,width,2.2],truth=inside))
    return result


def main():
    import unreal as u
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from mz115_zonal_tof import zone_geometry,measure_zone,FOV_DEG,GRID
    out=Path(os.environ['BA_DITHER_OUT']);out.mkdir(exist_ok=False)
    api=u.get_editor_subsystem(u.EditorActorSubsystem)
    world=u.EditorLoadingAndSavingUtils.new_blank_map(False)
    mesh=u.load_asset('/Engine/BasicShapes/Cube');actor=None;rows=[]
    try:
        for si,s in enumerate(scenes()):
            if actor: api.destroy_actor(actor)
            actor=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*(v*100 for v in s['center'])))
            actor.static_mesh_component.set_static_mesh(mesh)
            actor.static_mesh_component.set_collision_profile_name('BlockAll')
            actor.set_actor_scale3d(u.Vector(*s['size']))
            center,extent=actor.get_actor_bounds(False)
            native=dict(center=[center.x/100,center.y/100,center.z/100],extent=[extent.x/100,extent.y/100,extent.z/100])
            for arm,phases in (('repeat',(0,0,0)),('dither',(0,.25,-.25))):
                for phase,fraction in enumerate(phases):
                    yaw=fraction*FOV_DEG/GRID;y=math.radians(yaw)
                    rng=random.Random(182018+si*100+phase);zones=[];private=[]
                    origin=u.Vector(0,0,170)
                    for zid in range(64):
                        z,rays=zone_geometry(zid);hits=[]
                        for ray in rays:
                            a=math.tan(math.radians(ray['theta_deg']));b=math.tan(math.radians(ray['phi_deg']))
                            n=math.sqrt(1+a*a+b*b);d=((math.cos(y)-a*math.sin(y))/n,(math.sin(y)+a*math.cos(y))/n,b/n)
                            hit=dict(ray,range_m=None)
                            trace=u.SystemLibrary.line_trace_single(world,origin,origin+u.Vector(*(v*400 for v in d)),
                                    u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
                            if trace and trace.to_tuple()[0]:
                                f=trace.to_tuple();pt=f[5]
                                hit.update(range_m=math.sqrt(pt.x**2+pt.y**2+(pt.z-170)**2)/100,
                                    reflectance_proxy=.55,actor_id=s['id'] if f[10]==actor.static_mesh_component else 'other',
                                    hit_point_m=[pt.x/100,pt.y/100,pt.z/100])
                            hits.append(hit)
                        targets,details=measure_zone(hits,rng,True)
                        zones.append(dict(z,targets=targets))
                        private.append(dict(zone_id=zid,private_rays=hits,**details))
                    rows.append(dict(id=f"{s['id']}_{arm}_{phase}",scene_id=s['id'],arm=arm,phase=phase,yaw_deg=yaw,
                        tof_zones=zones,native=native,private=private))
            (out/'progress.json').write_text(json.dumps(dict(scenes=si+1,total=24)))
        public=[{k:v for k,v in r.items() if k not in ('native','private')} for r in rows]
        for name,data in (('raw.jsonl',public),('evaluator.jsonl',rows)):
            (out/name).write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
        (out/'spec.json').write_text(json.dumps(scenes(),indent=2))
        (out/'receipt.json').write_text(json.dumps(dict(status='PASS',frames=len(rows),engine=u.SystemLibrary.get_engine_version(),
            backend='UE_NATIVE_COLLISION_CPU_NULLRHI',exact_yaw=True,packet_received=True,
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('mz115_zonal_tof.py'))},
            hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('raw.jsonl','evaluator.jsonl','spec.json')}),indent=2))
    finally:
        if actor:api.destroy_actor(actor)
        u.SystemLibrary.quit_editor()


if __name__=='__main__': main()
