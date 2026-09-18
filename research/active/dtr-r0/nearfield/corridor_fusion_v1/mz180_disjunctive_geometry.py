"""MZ180 support-topology ceiling over sealed MZ177/MZ178 inputs."""
import argparse, hashlib, json, math
from pathlib import Path
from mz178_current_frame import ART, sha, write, support_readout
from mz115_spatial_allocation import slant_envelope, zone_box, possible

def finite(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)

def native_component_bounds(row, z, target, yaw, component):
    r=float(component['pre_noise_range_m']); span=max(0.0,float(component.get('private_hit_span_m',0.0)))
    ranges=(max(.02,r-span/2), r+span/2)
    b=slant_envelope(zone_box(z,row['rgb_intrinsics']),ranges,row['rgb_intrinsics'],
                     (row['camera_pitch_deg'],)*2,(yaw,)*2,0.)
    return [[lo+o,hi+o] for (lo,hi),o in zip(b,row['camera_in_body_m'])]

def readout(row, evalrow, yaw, mode):
    if not row.get('imu_valid'): return []
    ez={z['zone_id']:z for z in evalrow['zonal_tof_native']}
    out=[]
    for z in row.get('tof_zones',[]):
        for i,t in enumerate(z.get('targets',[])):
            if t.get('status') not in ('SIM_VALID','SIM_MERGED'): continue
            if t['status']=='SIM_MERGED':
                if mode=='ambiguous': continue
                comps=ez.get(z['zone_id'],{}).get('components',[])
                if mode=='union':
                    for c in comps:
                        b=native_component_bounds(row,z,t,yaw,c)
                        if possible(b): out.append({'sensor':'tof','zone_id':z['zone_id'],'target_index':i,'bounds':b,'status':'NATIVE_COMPONENT'})
                elif mode=='oracle':
                    comps=[c for c in comps if ez.get(z['zone_id'],{}).get('returned_lineage')]
                    for c in comps:
                        b=native_component_bounds(row,z,t,yaw,c)
                        if possible(b): out.append({'sensor':'tof','zone_id':z['zone_id'],'target_index':i,'bounds':b,'status':'SOURCE_ORACLE'})
            else:
                d,s=t.get('distance_m'),t.get('range_noise_sigma_m')
                if finite(d) and finite(s):
                    b=slant_envelope(zone_box(z,row['rgb_intrinsics']),(max(.02,d-3*s),d+3*s),row['rgb_intrinsics'],(row['camera_pitch_deg'],)*2,(yaw,)*2,0.)
                    b=[[lo+o,hi+o] for (lo,hi),o in zip(b,row['camera_in_body_m'])]
                    if possible(b): out.append({'sensor':'tof','zone_id':z['zone_id'],'target_index':i,'bounds':b,'status':'SIM_VALID'})
    if mode!='oracle':
        # Radar remains an independent branch, exactly as MZ178.
        for slot,(r,a,v) in enumerate(zip(row.get('radar_range_m',[]),row.get('radar_angle',[]),row.get('radar_valid',[]))):
            if v and finite(r) and finite(a) and r>0:
                x,y=r*math.cos(math.radians(a+yaw)),r*math.sin(math.radians(a+yaw))
                if .2<=x<=3.6 and abs(y)<=.3: out.append({'sensor':'radar','slot':slot,'x':x,'y':y})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--capture',type=Path,required=True); p.add_argument('--evaluator',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); assert out.is_relative_to(ART) and not out.exists(); out.mkdir(parents=True)
    rows=[json.loads(x) for x in a.capture.joinpath('raw.jsonl').read_text().splitlines()]
    ev=[json.loads(x) for x in a.evaluator.read_text().splitlines()]
    base=[json.loads(x) for x in a.baseline.joinpath('predictions.jsonl').read_text().splitlines()]
    assert len(rows)==len(ev)==len(base)
    result=[]; ep=None; yaw=0.
    for row,e,b in zip(rows,ev,base):
        if row['episode_id']!=ep: ep=row['episode_id']; yaw=0.
        if row.get('imu_valid'): yaw+=float(row.get('delta_yaw',0.))
        arms={}
        for mode in ('broad','union','ambiguous','oracle'):
            cs=support_readout(row,yaw).get('contributors',[]) if mode=='broad' else readout(row,e,yaw,mode)
            arms[mode]={'alert':bool(b['astar_alert'] or cs),'contributors':cs,'state':'POSSIBLE_OCCUPANCY' if cs else 'UNKNOWN'}
        result.append({'id':row['id'],'episode_id':ep,'time_s':row['time_s'],'yaw':yaw,'baseline_alert':b['astar_alert'],'arms':arms})
    (out/'predictions.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in result))
    write(out/'input-seal.json',{'evaluator_read':True,'modes':['broad','union','ambiguous','oracle'],'frames':len(result),'sources':['raw.jsonl','evaluator.jsonl','predictions.jsonl']})
    write(out/'completion.json',{'status':'PASS','frames':len(result),'evaluator_read':True,'outputs':{n:sha(out/n) for n in ('predictions.jsonl','input-seal.json')}})

if __name__=='__main__': main()
