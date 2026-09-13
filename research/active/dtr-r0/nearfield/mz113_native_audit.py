"""Evaluator-only native geometry/motion audit, after a matching prediction seal.

Never imported by predictors. Native actor centers are engine measurements;
evaluator camera positions are commanded source poses, not engine readbacks.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback


ROOT = Path(__file__).resolve().parents[4]
POSITION_TOLERANCE_M = 1e-5
MOTION_TOLERANCE_MPS = 1e-4


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def readrows(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def require(condition, message):
    if not condition:raise ValueError(message)


def difference(a,b):
    require(len(a)==len(b),'Vector length mismatch')
    return max((abs(x-y) for x,y in zip(a,b)),default=0.)


def native_intersects(bounds, origin):
    lo=[c-e-b for c,e,b in zip(bounds['center_m'],bounds['extent_m'],origin)]
    hi=[c+e-b for c,e,b in zip(bounds['center_m'],bounds['extent_m'],origin)]
    return (hi[0]>=.2 and lo[0]<=3.6 and hi[1]>=-.3 and lo[1]<=.3 and hi[2]>=.4 and lo[2]<=2.05)


def radial_from_displacements(center,camera,old_center,old_camera,dt):
    """Independent formulation: change of relative position projected on current LOS."""
    if old_center is None or old_camera is None or dt is None or dt<=0:return None
    relative=[center[k]-camera[k] for k in (0,1)]
    old_relative=[old_center[k]-old_camera[k] for k in (0,1)]
    radius=math.hypot(*relative)
    if radius<=1e-9:return None
    return sum((new-old)*new for new,old in zip(relative,old_relative))/(radius*dt)


def ray_hit_horizontal_range(bounds,camera):
    """Center-directed ray/AABB entry, matching the retained native ray-hit range."""
    near,far=0.,1.
    center=bounds['center_m']
    for c,e,o in zip(center,bounds['extent_m'],camera):
        direction=c-o
        if abs(direction)<1e-12:
            require(c-e<=o<=c+e,'Ray parallel outside native box')
            continue
        first,second=sorted(((c-e-o)/direction,(c+e-o)/direction))
        near=max(near,first);far=min(far,second)
    require(near<=far,'Center ray misses native AABB')
    return near*math.hypot(center[0]-camera[0],center[1]-camera[1])


def audit_rows(spec, raw, evaluations, provenance):
    frames=spec['frames']
    require(len(frames)==len(raw)==len(evaluations)==len(provenance)==240,'Expected all 240 frames')
    require(spec['seed']==113013,'Unexpected source seed')
    require(spec['rig']==dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,rgb_camera_count=1),'Rig changed')
    require(len({f['id'] for f in frames})==240,'Duplicate source IDs')
    audits={a['episode']:a for a in spec['source_audit']}
    require(len(audits)==20,'Expected 20 source episodes')
    states={};labels={};counts=Counter();motion_samples=0;moving_episodes=set()
    slot_kinds=Counter();first_missing_slots=0;native_missing=0;native_present=0
    maxima=dict(center_error_m=0.,extent_error_m=0.,actor_velocity_error_mps=0.,
                camera_velocity_error_mps=0.,radial_formula_error_mps=0.,ray_hit_range_error_m=0.)
    traces=[]
    for frame,row,ev,prov in zip(frames,raw,evaluations,provenance):
        ident=frame['id'];episode=frame['episode'];now=frame['time_s'];audit=audits[episode]
        require(all(r['id']==ident for r in (row,ev,prov)),ident+': ID mismatch')
        require(all(r['episode_id']==episode and r['time_s']==now for r in (row,ev,prov)),ident+': episode/time mismatch')
        require(ev['family']==frame['family']==audit['family'],ident+': family mismatch')
        require(ev['authority'].startswith('EVALUATOR_ONLY') and prov['authority'].startswith('EVALUATOR_ONLY'),ident+': authority mismatch')
        require(ev['camera']==frame['camera'],ident+': commanded camera mismatch')
        require(ev['body_origin_m']==frame['body_origin_m'],ident+': body origin mismatch')
        require(ev['camera']['pitch']==-3. and ev['camera']['roll']==0.,ident+': pitch/roll changed')
        require(row['camera_pitch_deg']==-3. and row['delta_pitch']==0.,ident+': sensor pitch changed')
        require(not any(any(token in key for token in ('native','truth','actor','source_audit')) for key in row),ident+': evaluator data in raw keys')
        camera=[ev['camera'][k] for k in ('x','y','z')]
        previous=states.get(episode)
        dt=None if previous is None else now-previous['time_s']
        require(dt is None or abs(dt-.25)<1e-12,ident+': wrong sample cadence')
        require(ev['native_motion_dt_s']==dt,ident+': native motion history mismatch')
        index=len(labels.setdefault(episode,[]))
        require(index<12 and abs(now-index*.25)<1e-12,ident+': wrong episode position')
        bounds=ev['native_bounds']
        require([b['name'] for b in bounds]==[o['name'] for o in frame['objects']]==[o['name'] for o in audit['objects']],ident+': actor identity/order mismatch')
        current={b['name']:b for b in bounds}
        expected_radials={}
        if previous is not None:
            camera_velocity=[(a-b)/dt for a,b in zip(camera,previous['camera'])]
            error=difference(camera_velocity,audit['camera_velocity_mps'])
            maxima['camera_velocity_error_mps']=max(maxima['camera_velocity_error_mps'],error)
            require(error<=MOTION_TOLERANCE_MPS,ident+': commanded camera velocity mismatch')
        for b,obj,intended in zip(bounds,frame['objects'],audit['objects']):
            require(all(math.isfinite(x) for x in b['center_m']+b['extent_m']),ident+': nonfinite native bounds')
            center_error=difference(b['center_m'],obj['center_m'])
            extent_error=difference(b['extent_m'],[s/2 for s in obj['size_m']])
            maxima['center_error_m']=max(maxima['center_error_m'],center_error)
            maxima['extent_error_m']=max(maxima['extent_error_m'],extent_error)
            require(center_error<=POSITION_TOLERANCE_M and extent_error<=POSITION_TOLERANCE_M,ident+': native/source geometry mismatch')
            old=None if previous is None else previous['bounds'][b['name']]['center_m']
            old_camera=None if previous is None else previous['camera']
            radial=radial_from_displacements(b['center_m'],camera,old,old_camera,dt)
            expected_radials[b['name']]=radial
            actual=b['native_relative_radial_velocity_mps']
            if radial is None:
                native_missing+=1
                require(actual is None,ident+': first native Doppler should be missing')
            else:
                native_present+=1
                require(actual is not None and math.isfinite(actual),ident+': native Doppler unexpectedly missing')
                error=abs(radial-actual)
                maxima['radial_formula_error_mps']=max(maxima['radial_formula_error_mps'],error)
                require(error<=1e-9,ident+': past-relative radial formula mismatch')
                velocity=[(a-b)/dt for a,b in zip(b['center_m'],old)]
                error=difference(velocity,intended['velocity_mps'])
                maxima['actor_velocity_error_mps']=max(maxima['actor_velocity_error_mps'],error)
                require(error<=MOTION_TOLERANCE_MPS,ident+': native actor motion disagrees with intended motion')
                motion_samples+=1
                if any(abs(v)>1e-5 for v in velocity):moving_episodes.add(episode)
        positive=any(native_intersects(b,ev['body_origin_m']) for b in bounds)
        require(positive==audit['source_aabb_labels'][index],ident+': native/source corridor label mismatch')
        labels[episode].append(positive);counts['positive' if positive else 'negative']+=1
        require(len(prov['radar_slots'])==4,ident+': wrong provenance slots')
        for key in ('radar_range_m','radar_angle','radar_velocity','radar_valid'):
            require(len(row[key])==4,ident+': wrong Radar slots')
        for k,slot in enumerate(prov['radar_slots']):
            observed=[row['radar_range_m'][k],row['radar_angle'][k],row['radar_velocity'][k]]
            require(bool(row['radar_valid'][k])==(slot is not None),ident+': raw/provenance validity mismatch')
            if slot is None:
                require(observed==[None,None,None],ident+': invalid slot contains measurements')
                continue
            require(slot['observed_triple']==observed,ident+': raw/provenance triple mismatch')
            kind=slot['kind'];slot_kinds[kind]+=1
            require(kind in ('real_actor','persistent_ghost','transient'),ident+': unknown provenance kind')
            require(math.isfinite(observed[0]) and math.isfinite(observed[1]),ident+': invalid observed range/angle')
            if dt is None:
                first_missing_slots+=1
                require(observed[2] is None and not slot['doppler_available'],ident+': first observed Doppler should be missing')
            require((observed[2] is not None)==bool(slot['doppler_available']),ident+': Doppler availability mismatch')
            if observed[2] is not None:
                require(math.isfinite(observed[2]),ident+': nonfinite observed Doppler')
                require(abs(observed[2]/.1-round(observed[2]/.1))<1e-7,ident+': Doppler quantization changed')
            if kind=='transient':
                require(slot['actor_id'] is None,ident+': transient has actor identity')
                continue
            if kind=='real_actor':
                actor=slot['actor_id'];require(actor.startswith(episode+'/'),ident+': wrong actor episode')
                name=actor[len(episode)+1:];require(name in current,ident+': unknown native actor')
                radial=expected_radials[name];center=current[name]['center_m']
                expected_range=ray_hit_horizontal_range(current[name],camera)
                error=abs(expected_range-slot['pre_noise_range_m'])
                maxima['ray_hit_range_error_m']=max(maxima['ray_hit_range_error_m'],error)
                require(error<=POSITION_TOLERANCE_M,ident+': native ray-hit range mismatch')
            else:
                require(slot['actor_id'] is None and frame['radar_ghost'] is not None,ident+': ghost identity mismatch')
                ghost=frame['radar_ghost'];center=[ghost['z'],ghost['x'],camera[2]]
                radial=radial_from_displacements(center,camera,center,None if previous is None else previous['camera'],dt)
                expected_range=math.hypot(center[0]-camera[0],center[1]-camera[1])
                require(abs(expected_range-slot['pre_noise_range_m'])<=1e-9,ident+': ghost range mismatch')
            recorded=slot['pre_noise_radial_velocity_mps']
            require((radial is None and recorded is None) or (radial is not None and recorded is not None and abs(radial-recorded)<=1e-9),ident+': provenance Doppler mismatch')
            angle=math.degrees(math.atan2(center[1]-camera[1],center[0]-camera[0]))-ev['camera']['yaw']
            require(abs(angle-slot['exact_angle_deg'])<=1e-8,ident+': provenance angle mismatch')
            require(slot['doppler_authority'].startswith('EVALUATOR_ONLY'),ident+': Doppler authority mismatch')
        traces.append(dict(id=ident,episode_id=episode,time_s=now,native_positive=positive,
                           native_radial_velocity_mps=expected_radials))
        states[episode]=dict(time_s=now,camera=camera,bounds=current)

    require(len(labels)==20 and all(len(v)==12 for v in labels.values()),'Wrong episode denominator')
    transitions=Counter()
    for episode,values in labels.items():
        enters=sum(not a and b for a,b in zip(values,values[1:]))
        exits=sum(a and not b for a,b in zip(values,values[1:]))
        require(enters==audits[episode]['enters'] and exits==audits[episode]['exits'],episode+': transition mismatch')
        transitions.update(enter=enters,exit=exits)
    require(counts==spec['source_design_aabb_frame_counts']==dict(positive=88,negative=152),'Native label denominator mismatch')
    require(transitions==spec['source_design_transition_counts']==dict(enter=13,exit=15),'Native transition count mismatch')
    require(len(moving_episodes)==16,'Expected 16 moving episodes')
    return dict(status='PASS',authority='EVALUATOR_ONLY_NATIVE_SOURCE_MOTION_AUDIT_NOT_PREDICTOR_INPUT',
                frames=240,episodes=20,native_aabb_frame_counts=dict(counts),native_transition_counts=dict(transitions),
                moving_episodes=len(moving_episodes),native_actor_motion_samples=motion_samples,
                native_doppler_present=native_present,native_doppler_missing=native_missing,
                first_frame_missing_doppler_slots=first_missing_slots,radar_slot_kinds=dict(slot_kinds),
                maxima=maxima,episode_labels=labels,
                camera_authority='COMMAND_SOURCE_CAMERA_IN_EVALUATOR_NOT_INDEPENDENT_ENGINE_CAMERA_READBACK',
                range_doppler_boundary='Native center LOS radial velocity; native center-directed ray-hit horizontal range; hypothetical sensor noise not RF.',
                traces=traces)


def run(capture,output,prediction_seal):
    capture=capture.resolve();output=output.resolve();prediction_seal=prediction_seal.resolve()
    artifacts=(ROOT/'artifacts.local').resolve()
    require(capture.is_relative_to(artifacts) and output.is_relative_to(artifacts),'Canonical artifacts only')
    require(not output.exists(),'Preserve existing audit output')
    seal=json.loads(prediction_seal.read_text())
    require(seal['status']=='ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE','All method predictions must be sealed first')
    for name,item in seal['panels'].items():
        require(sha(prediction_seal.parent/name/'predictions.json')==item['predictions_sha256'],name+': sealed predictions changed')
    panels=[p for p in seal['panels'].values() if Path(p['capture']).resolve()==capture]
    require(len(panels)==1,'Prediction seal must identify this capture exactly once')
    panel=panels[0]
    require(sha(capture/'receipt.json')==panel['receipt_sha256'],'Sealed capture receipt changed')
    require(sha(capture/'raw.jsonl')==panel['raw_sha256'],'Sealed raw observations changed')
    receipt=json.loads((capture/'receipt.json').read_text())
    require(receipt['status']=='PASS' and receipt['frames']==240,'Capture must pass all 240 frames')
    require(sha(capture/'spec.json')==receipt['spec_sha256'],'Capture spec changed')
    for name in ('raw.jsonl','evaluator.jsonl','provenance.jsonl','manifest.json'):
        require(sha(capture/name)==receipt['hashes'][name],name+': receipt hash mismatch')
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    try:
        result=audit_rows(json.loads((capture/'spec.json').read_text()),readrows(capture/'raw.jsonl'),
                          readrows(capture/'evaluator.jsonl'),readrows(capture/'provenance.jsonl'))
        result.update(capture=str(capture),prediction_seal=str(prediction_seal),prediction_seal_sha256=sha(prediction_seal),
                      input_hashes={n:sha(capture/n) for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl','provenance.jsonl')})
        traces=result.pop('traces');write(output/'native-audit-traces.json',traces)
        write(output/'summary.json',result)
        write(output/'completion.json',dict(status='PASS',source_sha256=sha(Path(__file__)),resources_started=[],
            output_hashes={p.name:sha(p) for p in output.glob('*.json')}))
        print(json.dumps({k:v for k,v in result.items() if k not in ('episode_labels','input_hashes')},indent=2))
    except Exception:
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),capture=str(capture),
                                        prediction_seal_sha256=sha(prediction_seal)))
        raise


def self_test():
    import copy
    import mz113_source_spec
    require(radial_from_displacements([3.5,0.,1.7],[.125,0.,1.7],[3.,0.,1.7],[0.,0.,1.7],.25)==1.5,'Receding test')
    require(radial_from_displacements([3.,0.,1.7],[0.,0.,1.7],None,None,None) is None,'First-frame test')
    bounds=dict(center_m=[3.,0.,1.7],extent_m=[.1,.2,.3])
    require(abs(ray_hit_horizontal_range(bounds,[0.,0.,1.7])-2.9)<1e-12,'Ray/AABB test')
    require(native_intersects(bounds,[0.,0.,0.]),'Positive geometry test')
    require(not native_intersects(dict(bounds,center_m=[3.,.6,1.7]),[0.,0.,0.]),'Negative geometry test')
    # Constructed native-schema fixture exercises all 240 labels, both motion
    # directions and raw/provenance checks without reading any actual capture.
    spec=mz113_source_spec.source();raw=[];evaluations=[];provenance=[];previous={}
    for frame in spec['frames']:
        episode=frame['episode'];now=frame['time_s'];camera=[frame['camera'][k] for k in ('x','y','z')]
        old=previous.get(episode);dt=None if old is None else now-old['time_s']
        bounds=[]
        for obj in frame['objects']:
            radial=radial_from_displacements(obj['center_m'],camera,None if old is None else old['centers'][obj['name']],
                                             None if old is None else old['camera'],dt)
            bounds.append(dict(name=obj['name'],center_m=obj['center_m'],extent_m=[s/2 for s in obj['size_m']],
                               native_relative_radial_velocity_mps=radial))
        native=bounds[0];radial=native['native_relative_radial_velocity_mps']
        distance=ray_hit_horizontal_range(native,camera)
        angle=math.degrees(math.atan2(native['center_m'][1]-camera[1],native['center_m'][0]-camera[0]))-frame['camera']['yaw']
        observed_v=None if radial is None else round(radial/.1)*.1
        ident=dict(id=frame['id'],episode_id=episode,time_s=now)
        row=dict(ident,camera_pitch_deg=-3.,delta_pitch=0.,radar_range_m=[distance,None,None,None],
                 radar_angle=[angle,None,None,None],radar_velocity=[observed_v,None,None,None],radar_valid=[True,False,False,False])
        ev=dict(ident,family=frame['family'],camera=frame['camera'],body_origin_m=frame['body_origin_m'],
                native_bounds=bounds,native_motion_dt_s=dt,authority='EVALUATOR_ONLY_SYNTHETIC_SELFTEST')
        slot=dict(kind='real_actor',actor_id=episode+'/'+native['name'],observed_triple=[distance,angle,observed_v],
                  doppler_available=observed_v is not None,pre_noise_radial_velocity_mps=radial,pre_noise_range_m=distance,
                  exact_angle_deg=angle,doppler_authority='EVALUATOR_ONLY_SYNTHETIC_SELFTEST')
        prov=dict(ident,radar_slots=[slot,None,None,None],authority='EVALUATOR_ONLY_SYNTHETIC_SELFTEST')
        raw.append(row);evaluations.append(ev);provenance.append(prov)
        previous[episode]=dict(time_s=now,camera=camera,centers={o['name']:o['center_m'] for o in frame['objects']})
    result=audit_rows(spec,raw,evaluations,provenance)
    require(result['status']=='PASS' and result['first_frame_missing_doppler_slots']==20,'Full synthetic fixture failed')
    for corruption in ('center','radial','first_doppler','provenance'):
        rs,es,ps=copy.deepcopy((raw,evaluations,provenance))
        if corruption=='center':es[0]['native_bounds'][0]['center_m'][0]+=.05
        elif corruption=='radial':es[1]['native_bounds'][0]['native_relative_radial_velocity_mps']+=.1
        elif corruption=='first_doppler':
            rs[0]['radar_velocity'][0]=0.;ps[0]['radar_slots'][0]['observed_triple'][2]=0.;ps[0]['radar_slots'][0]['doppler_available']=True
        else:ps[1]['radar_slots'][0]['observed_triple'][0]+=.1
        try:audit_rows(spec,rs,es,ps)
        except ValueError:pass
        else:raise ValueError('Failed to reject corruption: '+corruption)
    print('5 geometry checks + full240 synthetic audit +4 corruption rejections passed; no capture accessed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prediction-seal',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif not all((args.capture,args.output,args.prediction_seal)):parser.error('--capture, --output and --prediction-seal are required')
    else:run(args.capture,args.output,args.prediction_seal)
