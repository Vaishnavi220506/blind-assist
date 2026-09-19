"""One consumed96-frame nominal-FOV contrast; no neural model or new capture."""
import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from ba_camera_corridor import sample_native, rays, tof_readout
from evaluate_ba_camera_corridor import read, sha, require, summarize, validate_predictions, write_new
from cross_zone_anchor_core import zone_map
from tof_fov45_core import boxes45, simulate, geometry_description

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/ba-camera-corridor-20260919'
PRIOR=ROOT/'artifacts.local/work/ba-cross-zone-anchor-ceiling-20260920'
OUT=ROOT/'artifacts.local/work/ba-tof-fov45-20260920'


def verify(out):
    protocol=read(out/'protocol.json')
    for name,digest in protocol['code_hashes'].items():
        require(sha(Path(__file__).with_name(name))==digest,'Code changed: '+name)
    for name,digest in protocol['input_hashes'].items():
        require(sha(ROOT/name)==digest,'Input changed: '+name)
    return protocol


def serial_trace(trace):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in trace.items()}


def materialize(out):
    verify(out)
    require(not (out/'observations').exists(),'Observation attempt already exists')
    (out/'observations').mkdir()
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.sum(np.arange(8)),
        lambda _:DeviceObservation('cpu',platform.processor(),'NumPy',('CPU',))),
        cpu_reason='TASK_NOT_GPU_SUITABLE',capabilities={'scope':'Scalar simulation and geometry only'},record_path=out/'backend.json')
    _,_,_,observations,_=validate_predictions(OLD)
    sources=read(OLD/'evaluator-source.json');began=time.perf_counter();outputs=[];lineage=[]
    # Private native depth is used only by the declared observation simulator.
    # No target footprint, event label or old prediction enters simulate().
    for old,source in zip(observations,sources):
        require(old['id']==source['id'],'Identity changed')
        native=OLD/'capture/evaluator'/source['geometry']['native_path']
        require(sha(native)==source['sensor_native_sha256'],'Native changed')
        depth=sample_native(np.load(native,allow_pickle=False))
        seed=source['sensor_seed_identity']
        values,traces=simulate(depth,seed,boxes45())
        path=out/'observations'/(old['id']+'.npz')
        np.savez_compressed(path,boxes=boxes45(),values=values)
        outputs.append({**{k:old[k] for k in ('id','clip_id','frame_in_clip','time_s')},
            'path':path.relative_to(out).as_posix(),'sha256':sha(path),'native_sha256':sha(native),'seed':seed})
        lineage.append(dict(id=old['id'],traces=[serial_trace(t) for t in traces]))
    require(len(outputs)==96,'Frame budget changed')
    write_new(out/'private-lineage.json',lineage)
    write_new(out/'observations.json',outputs)
    write_new(out/'observation-seal.json',dict(status='COMPLETE',frames=96,protocol_sha256=sha(out/'protocol.json'),
        observations_sha256=sha(out/'observations.json'),private_lineage_sha256=sha(out/'private-lineage.json'),
        geometry=geometry_description(),elapsed_s=time.perf_counter()-began))


def predict(out):
    verify(out);seal=read(out/'observation-seal.json');began=time.perf_counter()
    require(seal['status']=='COMPLETE' and seal['frames']==96 and seal['protocol_sha256']==sha(out/'protocol.json'),'Observation protocol/status differs')
    require(seal['observations_sha256']==sha(out/'observations.json'),'Observations changed')
    predictions=[]
    for observation in read(out/'observations.json'):
        path=out/observation['path'];require(sha(path)==observation['sha256'],'Observation changed')
        with np.load(path,allow_pickle=False) as data:
            decision,anchors=tof_readout(data['boxes'],data['values'])
        predictions.append({**{k:observation[k] for k in ('id','clip_id','frame_in_clip','time_s')},
            'observation_sha256':sha(path),'decision':decision,'anchors':anchors})
    write_new(out/'predictions.json',predictions)
    write_new(out/'prediction-seal.json',dict(status='COMPLETE',frames=96,protocol_sha256=sha(out/'protocol.json'),
        observation_seal_sha256=sha(out/'observation-seal.json'),predictions_sha256=sha(out/'predictions.json'),
        elapsed_s=time.perf_counter()-began,model_calls=0))


def evaluate(out):
    verify(out);began=time.perf_counter();seal=read(out/'prediction-seal.json')
    observation_seal=read(out/'observation-seal.json')
    for stage in (seal,observation_seal):
        require(stage['status']=='COMPLETE' and stage['frames']==96 and stage['protocol_sha256']==sha(out/'protocol.json'),'Stage protocol/status differs')
    require(seal['predictions_sha256']==sha(out/'predictions.json'),'Predictions changed')
    require(seal['observation_seal_sha256']==sha(out/'observation-seal.json'),'Observation seal changed')
    require(observation_seal['observations_sha256']==sha(out/'observations.json'),'Observation index changed')
    observations=read(out/'observations.json');sealed_predictions=read(out/'predictions.json')
    require(len(observations)==len(sealed_predictions)==96,'Stage frame count changed')
    for i,(observation,prediction) in enumerate(zip(observations,sealed_predictions)):
        require(observation['id']==prediction['id']==f'f{i:04d}','Stage frame order changed')
        require(all(observation[k]==prediction[k] for k in ('clip_id','frame_in_clip','time_s')),'Stage timestamps differ')
        require(sha(out/observation['path'])==observation['sha256']==prediction['observation_sha256'],'Prediction input binding differs')
    require(read(out/'observation-seal.json')['private_lineage_sha256']==sha(out/'private-lineage.json'),'Lineage changed')
    from audit_cross_zone_corridor import target_footprint,frame_lineage,baseline_parity
    rows=read(OLD/'frame-results.json');prior=read(PRIOR/'corridor-results.json')
    baseline=baseline_parity(rows,read(OLD/'results.json'),.2)
    candidate=copy.deepcopy(rows);predictions=read(out/'predictions.json')
    lineage=read(out/'private-lineage.json');sources=read(OLD/'evaluator-source.json')
    require(len(rows)==len(predictions)==len(lineage)==len(sources)==96,'Cohort differs')
    znew=zone_map(boxes45(),(192,256));records=[];a,b=rays()
    with np.load(OLD/read(OLD/'observations.json')[0]['prepared'],allow_pickle=False) as data:
        zold=zone_map(data['boxes'],(192,256))
    require(np.all((znew<0)|(zold>=0)),'Candidate FOV must be nested')
    for row,new,pred,lin,source,previous in zip(rows,candidate,predictions,lineage,sources,prior['frames']):
        require(row['id']==pred['id']==lin['id']==source['id']==previous['id'],'Scoring identity differs')
        require(all(row[k]==pred[k] for k in ('clip_id','frame_in_clip','time_s')),'Temporal identity differs')
        new['predictions']['raw_tof']=pred['decision']
        native=OLD/'capture/evaluator'/source['geometry']['native_path'];require(sha(native)==source['sensor_native_sha256'],'Native changed')
        raw=np.load(native,allow_pickle=False);mask,_=target_footprint(raw,source);target=sample_native(mask)
        depth=sample_native(raw);known=np.isfinite(depth)&(depth>0)
        inside=known&(depth>=.3)&(depth<=3)&(abs(a*depth)<=.3)&(b*depth>=-.2)&(b*depth<=.9)
        traces=[{k:np.asarray(v) if k in ('pixel_indices','weights') else v for k,v in t.items()} for t in lin['traces']]
        newlin=frame_lineage(target,znew,traces)
        coverage={name:int(m.sum()) for name,m in dict(target=target,target_old=target&(zold>=0),target_new=target&(znew>=0),
            corridor_reference=inside,corridor_reference_old=inside&(zold>=0),corridor_reference_new=inside&(znew>=0),
            known_old=known&(zold>=0),known_new=known&(znew>=0)).items()}
        records.append(dict(id=row['id'],clip_id=row['clip_id'],source_clip_id=row['source_clip_id'],frame_in_clip=row['frame_in_clip'],
            time_s=row['time_s'],truth=row['truth'],boundary=row['boundary'],old=row['predictions']['raw_tof'],new=pred['decision'],
            old_lineage=previous['lineage'],new_lineage=newlin,coverage=coverage))
    changed=summarize(candidate,.2)['arms']['raw_tof'];events=[]
    for old_event,new_event in zip(baseline['arms']['raw_tof']['events'],changed['events']):
        require(all(old_event[k]==new_event[k] for k in ('clip_id','start_frame','end_frame','interior_frames','entry_time_s')),'Events changed')
        members=[r for r in records if r['clip_id']==old_event['clip_id'] and old_event['start_frame']<=r['frame_in_clip']<=old_event['end_frame']]
        entry=dict(old=old_event,new=new_event)
        for layout in ('old','new'):
            for kind in ('pure','mixed'):
                hits=[r for r in members if r[layout+'_lineage']['observed_range_lt_3m'][kind+'_anchor_available']]
                clip_hits=[r for r in records if r['clip_id']==old_event['clip_id'] and r[layout+'_lineage']['observed_range_lt_3m'][kind+'_anchor_available']]
                entry[layout+'_'+kind+'_near']=dict(frames=len(hits),interior_frames=sum(not r['boundary'] for r in hits),
                    first_time_s=hits[0]['time_s'] if hits else None,first_relative_to_entry_s=hits[0]['time_s']-old_event['entry_time_s'] if hits else None,
                    first_within_clip_s=clip_hits[0]['time_s'] if clip_hits else None,
                    first_within_clip_relative_to_entry_s=clip_hits[0]['time_s']-old_event['entry_time_s'] if clip_hits else None,
                    timing_scope='first_time_s is event-only; first_within_clip_s includes pre-entry observations')
        events.append(entry)
    write_new(out/'frame-results.json',records)
    result=dict(status='COMPLETE',scope='CONSUMED_CONTROLLED_SIMULATION_NOMINAL_FOV_ONLY',frames=96,geometry=geometry_description(),
        baseline=baseline['arms']['raw_tof'],candidate=changed,events=events,
        image_coverage=dict(old_pixels=int((zold>=0).sum()),new_pixels=int((znew>=0).sum()),lost_pixels=int(((zold>=0)&(znew<0)).sum()),
            image_pixels=192*256,outside_candidate_semantics='UNOBSERVED, never free space'),
        coverage={k:sum(r['coverage'][k] for r in records) for k in records[0]['coverage']},
        by_clip={clip:dict(old=summarize([r for r in rows if r['clip_id']==clip],.2)['arms']['raw_tof'],
            new=summarize([r for r in candidate if r['clip_id']==clip],.2)['arms']['raw_tof']) for clip in sorted({r['clip_id'] for r in rows})},
        original_baseline_exact_parity=True,frame_results_sha256=sha(out/'frame-results.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
        protocol_sha256=sha(out/'protocol.json'),elapsed_s=time.perf_counter()-began,
        warning='Point bins on fixed simulated optical-Z lattice, not calibrated VL53 range/PSF/photon simulation or physical efficacy. NFO not rerun.')
    write_new(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','image_coverage','coverage','elapsed_s')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=['materialize','predict','evaluate']);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();globals()[args.phase](args.out)
