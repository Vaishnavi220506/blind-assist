"""Posthoc consumed guard study: fixed recipe, authenticated observation caches."""
import argparse
import json
from pathlib import Path
import shutil
import time
from mz116_radar_resolution_guard import protect
from mz116_merged_collision import witness
from run_mz107_four_sensor import ROOT,readrows,sha,write,truth
from run_mz111_spatial_temporal import score
import mz111_spatial_evidence as radar
from mz115_spatial_allocation import legacy_empty_tof,raw_radar


def authenticate(source,item,predpath):
    assert sha(source/'raw.jsonl')==item['raw_sha256']
    assert sha(source/'receipt.json')==item['receipt_sha256']
    assert sha(predpath)==item['predictions_sha256']
    receipt=json.loads((source/'receipt.json').read_text());assert receipt['status']=='PASS'
    for name,digest in receipt['hashes'].items():assert sha(source/name)==digest,name
    return readrows(source/'raw.jsonl'),json.loads(predpath.read_text())


def run(output):
    output=output.resolve();assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    output.mkdir(parents=True);base=ROOT/'artifacts.local/work';packets={};seals={};costs={}
    names=['mz116_radar_resolution_guard.py','mz116_merged_collision.py','run_mz116_resolution_guard.py',
           'mz109_interval_extent.py','mz111_spatial_evidence.py','mz115_spatial_allocation.py','mz115_zonal_tof.py',
           'mz107_rgb_association.py','MZ116_PROTOCOL_20260913.md']
    for name in names:shutil.copyfile(Path(__file__).with_name(name),output/name)
    write(output/'method-freeze.json',dict(scope='OUTCOME_INFORMED_CONSUMED_DEVELOPMENT',hashes={n:sha(output/n) for n in names}))
    old=base/'mz114-spatial-evidence-20260913/analysis-v1'
    seal=json.loads((old/'prediction-seal.json').read_text())
    for panel,item in seal['panels'].items():
        source=Path(item['capture']);rows,cache=authenticate(source,item,old/panel/'predictions.json')
        start=time.perf_counter()
        guarded=[protect(r,p,s,n['integrated_yaw_deg']) for r,p,s,n in zip(rows,cache['combined_flow'],cache['filtered_plane'],cache['nominal'])]
        costs[panel]=(time.perf_counter()-start)/len(rows)
        arms=dict(baseline=cache['combined_flow'],nominal=cache['combined_flow'],guard=guarded)
        packets[panel]=(source,rows,{'3.6':arms})
    old=base/'mz115-zonal-allocation-20260913/analysis-v1';seal=json.loads((old/'prediction-seal.json').read_text())
    source=Path(seal['capture']);rows,cache=authenticate(source,seal,old/'predictions.json')
    empty=[legacy_empty_tof(r) for r in rows];primary=cache['3.6']['allocation']
    nominal=[dict(proposals=p['proposals'],integrated_yaw_deg=p['integrated_yaw_deg'],tof_support=False,
                  candidate=raw_radar(r,p['integrated_yaw_deg'],3.6),baseline=raw_radar(r,p['integrated_yaw_deg'],3.6)) for r,p in zip(rows,primary)]
    start=time.perf_counter();spatial=radar.predict(empty,nominal,surface='plane',filter_range=True)
    curves={}
    for distance,arms in cache.items():
        guarded=[protect(r,p,s,n['integrated_yaw_deg'],float(distance)) for r,p,s,n in zip(rows,arms['allocation'],spatial,nominal)]
        curves[distance]=dict(baseline=arms['allocation'],nominal=arms['allocation'],guard=guarded)
    costs['consumed_mz115']=(time.perf_counter()-start)/len(rows)
    packets['consumed_mz115']=(source,rows,curves)
    assert sum(len(rows) for _,rows,_ in packets.values())==1008
    for panel,(source,rows,curves) in packets.items():
        for distance,arms in curves.items():
            assert all(not b['candidate'] or a['candidate'] for a,b in zip(arms['guard'],arms['nominal']))
        target=output/panel;target.mkdir();write(target/'predictions.json',curves)
        seals[panel]=dict(capture=str(source),raw_sha256=sha(source/'raw.jsonl'),receipt_sha256=sha(source/'receipt.json'),predictions_sha256=sha(target/'predictions.json'))
    write(output/'prediction-seal.json',dict(status='FIXED_CONSUMED_PREDICTIONS_SERIALIZED_BEFORE_THIS_SCORING',panels=seals,
        limitation='Prior source outcomes informed this repair; not fresh confirmation'))
    write(output/'tof-only-collision.json',witness())
    summaries={};restored=[]
    for panel,(source,rows,curves) in packets.items():
        es=readrows(source/'evaluator.jsonl');ps=readrows(source/'provenance.jsonl')
        assert [r['id'] for r in rows]==[e['id'] for e in es]==[p['id'] for p in ps]
        summaries[panel]={}
        for distance,arms in curves.items():
            summaries[panel][distance]=score(rows,es,arms)
            for i,(r,e,p) in enumerate(zip(rows,es,arms['guard'])):
                if p['guard_added']:
                    restored.append(dict(panel=panel,distance_m=float(distance),id=r['id'],truth=bool(truth(e)),
                        guard_events=p['guard_events'],native_bounds=e['native_bounds'],provenance=ps[i]))
    result=dict(status='FIXED_GUARD_CONSUMED_STUDY_COMPLETE',frames=1008,panels=summaries,restored=restored,seconds_per_frame=costs,
                backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',workload='cached scalar box and current Radar readout'))
    write(output/'summary.json',result)
    write(output/'completion.json',dict(status='PASS',hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json')},resources_started=[]))
    print(json.dumps({panel:{arm:v['metrics'] for arm,v in distances['3.6'].items()} for panel,distances in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
