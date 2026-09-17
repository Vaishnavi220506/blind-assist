"""Post-seal descriptive accounting only; no new model or working point."""
import json
from pathlib import Path
home=Path(__file__).resolve().parents[5]/'artifacts.local/work/corridor-public-single-20260917'
out=home/'confirmation';cap=home/'source/returned-v1/capture-v1'
cases=json.loads((out/'cases.json').read_text())
native=[json.loads(s) for s in (cap/'evaluator.jsonl').read_text().splitlines()]
rows=[json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
spec=json.loads((cap/'spec.json').read_text())
targets={f['id']:f['objects'][0]['name'] for f in spec['frames']}
misses=[]
for c,e,r in zip(cases,native,rows):
    assert c['id']==e['id']==r['id']
    if c['stratum']!='positive' or c['A']:continue
    origin=e['body_origin_m'];inside=[];target=[];returned_target=[]
    for z in e['zonal_tof_native']:
        for hit in z['private_rays']:
            if hit.get('actor_id','').rsplit('/',1)[-1]==targets[c['id']]:target.append([z['zone_id'],hit['subray']])
            p=hit.get('hit_point_m')
            if p and all(lo<=p[k]-origin[k]<=hi for k,(lo,hi) in enumerate([(.2,3.6),(-.3,.3),(.4,2.05)])):
                inside.append([z['zone_id'],hit['subray']])
        for lineage in z['returned_lineage']:
            if any([z['zone_id'],i] in target for i in lineage['hit_indices']):
                returned_target.append([z['zone_id'],lineage['target_index']])
    misses.append(dict(id=c['id'],family=c['family'],packet=r['tof_packet_received'],usable_returns=c['usable_tof_returns'],
        sampled_witness=c['sampled_witness'],native_corridor_ray_hits=len(inside),native_target_ray_hits=len(target),
        target_return_lineages=returned_target,A_retrained=c['control']))
clear=[c for c in cases if c['stratum']!='boundary']
result=dict(scope='POST_SEAL_DESCRIPTIVE_AUDIT_NOT_SELECTION',clear_A_misses=misses,
    positive_branch_frames=sum(c['positive'] for c in cases),positive_branch_clear_frames=sum(c['positive'] for c in clear),
    positive_branch_clear_negative_frames=sum(c['positive'] and not c['truth'] for c in clear),
    native_sampled_positive_clear_frames=sum(c['sampled_witness'] for c in clear),
    branch_vs_native_clear_mismatches=[c['id'] for c in clear if c['positive']!=c['sampled_witness']])
(out/'case-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
