"""Four consumed Willow frames plus an explicitly artificial bilateral control."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from tof_directional_readout import readout, zone_weights


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.source/'receipt.json').read_text());assert receipt['status']=='PASS'
    for name,digest in receipt['hashes'].items():assert sha(a.source/name)==digest,name
    rows=[json.loads(s) for s in (a.source/'raw.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==['willow-clear','willow-left','willow-right','willow-center']
    baseline=json.loads(a.baseline.read_text());assert len(baseline)==len(rows)
    results=[dict(id=r['id'],rgb_path=r['rgb_path'],incumbent_alert=b['candidate'],
                  weighted=readout(r,equal_weights=False),equal=readout(r,equal_weights=True)) for r,b in zip(rows,baseline)]
    # No RGB is assigned to this control: separate captures are not a joint observation.
    bilateral=copy.deepcopy(rows[1]);bilateral['id']='ARTIFICIAL_BILATERAL_UNIT_CONTROL'
    for z,right in zip(bilateral['tof_zones'],rows[2]['tof_zones']):
        assert z['zone_id']==right['zone_id']
        if right['targets']:
            assert not z['targets'];z['targets']=copy.deepcopy(right['targets']);z['target_count']=len(z['targets'])
    control=readout(bilateral,equal_weights=False)
    assert [r['horizontal'] for r in control['regions']]==['LEFT','RIGHT']
    expected=[[],['LEFT'],['RIGHT'],['CENTER']]
    for r,want in zip(results,expected):
        assert [c['horizontal'] for c in r['weighted']['regions']]==want
        assert [c['horizontal'] for c in r['equal']['regions']]==want
    document=dict(scope='ENGINEERING_DEMO_CONSUMED_SIMULATION_NOT_NEW_VALIDATION',
        hypothesis='Preserved zone positions can add direction without collapsing bilateral support.',
        stop='Four existing frames plus one artificial bilateral control; no tuning/training/capture.',
        results=results,bilateral_control=dict(authority='ARTIFICIAL_OBSERVABLE_SPLICE_NOT_CAPTURE',readout=control),
        weight_map=[dict(zone_id=z['zone_id'],**zone_weights(z)) for z in rows[0]['tof_zones']],
        comparison='Same four directional outcomes with equal and quality weights; no demonstrated quality-weight gain.',
        hashes={str(p):sha(p) for p in [a.source/'raw.jsonl',a.source/'receipt.json',a.baseline,Path(__file__),Path(__file__).with_name('tof_directional_readout.py')]})
    (a.output/'results.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
    for r in results:
        print(r['id'],r['weighted']['state'],[(c['horizontal'],round(c['bearing_deg'],2),c['valid_slant_median_m']) for c in r['weighted']['regions']])
    print(document['comparison']);print('Artificial bilateral: LEFT + RIGHT, not CENTER')


if __name__=='__main__':main()
