"""BA-NFO data/label preflight; deliberately does not download or train."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
REQUIRED=('hypersim','sanpo-synthetic')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    roots={k:None for k in REQUIRED}
    for k in REQUIRED:
        candidates=[ROOT/'artifacts.local/datasets'/k,ROOT/'artifacts.local/datasets'/k.replace('-','_'),ROOT/'artifacts.local/work'/k]
        found=next((p for p in candidates if p.exists()),None)
        roots[k]=dict(path=str(found),exists=True) if found else dict(path=None,exists=False)
    ue=list((ROOT/'artifacts.local/unreal').glob('*.json')) if (ROOT/'artifacts.local/unreal').exists() else []
    old=list((ROOT/'artifacts.local/work/mz120-learned-occupancy-20260913').rglob('*.npz'))
    report=dict(id='ba-nfo-preflight-20260918',status='BLOCKED_DATA_PREREQUISITES',
        question='Can direct task-specific RGB+simulated-8x8-ToF near-field occupancy beat frozen DEPTHOR?',
        phase='EXPLORE_PREPARATION_ONLY',
        required_data=roots,
        ue_inventory=dict(json_files=len(ue),sample_paths=[str(p) for p in ue[:10]],dense_rgb_depth_pairs=False),
        existing_occupancy=dict(mz120_npz=len(old),label_type='45-cell native AABB volume occupancy, not pixelwise 1/1.5/2/3m masks'),
        available_zju='Official ZJU-L5 is retained in ba-depth-probe smoke artifacts; it lacks BODY/HEAD truth and is sanity-only, not BA-NFO training authority',
        frozen_protocol=dict(inputs=['RGB uint8 640x360','simulated 8x8 ToF zones with range/status/missingness'],outputs=['P(depth<1m)','P(depth<1.5m)','P(depth<2m)','P(depth<3m)'],loss='BCE or focal plus ordinal monotonicity penalty',arms=['RGB-only','ToF-only','RGB+ToF','RGB+ToF+ordinal'],split='whole scenes/sequences, source-disjoint',baseline='Raw ToF, RGB metric mono, frozen DEPTHOR',budget='10k-30k frames, one frozen recipe, no architecture sweep',gate='mixed-zone 2m recall >=95.69% and IoU >=78.40% (+3 points) plus no worse UE event/FP evidence',stop='one failed gate closes this public-data direct occupancy route'),
        no_action=['No Hypersim/SANPO download performed','No old MZ120 labels repurposed','No training or alert integration','No body/head corridor truth inferred from ZJU-L5'],
        next_input='Provide or authorize acquisition of Hypersim/SANPO-Synthetic and a dense UE RGB+depth+ToF manifest; then run one frozen BA-NFO fit')
    (out/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'preflight.sha256').write_text(sha(out/'preflight.json'),encoding='ascii')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)



