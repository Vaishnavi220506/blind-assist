"""One frozen Depth Pro reference: observations-only inference, no training.

Run seal, smoke, infer in order. Evaluation is a separate executable. Mechanical
resume verifies existing prediction receipts; it never changes the protocol.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/ba-nfo-depthpro-20260919'
SOURCE = ROOT / 'artifacts.local/work/ba-nfo-20260919'
BASE = ROOT / 'artifacts.local/work/ba-nfo-frozen-transfer500-20260919'
HYP = ROOT / 'artifacts.local/datasets/hypersim-ba-nfo'
WEIGHT_SHA = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
UPSTREAM = '9e65e4dbe9568d23c546fcec53302b10445e109e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def rectify(rgb, matrix):
    """Observation-only oblique projection -> centered pinhole, no ray cropping.

    Return maps from original pixels to rectified pixels for inverse-z sampling.
    Optical z stays in the original camera frame (no rotation of the camera).
    """
    h,w=rgb.shape[:2]; yy,xx=np.mgrid[:h,:w]
    uv=np.stack([(xx+.5)/w*2-1,1-(yy+.5)/h*2,np.ones_like(xx)],-1)
    rays=uv@np.asarray(matrix,dtype=np.float64).T
    assert np.all(rays[...,2]<0)
    rx=rays[...,0]/-rays[...,2]; ry=rays[...,1]/-rays[...,2]
    f=min((w/2-2)/np.abs(rx).max(),(h/2-2)/np.abs(ry).max())
    target=np.stack([(xx+.5-w/2)/f,(h/2-yy-.5)/f,-np.ones_like(xx)],-1)
    original=target@np.linalg.inv(matrix).T
    uv_original=original[...,:2]/original[...,2:]
    sx=((uv_original[...,0]+1)*w/2-.5).astype(np.float32)
    sy=((1-uv_original[...,1])*h/2-.5).astype(np.float32)
    image=cv2.remap(rgb,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    bx=(f*rx+w/2-.5).astype(np.float32); by=(h/2-f*ry-.5).astype(np.float32)
    assert bx.min()>=1 and bx.max()<=w-2 and by.min()>=1 and by.max()<=h-2
    return image,float(f),bx,by


def seal():
    assert not (OUT / 'protocol.json').exists(), 'One frozen protocol only'
    assert sha(OUT / 'depth_pro.pt') == WEIGHT_SHA
    revision = subprocess.check_output(['git', '-C', str(OUT / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == UPSTREAM
    rows = json.loads((BASE / 'manifest.json').read_text())
    assert len(rows) == 500 and all(r['split'] == 'val' for r in rows)
    cameras = {r['scene_name']: r for r in csv.DictReader((SOURCE / 'metadata_camera_parameters.csv').open())}
    observed = []
    for row in rows:
        assert sha(SOURCE / row['prepared']) == row['sha256']
        assert sha(HYP / row['rgb']) == row['rgb_sha256']
        matrix=np.array([[float(cameras[row['scene']][f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)])
        observed.append(dict(id=row['id'], prepared=row['prepared'], sha256=row['sha256'],
            rgb=row['rgb'], rgb_sha256=row['rgb_sha256'], camera_matrix=matrix.tolist()))
    g5root = ROOT / 'artifacts.local/nearfield/distinct-views-20260907-v1'
    g5 = json.loads((g5root / 'predictions/result.json').read_text())['rows']
    diagnostics = []
    for row in g5:
        i = row['sample_index']
        path = g5root / f'capture/model/sample/{i:04d}.png'
        assert sha(path) == row['rgb_sha256']
        diagnostics.append(dict(id=f'g5-{i:04d}', case=row['case_name'], rgb=str(path),
            rgb_sha256=row['rgb_sha256'], camera=row['camera'],
            fx_native=320 / np.tan(np.deg2rad(row['camera']['hfov_deg']) / 2)))
    write(OUT / 'manifest.json', rows)
    write(OUT / 'observations.json', observed)
    write(OUT / 'diagnostic-observations.json', diagnostics)
    protocol = dict(id='ba-nfo-depthpro-20260919', phase='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        question='Does a frozen detail-preserving monocular reference recover structure and near2m support?',
        model=dict(repository='https://github.com/apple/ml-depth-pro', revision=revision,
                   checkpoint_sha256=WEIGHT_SHA, precision='float16 official CLI', calibration='Full public M_cam_from_uv rectification to centered pinhole at same input size, focal fits all original rays with2pixel margin; remap inverse-z back. No GT alignment.'),
        inputs=['low: unchanged RGB256x192', 'native: original RGB1024x768'],
        output='Optical-axis metric z. Rectified prediction bilinear inverse-depth remapped to original rays. Low output192x256; native output768x1024 sampled [::4,::4] to original point-ray label grid. Native full depth retained.',
        budget='One weight, 500x2 scientific predictions +18x2 separate old G5 diagnostics; no training, model sweep, threshold search or successor. Smoke uses upstream example RGB only.',
        baseline='Original NFO binary predictions at0.081 all500; existing UniDepthV2 ViT-S cache only matched70 supplemental rows, no extra mono inference.',
        gates=dict(far_small_recall_min=.75, far_small_iou='>=original', mixed_recall_min=.945, pure_far_fp='<=original'),
        contours='Near2m-mask boundary F1 tolerance1 pixel vs NFO on500 (metric-dependent); directed adjacent depth-ratio>1.25 P/R/F1 vs UniDepth on matched70 (scale-independent, no NFO pseudo-depth). Exclude unknown endpoint pairs.',
        contour_support='Near-mask boundary F1>=original+.02 AND matched70 scale-independent F1>=UniDepth+.02. Fixed diagnostic effect sizes, not confidence intervals.',
        decision='All4 task gates plus contour support allow retaining this teacher as a candidate for a separate ToF-assignment experiment; contour-only supports investigating range evidence; no contour gain closes this reference role. Native gains include additional input information. No automatic fusion/training.',
        diagnostics='All G5 eighteen consumed views, optical z<2m only; old <=3m heading-forward alerts not recomputed or renamed. Keep separate from500 cohort.',
        unknown='Unknown truth excluded and counted, never far; any invalid model output fails inference receipt, never silently drops frame.',
        limits=['Public weight pretraining overlap unexcluded; capability diagnosis, not independent generalization',
                'Known full camera matrix supplies calibration; public rectification resamples RGB and introduces padded corners; official infer internally resizes to1536square. Low-input arm is information-matched, not pixel/compute-matched',
                'NFO receives ToF; DepthPro sees RGB+public calibration only; not an isolated architecture or fusion ablation',
                'No device, event, deployment, App or safety promotion'],
        mechanical_recovery='Resume only identical hashes. Source/API/memory defects may be repaired with receipts before scores; no quality-selected modifications.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'manifest.json', BASE/'predictions.npz', BASE/'results.json', SOURCE/'metadata_camera_parameters.csv', Path(__file__)]})
    write(OUT / 'protocol.json', protocol)
    print('SEALED', sha(OUT / 'protocol.json'), flush=True)


def model_and_transform(device):
    sys.path.insert(0, str(OUT / 'upstream/src'))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(OUT / 'depth_pro.pt'))
    model, transform = depth_pro.create_model_and_transforms(config, device=torch.device(device), precision=torch.float16)
    return model.eval().requires_grad_(False), transform


@torch.inference_mode()
def smoke():
    assert (OUT / 'protocol.json').exists()
    sys.path.insert(0, str(ROOT / 'tools'))
    from research_backend import BackendCandidate, Workload, select_backend, torch_observation
    torch.set_num_threads(4)
    model, transform = model_and_transform('cpu')
    rgb = cv2.cvtColor(cv2.imread(str(OUT/'upstream/data/example.jpg')), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256,192), interpolation=cv2.INTER_AREA)
    x = transform(rgb)
    # Same official half-precision model/input on both devices; one timing each.
    def probe(device):
        model.to(device)
        return model.infer(x.to(device), f_px=torch.tensor(220., device=device))
    def observe(output):
        assert torch.isfinite(output['depth']).all() and (output['depth'] > 0).all()
        return torch_observation(model=model, output=output)
    receipt = select_backend(Workload.MODEL_INFERENCE,
        cpu=BackendCandidate('torch-cpu','cpu',lambda:probe('cpu'),observe),
        gpu=BackendCandidate('torch-cuda','cuda',lambda:probe('cuda'),observe,torch.cuda.synchronize),
        warmups=0, repeats=1, record_path=OUT/'backend.json')
    assert receipt['selected_device_type'] == 'cuda', 'Inspect placement before scientific run'
    write(OUT/'smoke.json', dict(status='PASS', protocol_sha256=sha(OUT/'protocol.json'),
        torch=torch.__version__, numpy=np.__version__, peak_allocated=torch.cuda.max_memory_allocated(),
        note='Existing runtime NumPy2.4.4 exceeds upstream declared numpy<2; source import and actual official inference pass. No shared runtime modification.'))


@torch.inference_mode()
def infer():
    smoke_record=json.loads((OUT/'smoke.json').read_text())
    assert smoke_record['status'] == 'PASS'
    protocol_sha = sha(OUT/'protocol.json')
    assert smoke_record['protocol_sha256'] == protocol_sha
    assert sha(OUT/'depth_pro.pt') == WEIGHT_SHA
    revision=subprocess.check_output(['git','-C',str(OUT/'upstream'),'rev-parse','HEAD'],text=True).strip()
    assert revision==UPSTREAM
    assert not subprocess.check_output(['git','-C',str(OUT/'upstream'),'status','--porcelain','--untracked-files=no'],text=True).strip()
    bound=[Path(__file__),OUT/'observations.json',OUT/'diagnostic-observations.json',
           OUT/'manifest.json',OUT/'protocol.json']+list((OUT/'upstream/src').rglob('*.py'))
    launch=dict(protocol_sha256=protocol_sha,weight_sha256=WEIGHT_SHA,upstream_revision=revision,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in bound},
        purpose='Pre-inference mechanical integrity addendum; binds manifests and repaired launch checks before scientific predictions',
        alignment='Low area RGB centers correspond to native4j+1.5, original nearest depth labels and native output sample at4j. Preserve legacy labels; native differences include inherited sampling mismatch.')
    if (OUT/'launch-seal.json').exists():
        assert json.loads((OUT/'launch-seal.json').read_text())==launch, 'Resume inputs/code changed'
    else:
        assert not (OUT/'predictions').exists() and not (OUT/'diagnostics').exists()
        write(OUT/'launch-seal.json',launch)
    torch.set_num_threads(4)
    model, transform = model_and_transform('cuda')
    times = []; outputs = {}; start_all = time.perf_counter()
    rows = json.loads((OUT/'observations.json').read_text())
    diags = json.loads((OUT/'diagnostic-observations.json').read_text())
    for group, entries in [('predictions',rows), ('diagnostics',diags)]:
        for i, row in enumerate(entries):
            rgbpath = HYP/row['rgb'] if group=='predictions' else Path(row['rgb'])
            assert sha(rgbpath) == row['rgb_sha256']
            native = cv2.cvtColor(cv2.imread(str(rgbpath)), cv2.COLOR_BGR2RGB)
            if group=='predictions':
                assert sha(SOURCE/row['prepared']) == row['sha256']
                # npz lazy loading: only observation RGB is accessed, never depth.
                with np.load(SOURCE/row['prepared']) as a:
                    low = a['rgb'].copy()
                np.testing.assert_array_equal(low, cv2.resize(native,(256,192),interpolation=cv2.INTER_AREA))
            else:
                low = cv2.resize(native,(256,144),interpolation=cv2.INTER_AREA)
            for arm, rgb in [('low',low),('native',native)]:
                dest = OUT/group/arm; dest.mkdir(parents=True, exist_ok=True)
                path = dest/f'{row["id"]}.npz'; receipt_path = path.with_suffix('.json')
                if path.exists():
                    old = json.loads(receipt_path.read_text())
                    assert old['protocol_sha256']==protocol_sha and old['sha256']==sha(path)
                    outputs[f'{group}/{arm}/{row["id"]}'] = old
                    continue
                if group=='predictions':
                    observed_rgb,fx,bx,by=rectify(rgb,row['camera_matrix'])
                else:
                    observed_rgb=rgb; fx=row['fx_native']*rgb.shape[1]/native.shape[1]
                torch.cuda.synchronize(); t=time.perf_counter()
                pred = model.infer(transform(observed_rgb), f_px=torch.tensor(fx, device='cuda'))
                assert pred['depth'].device.type=='cuda'
                full = pred['depth'].float().cpu().numpy()
                torch.cuda.synchronize(); seconds=time.perf_counter()-t
                assert full.shape==rgb.shape[:2] and np.isfinite(full).all() and (full>0).all()
                if group=='predictions':
                    full=1/cv2.remap(1/full,bx,by,cv2.INTER_LINEAR)
                    depth = full[::4,::4] if arm=='native' else full
                    assert depth.shape==(192,256)
                else:
                    depth = full if arm=='native' else cv2.resize(full,(640,360),interpolation=cv2.INTER_LINEAR)
                np.savez_compressed(path, depth=depth, **({'native_depth':full} if arm=='native' else {}))
                receipt=dict(sha256=sha(path),protocol_sha256=protocol_sha,rgb_sha256=row['rgb_sha256'],
                    seconds=seconds,input_shape=list(rgb.shape),fx=float(fx),device='cuda',precision='float16')
                write(receipt_path,receipt); outputs[f'{group}/{arm}/{row["id"]}']=receipt
                times.append(seconds)
            if (i+1)%10==0 or i+1==len(entries):
                print(group,i+1,'/',len(entries),'elapsed_s',round(time.perf_counter()-start_all,1),flush=True)
    assert len(outputs)==1036
    write(OUT/'prediction-seal.json',dict(protocol_sha256=protocol_sha, outputs=outputs,
        status='COMPLETE',new_inferences=len(times),elapsed_seconds=time.perf_counter()-start_all,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    print('COMPLETE',len(outputs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['seal','smoke','infer'])
    globals()[parser.parse_args().command]()
