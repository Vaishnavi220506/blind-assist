"""M1-only candidate-mask readout with exact S2 transport covariance.

Existing simulated observations only. Privileged object IDs supply masks; oracle
ranges never enter this module. No M0/M2 recomputation and no old M1 reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cnh_segment_joint_readout import WINDOWS_R, query_membership

FAMILIES = {'v2': 'cnh-track-a-scale-v2-20260926', 'v4': 'cnh-track-a-scale-v4-20260926'}
BASE_KEYS = {'v2': 'G0', 'v4': 'S2__noisy@0.75'}


def linear_statistics(weights, total, variance, past):
    """Tensor linear mean/variance, including transport-induced offdiagonals.

    past = [(A[current <- previous], previous diagonal variance), ...].
    Pure tensor arithmetic also permits small CPU correctness tests.
    """
    mean = weights @ total
    var = weights.square() @ variance
    for matrix, previous_var in past:
        moved = weights @ matrix
        var = var + moved.square() @ previous_var
    return mean, var.clamp_min(1e-9)


def weights_for_coverage(coverage):
    coverage = np.asarray(coverage, dtype=float)
    if coverage.shape != (8, 8) or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError('Expected finite 8x8 coverage fractions')
    weights = np.zeros((len(WINDOWS_R), 8, 8, 16), dtype=np.float32)
    for i, (start, width) in enumerate(WINDOWS_R):
        weights[i, :, :, start:start+width] = coverage[..., None]
    return weights.reshape(len(WINDOWS_R), 1024)


def reduce_queries(z, membership):
    """Maximum candidate/window evidence; absent query remains UNKNOWN (-inf)."""
    return np.where(membership, np.asarray(z)[:, None], -np.inf).max(axis=0)


def sequence_scores(hist, ambient, bias, tq, noisy, fields):
    """Only public measurements/poses and privileged angular masks accepted."""
    import torch
    import cnh_track_a_gpu_readout as g
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no heavy CPU fallback')
    n = len(hist)
    if any(len(frames) != n for frames in fields.values()):
        raise ValueError('Candidate frames are not aligned with histogram')
    residual = (g.T(hist) - g.T(bias)).reshape(n, 1024)
    variance = (16*g.T(ambient)[..., None] + g.T(bias).clamp_min(0)).reshape(n, 1024)
    pairs = [(i,j) for i in range(1,n) for j in range(max(0,i-3),i)]
    pose = torch.as_tensor(noisy, dtype=g.D64, device=g.DEV)
    if pairs:
        ii = torch.tensor([i for i,j in pairs], device=g.DEV)
        jj = torch.tensor([j for i,j in pairs], device=g.DEV)
        matrix = g.transport(torch.linalg.inv(pose[ii]) @ pose[jj], 1).to(g.DT)
        total = residual.clone().index_add_(0, ii, torch.bmm(matrix, residual[jj].unsqueeze(-1)).squeeze(-1))
    else:
        matrix, total = [], residual
    out = {f'M1__{condition}': np.full((n,6), -np.inf) for condition in fields}
    for t in range(n):
        # Drop conditions and other repeated masks share weights/membership,
        # while their candidate presence remains condition-specific.
        coverage_index, templates, layouts, member_cache = {}, [], [], {}
        for condition, frames in fields.items():
            for candidate in frames[t]:
                coverage = np.asarray(candidate['coverage'], dtype=np.float64)
                key = coverage.tobytes()
                if key not in coverage_index:
                    coverage_index[key] = len(templates)
                    templates.append(weights_for_coverage(coverage))
                mask = np.asarray(candidate['mask'], dtype=bool)
                mk = mask.tobytes()
                if mk not in member_cache:
                    if mask.shape != (128,128):
                        raise ValueError('Mask raster must be128x128')
                    member_cache[mk] = query_membership(mask, tq[t])[0]
                layouts.append((condition, coverage_index[key], member_cache[mk]))
        if not templates:
            continue
        u = g.T(np.concatenate(templates))
        past = [(matrix[k], variance[j]) for k,(i,j) in enumerate(pairs) if i == t]
        mean, var = linear_statistics(u, total[t], variance[t], past)
        z = (mean / var.sqrt()).cpu().numpy().reshape(len(templates),len(WINDOWS_R))
        for condition,index,member in layouts:
            key = f'M1__{condition}'
            out[key][t] = np.maximum(out[key][t], reduce_queries(z[index], member))
    return out


def find_unit(root, unit, suffix='.npz'):
    for width in (2,3):
        p = Path(root)/f'unit{unit:0{width}d}{suffix}'
        if p.exists():
            return p
    raise FileNotFoundError(f'unit{unit} under {root}')


def cohort(dataset):
    return [u for u in range(96,192) if u != 143] if dataset == 'v2' else list(range(96))


def score_unit(args, unit):
    import torch
    import cnh_track_a_scale_evaluate as se
    from cnh_track_a_readout import noisy_poses
    from cnh_segment_learned_masks import CONDITIONS, eval_candidates
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    se.sensor_module.FAMILY = args.family
    start = time.monotonic()
    split, records, step = se.unit_records(args.geometry, args.sensor, unit, -10, 1)
    expected_split = ('calib' if unit <128 else 'audit') if args.dataset == 'v2' else ('calib' if unit<32 else 'audit')
    if split != expected_split:
        raise ValueError(f'Unit{unit} split mismatch: {split}')
    geometry_path = args.geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json'
    geo = json.loads(geometry_path.read_text(encoding='utf-8-sig'))
    objects = {c['config']:c['objects'] for c in geo['configs']}
    oracle_path = next((p/f'unit{unit:02d}-mount-10-oracle.npz' for p in args.oracle
                        if (p/f'unit{unit:02d}-mount-10-oracle.npz').exists()),None)
    if oracle_path is None:
        raise FileNotFoundError(f'No object_id source for unit{unit}')
    with np.load(oracle_path, allow_pickle=False) as f:
        oid = f['object_id']  # no oracle ranges, classes or contributor truth
    observation_path = args.sensor/f'unit{unit:02d}-mount-10-observations.npz'
    with np.load(observation_path, allow_pickle=False) as f:
        config, frame, rate = f['config'], f['frame'], int(f['rate'])
    if rate not in (5,10) or len(oid) != len(config):
        raise ValueError('Object IDs / observations not aligned')
    bias = np.load(args.bias, allow_pickle=False)
    pool = [np.asarray(x,int) for x in json.loads(args.pool.read_text(encoding='utf-8-sig'))]
    arrays, configs, frames = {}, [], []
    for rec in records:
        rows = np.flatnonzero(config == rec['config'])[::step]
        if len(rows) != len(rec['hist']) or len(rows) != 12:
            raise ValueError('Expected same12 sampled frames as sourceS2')
        fields = eval_candidates(oid[rows], objects[rec['config']], unit, rec['config'], args.family, pool)
        if set(fields) != set(CONDITIONS):
            raise ValueError('Candidate condition contract mismatch')
        scores = sequence_scores(rec['hist'], rec['ambient'], bias, rec['tq'],
                                 noisy_poses(rec['poses'], rec['ego_seed'], dt=.2), fields)
        for k,x in scores.items():
            arrays.setdefault(k,[]).append(x)
        configs.append(np.full(len(rows),rec['config']))
        frames.append(np.arange(len(rows)))
    result = {k:np.concatenate(v) for k,v in arrays.items()}
    result.update(config=np.concatenate(configs),frame=np.concatenate(frames),split=np.asarray(split),family=np.asarray(args.family))
    baseline_path = find_unit(args.baseline,unit)
    with np.load(baseline_path, allow_pickle=False) as f:
        for key in ('config','frame'):
            if not np.array_equal(f[key],result[key]):
                raise ValueError(f'SourceS2 row-order mismatch {key}')
        result['S2'] = f[BASE_KEYS[args.dataset]]
    if result['S2'].shape != (len(result['frame']),6):
        raise ValueError('Baseline score shape mismatch')
    output = args.output/'scores'/f'unit{unit:02d}.npz'
    np.savez_compressed(output,**result)
    receipt = dict(unit=unit,split=split,seconds=time.monotonic()-start,backend='cuda',device=torch.cuda.get_device_name(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),source_s2=str(baseline_path),
                   source_s2_key=BASE_KEYS[args.dataset],oracle_object_id_source=str(oracle_path),observations=str(observation_path),
                   family=args.family,dt=.2,snr_index=1,mount=-10,conditions=list(CONDITIONS),code_identity=args.code_identity,
                   source_contract=args.source_contract)
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','sensor','baseline','bias','pool','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--oracle',type=Path,nargs='+',required=True)
    p.add_argument('--dataset',choices=('v2','v4'),required=True)
    p.add_argument('--family',required=True)
    p.add_argument('--units',type=int,nargs='+')
    a=p.parse_args()
    if a.family != FAMILIES[a.dataset]:
        raise ValueError('Family/dataset mismatch')
    a.units = a.units or cohort(a.dataset)
    if len(a.units)!=len(set(a.units)) or not set(a.units)<=set(cohort(a.dataset)):
        raise ValueError('Unsupported or duplicate units')
    (a.output/'scores').mkdir(parents=True,exist_ok=True)
    own = Path(__file__)
    paths=[own,own.with_name('cnh_segment_learned_masks.py'),own.with_name('cnh_segment_joint_readout.py')]
    a.code_identity={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    a.source_contract={k:str(getattr(a,k).resolve()) for k in ('geometry','sensor','baseline','bias','pool')}
    a.source_contract['oracle']=[str(x.resolve()) for x in a.oracle]
    a.source_contract['bias_sha256']=hashlib.sha256(a.bias.read_bytes()).hexdigest()
    a.source_contract['pool_sha256']=hashlib.sha256(a.pool.read_bytes()).hexdigest()
    # Resume only completed files from the same code/source contract.
    todo=[]
    for u in a.units:
        path=a.output/'scores'/f'unit{u:02d}.json'
        if path.exists() and path.with_suffix('.npz').exists():
            r=json.loads(path.read_text())
            if (r.get('code_identity')!=a.code_identity or r.get('family')!=a.family
                    or r.get('source_contract')!=a.source_contract):
                raise ValueError(f'Resume mismatch unit{u}; use separate output')
        else: todo.append(u)
    progress=dict(status='running',dataset=a.dataset,total=len(a.units),complete=len(a.units)-len(todo),code_identity=a.code_identity)
    record=a.output/'progress.json'
    try:
        record.write_text(json.dumps(progress),encoding='utf-8')
        for u in todo:
            r=score_unit(a,u)
            progress.update(complete=progress['complete']+1,last=r)
            record.write_text(json.dumps(progress),encoding='utf-8')
            print(json.dumps(r),flush=True)
        progress['status']='complete'
    except BaseException as exc:
        progress.update(status='failed',error=str(exc)); raise
    finally:
        record.write_text(json.dumps(progress,indent=2),encoding='utf-8')

if __name__=='__main__': main()
