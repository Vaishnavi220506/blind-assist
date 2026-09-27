"""Descriptive H3 recording diagnostics; no independence or calibration claim."""
import json
from pathlib import Path
import numpy as np


def load_recording(path):
    """Read raw serial bytes in acquisition order, retaining every bad-line receipt."""
    path = Path(path)
    if path.is_dir():
        path = path / 'tof' / 'raw.bin'
    raw = path.read_bytes()
    lines = raw.split(b'\n') if raw else []
    if lines and lines[-1] == b'':
        lines.pop()  # a terminating newline is not an extra received line
    frames, good_lines, errors, other, blank = [], [], [], [], []
    for i, line in enumerate(lines, 1):
        try:
            if not line.strip():
                blank.append(i)
                raise ValueError('blank serial line')
            f = json.loads(line.decode('utf-8'))
            if not isinstance(f, dict):
                raise ValueError('JSON is not an object')
            if f.get('type') != 'cnh_frame':
                other.append(dict(line=i, type=f.get('type')))
                continue
            for key, shape in [('hist_raw',(64,16)), ('hist_scaler',(64,16)),
                               ('ambient_raw',(64,)), ('ambient_scaler',(64,)),
                               ('distance_mm',(64,)), ('target_status',(64,)), ('nb_target',(64,))]:
                a = np.asarray(f[key], dtype=float)
                if a.shape != shape or not np.isfinite(a).all():
                    raise ValueError(f'{key}: invalid shape or nonfinite values {a.shape}')
            for key in ('seq', 'ms'):
                if not isinstance(f[key], int) or isinstance(f[key], bool):
                    raise ValueError(f'{key}: expected integer')
            h = np.ldexp(np.asarray(f['hist_raw'], float), -np.asarray(f['hist_scaler'], int))
            a = np.ldexp(np.asarray(f['ambient_raw'], float), -np.asarray(f['ambient_scaler'], int))
            if not np.isfinite(h).all() or not np.isfinite(a).all():
                raise ValueError('nonfinite scaled values')
            frames.append((f,h,a)); good_lines.append(i)
        except (ValueError, KeyError, TypeError, OverflowError) as e:
            errors.append(dict(line=i, bytes=len(line), error=str(e), prefix_hex=line[:48].hex()))
    for e in errors:
        e['position'] = ('internal' if good_lines and good_lines[0] < e['line'] < good_lines[-1] else 'boundary')
    arr = lambda k: np.asarray([f[k] for f,_,_ in frames])
    seq, ms = arr('seq').astype(np.int64), arr('ms').astype(np.int64)
    events = []
    for i, delta in enumerate(np.diff(seq), 1):
        if delta != 1:
            events.append(dict(previous_line=good_lines[i-1], line=good_lines[i], previous_seq=int(seq[i-1]),
                               seq=int(seq[i]), kind='gap' if delta > 1 else 'duplicate' if delta == 0 else 'reset_or_reorder',
                               missing=int(delta-1) if delta > 1 else 0))
    quality = dict(source=str(path), total_lines=len(lines), valid_frames=len(frames), bad_lines=len(errors),
                   bad_line_rate=len(errors)/len(lines) if lines else None, blank_lines=len(blank),
                   valid_nonframe_lines=len(other), other_lines=other, errors=errors,
                   boundary_bad_lines=sum(e['position']=='boundary' for e in errors),
                   internal_bad_lines=sum(e['position']=='internal' for e in errors),
                   missing_seq=sum(e['missing'] for e in events),
                   duplicate_seq=sum(e['kind']=='duplicate' for e in events),
                   resets_or_reorders=sum(e['kind']=='reset_or_reorder' for e in events), sequence_events=events,
                   note='All newline-delimited byte records, including blanks and an unterminated final record. Boundary damage is not assumed transport loss. Missing sequences and bad lines may describe the same loss; do not add.')
    return dict(H=np.asarray([h for _,h,_ in frames]).reshape(-1,64,16),
                A=np.asarray([a for _,_,a in frames]).reshape(-1,64),
                D=arr('distance_mm').reshape(-1,64), S=arr('target_status').reshape(-1,64),
                N=arr('nb_target').reshape(-1,64), seq=seq, ms=ms, line_indices=np.asarray(good_lines, dtype=np.int64), quality=quality)


def distribution(values):
    a = np.asarray(values, float).reshape(-1)
    ok = np.isfinite(a)
    return dict(n=int(ok.sum()), excluded=int((~ok).sum()),
                quantiles={str(q):float(np.percentile(a[ok],q)) for q in (0,5,25,50,75,95,100)} if ok.any() else {},
                mean=float(a[ok].mean()) if ok.any() else None)


def contiguous_runs(d):
    """Require consecutive sequence and positive dt <= 1.5 times median period."""
    seq, ms = np.asarray(d['seq']), np.asarray(d['ms'])
    dt = np.diff(ms)
    eligible = dt[(np.diff(seq)==1) & (dt>0)]
    period = float(np.median(eligible)) if len(eligible) else None
    split = np.flatnonzero((np.diff(seq)!=1) | (dt<=0) | (dt > 1.5*period if period else True))+1
    return [a for a in np.split(np.arange(len(seq)), split) if len(a)], period


def channel_diagnostics(x, runs):
    """Run-demeaned descriptive Pearson correlations and nonoverlap sum variance."""
    x = np.asarray(x, float).reshape(len(x),-1)
    residual = np.zeros_like(x)
    for r in runs:
        residual[r] = x[r]-x[r].mean(0)
    usable = [r for r in runs if len(r)>=2]
    df = sum(len(r)-1 for r in usable)
    variance = sum((residual[r]**2).sum(0) for r in usable)/df if df else np.full(x.shape[1],np.nan)
    ac = {}
    for lag in (1,2,3,4):
        pairs = [(residual[r[:-lag]],residual[r[lag:]]) for r in runs if len(r)>lag]
        if not pairs:
            ac[str(lag)] = dict(pairs=0, distribution=distribution([])); continue
        a,b = (np.concatenate([p[j] for p in pairs]) for j in (0,1))
        a,b = a-a.mean(0), b-b.mean(0)
        den = np.sqrt((a*a).sum(0)*(b*b).sum(0))
        corr = np.divide((a*b).sum(0),den,out=np.full_like(den,np.nan),where=den>0)
        ac[str(lag)] = dict(pairs=len(a), distribution=distribution(corr))
    blocks = [residual[r[:len(r)//4*4]].reshape(-1,4,x.shape[1]).sum(1) for r in runs if len(r)>=4]
    # Pool within-run sample variances so run boundaries never create synthetic blocks.
    bdf = sum(len(b)-1 for b in blocks if len(b)>1)
    bv = sum(((b-b.mean(0))**2).sum(0) for b in blocks if len(b)>1)/bdf if bdf else np.full(x.shape[1],np.nan)
    ratio = np.divide(bv,4*variance,out=np.full_like(variance,np.nan),where=variance>0)
    return dict(channels=x.shape[1], variance_df=df, lag=ac,
                four_frame=dict(blocks=sum(len(b) for b in blocks), variance_df=bdf, distribution=distribution(ratio))), variance


def temporal_diagnostics(d):
    runs, period = contiguous_runs(d)
    out = dict(frames=len(d['H']), run_lengths=[len(r) for r in runs], median_period_ms=period,
               method='Run-demeaned Pearson per channel; nonoverlap 4-frame sum variance / (4 * single-frame variance), sample variances pooled within runs. Quantiles over channels are descriptive, not confidence intervals or proof of independence.', groups={})
    for name, bins in [('crosstalk_bin0',[0]), ('target_bins1_4',list(range(1,5))), ('far_bins8_13',list(range(8,14)))]:
        x = d['H'][:,:,bins]
        channels, var = channel_diagnostics(x,runs)
        summed, sumvar = channel_diagnostics(x.sum(2),runs)
        denominator = var.reshape(64,-1).sum(1)
        covratio = np.divide(sumvar,denominator,out=np.full(64,np.nan),where=denominator>0)
        out['groups'][name] = dict(bins=bins, per_channel=channels, window_sum=summed,
                                  window_variance_over_sum_bin_variances=distribution(covratio))
    return out
