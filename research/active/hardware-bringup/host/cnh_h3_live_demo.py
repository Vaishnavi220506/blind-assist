"""Frozen A2 engineering demo. Real-input replay is never described as live.

Background is a separate stationary acquisition, not a physical sensor calibration.
Camera overlay requires an explicit verified 64-zone polygon registration. Otherwise
the display is side by side. Camera matching is host receipt time, not exposure sync.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

NEAR = Path(__file__).resolve().parents[2] / 'dtr-r0' / 'nearfield'
sys.path.insert(0, str(NEAR))
from capture import validate_frame

MODEL_HASHES = (
    'b7ffb180ddde81a6c28e1e7c3922d9653e948a9023ba88fea9d1b8d7a9ee0209',
    '7bafc3201a395f60790dbdfc95b834e5cea43580acd15a61611c972656e33c8d',
    '9e8296de5a0af23ecb752299c4ea1f9f729237b560c3549c6bd472be34e13083',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth_logits(logits):
    scores = np.asarray(logits)[-5:]
    weights = .5**np.arange(len(scores)-1, -1, -1)
    return ((scores*weights[:, None]).sum(0)/weights.sum()).astype(scores.dtype)


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def causal_sample(records, fps=5.):
    """Latest received observation at each grid time; never interpolate/reuse a frame.

    Streaming emission waits for the next receipt to close a grid interval. This
    buffering is included in live receipt-to-render latency, not hidden as exposure
    synchronization. Empty grid intervals cause no fabricated sensor observation.
    """
    previous = None; tick = None; used = None; segment = 0
    period = round(1e9/fps)
    for row in records:
        stamp = row['host_received_monotonic_ns']
        if tick is None: tick = stamp
        if previous is not None and stamp < previous['host_received_monotonic_ns']:
            raise ValueError('decreasing host receipt timestamps')
        while tick < stamp:
            if previous is not None and previous['host_received_monotonic_ns'] != used:
                yield {**previous, 'sample_tick_ns':tick}
                used = previous['host_received_monotonic_ns']
            tick += period
        if previous is not None and (row['sensor']['seq'] != previous['sensor']['seq']+1 or
                not 0 < row['sensor']['ms']-previous['sensor']['ms'] <= 500):
            segment += 1
        previous = {**row, 'stream_segment':segment}
    if previous is not None and tick == previous['host_received_monotonic_ns']:
        yield {**previous, 'sample_tick_ns':tick}


def hist(row):
    d = validate_frame(row['sensor'])
    if (d.get('rows'), d.get('bins')) != (8, 16):
        raise ValueError('H3 8x8x16 required')
    h = np.asarray(d['hist_normalized'], float).reshape(8, 8, 16)
    if not np.isfinite(h).all():
        raise ValueError('nonfinite histogram')
    return h


def noise_model(background):
    """Estimate variance of sums directly; retain temporal covariance up to k=4."""
    records = list(causal_sample(rows(background)))
    if len(records) < 80:
        raise ValueError('At least 80 stationary background frames required')
    h = np.asarray([hist(r) for r in records])
    seq = np.array([r['sensor']['seq'] for r in records])
    ms = np.array([r['sensor']['ms'] for r in records])
    period = np.median(np.diff(ms)[np.diff(ms) > 0])
    ticks = np.array([r['sample_tick_ns'] for r in records])
    segments = np.array([r['stream_segment'] for r in records])
    good = (np.diff(segments) == 0) & (np.diff(seq) > 0) & (np.diff(ticks) == 200_000_000) & (np.diff(ms) > 0) & (np.diff(ms) < 500)
    mean = h.mean(0)
    variance, counts, floored = [], [], []
    for k in range(1, 5):
        sums = [h[i-k+1:i+1].sum(0) for i in range(k-1, len(h))
                if good[i-k+1:i].all()]
        if len(sums) < 30:
            raise ValueError('Insufficient continuous stationary background')
        v = np.var(sums, axis=0, ddof=1)
        floor = max(float(np.median(v[v > 0]))*1e-6, 1e-12)
        floored.append(int((v < floor).sum()))
        variance.append(np.maximum(v, floor)); counts.append(len(sums))
    return mean, np.asarray(variance), dict(frames=len(h), sum_windows=counts,
        variance_floor_cells=floored, median_period_ms=float(period),
        scope='Empirical stationary normalization, not physical noise/SNR calibration; overlapping windows are not independent')


class InputNotAvailable(ValueError):
    """Explicit simulation fields are absent/invalid; never substitute real estimates."""


class SimFloorFeatures:
    SCHEMA = 'cnh.sim-floor.v1'

    def __init__(self, path):
        if path is None or not Path(path).is_file():
            raise InputNotAvailable('sim-floor requires --sim-floor-fields NPZ')
        self.path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as d:
                if str(d['schema']) != self.SCHEMA or not str(d['source']).strip():
                    raise ValueError('schema/source missing')
                self.source = str(d['source'])
                self.bias = np.array(d['bias'])
                seq = np.array(d['seq']); ambient = np.array(d['ambient']); tq = np.array(d['T_Q_tof'])
            if seq.ndim != 1 or not len(seq) or not np.issubdtype(seq.dtype,np.integer) or len(np.unique(seq)) != len(seq):
                raise ValueError('unique integer seq[N] required')
            if self.bias.shape != (8,8,16) or ambient.shape != (len(seq),8,8) or tq.shape != (len(seq),4,4):
                raise ValueError('bias[8,8,16], ambient[N,8,8], T_Q_tof[N,4,4] required')
            if not all(np.isfinite(x).all() for x in (self.bias,ambient,tq)) or (ambient < 0).any():
                raise ValueError('finite fields and nonnegative ambient required')
            from cnh_track_a_readout import _poses
            _poses(tq,len(seq))
            self.by_seq = {int(s):(ambient[i],tq[i]) for i,s in enumerate(seq)}
        except (KeyError,ValueError,TypeError,OSError) as e:
            raise InputNotAvailable(f'invalid sim-floor fields: {e}') from e
        self.history = deque(maxlen=4)

    def require_sequences(self, seqs):
        missing = sorted(set(seqs)-self.by_seq.keys())
        if missing:
            raise InputNotAvailable(f'sim-floor ambient/query fields absent for {len(missing)} sequences; first={missing[0]}')

    def step(self, row, histogram, reset=False):
        self.require_sequences([row['sensor']['seq']])
        if reset: self.history.clear()
        ambient,tq = self.by_seq[row['sensor']['seq']]
        self.history.append((histogram,ambient,tq))
        from cnh_learned_features import sequence_features
        from cnh_learned_readout import squash
        h,a,t = (np.asarray([item[i] for item in self.history]) for i in range(3))
        poses = np.repeat(np.eye(4)[None],len(h),axis=0)
        z4,z1,sup = sequence_features(h,a,self.bias,t,poses)
        # Frozen unit_features stores float16; Readout.load restores float32 before squash.
        e4,e1 = (z[-1].astype(np.float16).astype(np.float32) for z in (z4,z1))
        x = np.stack((squash(e4),squash(e1)))[None]
        if not np.isfinite(x).all():
            raise InputNotAvailable('sim-floor feature float16 overflow; no clipping/fallback')
        return z4[-1],z1[-1],sup[-1],x


class Engine:
    def __init__(self, models, mean, variance, tq, sim_floor=None):
        import torch
        from cnh_learned_readout import Readout, squash
        from cnh_track_a_readout import query_weights
        self.torch, self.squash = torch, squash
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        if len(models) != 3:
            raise ValueError('Exactly three frozen models required')
        if tuple(sha(path) for path in models) != MODEL_HASHES:
            raise ValueError('Frozen seed0/1/2 checkpoint identity mismatch')
        for path in models:
            m = Readout().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.models.append(m.eval())
        self.support = torch.as_tensor((query_weights(tq) >= .75).reshape(1, 6, 8, 8, 16), device=self.device)
        self.mean, self.variance = mean, variance
        self.sim_floor = sim_floor
        self.history, self.logits = deque(maxlen=4), deque(maxlen=5)
        self.last = None

    def step(self, row):
        t0 = time.perf_counter()
        seq, ms = row['sensor']['seq'], row['sensor']['ms']
        tick = row['sample_tick_ns']
        reset = self.last is not None and (seq <= self.last[0] or ms <= self.last[1] or ms-self.last[1] > 500 or tick-self.last[2] != 200_000_000 or row['stream_segment'] != self.last[3])
        if reset:
            self.history.clear(); self.logits.clear()
        self.last = (seq, ms, tick, row['stream_segment'])
        if self.sim_floor is None:
            self.history.append(hist(row)-self.mean)
            z1 = self.history[-1]/np.sqrt(self.variance[0])
            z4 = np.sum(self.history, axis=0)/np.sqrt(self.variance[len(self.history)-1])
            x = np.stack((self.squash(z4), self.squash(z1)))[None].astype(np.float32)
        else:
            h = hist(row); self.history.append(h)
            z4,z1,sup,x = self.sim_floor.step(row,h,reset=reset)
            self.support = self.torch.as_tensor(sup[None],device=self.device)
        with self.torch.inference_mode():
            xt = self.torch.as_tensor(x, device=self.device)
            nn = self.torch.stack([m(xt, self.support) for m in self.models]).mean(0)[0].cpu().numpy()
        self.logits.append(nn)
        # Exactly cnh_learned_memory_fusion.causal_ewma(alpha=.5, window=5), including dtype.
        a2 = smooth_logits(self.logits)
        return dict(seq=seq, ms=ms, NN=nn.tolist(), A2=a2.tolist(), reset=reset,
                    history=len(self.history), compute_ms=(time.perf_counter()-t0)*1000,
                    **({'input_mode':'sim-floor'} if self.sim_floor is not None else {})), z4


def load_registration(path):
    if not path:
        return None
    r = json.loads(Path(path).read_text(encoding='utf-8'))
    p = np.asarray(r['zone_polygons_normalized'], float)
    if r.get('verified') is not True or not r.get('evidence') or p.shape != (64, 4, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Registration must have verified evidence and 64 normalized quadrilaterals')
    return r


def render(camera, z4, result, mode, registration, thresholds):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    im = np.zeros((720, 1120, 3), np.uint8)
    if camera is not None:
        im[110:590, :640] = cv2.resize(camera, (640, 480))
    zone_z = np.max(z4, axis=-1)
    heat_limit = max(10., float(np.max(zone_z)))
    heat = cv2.applyColorMap(np.uint8(np.clip(zone_z, 0, heat_limit)*255/heat_limit), cv2.COLORMAP_TURBO)
    im[110:590, 640:1120] = cv2.resize(heat, (480, 480), interpolation=cv2.INTER_NEAREST)
    if camera is not None and registration:
        overlay = im.copy()
        for zone, polygon in enumerate(registration['zone_polygons_normalized']):
            pts = np.int32(np.array(polygon)*[640, 480]+[0, 110])
            cv2.fillConvexPoly(overlay, pts, tuple(int(c) for c in heat.reshape(64, 3)[zone]))
        im = cv2.addWeighted(im, .65, overlay, .35, 0)
    pic = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(pic)
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else font
    for r in range(8):
        for c in range(8):
            draw.rectangle((640+c*60,110+r*60,699+c*60,169+r*60),outline='#bbbbbb')
            draw.text((643+c*60,130+r*60),f'{zone_z[r,c]:.0f}',font=small,fill='white',stroke_width=1,stroke_fill='black')
    lines = [f'{mode} | 模拟训练模型、真实传感器输入、无定量结论',
             ('恒等输运；显式仿真 bias/ambient/查询变换，非实测噪声标定' if result.get('input_mode') == 'sim-floor' else '静止传感器、恒等输运；名义安装角 -10°，非人体姿态标定'),
             '相机配准：'+('已提供验证文件（仅该固定装置）' if registration else '未完成；仅并排显示，不作像素定位'),
             f"seq {result['seq']} | 处理 {result['compute_ms']:.1f} ms | max-bin z4 色标 0–{heat_limit:.0f}（逐帧）；非距离/概率"]
    for y, text in zip((10, 40, 76, 600), lines):
        draw.text((12, y), text, font=font, fill='white')
    if camera is None:
        draw.text((30, 300), '相机帧不可用 / 无有效时间配对', font=font, fill='white')
    if thresholds and result['history'] < 4:
        draw.text((12, 635), '历史不足4帧：预热中，不报警', font=font, fill='#ffbf60')
    elif thresholds:
        alarms = [result['A2'][q] >= thresholds['A2_thresholds']['HEAD' if q % 2 == 0 else 'BODY'] for q in range(6)]
        status = ' / '.join(f'{q}:{"触发" if v else "未触发"}' for q, v in enumerate(alarms))
        draw.text((12, 635), 'A2 仿真阈值演示 '+status, font=font, fill='#ffbf60')
    else:
        draw.text((12, 635), 'A2 分数 '+', '.join(f'{v:.2f}' for v in result['A2'])+'；未加载阈值，不报警', font=font, fill='#ffbf60')
    draw.text((12, 675), '主机接收时间近邻配对，非曝光同步；A3 真实 S2 标定未完成', font=font, fill='white')
    return cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--replay', type=Path, help='segment directory containing tof and camera')
    source.add_argument('--port', help='explicit XIAO serial port; no automatic scan or flashing')
    p.add_argument('--background', type=Path, help='empirical mode: separate static tof/frames.jsonl, >=80 frames')
    p.add_argument('--input-mode', choices=('empirical','sim-floor'), default='empirical')
    p.add_argument('--sim-floor-fields', type=Path, help='explicit cnh.sim-floor.v1 NPZ; never estimated from real background')
    p.add_argument('--models', type=Path, nargs=3, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--video-fps', type=float, default=10)
    p.add_argument('--thresholds', type=Path)
    p.add_argument('--registration', type=Path)
    p.add_argument('--camera', help='explicit live camera URL or index; never auto-opens a camera')
    p.add_argument('--baud', type=int, default=115200)
    a = p.parse_args()
    if a.seconds <= 0 or a.video_fps <= 0:
        p.error('positive duration and video fps required')
    import cv2
    a.out.mkdir(parents=True, exist_ok=True)
    sim_floor = None
    if a.input_mode == 'sim-floor':
        try:
            sim_floor = SimFloorFeatures(a.sim_floor_fields)
            if a.replay:
                sim_floor.require_sequences(r['sensor']['seq'] for r in rows(a.replay/'tof/frames.jsonl'))
        except InputNotAvailable as e:
            status = dict(status='NOT_AVAILABLE',input_mode='sim-floor',reason=str(e),fallback=False)
            (a.out/'not_available.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            print(json.dumps(status)); return 2
        mean = variance = None
        noise_receipt = dict(median_period_ms=200.,scope='Explicit simulation fields; not derived from real background')
    else:
        if a.background is None: p.error('--background required for empirical input mode')
        mean, variance, noise_receipt = noise_model(a.background)
        np.savez_compressed(a.out/'background_model.npz', mean=mean, variance=variance)
    tq = np.eye(4); angle = np.deg2rad(-10)
    tq[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    engine = Engine(a.models, mean, variance, tq, sim_floor=sim_floor)
    registration = load_registration(a.registration)
    thresholds = json.loads(a.thresholds.read_text()) if a.thresholds else None
    if thresholds and (not thresholds.get('source') or not thresholds.get('scope') or
                       not all(np.isfinite(thresholds['A2_thresholds'][g]) for g in ('HEAD','BODY'))):
        raise ValueError('Thresholds need finite HEAD/BODY, source and scope')
    mode = '历史真实传感器回放 / REPLAY' if a.replay else 'LIVE 工程演示'
    camera = port = writer = raw = receipt = None
    results, pair_offsets, pending = [], [], []
    raw_stamps = []
    started = time.perf_counter(); next_video = 0.; source_start = None; previous_image = None
    camera_rows = list(rows(a.replay/'camera/frames.jsonl')) if a.replay else []
    camera_stamps = np.array([r['host_received_monotonic_ns'] for r in camera_rows])
    def source_rows():
        if a.replay:
            for row in rows(a.replay/'tof/frames.jsonl'):
                raw_stamps.append(row['host_received_monotonic_ns'])
                yield row
        else:
            while time.perf_counter()-started < a.seconds:
                line = port.readline()
                raw.write(line)
                if not line:
                    continue
                try:
                    sensor = json.loads(line)
                    if sensor.get('type') == 'cnh_frame':
                        stamp = time.monotonic_ns(); raw_stamps.append(stamp)
                        yield dict(sensor=sensor, host_received_monotonic_ns=stamp)
                except (ValueError, UnicodeError):
                    pending.append('invalid serial line')
    try:
        if a.port:
            import serial
            port = serial.Serial(a.port, a.baud, timeout=.5)
            raw = (a.out/'serial.bin').open('wb')
            receipt = (a.out/'acquired.jsonl').open('w',encoding='utf-8')
        if a.camera:
            camera = cv2.VideoCapture(int(a.camera) if a.camera.isdigit() else a.camera)
            if not camera.isOpened():
                raise RuntimeError('Explicit camera source unavailable')
        writer = cv2.VideoWriter(str(a.out/'demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), a.video_fps, (1120, 720))
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        for row in causal_sample(source_rows()):
            if receipt:
                receipt.write(json.dumps(row)+'\n'); receipt.flush()
            stamp = row['host_received_monotonic_ns']
            source_start = row['sample_tick_ns'] if source_start is None else source_start
            elapsed = (row['sample_tick_ns']-source_start)/1e9
            if elapsed >= a.seconds:
                break
            frame_start = time.perf_counter()
            try:
                result, z4 = engine.step(row)
            except InputNotAvailable as e:
                (a.out/'not_available.json').write_text(json.dumps(dict(status='NOT_AVAILABLE',input_mode='sim-floor',reason=str(e),fallback=False),indent=2),encoding='utf-8')
                raise
            except ValueError as e:
                pending.append(str(e)); continue
            result['sample_tick_ns'] = row['sample_tick_ns']
            result['sensor_host_received_monotonic_ns'] = stamp
            result['z4_min_max'] = [float(z4.min()),float(z4.max())]
            result['alarms'] = [bool(result['history'] >= 4 and result['A2'][q] >= thresholds['A2_thresholds']['HEAD' if q%2==0 else 'BODY']) for q in range(6)] if thresholds else None
            cam = None
            if camera_rows:
                j = int(np.argmin(abs(camera_stamps-stamp)))
                delta = float((camera_stamps[j]-stamp)/1e6)
                if abs(delta) <= 150:
                    camera_path = a.replay/'camera'/camera_rows[j]['filename']
                    if sha(camera_path) != camera_rows[j]['sha256']:
                        raise ValueError('Recorded camera file hash mismatch')
                    cam = cv2.imread(str(camera_path))
                    if cam is None:
                        raise ValueError('Recorded camera image cannot be decoded')
                    pair_offsets.append(delta)
                    result['camera_file'] = camera_rows[j]['filename']
                    result['camera_receipt_delta_ms'] = delta
            elif camera:
                ok, cam = camera.read()
                if not ok:
                    cam = None
            im = render(cam, z4, result, mode, registration, thresholds)
            if a.port:
                cv2.imshow('CNH H3 engineering demo - ESC to stop', im)
                if cv2.waitKey(1) == 27:
                    break
            # Timestamp-driven CFR; duplicate images do not count as new sensor frames.
            while next_video < elapsed and previous_image is not None:
                writer.write(previous_image); next_video += 1/a.video_fps
            previous_image = im
            result['source_elapsed_s'] = elapsed
            result['receipt_age_at_sample_tick_ms'] = (row['sample_tick_ns']-stamp)/1e6
            result['processing_render_encode_ms'] = (time.perf_counter()-frame_start)*1000
            if a.port:
                result['host_receipt_to_render_ms'] = (time.monotonic_ns()-stamp)/1e6
            results.append(result)
            if len(results) == 1:
                cv2.imwrite(str(a.out/'preview.jpg'), im)
        # Hold the last received frame only for one measured native period.
        end = min(a.seconds, (results[-1]['source_elapsed_s'] if results else 0)+noise_receipt['median_period_ms']/1000)
        while next_video < end and previous_image is not None:
            writer.write(previous_image); next_video += 1/a.video_fps
    finally:
        if writer: writer.release()
        if camera: camera.release()
        if port: port.close()
        if raw: raw.close()
        if receipt: receipt.close()
        if a.port: cv2.destroyAllWindows()
    wall = time.perf_counter()-started
    duration = results[-1]['source_elapsed_s'] if results else 0
    report = dict(mode='recorded-real-input-replay' if a.replay else 'live', frames=len(results),
        source_duration_s=duration, video_duration_s=next_video, requested_duration_s=a.seconds,
        complete_30s=next_video >= 30, wall_seconds=wall, processing_fps=len(results)/wall,
        used_fps=(len(results)-1)/duration if duration else None, sampling_grid_fps=5.,
        acquired_raw_frames=len(raw_stamps), resets=sum(r['reset'] for r in results),
        acquired_raw_fps=(len(raw_stamps)-1)*1e9/(raw_stamps[-1]-raw_stamps[0]) if len(raw_stamps)>1 else None,
        processing_render_encode_ms={str(q):float(np.percentile([r['processing_render_encode_ms'] for r in results], q)) for q in (50,95)} if results else {},
        paired_camera_frames=len(pair_offsets), pairing_abs_ms_p95=float(np.percentile(np.abs(pair_offsets),95)) if pair_offsets else None,
        models={str(m):sha(m) for m in a.models}, background=dict(path=str(a.background),sha256=sha(a.background),**noise_receipt) if sim_floor is None else None,
        source_sha256=sha(__file__), backend=str(engine.device),torch=engine.torch.__version__,
        thresholds=dict(path=str(a.thresholds),sha256=sha(a.thresholds),content=thresholds) if a.thresholds else None,
        registration=dict(path=str(a.registration),sha256=sha(a.registration)) if a.registration else None,
        replay_source=dict(path=str(a.replay/'tof/frames.jsonl'),sha256=sha(a.replay/'tof/frames.jsonl')) if a.replay else None,
        A3='NOT_AVAILABLE: simulation guard cannot be treated as real-sensor calibrated S2', errors=pending,
        scope='Engineering only; real noise and nominal query geometry differ from simulator; causal latest-observation sampling on 5 Hz host receipt grid, no interpolation or duplicate frame. No exposure-to-alert latency or quantitative accuracy claim.')
    if sim_floor is not None:
        report['input_mode'] = 'sim-floor'
        report['sim_floor_fields'] = dict(path=str(sim_floor.path),sha256=sha(sim_floor.path),source=sim_floor.source,
            schema=sim_floor.SCHEMA,feature_encoding='frozen float32 features -> float16 -> float32 -> sign*log1p',
            scope='Explicit supplied simulation fields; no real calibration or domain-transfer claim')
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.out/'inference.jsonl').open('w',encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sys.exit(main())
