"""Offline adapter for the fixed Android YOLO11n FP16 TFLite asset.

Class-score decoding, coordinate normalization, native clamping and class-aware
NMS follow YoloOutputDecoder.kt. Bitmap-path letterbox geometry is reproduced;
NumPy bilinear rasterization is an explicitly approximate substitute for Android
Canvas/Skia FILTER_BITMAP, not claimed bit-identical. No installation or download.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT/'app/src/main/assets'
INPUT_SIZE = 320
CONFIDENCE = np.float32(.35)
IOU_THRESHOLD = np.float32(.45)
RUNTIMES = ('ai_edge_litert', 'tflite_runtime', 'tensorflow')
DIFFERENCES = [
    'Bitmap-path geometry: float32 min scale, truncate resized dimensions, floating half padding, RGB/255, black background.',
    'NumPy bilinear pixel-centre sampling with nearest integer uint8 rounding approximates Android Canvas FILTER_BITMAP; Skia interpolation/edge rasterization is not bit-exactly verified.',
    'Replicates scale1 full-width padded-bitmap fast path with integer top offset; does not emulate camera RGBA nearest-neighbour preprocessing or display rotation.',
    'RGB file decoder is Pillow, with no EXIF transpose or ICC conversion; Android Bitmap decoding differences remain possible.',
    'CPU Python TFLite runtime/default delegate may differ numerically from Android LiteRT; no Android speed/parity claim.',
    'Nonfinite model output is an explicit error; no unknown or failed frame is silently reported empty.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime_probe():
    return dict(python=sys.executable, numpy=np.__version__,
                available={name: importlib.util.find_spec(name) is not None for name in RUNTIMES},
                image_decoder=importlib.util.find_spec('PIL') is not None)


def calculate_letterbox(width, height):
    if width <= 0 or height <= 0:
        raise ValueError('positive native dimensions required')
    scale = np.minimum(np.float32(INPUT_SIZE)/np.float32(width), np.float32(INPUT_SIZE)/np.float32(height))
    rw = max(1, int(np.float32(width)*scale)); rh = max(1, int(np.float32(height)*scale))
    return dict(scale=float(scale), dx=float(np.float32(INPUT_SIZE-rw)/2), dy=float(np.float32(INPUT_SIZE-rh)/2),
                source_width=int(width), source_height=int(height), input_size=INPUT_SIZE,
                resized_width=rw, resized_height=rh)


def preprocess_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB uint8 [height,width,3] required')
    h, w = rgb.shape[:2]
    info = calculate_letterbox(w, h)
    rw, rh, dx, dy = (info[k] for k in ('resized_width', 'resized_height', 'dx', 'dy'))
    result = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    if info['scale'] == 1. and rw == INPUT_SIZE and rh < INPUT_SIZE:
        result[int(dy):int(dy)+rh] = rgb
        raster = 'Android padded-bitmap fast-path geometry'
    else:
        yy, xx = np.indices((INPUT_SIZE, INPUT_SIZE), dtype=np.float64)
        inside = (xx+.5 >= dx) & (xx+.5 < dx+rw) & (yy+.5 >= dy) & (yy+.5 < dy+rh)
        sx = np.clip((xx+.5-dx)*w/rw-.5, 0, w-1)
        sy = np.clip((yy+.5-dy)*h/rh-.5, 0, h-1)
        x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
        x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
        wx = (sx-x0)[..., None]; wy = (sy-y0)[..., None]
        values = ((1-wy)*((1-wx)*rgb[y0, x0]+wx*rgb[y0, x1])+
                  wy*((1-wx)*rgb[y1, x0]+wx*rgb[y1, x1]))
        result[inside] = np.floor(values[inside]+.5).astype(np.uint8)
        raster = 'NumPy bilinear approximation to Canvas filtered bitmap draw'
    info['rasterization'] = raster
    return result[None].astype(np.float32)*np.float32(1./255.), info


def _iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    wh = np.maximum(np.float32(0), np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = wh[0]*wh[1]
    awh = np.maximum(np.float32(0), a[2:]-a[:2]); bwh = np.maximum(np.float32(0), b[2:]-b[:2])
    union = awh[0]*awh[1]+bwh[0]*bwh[1]-intersection
    return np.float32(0) if union <= 0 else intersection/union


def decode_output(raw, letterbox, labels):
    raw = np.asarray(raw)
    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError('raw YOLO FLOAT32 [1,channels,predictions] or transpose required')
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite model output')
    d1, d2 = raw.shape[1:]
    channels_first = d1 <= d2 and d1 >= 5
    matrix = raw[0].T if channels_first else raw[0]
    nclass = min(len(labels), matrix.shape[1]-4)
    if nclass <= 0:
        raise ValueError('no class channels')
    scale = np.float32(letterbox['scale'])
    offset = np.array([letterbox['dx'], letterbox['dy']]*2, np.float32)
    maximum = np.array([letterbox['source_width'], letterbox['source_height']]*2, np.float32)
    detections = []
    for index, row in enumerate(matrix):
        klass = int(np.argmax(row[4:4+nclass])); score = row[4+klass]
        # Strict best-score >0 and score>=confidence as in Kotlin class loop.
        if score <= 0 or score < CONFIDENCE:
            continue
        xywh = np.where(row[:4] <= np.float32(1.5), row[:4]*np.float32(INPUT_SIZE), row[:4])
        cx, cy, width, height = xywh
        box = np.array([cx-width/np.float32(2), cy-height/np.float32(2),
                        cx+width/np.float32(2), cy+height/np.float32(2)], np.float32)
        box = np.minimum(np.maximum((box-offset)/scale, np.float32(0)), maximum)
        if box[2]-box[0] <= 1 or box[3]-box[1] <= 1:
            continue
        detections.append(dict(class_id=klass, label=labels[klass], score=float(score),
                               xyxy=[float(x) for x in box], prediction_index=index))
    ordered = sorted(detections, key=lambda d: -d['score'])  # stable ties preserve original order
    keep = []
    for candidate in ordered:
        if any(old['class_id'] == candidate['class_id'] and _iou(old['xyxy'], candidate['xyxy']) > IOU_THRESHOLD for old in keep):
            continue
        keep.append(candidate)
    return keep


class SemanticDetector:
    def __init__(self, model=ASSETS/'yolo11n_fp16_320.tflite', labels=ASSETS/'coco_labels.txt', runtime='auto'):
        self.model_path, self.labels_path = Path(model), Path(labels)
        self.labels = [s.strip() for s in self.labels_path.read_text(encoding='utf-8').splitlines() if s.strip()]
        if runtime == 'auto':
            runtime = next((name for name in RUNTIMES if importlib.util.find_spec(name) is not None), None)
        if runtime not in RUNTIMES:
            raise RuntimeError('No installed TensorFlow/LiteRT/tflite-runtime interpreter; use an already provisioned environment. This adapter does not install packages.')
        if runtime == 'tensorflow':
            module = importlib.import_module(runtime)
            interpreter_class = module.lite.Interpreter
        else:
            module = importlib.import_module(runtime+'.interpreter')
            interpreter_class = module.Interpreter
        self.interpreter = interpreter_class(model_path=str(self.model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        ins, outs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError('fixed detector expects one input and one raw output')
        self.input, self.output = ins[0], outs[0]
        assert list(self.input['shape']) == [1, 320, 320, 3] and self.input['dtype'] == np.float32
        shape = list(map(int, self.output['shape']))
        assert len(shape) == 3 and shape[0] == 1 and self.output['dtype'] == np.float32
        assert (shape[1] >= 5 and shape[2] > shape[1]) or (shape[2] >= 5 and shape[1] > shape[2])
        distribution = {'ai_edge_litert': 'ai-edge-litert', 'tflite_runtime': 'tflite-runtime', 'tensorflow': 'tensorflow'}[runtime]
        try:
            runtime_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            runtime_version = getattr(module, '__version__', 'UNKNOWN')
        self.provenance = dict(model_sha256=sha(self.model_path), labels_sha256=sha(self.labels_path),
            adapter_sha256=sha(__file__), runtime=runtime, runtime_version=runtime_version,
            python=sys.executable, numpy=np.__version__, threads=4, device='CPU/default TFLite delegates',
            placement_reason='GPU_BACKEND_UNAVAILABLE: installed Windows LiteRT Python interpreter exposes the CPU/XNNPACK backend; no compatible GPU delegate configured; fixed asset not converted',
            input_shape=[1, 320, 320, 3], output_shape=shape, confidence=float(CONFIDENCE), iou=float(IOU_THRESHOLD),
            nms='stable descending score; same-class suppression iff IoU>0.45; no max detections',
            decoding='class max only, no objectness/sigmoid; each xywh coordinate <=1.5 individually scaled by320; source clamp; reject width/height<=1',
            differences=DIFFERENCES)

    def detect_rgb_file(self, path):
        from PIL import Image
        path = Path(path)
        with Image.open(path) as image:
            if 'A' in image.getbands():
                rgba = image.convert('RGBA')
                background = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
                rgb = np.asarray(Image.alpha_composite(background, rgba).convert('RGB'))
                alpha = 'composited on black'
            else:
                rgb = np.asarray(image.convert('RGB'))
                alpha = 'absent'
        started = time.perf_counter()
        tensor, letterbox = preprocess_rgb(rgb)
        prepared = time.perf_counter()
        self.interpreter.set_tensor(self.input['index'], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output['index'])
        inferred = time.perf_counter()
        detections = decode_output(raw, letterbox, self.labels)
        return dict(rgb_sha256=sha(path), native_shape=list(rgb.shape), letterbox=letterbox,
                    alpha_handling=alpha, detections=detections, provenance=self.provenance,
                    seconds=dict(preprocess=prepared-started, inference=inferred-prepared,
                                 postprocess=time.perf_counter()-inferred))

    def close(self):
        self.interpreter = None
        gc.collect()


def write_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'preserve existing/orphan result: {path}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_observations(observations, output, model, labels, root=ROOT, runtime='auto'):
    observations, output, root = Path(observations), Path(output), Path(root)
    rows = json.loads(observations.read_text(encoding='utf-8'))
    assert isinstance(rows, list) and rows
    ids = [row['id'] for row in rows]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', x) for x in ids)
    output.mkdir(parents=True, exist_ok=True)
    detector = SemanticDetector(model, labels, runtime)
    try:
        results = {}
        observation_sha = sha(observations)
        for row in rows:
            rgb = Path(row['rgb_path']); rgb = rgb if rgb.is_absolute() else root/rgb
            assert sha(rgb) == row['rgb_sha256'], f'RGB input changed: {row["id"]}'
            destination = output/f'{row["id"]}.json'
            if destination.with_name(destination.name+'.partial').exists():
                raise RuntimeError(f'orphan partial requires inspection: {destination}')
            if destination.exists():
                result = json.loads(destination.read_text(encoding='utf-8'))
                assert result['provenance'] == detector.provenance
                assert result['rgb_sha256'] == row['rgb_sha256'] and result['observations_sha256'] == observation_sha
            else:
                result = detector.detect_rgb_file(rgb)
                result.update(id=row['id'], observations_sha256=observation_sha)
                write_json(destination, result)
            results[row['id']] = dict(receipt=destination.name, receipt_sha256=sha(destination), detections=len(result['detections']))
            print(f'{row["id"]}: {len(result["detections"])} detections', flush=True)
        summary = dict(status='COMPLETE', frames=len(rows), observations_sha256=observation_sha,
                       provenance=detector.provenance, outputs=results)
        path = output/'inference-result.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8')) == summary
        else:
            write_json(path, summary)
        return summary
    finally:
        detector.close()


def fixtures():
    # Mirrors the Android decoder unit test, adds cross-class NMS and transpose.
    raw = np.zeros((1, 6, 8), np.float32)
    raw[0, :, 0] = [160, 160, 100, 100, .92, .1]
    raw[0, :, 1] = [164, 164, 100, 100, .8, .05]
    raw[0, :, 2] = [250, 120, 40, 40, .1, .7]
    raw[0, :, 3] = [40, 40, 20, 20, .2, .1]
    raw[0, :, 4] = [160, 160, 100, 100, .1, .6]
    info = calculate_letterbox(640, 480)
    result = decode_output(raw, info, ['person', 'chair'])
    assert [d['prediction_index'] for d in result] == [0, 2, 4]
    np.testing.assert_array_equal(result[0]['xyxy'], [220, 140, 420, 340])
    assert decode_output(raw.transpose(0, 2, 1).copy(), info, ['person', 'chair']) == result
    normalized = raw.copy(); normalized[0, :4] /= 320
    result_norm = decode_output(normalized, info, ['person', 'chair'])
    np.testing.assert_allclose(result_norm[0]['xyxy'], result[0]['xyxy'], atol=1e-4)
    image = np.full((480, 640, 3), [255, 128, 0], np.uint8)
    tensor, info2 = preprocess_rgb(image)
    assert info2['dx'] == 0 and info2['dy'] == 40 and info2['scale'] == .5
    assert tensor.dtype == np.float32 and tensor.shape == (1, 320, 320, 3)
    assert np.all(tensor[:, :40] == 0) and np.all(tensor[:, 280:] == 0)
    np.testing.assert_allclose(tensor[0, 40, 0], [1., 128/255, 0.], atol=1e-7)
    padded, pad_info = preprocess_rgb(np.full((239, 320, 3), 255, np.uint8))
    assert pad_info['dy'] == 40.5 and np.all(padded[0, 40:279] == 1) and np.all(padded[0, 279:] == 0)
    return dict(android_decoder_reference=True, transpose=True, normalized_xywh=True,
                class_aware_nms=True, black_letterbox_rgb=True, fractional_padding_fastpath=True,
                tensor_model_invoked=False, scientific_images_accessed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['probe', 'fixtures', 'infer'], default='infer')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path, default=ASSETS/'yolo11n_fp16_320.tflite')
    parser.add_argument('--labels', type=Path, default=ASSETS/'coco_labels.txt')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--runtime', choices=['auto', *RUNTIMES], default='auto')
    args = parser.parse_args()
    if args.stage == 'probe':
        print(json.dumps(runtime_probe()))
    elif args.stage == 'fixtures':
        print(json.dumps(fixtures()))
    else:
        if args.observations is None or args.output is None:
            parser.error('--observations and --output required for inference')
        run_observations(args.observations, args.output, args.model, args.labels, args.root, args.runtime)
