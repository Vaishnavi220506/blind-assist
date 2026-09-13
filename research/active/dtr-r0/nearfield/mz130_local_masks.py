"""Exact component support from the frozen MZ125 foreground, without labels.

Each matched component is dilated independently by the inherited two-pixel box
pad and clipped to that padded proposal. Half-open pixel run rectangles encode
the resulting binary mask exactly; missing/ambiguous components mean fallback.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import mz125_observable_correction as correction


ROOT = Path(__file__).resolve().parents[4]
PAD = 2
KERNEL = np.ones((2*PAD+1, 2*PAD+1), np.uint8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_tiles(binary, offset=(0, 0)):
    """Partition true pixels into disjoint rectangles with exact run merging."""
    assert binary.ndim == 2
    active, rectangles = {}, []
    ox, oy = offset
    for y, row in enumerate(binary):
        edges = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        assert len(edges) % 2 == 0
        runs = {(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])}
        for key in list(active):
            if key not in runs:
                rectangles.append(active.pop(key))
        for a, b in sorted(runs):
            key = (a, b)
            if key in active:
                active[key][3] = oy+y+1
            else:
                active[key] = [ox+a, oy+y, ox+b, oy+y+1]
    rectangles.extend(active.values())
    rectangles.sort(key=lambda r: (r[1], r[0], r[3], r[2]))
    # Exact binary equality is the contract; a bbox, hull or lost hole would fail.
    rebuilt = np.zeros(binary.shape, dtype=np.uint8)
    for x0, y0, x1, y1 in rectangles:
        region = rebuilt[y0-oy:y1-oy, x0-ox:x1-ox]
        assert not region.any()
        region[:] = 1
    assert np.array_equal(rebuilt.astype(bool), binary.astype(bool))
    return rectangles


def fallback(index, box, reason):
    return dict(proposal=index, box=list(box), status='FALLBACK',
                tiles=[], pixel_count=0, fallback_reason=reason)


def get_components(image, cached):
    """Return one entry per cached proposal; FALLBACK entries never exclude.

    Boxes must reproduce cached recovered_boxes exactly. Coordinates are image
    pixels [left, top, right, bottom), matching MZ125 connected-component boxes.
    """
    proposals = cached['proposals']
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return [fallback(j, b, 'IMAGE_UNAVAILABLE') for j, b in enumerate(proposals)]
    boxes, mask = correction.foreground(image)
    assert boxes == cached['recovered_boxes'], 'MZ125 recovered-box replay mismatch'
    assert bool(cached['fallback']) == (not bool(boxes))
    if cached['fallback']:
        return [fallback(j, b, 'MZ125_ORIGINAL_BOX_FALLBACK') for j, b in enumerate(proposals)]
    assert proposals == boxes, 'Corrected proposal namespace mismatch'
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    by_box = {}
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= 12 and w < .8*width:
            by_box.setdefault((x, y, x+w, y+h), []).append(label)
    result = []
    for j, box in enumerate(proposals):
        matches = by_box.get(tuple(box), [])
        if len(matches) != 1:
            result.append(fallback(j, box, 'COMPONENT_MATCH_MISSING_OR_AMBIGUOUS'))
            continue
        label = matches[0]
        x0, y0 = max(0, box[0]-PAD), max(0, box[1]-PAD)
        x1, y1 = min(width, box[2]+PAD), min(height, box[3]+PAD)
        local = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        if local.size == 0 or not local.any():
            result.append(fallback(j, box, 'EMPTY_MATCHED_COMPONENT'))
            continue
        # Independent masks: adjacent components never seed each other's dilation.
        padded = cv2.dilate(local, KERNEL, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        tiles = run_tiles(padded, (x0, y0))
        pixels = int(padded.sum())
        assert pixels == sum((r[2]-r[0])*(r[3]-r[1]) for r in tiles)
        assert all(x0 <= r[0] < r[2] <= x1 and y0 <= r[1] < r[3] <= y1 for r in tiles)
        result.append(dict(proposal=j, box=list(box), status='MASK_COMPONENT',
            clip_box=[x0, y0, x1, y1], tiles=tiles, pixel_count=pixels,
            component_pixel_count=int(local.sum()), dilation_pad_px=PAD,
            fallback_reason=None))
    assert len(result) == len(proposals)
    return result


def run(raw_path, correction_path, image_root, output):
    raw_path, correction_path, image_root, output = [p.resolve() for p in
                                                   (raw_path, correction_path, image_root, output)]
    assert not output.exists()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve())
    seal_path = correction_path.parent/'prediction-seal.json'
    old_seal = read(seal_path)
    assert sha(raw_path) == old_seal['raw_sha256']
    assert sha(correction_path) == old_seal['predictions_sha256']
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    cached = read(correction_path)
    assert len(rows) == len(cached) == 288
    assert [r['id'] for r in rows] == list(old_seal['rgb_sha256'])
    input_paths = [raw_path, correction_path, seal_path]
    input_hashes = {str(p):sha(p) for p in input_paths}
    records = []
    started = time.perf_counter()
    for index, (row, cache) in enumerate(zip(rows, cached)):
        image_path = image_root/row['rgb_path']
        digest = sha(image_path)
        assert digest == old_seal['rgb_sha256'][row['id']]
        input_hashes[str(image_path)] = digest
        image = cv2.imread(str(image_path))
        assert image is not None, row['id']
        components = get_components(image, cache)
        records.append(dict(id=row['id'], rgb_sha256=digest,
            image_size=[int(image.shape[1]), int(image.shape[0])],
            recovered_boxes_reproduced=True, components=components))
        if (index+1) % 48 == 0:
            print(f'MZ130 masks: {index+1}/{len(rows)} observable frames', flush=True)
    elapsed = time.perf_counter()-started
    all_components = [c for r in records for c in r['components']]
    summary = dict(frames=len(records), boxes_reproduced_frames=len(records),
        components=len(all_components), statuses=dict(Counter(c['status'] for c in all_components)),
        fallback_reasons=dict(Counter(c['fallback_reason'] for c in all_components if c['status']=='FALLBACK')),
        empty_proposal_frames=sum(not r['components'] for r in records),
        total_tiles=sum(len(c['tiles']) for c in all_components),
        seconds=elapsed, python=sys.executable, numpy=np.__version__, opencv=cv2.__version__,
        backend='CPU: GPU_BACKEND_UNAVAILABLE for unchanged MZ125 foreground/connected components',
        authority='OBSERVABLE_COMPONENT_MASKS_NO_LABELS_OR_GEOMETRY_EVALUATION',
        geometry='HALF_OPEN_PIXEL_RECTS_EXACT_DILATED_COMPONENT_CLIPPED_TO_PROPOSAL_PLUS_2PX',
        fallback='No exclusion from FALLBACK or empty component; preserve inherited proposal support')
    output.mkdir(parents=True)
    write(output/'components.json', records)
    write(output/'summary.json', summary)
    sources = {str(Path(module.__file__).resolve()):sha(module.__file__)
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and 'nearfield' in Path(module.__file__).parts}
    own = Path(__file__).resolve()
    sources[str(own)] = sha(own)
    write(output/'prediction-seal.json', dict(raw_sha256=sha(raw_path),
        correction_sha256=sha(correction_path), correction_seal_sha256=sha(seal_path),
        components_sha256=sha(output/'components.json'), summary_sha256=sha(output/'summary.json'),
        code_sha256=sha(own), source_hashes=sources, input_hashes=input_hashes,
        rgb_sha256=old_seal['rgb_sha256'], authority='SAVED_BEFORE_THIS_RUN_LABEL_PARSE',
        limits='Old mask PNGs were not used as authority; recomputed boxes and authenticated RGB were checked. Controlled foreground is not a complete physical surface segmentation.'))
    assert input_hashes == {path:sha(path) for path in input_hashes}
    write(output/'completion.json', dict(status='PASS', inputs_unchanged=True,
        seal_sha256=sha(output/'prediction-seal.json'), resources='No persistent process or allocation'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('raw', 'correction', 'image-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.correction, args.image_root, args.output)
