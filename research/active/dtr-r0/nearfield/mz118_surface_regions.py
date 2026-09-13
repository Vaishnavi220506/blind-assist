"""Auxiliary RGB surface regions: restore size-filtered Otsu/MSER candidates.

Legacy regions retain their identity/order. Additional broad or clipped regions
may contain them and remain explicit alternatives. This function is for the
surface branch only; it does not replace the frozen Radar association frontend.
No line-derived boxes, source geometry, native truth or sensor inputs are used.
"""
import cv2
import numpy as np

import mz108_competitive_association as legacy


def _append_unique(legacy_boxes, additions):
    """One-pixel edge-equivalent additions coalesce; containment is not equality."""
    retained = [list(box) for box in legacy_boxes]
    for box in sorted(additions, key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]), b)):
        if not any(max(abs(a-b) for a, b in zip(box, prior)) <= 1 for prior in retained):
            retained.append(list(box))
    return retained


def proposals(image):
    if image is None or image.size == 0: return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.min(gray) == np.max(gray): return []
    h, w = gray.shape
    retained = legacy.proposals(image)
    _, mask = cv2.threshold(cv2.medianBlur(gray, 3), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    # The original MSER detector settings, including its 50%-image-area cap.
    _, regions = cv2.MSER_create(5, 12, int(h*w*.5), .25, .2).detectRegions(gray)
    additions = [[int(x), int(y), int(x+bw), int(y+bh)]
                 for x, y, bw, bh, area in stats[1:n]
                 if area >= 12 and (bw >= .8*w or bh >= .95*h)]
    additions += [[int(x), int(y), int(x+bw), int(y+bh)]
                  for x, y, bw, bh in regions if bw >= .8*w or bh >= .95*h]
    return _append_unique(retained, additions)
