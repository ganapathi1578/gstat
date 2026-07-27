from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import math
import numpy as np
import cv2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_from_points(points: np.ndarray, pad: int = 10):
    xs = points[:, 0]
    ys = points[:, 1]
    x1 = int(xs.min()) - pad
    y1 = int(ys.min()) - pad
    x2 = int(xs.max()) + pad
    y2 = int(ys.max()) + pad
    return [x1, y1, x2, y2]


def distance(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0


def smooth_point(prev, new, alpha=0.65):
    if prev is None:
        return new
    return (
        alpha * prev[0] + (1 - alpha) * new[0],
        alpha * prev[1] + (1 - alpha) * new[1],
    )


def draw_label(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def within_bounds(pt, width, height, margin=0):
    x, y = pt
    return margin <= x < width - margin and margin <= y < height - margin


def linear_interpolate_points(p0, p1, n):
    if n <= 0:
        return []
    pts = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        x = p0[0] * (1 - t) + p1[0] * t
        y = p0[1] * (1 - t) + p1[1] * t
        pts.append((x, y))
    return pts


def extract_wrist_points_from_keypoints(kpts_xy):
    """
    YOLO pose keypoints:
    9  = left wrist
    10 = right wrist
    """
    if kpts_xy is None or len(kpts_xy) < 11:
        return None, None
    left_wrist = tuple(map(float, kpts_xy[9]))
    right_wrist = tuple(map(float, kpts_xy[10]))
    return left_wrist, right_wrist


def mean_point(points: Iterable[Tuple[float, float]]):
    pts = list(points)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))