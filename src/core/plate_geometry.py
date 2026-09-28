"""Pure bounding-box geometry shared by image and video plate stages."""

from __future__ import annotations

from math import isfinite

import numpy as np

BBox = tuple[int, int, int, int]


def valid_bbox(bbox: BBox) -> bool:
    if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def intersection_area(box_a: BBox, box_b: BBox) -> float:
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return intersection_width * intersection_height


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    if not valid_bbox(box_a) or not valid_bbox(box_b):
        return 0.0
    intersection = intersection_area(box_a, box_b)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_plate_area(plate_bbox: BBox, vehicle_bbox: BBox) -> float:
    if not valid_bbox(plate_bbox) or not valid_bbox(vehicle_bbox):
        return 0.0
    plate_area = (plate_bbox[2] - plate_bbox[0]) * (
        plate_bbox[3] - plate_bbox[1]
    )
    if plate_area <= 0:
        return 0.0
    return intersection_area(plate_bbox, vehicle_bbox) / plate_area


def overlap_over_smaller(box_a: BBox, box_b: BBox) -> float:
    smaller = min(
        max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1]),
        max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1]),
    )
    return intersection_area(box_a, box_b) / smaller if smaller else 0.0


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    return tuple(
        max(0, min(limit, int(value)))
        for value, limit in zip(bbox, (width, height, width, height))
    )  # type: ignore[return-value]


def bbox_inside(inner: BBox, outer: BBox) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def crop_vehicle_roi(frame: np.ndarray, bbox: BBox) -> tuple[np.ndarray, BBox] | None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be an HxWxC NumPy array")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    clipped = clip_bbox(bbox, width, height)
    if not valid_bbox(clipped):
        return None
    x1, y1, x2, y2 = clipped
    roi = frame[y1:y2, x1:x2]
    return (roi, clipped) if roi.size else None


def local_bbox_to_global(
    local_bbox: BBox,
    parent_bbox: BBox,
    frame_width: int,
    frame_height: int,
    *,
    clip_to_vehicle: bool = False,
) -> BBox:
    """Map ROI coordinates, preserving each pipeline's existing clipping policy."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    if clip_to_vehicle:
        px1, py1, px2, py2 = clip_bbox(parent_bbox, frame_width, frame_height)
        px2, py2 = max(px1, px2), max(py1, py2)
        local = clip_bbox(local_bbox, px2 - px1, py2 - py1)
        return (px1 + local[0], py1 + local[1],
                px1 + local[2], py1 + local[3])
    px1, py1 = parent_bbox[:2]
    return (
        max(0, min(frame_width - 1, int(px1 + local_bbox[0]))),
        max(0, min(frame_height - 1, int(py1 + local_bbox[1]))),
        max(0, min(frame_width, int(px1 + local_bbox[2]))),
        max(0, min(frame_height, int(py1 + local_bbox[3]))),
    )
