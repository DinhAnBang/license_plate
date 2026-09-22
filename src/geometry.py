"""Pure bounding-box geometry shared by ownership implementations."""

from __future__ import annotations

from math import isfinite

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
