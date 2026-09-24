import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

def generate_dbnet_prob_map(image_shape, bboxes, shrink_ratio=0.4):
    """
    Converts a list of word bounding boxes into a DBNet Probability Map.
    image_shape: tuple of (Height, Width)
    bboxes: list of polygons, e.g., [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ... ]
    """
    prob_map = np.zeros(image_shape[:2], dtype=np.float32)
    
    for box in bboxes:
        polygon = np.array(box)
        poly_obj = Polygon(polygon)
        
        # If the box is a straight line or invalid, skip it
        if poly_obj.area <= 0:
            continue
            
        area = poly_obj.area
        perimeter = poly_obj.length
        
        # 1. Calculate the DBNet shrink distance (D)
        # Formula from the paper: D = Area * (1 - r^2) / Perimeter
        distance = area * (1 - shrink_ratio**2) / (perimeter + 1e-6)
        
        # 2. Use Pyclipper to shrink the bounding box inward
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(polygon.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        
        # Execute the shrink (negative distance shrinks)
        shrunk_polygons = pco.Execute(-distance)
        
        # 3. Draw the shrunk polygon onto the probability map as solid white (1.0)
        if len(shrunk_polygons) > 0:
            shrunk_poly = np.array(shrunk_polygons[0]).reshape(-1, 1, 2)
            cv2.fillPoly(prob_map, [shrunk_poly.astype(np.int32)], 1.0)
            
    return prob_map

def _segment_distance(xs, ys, a, b):
    """Distance from every (xs, ys) grid point to the segment a-b."""
    ab = b - a
    t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / (ab @ ab + 1e-6), 0, 1)
    return np.hypot(xs - (a[0] + t * ab[0]), ys - (a[1] + t * ab[1]))


def generate_dbnet_thresh_map(image_shape, bboxes, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
    """
    Converts a list of word bounding boxes into a DBNet Threshold Map and its mask.
    The map peaks (thresh_max) on each box border and decays to thresh_min at distance D.
    The mask marks the dilated border region where the L1 threshold loss is applied.
    """
    h, w = image_shape[:2]
    canvas = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    for box in bboxes:
        polygon = np.array(box, dtype=np.float32)
        poly_obj = Polygon(polygon)

        if poly_obj.area <= 0:
            continue

        # Same distance D as the shrink, but applied outward (dilation)
        distance = poly_obj.area * (1 - shrink_ratio**2) / (poly_obj.length + 1e-6)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(polygon.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated_polygons = pco.Execute(distance)

        if len(dilated_polygons) == 0:
            continue

        dilated = np.array(dilated_polygons[0], dtype=np.int32)
        cv2.fillPoly(mask, [dilated], 1.0)

        # Only compute distances inside the dilated box's bounding rectangle
        x_min, y_min = np.clip(dilated.min(axis=0), 0, [w - 1, h - 1])
        x_max, y_max = np.clip(dilated.max(axis=0), 0, [w - 1, h - 1])
        ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1].astype(np.float32)

        n = len(polygon)
        dist = np.min(
            [_segment_distance(xs, ys, polygon[i], polygon[(i + 1) % n]) for i in range(n)], axis=0
        )
        region = canvas[y_min:y_max + 1, x_min:x_max + 1]
        canvas[y_min:y_max + 1, x_min:x_max + 1] = np.fmax(region, 1 - np.clip(dist / distance, 0, 1))

    return canvas * (thresh_max - thresh_min) + thresh_min, mask
