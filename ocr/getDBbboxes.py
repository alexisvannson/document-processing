import cv2
import numpy as np
import pyclipper

def extract_bounding_boxes(prob_map, thresh=0.3, unclip_ratio=1.5, min_area=16):
    """
    Converts a DBNet probability map into full-sized text bounding boxes.
    
    Args:
        prob_map: 2D numpy array (Height, Width) with values between 0.0 and 1.0
        thresh: The probability threshold to consider a pixel as text
        unclip_ratio: The expansion factor (usually 1.5 to 2.0)
        min_area: Ignore tiny noise blobs smaller than this area
        
    Returns:
        list of numpy arrays, where each array is [4, 2] containing the (x, y) 
        coordinates of the bounding box corners.
    """
    # 1. Apply hard threshold to create a binary mask (0 or 255)
    binary_map = (prob_map > thresh).astype(np.uint8) * 255
    
    # 2. Find all connected components (contours) in the binary mask
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    
    for contour in contours:
        # Calculate standard area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filter out tiny specks of noise
        if area < min_area or perimeter == 0:
            continue
            
        # 3. Calculate the expansion distance (Reverse of the training formula)
        # D' = Area * unclip_ratio / Perimeter
        expand_distance = (area * unclip_ratio) / perimeter
        
        # Format the contour for pyclipper (needs a flat list of (x,y) points)
        contour_points = contour.reshape(-1, 2).tolist()
        
        # 4. Use Pyclipper to expand the polygon outward
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contour_points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        
        # Execute the expansion (positive distance expands)
        expanded_polygons = pco.Execute(expand_distance)
        
        # 5. Convert the expanded irregular polygon into a clean 4-point rectangle
        if len(expanded_polygons) > 0:
            # Pyclipper can sometimes return multiple polygons if the expansion splits, 
            # we take the largest/first one
            expanded_poly = np.array(expanded_polygons[0])
            
            # Find the minimum area rotated rectangle that fits the expanded polygon
            rect = cv2.minAreaRect(expanded_poly)
            
            # Get the 4 corners of that rectangle
            box_points = cv2.boxPoints(rect)
            
            # Ensure coordinates are integers and strictly positive
            box_points = np.clip(np.int32(box_points), 0, None)
            
            boxes.append(box_points)
            
    return boxes