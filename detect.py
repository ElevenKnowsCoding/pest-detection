import cv2
import numpy as np
from leaf_structure import extract_leaf_structure
import time

class LeafPestDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
    
    def detect(self, image):
        """Detect pests using leaf structure analysis"""
        start_time = time.time()
        
        # Extract leaf structure
        leaf_contour, leaf_mask, vein_mask = extract_leaf_structure(image)
        
        if leaf_contour is None:
            return self._empty_result(time.time() - start_time)
        
        # Detect pests within leaf boundary, excluding veins
        pest_detections = self._detect_pests(image, leaf_mask, vein_mask)
        
        # Format results
        results = {
            'total_pests': len(pest_detections),
            'detections': pest_detections,
            'class_counts': self._count_classes(pest_detections),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'leaf_structure': {
                'outline': leaf_contour.tolist() if leaf_contour is not None else None,
                'has_veins': vein_mask is not None
            }
        }
        
        return results
    
    def _detect_pests(self, img, leaf_mask, vein_mask):
        """Detect obvious pests, excluding veins and shadows"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Erode leaf mask to exclude border shadows
        kernel_erode = np.ones((25,25), np.uint8)
        leaf_mask_inner = cv2.erode(leaf_mask, kernel_erode)
        
        # Detect dark areas with moderate saturation
        lower_dark = np.array([0, 50, 0])
        upper_dark = np.array([180, 255, 60])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Apply inner leaf mask
        pest_mask = cv2.bitwise_and(dark_mask, leaf_mask_inner)
        
        # Exclude veins
        if vein_mask is not None:
            pest_mask = cv2.subtract(pest_mask, vein_mask)
        
        # Moderate noise removal
        kernel = np.ones((4,4), np.uint8)
        pest_mask = cv2.morphologyEx(pest_mask, cv2.MORPH_OPEN, kernel)
        pest_mask = cv2.morphologyEx(pest_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(pest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 55 or area > 1500:  # Even lower minimum
                continue
            
            # Classify: eggs are smaller (55-150), pests are larger (150+)
            if area < 150:
                pest_type = 'egg'
                color = (255, 165, 0)  # Orange
            else:
                pest_type = 'pest'
                color = (0, 0, 255)  # Red
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Calculate confidence
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            confidence = min(0.95, 0.5 + (circularity * 0.3) + (min(area/1000, 1) * 0.2))
            
            detections.append({
                'id': idx + 1,
                'class': pest_type,
                'confidence': confidence,
                'bbox': {
                    'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h,
                    'center_x': cx, 'center_y': cy,
                    'width': w, 'height': h
                },
                'area': area,
                'circularity': circularity,
                'color': color
            })
        
        return detections
    
    def visualize(self, image, show_conf=True):
        """Visualize detection results with leaf structure"""
        result = image.copy()
        
        # Get leaf structure
        leaf_contour, leaf_mask, vein_mask = extract_leaf_structure(image)
        
        # Draw leaf outline
        if leaf_contour is not None:
            cv2.drawContours(result, [leaf_contour], -1, (0, 255, 0), 2)
        
        # Detect and draw pests
        pest_detections = self._detect_pests(image, leaf_mask, vein_mask)
        
        for detection in pest_detections:
            bbox = detection['bbox']
            cx, cy = bbox['center_x'], bbox['center_y']
            color = detection.get('color', (0, 0, 255))
            
            # Draw filled circle
            cv2.circle(result, (cx, cy), 8, color, -1)
            cv2.circle(result, (cx, cy), 8, (255, 255, 255), 2)
            
            # Add number
            label = f"{detection['id']}"
            cv2.putText(result, label, (cx-8, cy+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def _count_classes(self, detections):
        """Count detections by class"""
        counts = {}
        for det in detections:
            cls = det['class']
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    def _empty_result(self, elapsed_time):
        """Return empty result structure"""
        return {
            'total_pests': 0,
            'detections': [],
            'class_counts': {},
            'processing_time_ms': elapsed_time * 1000,
            'leaf_structure': None
        }
