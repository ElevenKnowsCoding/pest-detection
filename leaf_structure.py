import cv2
import numpy as np

def extract_leaf_structure(img):
    """Extract leaf outline and vein structure"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract leaf mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_yellow = np.array([15, 30, 30])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)
    kernel = np.ones((5,5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    # Get largest contour as leaf outline
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_mask_clean = np.zeros_like(leaf_mask)
    cv2.fillPoly(leaf_mask_clean, [leaf_contour], 255)
    
    # Extract veins using green channel and morphology
    leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask_clean)
    gray = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=leaf_mask_clean)
    
    # Enhance veins with morphological operations
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)
    
    # Threshold to get vein structure
    _, vein_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    
    return leaf_contour, leaf_mask_clean, vein_mask

def visualize_leaf_structure(img, leaf_contour, leaf_mask, vein_mask):
    """Create visualization of leaf structure"""
    result = img.copy()
    
    # Draw leaf outline in green
    cv2.drawContours(result, [leaf_contour], -1, (0, 255, 0), 3)
    
    # Overlay veins in darker green
    vein_overlay = cv2.cvtColor(vein_mask, cv2.COLOR_GRAY2BGR)
    vein_overlay[vein_mask > 0] = [0, 150, 0]
    result = cv2.addWeighted(result, 1, vein_overlay, 0.6, 0)
    
    return result

if __name__ == "__main__":
    img = cv2.imread('uploads/Test 1.jpeg')
    if img is None:
        print("Could not load image")
    else:
        leaf_contour, leaf_mask, vein_mask = extract_leaf_structure(img)
        if leaf_contour is not None:
            result = visualize_leaf_structure(img, leaf_contour, leaf_mask, vein_mask)
            cv2.imwrite('uploads/leaf_structure.jpg', result)
            cv2.imwrite('uploads/leaf_outline.jpg', leaf_mask)
            cv2.imwrite('uploads/leaf_veins.jpg', vein_mask)
            print("Leaf structure extracted and saved")
