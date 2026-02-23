from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import importlib
import sys
from dotenv import load_dotenv
from auth import login_required, check_auth

# Load environment variables
load_dotenv()

# Force reload detect module
if 'detect' in sys.modules:
    importlib.reload(sys.modules['detect'])
from detect import LeafPestDetector

# Conservative pest detection - only obvious pests
detection_config = {
    'min_pest_pixels': 30,  # Larger minimum for obvious pests only
}

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pest-detection-secret-2024')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_leaf_mask(img):
    """
    Extract the leaf outline and create a mask for the leaf area
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for green colors (healthy leaf)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Also include yellowish-green areas
    lower_yellow_green = np.array([15, 30, 30])
    upper_yellow_green = np.array([35, 255, 255])
    yellow_green_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
    
    # Combine masks
    leaf_mask = cv2.bitwise_or(green_mask, yellow_green_mask)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes in the leaf
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (main leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        leaf_mask = np.zeros_like(leaf_mask)
        cv2.fillPoly(leaf_mask, [largest_contour], 255)
    
    return leaf_mask, largest_contour if contours else None

def detect_healthy_leaf_color(img, leaf_mask):
    """
    Analyze the leaf to determine the dominant healthy color
    """
    # Apply mask to get only leaf pixels
    leaf_pixels = img[leaf_mask > 0]
    
    if len(leaf_pixels) == 0:
        return None
    
    # Convert to HSV for better analysis
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    leaf_hsv_pixels = hsv_img[leaf_mask > 0]
    
    # Calculate mean and std of healthy leaf color
    mean_hsv = np.mean(leaf_hsv_pixels, axis=0)
    std_hsv = np.std(leaf_hsv_pixels, axis=0)
    
    return mean_hsv, std_hsv

def find_discolored_pixels(img, leaf_mask, healthy_color_stats):
    """
    Find brown discolored pixels that are clearly visible pests
    More inclusive detection to catch all brown variations
    """
    if healthy_color_stats is None:
        return np.zeros_like(leaf_mask)
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a mask for brown discolored pixels only
    discolored_mask = np.zeros_like(leaf_mask)
    
    # Expanded brown color ranges to catch more variations
    # Brown range 1: Dark brown (expanded)
    lower_brown1 = np.array([5, 30, 15])
    upper_brown1 = np.array([25, 255, 180])
    
    # Brown range 2: Reddish brown (expanded)
    lower_brown2 = np.array([0, 30, 15])
    upper_brown2 = np.array([15, 255, 180])
    
    # Brown range 3: Yellowish brown (expanded)
    lower_brown3 = np.array([15, 30, 15])
    upper_brown3 = np.array([35, 255, 180])
    
    # Brown range 4: Very dark brown/black spots
    lower_brown4 = np.array([0, 0, 10])
    upper_brown4 = np.array([180, 255, 80])
    
    # Create masks for different brown ranges
    brown_mask1 = cv2.inRange(hsv_img, lower_brown1, upper_brown1)
    brown_mask2 = cv2.inRange(hsv_img, lower_brown2, upper_brown2)
    brown_mask3 = cv2.inRange(hsv_img, lower_brown3, upper_brown3)
    brown_mask4 = cv2.inRange(hsv_img, lower_brown4, upper_brown4)
    
    # Combine all brown masks
    brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
    brown_mask = cv2.bitwise_or(brown_mask, brown_mask3)
    brown_mask = cv2.bitwise_or(brown_mask, brown_mask4)
    
    # Only keep brown pixels that are within the leaf boundary
    discolored_mask = cv2.bitwise_and(brown_mask, leaf_mask)
    
    # More lenient filtering - only remove obvious noise
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if discolored_mask[y, x] > 0:
                pixel_hsv = hsv_img[y, x]
                # Only remove extremely bright pixels (strong reflections)
                if pixel_hsv[2] > detection_config['max_brightness']:
                    discolored_mask[y, x] = 0
                # Keep more saturated areas, only remove very gray areas
                elif pixel_hsv[1] < detection_config['min_saturation'] and pixel_hsv[2] > 100:
                    discolored_mask[y, x] = 0
    
    return discolored_mask

def group_connected_pixels(discolored_mask):
    """
    Group connected brown discolored pixels into individual pest clusters
    More inclusive grouping to catch all pest shapes
    """
    # Apply lighter morphological operations to preserve more details
    kernel_small = np.ones((2,2), np.uint8)
    kernel_medium = np.ones((3,3), np.uint8)
    
    # Remove very small noise but preserve pest details
    discolored_mask = cv2.morphologyEx(discolored_mask, cv2.MORPH_OPEN, kernel_small)
    # Fill small gaps in pest areas
    discolored_mask = cv2.morphologyEx(discolored_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Use connected components to group touching pixels
    num_labels, labels = cv2.connectedComponents(discolored_mask)
    
    pest_clusters = []
    for label in range(1, num_labels):  # Skip background (label 0)
        cluster_mask = (labels == label).astype(np.uint8) * 255
        
        # Find contour of this cluster
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            # More lenient filtering - focus mainly on size
            if area > detection_config['min_pest_area']:
                # Less strict shape filtering - accept more varied shapes
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Calculate circularity but be more accepting
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Very lenient circularity check - mainly to filter out extremely thin lines
                    if circularity > detection_config['min_circularity'] or area > 50:
                        pest_clusters.append(contour)
                else:
                    # If perimeter calculation fails, just use area
                    pest_clusters.append(contour)
    
    return pest_clusters

def detect_pests(image_path):
    """Detect pests using new leaf structure analysis"""
    img = cv2.imread(image_path)
    if img is None:
        return {'total': 0, 'pests': 0, 'eggs': 0}, None
    
    # Use new detector
    detector = LeafPestDetector(conf_threshold=0.25)
    results = detector.detect(img)
    
    # Count pests and eggs separately
    pest_count = sum(1 for d in results['detections'] if d['class'] == 'pest')
    egg_count = sum(1 for d in results['detections'] if d['class'] == 'egg')
    
    # Visualize
    result_img = detector.visualize(img, show_conf=False)
    
    result_path = image_path.replace('.', '_result.')
    cv2.imwrite(result_path, result_img)
    
    return {'total': results['total_pests'], 'pests': pest_count, 'eggs': egg_count}, result_path

def remove_overlapping_contours(contours, overlap_threshold=0.5):
    """Remove overlapping contours to avoid duplicate detections"""
    if len(contours) <= 1:
        return contours
    
    rects = [cv2.boundingRect(c) for c in contours]
    centers = []
    for rect in rects:
        x, y, w, h = rect
        centers.append((x + w//2, y + h//2))
    
    keep = [True] * len(contours)
    
    for i in range(len(contours)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(contours)):
            if not keep[j]:
                continue
            
            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            
            area_i = cv2.contourArea(contours[i])
            area_j = cv2.contourArea(contours[j])
            radius_i = np.sqrt(area_i / np.pi)
            radius_j = np.sqrt(area_j / np.pi)
            avg_radius = (radius_i + radius_j) / 2
            
            if dist < avg_radius * overlap_threshold:
                if area_i >= area_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [contours[i] for i in range(len(contours)) if keep[i]]

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        auth_key = request.form.get('auth_key')
        if check_auth(auth_key):
            session['authenticated'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid authentication key')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect pests and eggs
        detection_result, result_path = detect_pests(filepath)
        
        return jsonify({
            'success': True,
            'total_count': detection_result['total'],
            'pest_count': detection_result['pests'],
            'egg_count': detection_result['eggs'],
            'original_image': f'/uploads/{filename}',
            'result_image': f'/uploads/{os.path.basename(result_path)}'
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/scan_camera', methods=['POST'])
@login_required
def scan_camera():
    try:
        # Get image data from camera
        image_data = request.json['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image and then to OpenCV format
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'camera_capture.jpg')
        cv2.imwrite(temp_path, cv_image)
        
        # Detect pests and eggs
        detection_result, result_path = detect_pests(temp_path)
        
        return jsonify({
            'success': True,
            'total_count': detection_result['total'],
            'pest_count': detection_result['pests'],
            'egg_count': detection_result['eggs'],
            'result_image': f'/uploads/{os.path.basename(result_path)}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_config', methods=['POST'])
def update_config():
    global detection_config
    try:
        data = request.json
        if 'min_pest_pixels' in data:
            detection_config['min_pest_pixels'] = int(data['min_pest_pixels'])
        return jsonify({'success': True, 'config': detection_config})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)