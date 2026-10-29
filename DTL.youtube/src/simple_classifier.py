"""
Simple Hand Sign Classifier
Uses template matching on collected images for immediate recognition
"""

import cv2
import os
import numpy as np
import config
from hand_detector import HandDetector

class SimpleHandSignClassifier:
    """
    Simple classifier that matches hand signs against collected templates.
    Works with collected data without needing model retraining.
    """
    
    def __init__(self):
        """Initialize the simple classifier"""
        self.detector = HandDetector()
        self.templates = {}  # sign_name -> list of template images
        self.img_size = (224, 224)
        self._load_templates()
    
    def _load_templates(self):
        """Load all collected hand sign templates"""
        self.templates = {}
        if not os.path.exists(config.DATA_IMG_BASE_PATH):
            return
        
        for sign_name in os.listdir(config.DATA_IMG_BASE_PATH):
            sign_path = os.path.join(config.DATA_IMG_BASE_PATH, sign_name)
            if not os.path.isdir(sign_path):
                continue
            
            templates = []
            for img_file in os.listdir(sign_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(sign_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize to standard size
                        img = cv2.resize(img, self.img_size)
                        templates.append(img)
            
            if templates:
                self.templates[sign_name] = templates
                print(f"Loaded {len(templates)} templates for '{sign_name}'")
    
    def predict(self, frame):
        """
        Detect hand and match against templates
        Returns dict with 'detected', 'label', 'confidence', 'bbox'
        """
        result = {
            'detected': False,
            'label': 'No Sign',
            'confidence': 0.0,
            'bbox': None
        }
        
        # Reload templates in case new ones were added
        self._load_templates()
        
        if not self.templates:
            result['label'] = 'Collect signs first'
            return result
        
        try:
            # Find hand in frame
            detect_res = self.detector.find_hand(frame)
            
            if not detect_res['detected']:
                return result
            
            # Get bounding box
            x, y, w, h = detect_res['bbox']
            
            # Safe cropping with offset
            y1 = max(0, y - config.OFFSET)
            y2 = min(frame.shape[0], y + h + config.OFFSET)
            x1 = max(0, x - config.OFFSET)
            x2 = min(frame.shape[1], x + w + config.OFFSET)
            
            img_crop = frame[y1:y2, x1:x2]
            
            if img_crop.size == 0:
                return result
            
            # Preprocess to match template format
            img_processed = self._preprocess_for_matching(img_crop, h, w)
            
            if img_processed is None:
                return result
            
            # Match against templates
            best_match, best_score = self._match_template(img_processed)
            
            result.update({
                'detected': True,
                'label': best_match if best_score > 0.6 else 'Unknown',
                'confidence': best_score,
                'bbox': (x1, y1, x2, y2)
            })
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            result['label'] = 'Error'
            return result
    
    def _preprocess_for_matching(self, img_crop, h, w):
        """Preprocess image to standard size on white background (same as collector)"""
        try:
            import math
            
            if w == 0 or h == 0:
                return None
            
            img_white = np.ones((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), np.uint8) * 255
            aspect_ratio = h / w
            
            if aspect_ratio > 1:
                k = config.IMAGE_SIZE / h
                w_cal = math.ceil(k * w)
                if w_cal == 0:
                    return None
                img_resize = cv2.resize(img_crop, (w_cal, config.IMAGE_SIZE))
                w_gap = math.ceil((config.IMAGE_SIZE - w_cal) / 2)
                img_white[:, w_gap:w_cal + w_gap] = img_resize
            else:
                k = config.IMAGE_SIZE / w
                h_cal = math.ceil(k * h)
                if h_cal == 0:
                    return None
                img_resize = cv2.resize(img_crop, (config.IMAGE_SIZE, h_cal))
                h_gap = math.ceil((config.IMAGE_SIZE - h_cal) / 2)
                img_white[h_gap:h_cal + h_gap, :] = img_resize
            
            # Resize to standard template size
            return cv2.resize(img_white, self.img_size)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def _match_template(self, img_processed):
        """Compare processed image against all templates using SSIM"""
        best_match = 'Unknown'
        best_score = 0.0
        
        # Convert to grayscale for comparison
        img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        
        for sign_name, templates in self.templates.items():
            scores = []
            for template in templates:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Use template matching
                result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                scores.append(max_val)
            
            # Average score for this sign
            avg_score = sum(scores) / len(scores) if scores else 0
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = sign_name
        
        return best_match, best_score
