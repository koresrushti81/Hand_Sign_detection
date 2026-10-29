"""
Hand Sign Data Collector Module
Handles data collection for training new hand signs
"""

import cv2
import os
import time
import numpy as np
import math
import config
from hand_detector import HandDetector


class HandSignDataCollector:
    """Collects hand sign images for training"""
    
    def __init__(self):
        """Initialize data collector"""
        self.counter = 0
        self.current_folder = None
        self.collecting = False
        self.detector = HandDetector()
    
    def start_collection(self, hand_sign_name):
        """Start collecting data for a new hand sign"""
        try:
            self.current_folder = os.path.join(config.DATA_IMG_BASE_PATH, hand_sign_name)
            os.makedirs(self.current_folder, exist_ok=True)
            self.counter = len([f for f in os.listdir(self.current_folder) if f.endswith('.jpg')])
            self.collecting = True
            return True
        except Exception as e:
            print(f"Error starting collection: {e}")
            return False
    
    def stop_collection(self):
        """Stop collecting data"""
        self.collecting = False
        images_saved = self.counter
        return images_saved
    
    def collect_frame(self, frame):
        """Process frame and extract hand image"""
        result = {
            'detected': False,
            'cropped_image': None,
            'processed_image': None,
            'bbox': None,
            'count': self.counter
        }
        
        if not self.collecting or self.current_folder is None:
            return result
        
        try:
            # Find hand
            detect_res = self.detector.find_hand(frame)
            
            if detect_res['detected']:
                x, y, w, h = detect_res['bbox']
                
                # Safe cropping
                y1 = max(0, y - config.OFFSET)
                y2 = min(frame.shape[0], y + h + config.OFFSET)
                x1 = max(0, x - config.OFFSET)
                x2 = min(frame.shape[1], x + w + config.OFFSET)
                
                img_crop = frame[y1:y2, x1:x2]
                
                if img_crop.size > 0:
                    # Preprocess
                    img_white = self._preprocess_image(img_crop, h, w)
                    
                    if img_white is not None:
                        result.update({
                            'detected': True,
                            'cropped_image': img_crop,
                            'processed_image': img_white,
                            'bbox': (x1, y1, x2, y2)
                        })
            
            return result
        except Exception as e:
            print(f"Error collecting frame: {e}")
            return result
    
    def save_image(self, img_white):
        """Save the processed hand sign image"""
        try:
            if self.current_folder is None or img_white is None:
                return False
            
            filename = os.path.join(self.current_folder, f'Image_{int(time.time()*1000)}.jpg')
            cv2.imwrite(filename, img_white)
            self.counter += 1
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def _preprocess_image(self, img_crop, h, w):
        """Preprocess cropped image to standard size on white background"""
        try:
            if w == 0 or h == 0:
                return None
            img_white = np.ones((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), np.uint8) * 255
            aspect_ratio = h / w
            
            if aspect_ratio > 1:
                k = config.IMAGE_SIZE / h
                w_cal = math.ceil(k * w)
                if w_cal == 0: return None
                img_resize = cv2.resize(img_crop, (w_cal, config.IMAGE_SIZE))
                w_gap = math.ceil((config.IMAGE_SIZE - w_cal) / 2)
                img_white[:, w_gap:w_cal + w_gap] = img_resize
            else:
                k = config.IMAGE_SIZE / w
                h_cal = math.ceil(k * h)
                if h_cal == 0: return None
                img_resize = cv2.resize(img_crop, (config.IMAGE_SIZE, h_cal))
                h_gap = math.ceil((config.IMAGE_SIZE - h_cal) / 2)
                img_white[h_gap:h_cal + h_gap, :] = img_resize
            
            return img_white
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None
    
    def get_available_hand_signs(self):
        """Get list of hand signs already in data collection"""
        try:
            if not os.path.exists(config.DATA_IMG_BASE_PATH):
                os.makedirs(config.DATA_IMG_BASE_PATH, exist_ok=True)
            return sorted([d for d in os.listdir(config.DATA_IMG_BASE_PATH) 
                         if os.path.isdir(os.path.join(config.DATA_IMG_BASE_PATH, d))])
        except Exception as e:
            print(f"Error getting hand signs: {e}")
            return []
    
    def get_collection_stats(self, hand_sign_name):
        """Get number of images collected for a hand sign"""
        try:
            folder = os.path.join(config.DATA_IMG_BASE_PATH, hand_sign_name)
            if os.path.exists(folder):
                return len([f for f in os.listdir(folder) if f.endswith('.jpg')])
            return 0
        except Exception as e:
            print(f"Error getting stats: {e}")
            return 0
