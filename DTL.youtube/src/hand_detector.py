"""
Hand Detector Module
Wraps cvzone's HandTrackingModule for consistent hand detection across the project.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector as CVZoneHandDetector
import config

class HandDetector:
    """Unified Hand Detection class"""
    
    def __init__(self, max_hands=config.MAX_HANDS, detection_con=config.DETECTION_CON, min_track_con=config.MIN_TRACK_CON):
        """Initialize detector"""
        self.detector = CVZoneHandDetector(
            maxHands=max_hands, 
            detectionCon=detection_con, 
            minTrackCon=min_track_con
        )
    
    def find_hand(self, frame):
        """
        Find hand in frame and return bbox and cropped image
        
        Args:
            frame: Input image
            
        Returns:
            dict: {
                'detected': bool,
                'bbox': (x, y, w, h),
                'hands': list of hands from cvzone
            }
        """
        try:
            hands, img = self.detector.findHands(frame, draw=False)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                return {
                    'detected': True,
                    'bbox': (x, y, w, h),
                    'hands': hands
                }
            
            return {'detected': False, 'bbox': None, 'hands': []}
        
        except Exception as e:
            print(f"Error in hand detection: {e}")
            return {'detected': False, 'bbox': None, 'hands': []}
