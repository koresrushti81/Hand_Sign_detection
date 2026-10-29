import os

# Set Keras backend to tensorflow and force CPU for stability on native Windows with RTX 50-series
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import keras
import config
from hand_detector import HandDetector

class HandSignClassifier:
    """
    A professional class for Hand Sign Classification using a pre-trained Keras model.
    Integrates hand detection and classification with confidence thresholding.
    """
    def __init__(self, model_path, labels_path, confidence_threshold=0.4):
        """
        Initialize the classifier.
        :param model_path: Path to the .keras or .h5 model file.
        :param labels_path: Path to the .txt labels file.
        :param confidence_threshold: Minimum confidence to return a prediction.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at: {labels_path}")

        # Load model using Keras 3 (handles .keras and .h5 automatically)
        print(f"Loading model: {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.labels = self.load_labels(labels_path)
        self.confidence_threshold = confidence_threshold
        self.img_size = (224, 224)
        
        # Initialize hand detector
        self.detector = HandDetector()

    def load_labels(self, labels_path):
        """Load labels from a text file."""
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def predict(self, frame):
        """
        Detect hand in frame and predict the hand sign.
        :param frame: Full camera frame (OpenCV image).
        :return: dict with keys: 'detected', 'label', 'confidence', 'bbox'
        """
        result = {
            'detected': False,
            'label': 'Searching...',
            'confidence': 0.0,
            'bbox': None
        }
        
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
            
            # Preprocessing for classification
            img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, self.img_size)
            img_reshaped = np.reshape(img_resized, (1, 224, 224, 3))
            
            # MobileNetV2 expects [-1, 1] preprocessing
            img_preprocessed = keras.applications.mobilenet_v2.preprocess_input(img_reshaped)

            # Prediction
            prediction = self.model.predict(img_preprocessed, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]

            # Determine label
            if confidence >= self.confidence_threshold:
                label = self.labels[index]
            else:
                label = "Unknown"

            # Update result
            result.update({
                'detected': True,
                'label': label,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
            
            return result

        except Exception as e:
            print(f"Prediction Error: {e}")
            result['label'] = 'Error'
            return result

if __name__ == "__main__":
    # Quick test if run directly
    try:
        classifier = HandSignClassifier("keras_model.keras", "labels.txt")
        print("Model and labels loaded successfully.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
