"""
Configuration file for Hand Sign Recognition Project
"""
import os

# Base path for relative imports if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Configuration
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# Image Processing Configuration
IMAGE_SIZE = 300
OFFSET = 20
MAX_HANDS = 1
DETECTION_CON = 0.8
MIN_TRACK_CON = 0.8

# Camera Configuration
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Data Collection Configuration
DATA_IMG_BASE_PATH = os.path.join(BASE_DIR, "data_img")
DEFAULT_HAND_SIGNS = ["one", "two", "three", "four", "five"]

# UI Configuration
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS_UPDATE_INTERVAL = 100  # milliseconds

# Colors (BGR for OpenCV)
COLOR_PURPLE = (255, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)

# UI Colors (Hex for Tkinter)
THEME_BG = "#121212"
THEME_FG = "#E0E0E0"
THEME_ACCENT = "#00B4D8"
THEME_ACCENT_HOVER = "#0096C7"
THEME_SUCCESS = "#2D6A4F"
THEME_DANGER = "#C1121F"
THEME_PANEL = "#1E1E1E"

# Detection Confidence Threshold
CONFIDENCE_THRESHOLD = 0.8
