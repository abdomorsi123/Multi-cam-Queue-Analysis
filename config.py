"""
Configuration parameters for the queue analysis system.
All configurable parameters are centralized here.
"""

# Input/Output paths
VIDEO_PATH1 = "scene_1/camera_1.mp4"
VIDEO_PATH2 = "scene_1/camera_2.mp4"
OUTPUT_PATH = "scene_1/output.mp4"

# YOLO model settings
YOLO_MODEL = "yolov10n"
CONFIDENCE_THRESHOLD = 0.70

# DeepSORT tracker settings
MAX_AGE = 50       # Frames to keep a track without detections
N_INIT = 3         # Detections needed to confirm a track
NN_BUDGET = 100    # Max size of appearance descriptor queue per track

# Display settings
MAX_DISPLAY_WIDTH = 1536
MAX_DISPLAY_HEIGHT = 1016

# Time estimation settings
INITIAL_ESTIMATED_TIME = 20      # Initial estimated waiting time in seconds
MAX_HISTORICAL_DIFFERENCES = 10  # Number of historical differences to store
MIN_ESTIMATED_TIME = 5           # Minimum estimated waiting time in seconds
MAX_ESTIMATED_TIME = 60          # Maximum estimated waiting time in seconds

# Cross-camera tracking settings
FEATURE_MATCH_THRESHOLD = 0.6   # Minimum similarity score to match across cameras (0.0-1.0)