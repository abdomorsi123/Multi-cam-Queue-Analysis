# Multi-Camera Queue Analysis System

A real-time queue monitoring system that tracks people across multiple camera views using computer vision and deep learning techniques.

## Features

- Real-time person detection using YOLO
- Cross-camera person tracking using DeepSORT and feature matching
- Waiting time estimation for each person in the queue
- Adaptive time estimation based on historical data
- Unique color coding for each tracked individual
- Combined view display from multiple cameras
- Real-time queue length monitoring

## Technical Stack

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- DeepSORT with ReID
- ResNet50 for feature extraction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Multi-cam-Queue-Analysis.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:

- Video input/output paths
- Detection confidence threshold
- Tracking parameters
- Display settings
- Time estimation parameters

## Usage

1. Place your input videos in the project directory
2. Update video paths in `config.py`
3. Run the analysis:

```bash
python queue_counter.py
```

## How it Works

1. **Person Detection**: YOLO detects people in each camera feed
2. **Feature Extraction**: ResNet50 extracts appearance features for each detection
3. **Cross-Camera Tracking**: DeepSORT and feature matching maintain consistent IDs across cameras
4. **Queue Analysis**:
   - Tracks waiting time for each person
   - Estimates remaining wait time
   - Adapts estimates based on historical data

## Output

- Real-time visualization showing:
  - Bounding boxes around detected people
  - Global ID for each person
  - Estimated waiting time
  - Total queue length
- Combined view from all cameras
- MP4 video output with tracking results

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Authors

Abdelrahman Elsisy

THWS
