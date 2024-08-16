
# TensorFlow Lite Obstacle Detection
This project implements an obstacle detection system using TensorFlow Lite. It analyzes video feed in real-time, dividing it into three segments and prompting the user to change direction based on detected obstacles.

## Features

- Real-time video feed analysis
- Video frame segmentation (3 parts)
- Obstacle detection using TensorFlow Lite
- User direction prompts based on obstacle location and size

## Requirements

- TensorFlow Lite
- OpenCV
- Python 3.7+
- Compatible camera or video input device

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tflite-obstacle-detection.git
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```
python obstacle_detection.py
```

## How it works

1. The video feed is divided into three vertical segments: left, center, and right.
2. TensorFlow Lite model detects obstacles in each frame.
3. The system calculates the area of the bounding box for each detected obstacle.
4. Based on the location and size of obstacles, the system prompts the user to change direction:
   - "Move right" if large obstacles are in the left or center segments
   - "Move left" if large obstacles are in the right or center segments
   - "Stop" if large obstacles are detected in all segments

## Configuration

Adjust detection sensitivity and prompt thresholds in `config.py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
