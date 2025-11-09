# Quick Start Guide

Get your robot camera detector running in minutes!

## Prerequisites Check

1. **Jetson Nano Orin** with JetPack installed
2. **Camera** connected (USB or CSI)
3. **nanoowl** installed (see below)

## Step 1: Install NanoOWL

```bash
# Clone nanoowl
cd ~
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
cd nanoowl

# Install nanoowl
python3 setup.py develop --user

# Build TensorRT engine (this may take several minutes)
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
```

## Step 2: Setup This Project

```bash
# Clone the repository (if not already done)
git clone https://github.com/anh-dong223/Hexapod.git
cd Hexapod

# Run setup script
./setup.sh

# Or manually install dependencies
pip3 install -r requirements.txt

# Verify setup (recommended)
python3 verify_setup.py
```

## Step 3: Configure Detection Targets

Edit `config.json` to specify what objects you want to detect:

```json
{
  "detection_texts": ["person", "box", "forklift", "robot"]
}
```

You can use natural language descriptions like:
- `"a person"`
- `"a red box"`
- `"a forklift truck"`
- `"an obstacle"`

## Step 4: Run the Detector

### Basic Usage

```bash
python3 robot_camera_detector.py
```

### With Custom Settings

```bash
python3 robot_camera_detector.py \
    --camera 0 \
    --texts person box forklift \
    --threshold 0.1
```

### Headless Mode (no display)

```bash
python3 robot_camera_detector.py --no-display
```

## Step 5: Use in Your Robot Code

```python
from robot_camera_detector import RobotCameraDetector
import cv2

# Initialize
detector = RobotCameraDetector(
    image_encoder_engine="../nanoowl/data/owl_image_encoder_patch32.engine",
    detection_texts=["person", "obstacle"],
    show_display=False
)

detector.initialize_camera()

# Process frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = detector.get_detections(frame)
    
    # Use detections for robot control
    for det in results["detections"]:
        print(f"Found {det['label']} at {det['center']}")
```

## Troubleshooting

### Engine File Not Found

Make sure you've built the engine:
```bash
cd ~/nanoowl
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
```

Update `config.json` with the correct path:
```json
{
  "image_encoder_engine": "/home/user/nanoowl/data/owl_image_encoder_patch32.engine"
}
```

### Camera Not Found

List available cameras:
```bash
ls /dev/video*
```

Try different camera indices:
```bash
python3 robot_camera_detector.py --camera 1
```

### Low Performance

- Reduce camera resolution in code
- Increase threshold to filter weak detections
- Close other applications to free GPU memory

## Next Steps

- See `example_usage.py` for more examples
- Read `README.md` for detailed documentation
- Customize detection targets in `config.json`
- Integrate with your robot control system

## Example: Robot Avoidance

```python
from robot_camera_detector import RobotCameraDetector

detector = RobotCameraDetector(
    detection_texts=["person", "obstacle"],
    show_display=False
)
detector.initialize_camera()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = detector.get_detections(frame)
    
    for det in results["detections"]:
        if det["label"] == "person":
            # Stop robot
            stop_robot()
        elif det["label"] == "obstacle":
            # Avoid obstacle
            avoid_obstacle(det["center"])
```

## Support

For issues:
1. Check `README.md` for detailed docs
2. Check nanoowl repository: https://github.com/NVIDIA-AI-IOT/nanoowl
3. Verify your Jetson setup and camera connection

