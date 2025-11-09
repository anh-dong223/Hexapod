# Robot Camera Detector with NVIDIA NanoOWL

Real-time object detection system for robot cameras using NVIDIA NanoOWL on Jetson Nano Orin.

## Overview

This project integrates NVIDIA's NanoOWL model for open-vocabulary object detection with robot camera systems. It provides a Python API for real-time detection that can be easily integrated into robot control systems.

## Prerequisites

### Hardware
- NVIDIA Jetson Nano Orin
- USB camera or CSI camera module

### Software Setup

1. **Install JetPack** (if not already installed)
   - Ensure you have JetPack 5.x installed on your Jetson Nano Orin

2. **Install PyTorch for Jetson**
   ```bash
   # Install PyTorch for Jetson (check NVIDIA's website for latest version)
   wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
   pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
   ```

3. **Install NanoOWL**
   ```bash
   # Clone the nanoowl repository
   git clone https://github.com/NVIDIA-AI-IOT/nanoowl
   cd nanoowl
   
   # Install nanoowl
   python3 setup.py develop --user
   
   # Build the TensorRT engine
   mkdir -p data
   python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
   ```

4. **Install Project Dependencies**
   ```bash
   cd /path/to/robott
   pip3 install -r requirements.txt
   ```

## Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/anh-dong223/Hexapod.git
   cd Hexapod
   ```

2. **Verify Setup** (Optional but recommended)
   ```bash
   python3 verify_setup.py
   ```
   This will check if all dependencies are installed and configured correctly.

2. **Build the TensorRT engine** (if not already done in nanoowl setup)
   ```bash
   # Make sure you're in the nanoowl directory
   cd ~/nanoowl
   python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
   ```

3. **Update configuration**
   - Edit `config.json` to set your detection targets and camera settings
   - Update the `image_encoder_engine` path to match your engine file location

## Usage

### Basic Usage

Run the detector with default settings:
```bash
python3 robot_camera_detector.py
```

### Command Line Options

```bash
python3 robot_camera_detector.py \
    --camera 0 \
    --engine /path/to/owl_image_encoder_patch32.engine \
    --texts person box forklift \
    --threshold 0.1 \
    --save-output
```

**Arguments:**
- `--config`: Path to configuration file (default: `config.json`)
- `--camera`: Camera device index (default: 0)
- `--model`: HuggingFace model name (default: `google/owlvit-base-patch32`)
- `--engine`: Path to TensorRT engine file
- `--texts`: Objects to detect (space-separated list)
- `--threshold`: Detection confidence threshold (default: 0.1)
- `--no-display`: Disable video display
- `--save-output`: Save detection results to files
- `--output-dir`: Output directory for saved results (default: `output`)

### Using Configuration File

Edit `config.json` to customize settings:

```json
{
  "model_name": "google/owlvit-base-patch32",
  "image_encoder_engine": "data/owl_image_encoder_patch32.engine",
  "camera_index": 0,
  "detection_texts": ["person", "box", "forklift"],
  "threshold": 0.1,
  "save_output": false,
  "output_dir": "output"
}
```

Then run:
```bash
python3 robot_camera_detector.py --config config.json
```

### Using as a Library

You can import and use the detector in your own scripts:

```python
from robot_camera_detector import RobotCameraDetector
import cv2

# Initialize detector
detector = RobotCameraDetector(
    image_encoder_engine="data/owl_image_encoder_patch32.engine",
    detection_texts=["person", "box"],
    threshold=0.1,
    show_display=False  # Set to False for headless operation
)

# Initialize camera
detector.initialize_camera(width=640, height=480)

# Process frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get detections
    results = detector.get_detections(frame)
    
    # Process results
    for detection in results["detections"]:
        bbox = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        center = detection["center"]
        
        print(f"Found {label} at {center} with confidence {confidence}")
        
        # Use detections for robot control
        # ...

# Cleanup
detector.cleanup()
cap.release()
```

### Integration with Robot Control Systems

The detector can be integrated into ROS 2, or any other robot control framework:

```python
from robot_camera_detector import RobotCameraDetector

class RobotController:
    def __init__(self):
        self.detector = RobotCameraDetector(
            detection_texts=["person", "obstacle"],
            show_display=False
        )
        self.detector.initialize_camera()
    
    def process_frame(self, frame):
        results = self.detector.get_detections(frame)
        
        # React to detections
        for detection in results["detections"]:
            if detection["label"] == "person":
                # Stop robot if person detected
                self.stop_robot()
            elif detection["label"] == "obstacle":
                # Avoid obstacle
                self.avoid_obstacle(detection["center"])
```

## Output Format

Detection results are returned as a dictionary:

```python
{
    "frame_number": 123,
    "timestamp": 1234567890.123,
    "process_time_ms": 45.2,
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "label": "person",
            "confidence": 0.85,
            "center": [cx, cy],
            "area": 12345.67
        },
        # ... more detections
    ],
    "frame_shape": [480, 640, 3]
}
```

## Troubleshooting

### Camera Not Found
- Check camera index: `ls /dev/video*`
- Try different camera indices: `--camera 1`, `--camera 2`, etc.
- For CSI cameras, you may need to use `gstreamer` pipeline instead

### Engine File Not Found
- Make sure you've built the TensorRT engine
- Update the path in `config.json` or use `--engine` argument
- The engine file should be in the `data/` directory of nanoowl

### Low Performance
- Reduce frame resolution in camera initialization
- Increase detection threshold to filter out weak detections
- Use a smaller model if available
- Ensure TensorRT engine is properly built

### Memory Issues
- Close other applications
- Reduce batch size if processing multiple frames
- Use `--no-display` to save GPU memory

## Camera Setup for Jetson

### USB Camera
Most USB cameras should work out of the box:
```bash
# List available cameras
ls /dev/video*
```

### CSI Camera (Raspberry Pi Camera Module)
For CSI cameras, you may need to use GStreamer:
```python
# Example GStreamer pipeline for CSI camera
gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
```

## Advanced Usage

### Dynamic Query Updates
You can change detection targets at runtime:

```python
detector.update_detection_texts(["person", "dog", "car"])
```

### Saving Detection Results
Enable saving to get JSON files with detection data:
```bash
python3 robot_camera_detector.py --save-output --output-dir results
```

### Headless Operation
For robots without displays:
```bash
python3 robot_camera_detector.py --no-display
```

## License

This project uses NVIDIA NanoOWL, which is licensed under Apache 2.0.

## References

- [NVIDIA NanoOWL GitHub](https://github.com/NVIDIA-AI-IOT/nanoowl)
- [ROS2-NanoOWL](https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL)
- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)

## Support

For issues related to:
- **NanoOWL**: Check the [NVIDIA NanoOWL repository](https://github.com/NVIDIA-AI-IOT/nanoowl)
- **Jetson Setup**: Refer to [NVIDIA Jetson Developer Guide](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- **This Script**: Open an issue in this repository

