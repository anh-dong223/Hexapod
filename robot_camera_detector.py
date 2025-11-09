#!/usr/bin/env python3
"""
Robot Camera Detector using NVIDIA NanoOWL
Real-time object detection for robot camera systems on Jetson Nano Orin
"""

import cv2
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

try:
    from nanoowl.owl_predictor import OwlPredictor
except ImportError:
    print("Error: nanoowl not installed. Please install it first:")
    print("  git clone https://github.com/NVIDIA-AI-IOT/nanoowl")
    print("  cd nanoowl && python3 setup.py develop --user")
    exit(1)


class RobotCameraDetector:
    """Real-time object detection using NanoOWL for robot cameras"""
    
    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        image_encoder_engine: str = "data/owl_image_encoder_patch32.engine",
        camera_index: int = 0,
        detection_texts: Optional[List[str]] = None,
        threshold: float = 0.1,
        show_display: bool = True,
        save_output: bool = False,
        output_dir: str = "output"
    ):
        """
        Initialize the robot camera detector
        
        Args:
            model_name: HuggingFace model name for OWL-ViT
            image_encoder_engine: Path to TensorRT engine file
            camera_index: Camera device index (0 for default camera)
            detection_texts: List of objects to detect (e.g., ["person", "box", "forklift"])
            threshold: Detection confidence threshold
            show_display: Whether to display video output
            save_output: Whether to save detection results
            output_dir: Directory to save output files
        """
        self.model_name = model_name
        self.image_encoder_engine = image_encoder_engine
        self.camera_index = camera_index
        self.detection_texts = detection_texts or ["person", "object"]
        self.threshold = threshold
        self.show_display = show_display
        self.save_output = save_output
        self.output_dir = Path(output_dir)
        
        # Create output directory if saving
        if self.save_output:
            self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the predictor
        print(f"Loading NanoOWL model: {model_name}")
        
        # Resolve engine path (handle relative paths)
        engine_path = Path(image_encoder_engine)
        if not engine_path.is_absolute():
            # Try relative to current directory
            if engine_path.exists():
                image_encoder_engine = str(engine_path.resolve())
            # Try relative to nanoowl directory
            elif Path("../nanoowl") / engine_path.name.exists():
                image_encoder_engine = str((Path("../nanoowl") / engine_path.name).resolve())
            elif Path("nanoowl") / engine_path.name.exists():
                image_encoder_engine = str((Path("nanoowl") / engine_path.name).resolve())
        
        print(f"Using engine: {image_encoder_engine}")
        
        if not Path(image_encoder_engine).exists():
            raise FileNotFoundError(
                f"Engine file not found: {image_encoder_engine}\n"
                f"Please build it using: python3 -m nanoowl.build_image_encoder_engine {image_encoder_engine}"
            )
        
        try:
            self.predictor = OwlPredictor(
                model_name,
                image_encoder_engine=image_encoder_engine
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize camera
        self.cap = None
        self.frame_count = 0
        self.detection_history = []
        
    def initialize_camera(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the camera capture
        
        Args:
            width: Camera frame width
            height: Camera frame height
            fps: Frames per second
        """
        print(f"Initializing camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for object detection
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Dictionary containing detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Perform prediction
        output = self.predictor.predict(
            image=pil_image,
            text=self.detection_texts,
            threshold=self.threshold
        )
        
        # Format results
        results = {
            "frame_number": self.frame_count,
            "timestamp": time.time(),
            "detections": [],
            "frame_shape": frame.shape
        }
        
        if output and len(output) > 0:
            boxes = output[0].boxes if hasattr(output[0], 'boxes') else []
            labels = output[0].labels if hasattr(output[0], 'labels') else []
            scores = output[0].scores if hasattr(output[0], 'scores') else []
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                # Convert box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box
                
                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "label": str(label),
                    "confidence": float(score),
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                    "area": float((x2 - x1) * (y2 - y1))
                }
                results["detections"].append(detection)
        
        return results
    
    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: Detection results dictionary
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for detection in results["detections"]:
            x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
            label = detection["label"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label and confidence
            label_text = f"{label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(
                frame_copy,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_copy,
                label_text,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        # Draw FPS and frame info
        fps_text = f"Frame: {results['frame_number']}"
        cv2.putText(
            frame_copy,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        detection_count = len(results["detections"])
        count_text = f"Detections: {detection_count}"
        cv2.putText(
            frame_copy,
            count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return frame_copy
    
    def save_detection_results(self, results: Dict):
        """Save detection results to file"""
        if not self.save_output:
            return
        
        # Save JSON results
        json_file = self.output_dir / f"detection_{results['frame_number']:06d}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Keep history for summary
        self.detection_history.append(results)
    
    def run(self):
        """Main detection loop"""
        if self.cap is None:
            self.initialize_camera()
        
        print("Starting detection loop...")
        print("Press 'q' to quit, 's' to save current frame")
        print(f"Detection targets: {', '.join(self.detection_texts)}")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                start_time = time.time()
                results = self.process_frame(frame)
                process_time = time.time() - start_time
                
                # Add processing time to results
                results["process_time_ms"] = process_time * 1000
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results)
                
                # Save results if enabled
                if self.save_output:
                    self.save_detection_results(results)
                
                # Display frame
                if self.show_display:
                    cv2.imshow("Robot Camera Detector", annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        frame_file = self.output_dir / f"frame_{self.frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_file), annotated_frame)
                        print(f"Saved frame to {frame_file}")
                
                # Print detection info
                if results["detections"]:
                    print(f"Frame {self.frame_count}: Found {len(results['detections'])} objects "
                          f"(Processed in {process_time*1000:.1f}ms)")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def get_detections(self, frame: np.ndarray) -> Dict:
        """
        Get detections for a single frame (for external use)
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Detection results dictionary
        """
        return self.process_frame(frame)
    
    def update_detection_texts(self, texts: List[str]):
        """
        Update the objects to detect
        
        Args:
            texts: List of object descriptions
        """
        self.detection_texts = texts
        print(f"Updated detection targets: {', '.join(texts)}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Save summary if output enabled
        if self.save_output and self.detection_history:
            summary = {
                "total_frames": self.frame_count,
                "total_detections": sum(len(r["detections"]) for r in self.detection_history),
                "detection_history": self.detection_history[-100:]  # Last 100 frames
            }
            summary_file = self.output_dir / "detection_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved detection summary to {summary_file}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Robot Camera Detector using NVIDIA NanoOWL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/owlvit-base-patch32",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="data/owl_image_encoder_patch32.engine",
        help="Path to TensorRT engine file"
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        help="Objects to detect (e.g., --texts person box forklift)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save detection results to files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for saved results"
    )
    
    args = parser.parse_args()
    
    # Load config if exists
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    
    # Create detector
    detector = RobotCameraDetector(
        model_name=config.get("model_name", args.model),
        image_encoder_engine=config.get("image_encoder_engine", args.engine),
        camera_index=config.get("camera_index", args.camera),
        detection_texts=config.get("detection_texts", args.texts),
        threshold=config.get("threshold", args.threshold),
        show_display=not args.no_display,
        save_output=args.save_output or config.get("save_output", False),
        output_dir=config.get("output_dir", args.output_dir)
    )
    
    # Run detector
    detector.run()


if __name__ == "__main__":
    main()

