#!/usr/bin/env python3
"""
Example usage of RobotCameraDetector for robot integration
"""

from robot_camera_detector import RobotCameraDetector
import cv2
import time


def example_basic_usage():
    """Basic example of using the detector"""
    print("Example 1: Basic Usage")
    
    # Initialize detector
    detector = RobotCameraDetector(
        image_encoder_engine="data/owl_image_encoder_patch32.engine",
        detection_texts=["person", "box"],
        threshold=0.1,
        show_display=True
    )
    
    # Run detection loop
    detector.run()


def example_library_usage():
    """Example of using detector as a library"""
    print("Example 2: Library Usage")
    
    # Initialize detector (headless mode)
    detector = RobotCameraDetector(
        image_encoder_engine="data/owl_image_encoder_patch32.engine",
        detection_texts=["person", "object"],
        threshold=0.1,
        show_display=False
    )
    
    # Initialize camera
    detector.initialize_camera(width=640, height=480)
    
    # Alternative: Use OpenCV directly
    cap = cv2.VideoCapture(0)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections
            results = detector.get_detections(frame)
            
            # Process results
            if results["detections"]:
                print(f"\nFrame {frame_count}:")
                for det in results["detections"]:
                    print(f"  - {det['label']}: confidence={det['confidence']:.2f}, "
                          f"center={det['center']}, bbox={det['bbox']}")
            
            frame_count += 1
            
            # Exit after 100 frames for demo
            if frame_count >= 100:
                break
            
            time.sleep(0.1)  # Small delay
            
    finally:
        cap.release()
        detector.cleanup()


def example_robot_integration():
    """Example integration with robot control logic"""
    print("Example 3: Robot Integration")
    
    class SimpleRobotController:
        def __init__(self):
            self.detector = RobotCameraDetector(
                image_encoder_engine="data/owl_image_encoder_patch32.engine",
                detection_texts=["person", "obstacle", "box"],
                threshold=0.2,
                show_display=False
            )
            self.detector.initialize_camera()
            self.running = True
        
        def process_detections(self, results):
            """Process detection results and make decisions"""
            person_detected = False
            obstacles = []
            
            for det in results["detections"]:
                if det["label"] == "person":
                    person_detected = True
                    print(f"Person detected at {det['center']}")
                    # Stop robot or take evasive action
                    self.stop_robot()
                elif det["label"] == "obstacle":
                    obstacles.append(det)
                    print(f"Obstacle detected at {det['center']}")
                    # Calculate avoidance path
                    self.avoid_obstacle(det)
            
            return person_detected, obstacles
        
        def stop_robot(self):
            """Stop robot movement"""
            print("  -> Stopping robot")
            # Add your robot stop logic here
        
        def avoid_obstacle(self, obstacle_detection):
            """Avoid obstacle based on detection"""
            center_x, center_y = obstacle_detection["center"]
            frame_width = 640  # Adjust based on your camera
        
            if center_x < frame_width / 3:
                print("  -> Obstacle on left, turning right")
            elif center_x > 2 * frame_width / 3:
                print("  -> Obstacle on right, turning left")
            else:
                print("  -> Obstacle ahead, backing up")
        
        def run(self):
            """Main robot control loop"""
            cap = cv2.VideoCapture(0)
            
            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Get detections
                    results = self.detector.get_detections(frame)
                    
                    # Process and react to detections
                    self.process_detections(results)
                    
                    # Small delay
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nStopping robot...")
            finally:
                cap.release()
                self.detector.cleanup()
    
    # Run robot controller
    robot = SimpleRobotController()
    robot.run()


def example_dynamic_queries():
    """Example of dynamically changing detection queries"""
    print("Example 4: Dynamic Query Updates")
    
    detector = RobotCameraDetector(
        image_encoder_engine="data/owl_image_encoder_patch32.engine",
        detection_texts=["person"],
        threshold=0.1,
        show_display=False
    )
    
    detector.initialize_camera()
    cap = cv2.VideoCapture(0)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Change detection targets every 100 frames
            if frame_count % 100 == 0:
                if frame_count % 200 == 0:
                    detector.update_detection_texts(["person", "box"])
                else:
                    detector.update_detection_texts(["person", "forklift"])
            
            results = detector.get_detections(frame)
            
            if results["detections"]:
                print(f"Frame {frame_count}: {len(results['detections'])} detections")
            
            frame_count += 1
            
            if frame_count >= 300:
                break
                
    finally:
        cap.release()
        detector.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
    else:
        example_num = 1
    
    examples = {
        1: example_basic_usage,
        2: example_library_usage,
        3: example_robot_integration,
        4: example_dynamic_queries
    }
    
    if example_num in examples:
        examples[example_num]()
    else:
        print("Available examples:")
        print("  1: Basic Usage")
        print("  2: Library Usage")
        print("  3: Robot Integration")
        print("  4: Dynamic Query Updates")
        print("\nUsage: python3 example_usage.py <example_number>")

