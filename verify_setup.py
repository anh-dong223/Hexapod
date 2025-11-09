#!/usr/bin/env python3
"""
Verification script to check if Robot Camera Detector is properly set up
Run this on your Jetson Nano Orin to verify the installation
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 6:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need Python 3.6+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    dependencies = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'numpy': 'numpy'
    }
    all_ok = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed (pip3 install {package})")
            all_ok = False
    return all_ok

def check_nanoowl():
    """Check if nanoowl is installed"""
    print("\nChecking nanoowl...")
    try:
        import nanoowl
        print("✓ nanoowl installed")
        try:
            from nanoowl.owl_predictor import OwlPredictor
            print("✓ OwlPredictor can be imported")
            return True
        except ImportError as e:
            print(f"✗ Cannot import OwlPredictor: {e}")
            return False
    except ImportError:
        print("✗ nanoowl NOT installed")
        print("  Install with: git clone https://github.com/NVIDIA-AI-IOT/nanoowl")
        print("  cd nanoowl && python3 setup.py develop --user")
        return False

def check_engine_file():
    """Check if TensorRT engine file exists"""
    print("\nChecking TensorRT engine file...")
    possible_paths = [
        "data/owl_image_encoder_patch32.engine",
        "../nanoowl/data/owl_image_encoder_patch32.engine",
        "nanoowl/data/owl_image_encoder_patch32.engine",
        os.path.expanduser("~/nanoowl/data/owl_image_encoder_patch32.engine"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Engine file found: {abs_path} ({size_mb:.1f} MB)")
            return True, abs_path
    
    print("✗ Engine file NOT found")
    print("  Build it with: python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine")
    return False, None

def check_config():
    """Check if config.json exists and is valid"""
    print("\nChecking config.json...")
    if os.path.exists("config.json"):
        try:
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)
            print("✓ config.json exists and is valid")
            if "image_encoder_engine" in config:
                engine_path = config["image_encoder_engine"]
                if os.path.exists(engine_path):
                    print(f"✓ Engine path in config is valid: {engine_path}")
                else:
                    print(f"⚠ Engine path in config not found: {engine_path}")
            return True
        except json.JSONDecodeError:
            print("✗ config.json is invalid JSON")
            return False
    else:
        print("⚠ config.json not found (will use defaults)")
        return True

def check_camera():
    """Check if camera is available"""
    print("\nChecking camera...")
    try:
        import cv2
        if os.path.exists("/dev/video0"):
            print("✓ Camera device found: /dev/video0")
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✓ Camera can be opened")
                cap.release()
                return True
            else:
                print("⚠ Camera device exists but cannot be opened")
                cap.release()
                return False
        else:
            # Check for other video devices
            import glob
            video_devices = glob.glob("/dev/video*")
            if video_devices:
                print(f"⚠ /dev/video0 not found, but found: {', '.join(video_devices)}")
                return False
            else:
                print("✗ No camera devices found")
                return False
    except ImportError:
        print("⚠ Cannot check camera (opencv not installed)")
        return False

def check_project_files():
    """Check if all project files exist"""
    print("\nChecking project files...")
    required_files = [
        "robot_camera_detector.py",
        "config.json",
        "requirements.txt",
        "setup.sh",
        "README.md"
    ]
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT found")
            all_ok = False
    return all_ok

def main():
    """Run all checks"""
    print("=" * 50)
    print("Robot Camera Detector - Setup Verification")
    print("=" * 50)
    print()
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Project Files": check_project_files(),
        "Config File": check_config(),
        "NanoOWL": check_nanoowl(),
        "Engine File": check_engine_file()[0],
        "Camera": check_camera(),
    }
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:20s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All checks passed! You're ready to use the detector.")
        print("  Run: python3 robot_camera_detector.py")
        return 0
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("  Run: ./setup.sh to help with setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())

