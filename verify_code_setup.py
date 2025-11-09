#!/usr/bin/env python3
"""
Verify that all code is properly set up and ready for Jetson deployment
This can be run on Mac/PC to verify code quality before deploying to Jetson
"""

import os
import json
import ast
import sys
from pathlib import Path

def check_file_exists(filename, description):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✓ {description}: {filename}")
        return True
    else:
        print(f"✗ {description}: {filename} NOT FOUND")
        return False

def check_python_syntax(filename):
    """Check if Python file has valid syntax"""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False

def check_imports_match_requirements():
    """Check if imports in code match requirements.txt"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    # Read requirements.txt
    required_packages = {}
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name (handle >=, ==, etc.)
                    package = line.split('>=')[0].split('==')[0].split('<=')[0].strip()
                    required_packages[package.lower()] = package
        print(f"✓ Found {len(required_packages)} packages in requirements.txt")
    else:
        print("✗ requirements.txt not found")
        return False
    
    # Check imports in main script
    imports_needed = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'Pillow': 'Pillow',
        'numpy': 'numpy',
        'nanoowl': 'nanoowl (install from source)'
    }
    
    all_ok = True
    with open("robot_camera_detector.py", 'r') as f:
        content = f.read()
        for module, package in imports_needed.items():
            if f"import {module}" in content or f"from {module}" in content:
                if module == 'nanoowl':
                    print(f"✓ {module} import found (install from source)")
                elif package.lower() in [p.lower() for p in required_packages.values()]:
                    print(f"✓ {module} import matches requirements.txt")
                else:
                    print(f"⚠ {module} imported but not in requirements.txt (may be Jetson-specific)")
    
    return True

def check_config_file():
    """Check if config.json is valid"""
    print("\n" + "="*60)
    print("Checking Configuration")
    print("="*60)
    
    if not os.path.exists("config.json"):
        print("✗ config.json not found")
        return False
    
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        required_keys = ["model_name", "image_encoder_engine", "camera_index", "detection_texts", "threshold"]
        all_present = True
        
        for key in required_keys:
            if key in config:
                print(f"✓ {key}: {config[key]}")
            else:
                print(f"✗ Missing key: {key}")
                all_present = False
        
        return all_present
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False

def check_script_structure():
    """Check if main script has required components"""
    print("\n" + "="*60)
    print("Checking Script Structure")
    print("="*60)
    
    with open("robot_camera_detector.py", 'r') as f:
        content = f.read()
    
    checks = {
        "RobotCameraDetector class": "class RobotCameraDetector" in content,
        "OwlPredictor import": "from nanoowl.owl_predictor import OwlPredictor" in content,
        "process_frame method": "def process_frame" in content,
        "get_detections method": "def get_detections" in content,
        "run method": "def run" in content,
        "main function": "def main()" in content,
        "if __name__ == '__main__'": "__name__" in content and "__main__" in content,
        "Error handling for nanoowl": "except ImportError" in content,
    }
    
    all_ok = True
    for check, result in checks.items():
        if result:
            print(f"✓ {check}")
        else:
            print(f"✗ {check} MISSING")
            all_ok = False
    
    return all_ok

def check_file_structure():
    """Check if all necessary files exist"""
    print("\n" + "="*60)
    print("Checking File Structure")
    print("="*60)
    
    required_files = {
        "robot_camera_detector.py": "Main detection script",
        "config.json": "Configuration file",
        "requirements.txt": "Python dependencies",
        "setup.sh": "Setup script",
        "verify_setup.py": "Setup verification script",
        "example_usage.py": "Usage examples",
        "README.md": "Documentation",
        "QUICKSTART.md": "Quick start guide",
        ".gitignore": "Git ignore file"
    }
    
    all_present = True
    for filename, description in required_files.items():
        if not check_file_exists(filename, description):
            all_present = False
    
    return all_present

def check_executable_permissions():
    """Check if scripts have executable permissions"""
    print("\n" + "="*60)
    print("Checking File Permissions")
    print("="*60)
    
    scripts = ["robot_camera_detector.py", "setup.sh", "example_usage.py", "verify_setup.py"]
    all_ok = True
    
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✓ {script} is executable")
            else:
                print(f"⚠ {script} is not executable (run: chmod +x {script})")
                all_ok = False
    
    return all_ok

def main():
    """Run all verification checks"""
    print("="*60)
    print("Robot Camera Detector - Code Setup Verification")
    print("="*60)
    print("\nThis script verifies that all code is properly set up")
    print("before deploying to Jetson device.\n")
    
    results = {}
    
    # Check file structure
    results["File Structure"] = check_file_structure()
    
    # Check Python syntax
    print("\n" + "="*60)
    print("Checking Python Syntax")
    print("="*60)
    python_files = ["robot_camera_detector.py", "example_usage.py", "verify_setup.py"]
    syntax_ok = True
    for py_file in python_files:
        if os.path.exists(py_file):
            if check_python_syntax(py_file):
                print(f"✓ {py_file}: Valid syntax")
            else:
                print(f"✗ {py_file}: Invalid syntax")
                syntax_ok = False
    results["Python Syntax"] = syntax_ok
    
    # Check dependencies
    results["Dependencies"] = check_imports_match_requirements()
    
    # Check config
    results["Configuration"] = check_config_file()
    
    # Check script structure
    results["Script Structure"] = check_script_structure()
    
    # Check permissions
    results["File Permissions"] = check_executable_permissions()
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:20s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All code checks passed!")
        print("\nCode is ready for deployment to Jetson device.")
        print("\nNext steps:")
        print("  1. Push to GitHub (if not already done)")
        print("  2. Clone on Jetson: git clone https://github.com/anh-dong223/Hexapod.git")
        print("  3. Run setup on Jetson: ./setup.sh")
        return 0
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

