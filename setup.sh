#!/bin/bash
# Setup script for Robot Camera Detector

set -e

echo "========================================="
echo "Robot Camera Detector Setup"
echo "========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Check Python version
echo "Checking Python version..."
python3 --version

# Install basic dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if nanoowl is installed
echo "Checking for nanoowl installation..."
if ! python3 -c "import nanoowl" 2>/dev/null; then
    echo "nanoowl not found. Please install it:"
    echo "  1. git clone https://github.com/NVIDIA-AI-IOT/nanoowl"
    echo "  2. cd nanoowl"
    echo "  3. python3 setup.py develop --user"
    echo "  4. python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine"
    echo ""
    echo "Would you like to install nanoowl now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        echo "Cloning nanoowl..."
        if [ ! -d "../nanoowl" ]; then
            cd ..
            git clone https://github.com/NVIDIA-AI-IOT/nanoowl
            cd nanoowl
            echo "Installing nanoowl..."
            python3 setup.py develop --user
            echo "Building TensorRT engine..."
            mkdir -p data
            python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
            cd ../robott
        else
            echo "nanoowl directory already exists at ../nanoowl"
            echo "Building TensorRT engine..."
            cd ../nanoowl
            mkdir -p data
            python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
            cd ../robott
        fi
    fi
else
    echo "nanoowl is installed!"
fi

# Check for engine file
echo "Checking for TensorRT engine file..."
ENGINE_PATH=""
if [ -f "../nanoowl/data/owl_image_encoder_patch32.engine" ]; then
    ENGINE_PATH="../nanoowl/data/owl_image_encoder_patch32.engine"
    echo "Found engine at: $ENGINE_PATH"
elif [ -f "data/owl_image_encoder_patch32.engine" ]; then
    ENGINE_PATH="data/owl_image_encoder_patch32.engine"
    echo "Found engine at: $ENGINE_PATH"
else
    echo "Engine file not found. Please build it:"
    echo "  python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine"
fi

# Update config if engine found
if [ -n "$ENGINE_PATH" ]; then
    echo "Updating config.json with engine path..."
    # Use python to update JSON (more reliable than sed)
    python3 << EOF
import json
import os

config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert to absolute path
    engine_abs = os.path.abspath("$ENGINE_PATH")
    config["image_encoder_engine"] = engine_abs
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated config.json with engine path: {engine_abs}")
else:
    print("config.json not found, skipping update")
EOF
fi

# Check for camera
echo "Checking for camera..."
if [ -e /dev/video0 ]; then
    echo "Camera found at /dev/video0"
else
    echo "Warning: No camera found at /dev/video0"
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "No video devices found"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config.json to customize detection targets"
echo "  2. Run: python3 robot_camera_detector.py"
echo "  3. Or see examples: python3 example_usage.py"
echo ""

