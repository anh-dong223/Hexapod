# Jetson Setup Instructions

## Why Jetson-Only?

This project uses **NVIDIA NanoOWL**, which requires:
- **TensorRT** - Only available on NVIDIA Jetson devices
- **Jetson PyTorch** - Special builds optimized for ARM architecture
- **CUDA** - Requires NVIDIA GPU hardware (Jetson has integrated GPU)

## Setup on Jetson Nano Orin

### Step 1: Connect to Your Jetson

If developing on Mac/PC, connect via SSH:

```bash
# On Mac/PC
ssh jetson@<jetson-ip-address>
# Or
ssh <your-username>@<jetson-ip-address>
```

### Step 2: Clone Repository on Jetson

```bash
# On Jetson device
cd ~
git clone https://github.com/anh-dong223/Hexapod.git
cd Hexapod
```

### Step 3: Run Setup Script

```bash
# This will check if you're on Jetson and install everything
./setup.sh
```

### Step 4: Verify Setup

```bash
python3 verify_setup.py
```

### Step 5: Run Detector

```bash
python3 robot_camera_detector.py
```

## Development on Mac/PC + Jetson

### Option 1: Edit Locally, Run on Jetson

1. **Edit code on Mac/PC**
   ```bash
   # On Mac/PC
   git clone https://github.com/anh-dong223/Hexapod.git
   # Edit files in your favorite editor
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```

3. **Pull on Jetson**
   ```bash
   # On Jetson
   cd ~/Hexapod
   git pull
   python3 robot_camera_detector.py
   ```

### Option 2: Direct SSH File Transfer

1. **Edit on Mac/PC**
2. **Transfer files via SCP**
   ```bash
   # On Mac/PC
   scp robot_camera_detector.py jetson@<jetson-ip>:~/Hexapod/
   ```

3. **Run on Jetson via SSH**
   ```bash
   # On Mac/PC
   ssh jetson@<jetson-ip> "cd ~/Hexapod && python3 robot_camera_detector.py"
   ```

### Option 3: VS Code Remote SSH

1. Install **Remote - SSH** extension in VS Code
2. Connect to Jetson
3. Edit files directly on Jetson
4. Run and debug on Jetson

## Common Issues

### "ModuleNotFoundError: No module named 'torch'"

**On Mac/PC:** This is expected! PyTorch for Jetson is different from desktop PyTorch.

**Solution:** Install on Jetson device, not Mac/PC.

### "TensorRT not found"

**On Mac/PC:** TensorRT is Jetson-specific.

**Solution:** This must be run on Jetson device.

### "CUDA not available"

**On Mac/PC:** Mac doesn't have CUDA support.

**Solution:** Run on Jetson device which has CUDA built-in.

## Testing Without Jetson

Unfortunately, you **cannot** test the full detection pipeline without a Jetson device. However, you can:

1. **Test code syntax** on Mac/PC
   ```bash
   python3 -m py_compile robot_camera_detector.py
   ```

2. **Review and edit code** on Mac/PC

3. **Use mock/dummy data** for logic testing (without actual detection)

4. **Run full tests on Jetson** only

## Getting Jetson IP Address

If you need to find your Jetson's IP address:

```bash
# On Jetson
hostname -I
# Or
ifconfig
```

## Next Steps

Once setup is complete on Jetson:
1. Test camera connection
2. Configure detection targets in `config.json`
3. Run detector: `python3 robot_camera_detector.py`
4. Integrate with your robot control system

