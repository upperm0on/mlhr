# MLHR - MediaPipe Hand Recognition & Accessibility Tools

A collection of computer vision and accessibility tools using MediaPipe for hand tracking and gesture recognition, designed to run on multi-core systems for optimal performance.

## 🚀 Features

### Hand Tracking Scripts
- **Real-time hand landmark detection** using MediaPipe
- **Multi-core parallel processing** for maximum performance
- **Handedness-aware finger detection** (left/right hand thumb logic)
- **Exponential moving average smoothing** for stable landmark tracking
- **Visual feedback** with glowing fingertips and FPS overlay
- **Palm-facing camera detection** for natural interaction

### Accessibility Integration
- **AT-SPI integration** for desktop accessibility control
- **Focus element tracking** for UI automation
- **Gesture-to-action mapping** foundation

## 📁 Project Structure

```
mlhr/
├── hand_tracking.py              # Original hand tracking implementation
├── hand_tracking_parallel.py     # Multi-core optimized version
├── accessibility.py              # AT-SPI accessibility controller
├── requirements.txt              # Python dependencies
├── get-pip.py                   # pip bootstrapper
└── README.md                    # This file
```

## 🛠 Installation

### Prerequisites
- Python 3.10+
- Linux (tested on Kali Linux)
- Webcam/camera device

### Setup
1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/mlhr
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or if using a virtual environment:
   ```bash
   # Activate your virtual environment first
   pip install -r requirements.txt
   ```

### Required Packages
- `mediapipe==0.10.14` - Google's ML pipeline for hand tracking
- `opencv-python==4.10.0.84` - Computer vision library
- `numpy` - Numerical computing
- `pyautogui` - GUI automation (for accessibility features)

## 🎯 Usage

### Hand Tracking (Parallel Version - Recommended)
```bash
python hand_tracking_parallel.py
```

**Features:**
- Automatically detects and utilizes all CPU cores
- Threaded camera capture for reduced latency
- Parallel MediaPipe inference
- Async drawing for smooth UI
- Exponential moving average landmark smoothing
- 1280x720 HD capture with 1.5x display scaling

**Controls:**
- Press `q` to quit
- Shows FPS, finger count per hand, and visual landmarks

### Hand Tracking (Original Version)
```bash
python hand_tracking.py
```

**Features:**
- Single-threaded implementation
- Basic landmark smoothing
- Core functionality without parallel optimization

### Accessibility Controller
```bash
python accessibility.py
```

**Features:**
- Monitors focused desktop elements using AT-SPI
- Foundation for gesture-to-UI-action mapping
- Currently displays focused application names

## 🔧 Configuration

### Hand Tracking Parameters (in script constants)
```python
CAM_INDEX = 0                    # Camera device index
MODEL_COMPLEXITY = 1            # MediaPipe model complexity (0-2)
MAX_NUM_HANDS = 2               # Maximum hands to track
MIN_DET_CONF = 0.5              # Detection confidence threshold
MIN_TRACK_CONF = 0.5            # Tracking confidence threshold
SMOOTHING_ALPHA = 0.3           # EMA smoothing factor (0.1-0.5)
FRAME_WIDTH = 1280              # Camera resolution width
FRAME_HEIGHT = 720              # Camera resolution height
DISPLAY_SCALE = 1.5             # UI display scaling factor
```

### Performance Tuning
- **CPU Cores**: Automatically detected and utilized
- **OpenCV Threads**: Set to use all available cores
- **Queue Size**: Configured for latest-frame processing
- **ThreadPool**: Sized to match CPU core count

## 🎨 Visual Features

### Hand Landmarks
- **21 landmark points** per detected hand
- **Hand connections** visualization
- **Color-coded overlays** for different hands

### Finger Detection
- **Thumb**: Handedness-aware (left/right logic)
- **Other fingers**: Y-coordinate based extension detection
- **Real-time counting**: "Left 3/5" or "Right 4/5" display

### Special Effects
- **Glowing fingertips**: Cyan-colored circles at finger tips
- **FPS counter**: Real-time performance monitoring
- **Status messages**: "No hands detected" when appropriate

## 🏗 Architecture

### Parallel Processing Pipeline
1. **Camera Thread**: Continuous frame capture → queue
2. **Inference Thread Pool**: Parallel MediaPipe processing
3. **Drawing Thread**: Async rendering and display
4. **Main Coordinator**: Queue management and thread synchronization

### Data Flow
```
Camera → Frame Queue → ThreadPool(Inference) → Display Queue → Drawing Thread → Screen
```

## 🔍 Technical Details

### Landmark Smoothing
- **Exponential Moving Average (EMA)**: Reduces jitter while maintaining responsiveness
- **Per-hand tracking**: Independent smoothers for multiple hands
- **Real-time updates**: No frame buffering delays

### Handedness Detection
- **MediaPipe classification**: Automatic left/right determination
- **Thumb logic**: Mirror-image detection for natural thumb extension
- **Confidence thresholding**: Fallback detection for uncertain classifications

### Performance Optimizations
- **Multi-threading**: Parallel camera, inference, and drawing
- **Latest-frame processing**: Always work with most recent camera input
- **Memory efficiency**: Minimal image copying and queue management
- **CPU utilization**: All cores engaged for maximum throughput

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'mediapipe'"**
```bash
pip install mediapipe==0.10.14
```

**Camera not found**
- Check `CAM_INDEX` in script (usually 0 for built-in webcam)
- Ensure camera permissions are granted

**Low FPS**
- Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` for lower resolution
- Increase `SMOOTHING_ALPHA` for less processing
- Check CPU usage and available cores

**Hand detection issues**
- Ensure good lighting
- Position hands clearly in frame
- Adjust `MIN_DET_CONF` if needed
- Verify palm is facing camera

### Virtual Environment
If using a virtual environment, make sure it's activated before installing dependencies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on multi-core systems
4. Submit a pull request

## 📄 License

This project is open-source. See individual script headers for license information.

## 🔗 Dependencies

- [MediaPipe](https://mediapipe.dev/) - ML pipeline framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Scientific computing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - GUI automation
- [AT-SPI](https://wiki.gnome.org/Accessibility/AT-SPI2) - Accessibility framework

---

**Note**: This project is designed for Linux environments and has been tested on Kali Linux. Performance may vary on different systems and hardware configurations.
# mlhr
