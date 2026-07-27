# G-Stat: AI Sports Analytics & 3D WebGL Replay Engine

G-Stat is a state-of-the-art computer vision pipeline and 3D WebGL rendering engine designed for advanced sports analytics (specifically Badminton). It leverages deep learning for multi-object tracking, custom camera calibration via Homography, and a Two-Pass Offline Smoothing architecture to deliver broadcast-quality 3D cinematic replays and live physics calculations.

## Features

- **Hybrid Deep Learning Pipeline:** Utilizes YOLOv11 for human pose estimation and bounding box tracking alongside a custom TrackNet architecture for high-speed, small-object (shuttlecock) detection.
- **Two-Pass Offline Smoothing:** Eliminates jitter and tracking artifacts caused by motion blur or occlusions. The pipeline mathematically interpolates missing frames using spatial smoothing algorithms to produce flawless object trajectories.
- **Dynamic Camera Calibration:** Projects 2D pixel coordinates into real-world 3D court coordinates using Homography matrices mapped to standard Badminton court dimensions.
- **Physics & Event Engine:** Automatically detects interactions (e.g., racket hits) by calculating spatial distances between the shuttlecock and the players' wrists, triggering events that calculate impact speed in km/h.
- **3D WebGL Broadcast Engine:** A decoupled Three.js web application that visualizes the JSON output of the python pipeline. Features interactive 3D camera controls, billboarded skeletal player representations, dynamic ball shadows, and a live Heads-Up Display (HUD) for hit speeds.

## Architecture

1. **Python Computer Vision Engine (`dev/main.py`)**
   - **Trackers (`dev/ball_tracker.py`, `dev/player_tracker.py`):** Handles object detection, spatial memory ID recovery (handling players stepping out of bounds), and Kalman filtering.
   - **Pipeline (`dev/pipeline.py`):** Orchestrates the Two-Pass logic. Extracts raw data, applies the `_smooth_ball_trajectory` algorithm, and serializes the final clean data to JSON.
   - **Game Engine (`dev/game_engine.py`):** Calculates physics events based on the smoothed data.

2. **Web Application (`app/index.html`)**
   - Built with raw HTML/JS and Three.js.
   - Loads the emitted `_hybrid.json` and syncs it with the H.264 video feed for a side-by-side synchronized dashboard.

## Getting Started

### Prerequisites

- Python 3.9+
- Conda (Recommended)
- Node.js / SimpleHTTP Server (for running the WebGL App)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:ganapathi1578/gstat.git
   cd gstat
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n gstat_env python=3.10
   conda activate gstat_env
   pip install -r requirements.txt
   ```

3. Download required weights:
   - Place your trained TrackNet weights in `models/ckpts/TrackNet_best.pt`
   - Place your YOLOv11 pose weights in `models/yolo26n-pose.pt`

### Usage

**1. Run the Computer Vision Pipeline**
```bash
python dev/main.py --video_path path/to/your/video.mp4 --output_dir ./output
```
*This will generate a `_hybrid.mp4` video with 2D overlays and a `_hybrid.json` file containing all spatial tracking data and events.*

**2. Launch the 3D Replay Dashboard**
```bash
python dev/server.py
```
*Navigate to `http://localhost:8000/app/` in your browser. Ensure that `app/index.html` is pointing to the correct JSON and video file generated in step 1.*

## License

This project is proprietary and confidential.
