# G-Stat Quickstart

## 1. Clone the Repository
*This repository uses Git Large File Storage (LFS) to store AI checkpoints and high-resolution videos. You must install Git LFS first.*
```bash
git lfs install
git clone git@github.com:ganapathi1578/gstat.git
cd gstat
git lfs pull
```

## 2. Install Dependencies
```bash
conda create -n gstat_env python=3.10
conda activate gstat_env
pip install -r requirements.txt
```

## 3. Calibrate the Court
*Run this once per video to map the 2D court to 3D space.*
```bash
python dev/calibrate.py --video_path assets/videos/match.mp4
```
*Note: This will open a UI. Drag the handles to the court corners/net, press `[s]` to save, and `[q]` to quit. It will save `match_calib.json` automatically.*

## 4. Run the Pipeline
*Once calibrated, run the automated pipeline to generate the 3D data and 2D overlay.*
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
set KMP_DUPLICATE_LIB_OK=TRUE
python dev/main.py --video_path assets/videos/match.mp4
```

## 5. Launch the 3D WebGL Dashboard
*Start the local server to view the 3D cinematic replay.*
```bash
python dev/server.py
```
*Open `http://localhost:8000/app/` in your browser.*
