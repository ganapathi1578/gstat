from __future__ import annotations

import argparse
import sys
from pathlib import Path
from config import Config
from pipeline import SportsHybridPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton hybrid tracking pipeline")
    parser.add_argument("--video_path",    type=str, required=True)
    parser.add_argument("--output_dir",    type=str, default="./output")
    parser.add_argument("--pose_model_path",          type=str,   default="models/yolo26n-pose.pt")
    parser.add_argument("--ball_seed_detector_path",  type=str,   default=None)
    parser.add_argument("--tracknet_model_path",      type=str,   default="models/ckpts/TrackNet_best.pt")
    parser.add_argument("--tracknet_heatmap_threshold",type=int,  default=64)
    parser.add_argument("--ball_lost_reset",           type=int,  default=10)
    parser.add_argument("--ball_conf_threshold",       type=float,default=0.25)
    parser.add_argument("--hit_radius",                type=float,default=60.0)
    parser.add_argument("--speed_jump_threshold",      type=float,default=18.0)
    parser.add_argument("--min_hit_gap_frames",        type=int,  default=6)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    calibration_path = video_path.parent / f"{video_path.stem}_calib.json"
    if not calibration_path.exists():
        print(f"\n[ERROR] Missing calibration file: {calibration_path}")
        print(f"Please run the calibration tool first:\n    python dev/calibrate.py --video_path \"{video_path}\"\n")
        sys.exit(1)

    cfg = Config(
        video_path                  = args.video_path,
        calibration_path            = str(calibration_path),
        output_dir                  = args.output_dir,
        pose_model_path             = args.pose_model_path,
        ball_seed_detector_path     = args.ball_seed_detector_path,
        tracknet_model_path         = args.tracknet_model_path,
        tracknet_heatmap_threshold  = args.tracknet_heatmap_threshold,
        ball_lost_reset             = args.ball_lost_reset,
        ball_conf_threshold         = args.ball_conf_threshold,
        hit_radius                  = args.hit_radius,
        speed_jump_threshold        = args.speed_jump_threshold,
        min_hit_gap_frames          = args.min_hit_gap_frames,
    )

    pipeline = SportsHybridPipeline(cfg)
    out_video, out_json = pipeline.run()
    print(f"\nDone!\n  Video : {out_video}\n  JSON  : {out_json}")


if __name__ == "__main__":
    main()