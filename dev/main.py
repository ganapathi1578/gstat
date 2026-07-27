from __future__ import annotations

import argparse
from config import Config
from pipeline import SportsHybridPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton hybrid tracking pipeline")
    parser.add_argument("--video_path",    type=str, required=True)
    parser.add_argument("--output_dir",    type=str, default="./output")
    parser.add_argument("--pose_model_path",          type=str,   default="models/yolov11n-pose.pt")
    parser.add_argument("--ball_seed_detector_path",  type=str,   default=None)
    parser.add_argument("--tracknet_model_path",      type=str,   default="models/best_model_base_new_data_e28.keras")
    parser.add_argument("--tracknet_heatmap_threshold",type=int,  default=64)
    parser.add_argument("--ball_lost_reset",           type=int,  default=10)
    parser.add_argument("--ball_conf_threshold",       type=float,default=0.25)
    parser.add_argument("--hit_radius",                type=float,default=60.0)
    parser.add_argument("--speed_jump_threshold",      type=float,default=18.0)
    parser.add_argument("--min_hit_gap_frames",        type=int,  default=6)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        video_path                  = args.video_path,
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