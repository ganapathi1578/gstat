from dataclasses import dataclass

@dataclass
class Config:
    video_path: str
    output_dir: str = "./output"

    # Player pose model (yolov11n-pose.pt recommended, auto-downloaded if missing)
    pose_model_path: str = "models/yolo26n-pose.pt"

    # Optional: pretrained YOLO shuttle detector for seeding/recovery
    ball_seed_detector_path: str | None = None

    # ── TrackNet shuttle detector ───────────────────────────────────────────
    tracknet_model_path: str = "models/ckpts/TrackNet_best.pt"
    tracknet_heatmap_threshold: int  = 64   # 0-255; lower = more sensitive

    # Ball tracker (Kalman)
    ball_lost_reset:    int   = 10
    ball_conf_threshold: float = 0.25

    # Player tracker
    player_match_dist:  float = 80.0
    player_max_missed:  int   = 10

    # Game engine
    hit_radius:             float = 60.0
    speed_jump_threshold:   float = 18.0
    min_hit_gap_frames:     int   = 6

    # Visualization
    draw_prediction_when_missing: bool = True