from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from config import Config
from player_tracker import PlayerTracker
from ball_tracker import HybridBallTracker
from game_engine import GameEngine
from visualizer import VideoVisualizer
from utils import draw_label

SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # legs
    (5, 11), (6, 12), (5, 6),  # torso
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # face/head
]

class SportsHybridPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.player_tracker = PlayerTracker(
            pose_model_path=cfg.pose_model_path
        )
        self.active_player_ids = set()

        # 1. Load Calibration
        with open("court_calibration_13handles.json", "r") as f:
            calib = json.load(f)
            
        # 2. Extract screen corners (Points 0, 4, 24, 20)
        pts_dict = {p["id"]: p["image"] for p in calib["points_31"]}
        screen_corners = np.array([pts_dict[0], pts_dict[4], pts_dict[24], pts_dict[20]], dtype=np.float32)
        
        # Define real-world dimensions (6.10m x 13.40m)
        real_corners = np.array([[0,0], [6.10, 0], [6.10, 13.40], [0, 13.40]], dtype=np.float32)
        
        # 3. Create Homography Matrix
        self.H, _ = cv2.findHomography(screen_corners, real_corners)
        
        # 4. Create a Shapely Polygon for the court boundaries
        buffer_pixels = 250 
        far_buffer_pixels = 500 # Extra buffer for the far side to capture extremely deep players
        self.court_polygon = Polygon([
            (pts_dict[0][0] - buffer_pixels, pts_dict[0][1] - far_buffer_pixels),
            (pts_dict[4][0] + buffer_pixels, pts_dict[4][1] - far_buffer_pixels),
            (pts_dict[24][0] + buffer_pixels, pts_dict[24][1] + buffer_pixels),
            (pts_dict[20][0] - buffer_pixels, pts_dict[20][1] + buffer_pixels)
        ])

        self.heavy_ball_detector = None
        if cfg.ball_seed_detector_path:
            self.heavy_ball_detector = YOLO(cfg.ball_seed_detector_path)

        self.ball_tracker = HybridBallTracker(
            tracknet_model_path=cfg.tracknet_model_path,
            heatmap_threshold=cfg.tracknet_heatmap_threshold,
            heavy_detector=self.heavy_ball_detector,
            lost_reset=cfg.ball_lost_reset,
            conf_threshold=cfg.ball_conf_threshold,
        )

        self.game_engine = GameEngine(
            hit_radius=cfg.hit_radius,
            speed_jump_threshold=cfg.speed_jump_threshold,
            min_hit_gap_frames=cfg.min_hit_gap_frames,
            homography_matrix=self.H
        )
        
        self.visualizer = VideoVisualizer(self.H)

    def _smooth_ball_trajectory(self, timeline, max_gap=20):
        # timeline is list of dicts: {"frame_idx", "ball", "players"}
        # 1. Extract true detections
        true_anchors = []
        for i, data in enumerate(timeline):
            ball = data["ball"]
            if ball is not None and not ball.get("is_predicted", False):
                true_anchors.append((i, ball))
                
        # 2. Interpolate
        for idx in range(len(true_anchors) - 1):
            start_i, start_ball = true_anchors[idx]
            end_i, end_ball = true_anchors[idx+1]
            
            gap = end_i - start_i
            if 1 < gap <= max_gap:
                for j in range(start_i + 1, end_i):
                    alpha = (j - start_i) / gap
                    interp_x = start_ball["x"] * (1 - alpha) + end_ball["x"] * alpha
                    interp_y = start_ball["y"] * (1 - alpha) + end_ball["y"] * alpha
                    timeline[j]["ball"] = {
                        "frame_idx": timeline[j]["frame_idx"],
                        "x": interp_x,
                        "y": interp_y,
                        "conf": 1.0, 
                        "is_predicted": False 
                    }
            elif gap > max_gap:
                # Break trail for camera cuts
                for j in range(start_i + 1, end_i):
                    timeline[j]["ball"] = None
                    
        # Handle edges: erase predicted balls outside anchors
        if true_anchors:
            first_i = true_anchors[0][0]
            last_i = true_anchors[-1][0]
            for j in range(0, first_i):
                timeline[j]["ball"] = None
            for j in range(last_i + 1, len(timeline)):
                timeline[j]["ball"] = None
                
        return timeline

    def run(self):
        video_path = Path(self.cfg.video_path)
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = out_dir / f"{video_path.stem}_hybrid.mp4"
        out_json_path = out_dir / f"{video_path.stem}_hybrid.json"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        raw_timeline = []

        print("Starting PASS 1: Extraction...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            raw_players = self.player_tracker.update(frame, frame_idx)
            players = []
            
            for p in raw_players:
                x1, y1, x2, y2 = p["bbox"]
                feet_x = (x1 + x2) / 2
                feet_y = y2
                
                in_court = self.court_polygon.contains(Point(feet_x, feet_y))
                
                if not in_court and p["track_id"] not in self.active_player_ids:
                    if hasattr(self, 'last_active_positions'):
                        current_ids = {rp["track_id"] for rp in raw_players}
                        for old_id, last_pos in self.last_active_positions.items():
                            if old_id not in current_ids:
                                dist = ((feet_x - last_pos[0])**2 + (feet_y - last_pos[1])**2)**0.5
                                if dist < 200:
                                    self.active_player_ids.add(p["track_id"])
                                    break
                
                if in_court or p["track_id"] in self.active_player_ids:
                    if in_court:
                        self.active_player_ids.add(p["track_id"])
                        
                    pt = np.array([[[feet_x, feet_y]]], dtype=np.float32)
                    real_world_coords = cv2.perspectiveTransform(pt, self.H)[0][0]
                    p["real_world_x"] = float(real_world_coords[0])
                    p["real_world_y"] = float(real_world_coords[1])
                    
                    players.append(p)
                    
            self.last_active_positions = {p["track_id"]: ((p["bbox"][0]+p["bbox"][2])/2, p["bbox"][3]) for p in players}
            
            ball = self.ball_tracker.update(frame, frame_idx)
            
            raw_timeline.append({
                "frame_idx": frame_idx,
                "ball": ball,
                "players": players
            })

            if frame_idx % 60 == 0:
                print(f"Extraction: Processed frame {frame_idx}")

        cap.release()

        print("Starting PASS 2: Offline Smoothing...")
        smoothed_timeline = self._smooth_ball_trajectory(raw_timeline)

        print("Starting PASS 3: Game Engine & Rendering...")
        cap = cv2.VideoCapture(str(video_path))
        final_timeline = []

        for frame_record in smoothed_timeline:
            ret, frame = cap.read()
            if not ret: break
            
            ball = frame_record["ball"]
            players = frame_record["players"]
            f_idx = frame_record["frame_idx"]

            events = self.game_engine.update(f_idx, ball, players)
            vis = self.visualizer.render(frame, players, ball, events)
            
            writer.write(vis)

            final_timeline.append({
                "frame": f_idx,
                "ball": ball,
                "players": [
                    {
                        "track_id": p["track_id"],
                        "bbox": p["bbox"],
                        "keypoints": p.get("keypoints", []),
                        "missed": p.get("missed", 0),
                        "real_world_x": p.get("real_world_x"),
                        "real_world_y": p.get("real_world_y"),
                    }
                    for p in players
                ],
                "events": [
                    {
                        "frame_idx": e.frame_idx,
                        "event_type": e.event_type,
                        "details": e.details,
                    }
                    for e in events
                ],
            })

        cap.release()
        writer.release()

        output_data = {
            "metadata": {
                "fps": fps,
                "width": width,
                "height": height,
                "homography": self.H.tolist() if self.H is not None else None
            },
            "frames": final_timeline
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved video: {out_video_path}")
        print(f"Saved JSON : {out_json_path}")

        return str(out_video_path), str(out_json_path)