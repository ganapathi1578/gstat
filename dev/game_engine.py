from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

from utils import distance


import numpy as np
import cv2

@dataclass
class GameEvent:
    frame_idx: int
    event_type: str
    details: dict = field(default_factory=dict)


class GameEngine:
    """
    Soft-constraint event engine:
    - hit detection from ball-wrist proximity + speed jump
    - no hard rejection of weird physics
    """

    def __init__(
        self,
        hit_radius: float = 60.0,
        speed_jump_threshold: float = 18.0,
        min_hit_gap_frames: int = 6,
        homography_matrix: np.ndarray = None,
        fps: float = 30.0
    ):
        self.hit_radius = hit_radius
        self.speed_jump_threshold = speed_jump_threshold
        self.min_hit_gap_frames = min_hit_gap_frames
        self.H = homography_matrix
        self.fps = fps

        self.prev_ball = None
        self.prev_speed = None
        self.last_hit_frame = -10**9
        self.events: List[GameEvent] = []
        
        self.ball_history = []

    def _ball_speed(self, history):
        if len(history) < 3 or self.H is None:
            return None
            
        b1 = history[0]
        b2 = history[-1]
        frames_diff = b2["frame_idx"] - b1["frame_idx"]
        if frames_diff == 0: return None
        
        # Transform to real world coords
        pt1 = np.array([[[b1["x"], b1["y"]]]], dtype=np.float32)
        pt2 = np.array([[[b2["x"], b2["y"]]]], dtype=np.float32)
        
        rw1 = cv2.perspectiveTransform(pt1, self.H)[0][0]
        rw2 = cv2.perspectiveTransform(pt2, self.H)[0][0]
        
        dist_meters = float(np.linalg.norm(rw2 - rw1))
        
        # Perspective multiplier (height compensation for flying objects)
        height_compensation = 0.8
        dist_meters *= height_compensation
        
        meters_per_sec = (dist_meters / frames_diff) * self.fps
        km_per_hour = meters_per_sec * 3.6
        return km_per_hour

    def update(self, frame_idx: int, ball_state: dict | None, player_tracks: list):
        if ball_state is None:
            self.prev_ball = None if self.prev_ball is None else self.prev_ball
            return []

        ball_state["frame_idx"] = frame_idx
        self.ball_history.append(ball_state)
        if len(self.ball_history) > 5:
            self.ball_history.pop(0)

        current_speed = self._ball_speed(self.ball_history)

        triggered = []

        # Hit detection: ball very close to a wrist and speed changes sharply
        if self.prev_ball is not None and current_speed is not None and self.prev_speed is not None:
            # We now use the pixel speed for hit detection as it's more stable for thresholds
            pix_speed1 = distance((self.prev_ball["x"], self.prev_ball["y"]), (self.ball_history[-2]["x"], self.ball_history[-2]["y"]))
            pix_speed2 = distance((ball_state["x"], ball_state["y"]), (self.ball_history[-2]["x"], self.ball_history[-2]["y"]))
            speed_jump = abs(pix_speed2 - pix_speed1)

            if (frame_idx - self.last_hit_frame) >= self.min_hit_gap_frames:
                nearest_wrist_dist = 10**9
                nearest_player = None
                nearest_wrist_side = None

                for p in player_tracks:
                    kpts = p.get("keypoints", [])
                    left = kpts[9] if len(kpts) >= 11 and kpts[9][0] > 0 else None
                    right = kpts[10] if len(kpts) >= 11 and kpts[10][0] > 0 else None
                    ball_xy = (ball_state["x"], ball_state["y"])

                    if left is not None:
                        d = distance(ball_xy, left)
                        if d < nearest_wrist_dist:
                            nearest_wrist_dist = d
                            nearest_player = p["track_id"]
                            nearest_wrist_side = "left"

                    if right is not None:
                        d = distance(ball_xy, right)
                        if d < nearest_wrist_dist:
                            nearest_wrist_dist = d
                            nearest_player = p["track_id"]
                            nearest_wrist_side = "right"

                if nearest_wrist_dist <= self.hit_radius and speed_jump >= self.speed_jump_threshold:
                    ev = GameEvent(
                        frame_idx=frame_idx,
                        event_type="hit",
                        details={
                            "player_id": nearest_player,
                            "wrist": nearest_wrist_side,
                            "ball_conf": ball_state.get("conf", 0.0),
                            "speed_jump": speed_jump,
                            "wrist_dist": nearest_wrist_dist,
                            "speed_kmh": current_speed
                        },
                    )
                    self.events.append(ev)
                    triggered.append(ev)
                    self.last_hit_frame = frame_idx

        self.prev_speed = pix_speed2 if 'pix_speed2' in locals() else None
        self.prev_ball = ball_state
        return triggered