from __future__ import annotations

"""
TrackNet Shuttlecock Tracker
-----------------------------
Primary detector  : best_model_base_new_data_e28.keras (TrackNet-style)
  - Input  : (1, 9, 288, 512)  — 3 consecutive BGR frames, channels-first
  - Output : (1, 3, 288, 512)  — 3 heatmaps; we use heatmap index [2] (frame t)

Fallback chain:
  1. TrackNet heatmap peak (TensorFlow / Keras)
  2. Multi-frame diff blob detection (no DL required)
  3. Kalman prediction (coasting on last known velocity)
  4. YOLO re-seed (if heavy_detector supplied)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import cv2


# ─── Model constants (must match training) ────────────────────────────────────
TRACKNET_W   = 512
TRACKNET_H   = 288
TRACKNET_FRAMES = 9   # number of frames stacked
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BallObservation:
    frame_idx:    int
    x:            float
    y:            float
    conf:         float
    is_predicted: bool = False


class HybridBallTracker:
    """
    Drop-in replacement for the old MOG2-based tracker.
    Uses a pre-trained TrackNet Keras model as the primary shuttle detector.
    """

    def __init__(
        self,
        tracknet_model_path: str  = "models/best_model_base_new_data_e28.keras",
        heatmap_threshold:   int  = 64,    # 0-255; lower → more sensitive
        heavy_detector            = None,  # optional YOLO model for seeding
        # Legacy API kwargs (kept for backward compat with config.py)
        roi_size:            int  = 256,
        min_area:            int  = 4,
        max_area:            int  = 400,
        lost_reset:          int  = 10,
        conf_threshold:      float= 0.25,
        seed_interval:       int  = 60,
    ):
        self.heatmap_threshold = heatmap_threshold
        self.heavy_detector    = heavy_detector
        self.lost_reset        = lost_reset
        self.conf_threshold    = conf_threshold

        # Rolling frame buffer (BGR full-res, keeps last 3)
        self._buf: deque = deque(maxlen=TRACKNET_FRAMES)
        # Grayscale buffer for diff-fallback
        self._gray_buf: deque = deque(maxlen=TRACKNET_FRAMES)

        # TrackNet inference callable (set by _load_model)
        self._infer = None

        # Load TrackNet
        self._model       = None
        self._tf_ok       = False
        self._load_model(tracknet_model_path)

        # Kalman filter — constant-velocity, tuned for fast shuttle
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 5e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)

        self.initialized  = False
        self.frame_idx    = 0
        self.last_obs: Optional[BallObservation] = None
        self.lost_count   = 0
        self.track_buffer: List[BallObservation] = []

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self, path: str):
        try:
            import torch
            from tracknet_model import TrackNet

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[TrackNet] Using device: {self.device}")

            if not os.path.exists(path):
                print(f"[TrackNet] Model not found at '{path}', using diff fallback.")
                return

            print(f"[TrackNet] Loading PyTorch model: {path} ...")
            self._model = TrackNet(in_dim=27, out_dim=8).to(self.device)
            
            # Load weights (the ckpt from TrackNetV3 contains a param_dict with 'model' or similar)
            ckpt = torch.load(path, map_location=self.device)
            if 'model' in ckpt:
                self._model.load_state_dict(ckpt['model'])
            elif 'model_state_dict' in ckpt:
                self._model.load_state_dict(ckpt['model_state_dict'])
            else:
                self._model.load_state_dict(ckpt)
            
            self._model.eval()
            self._tf_ok = True
            print(f"[TrackNet] Ready (PyTorch).")

        except ImportError:
            print("[TrackNet] PyTorch not installed — using frame-diff fallback.")
        except Exception as e:
            print(f"[TrackNet] Load error: {e} — using frame-diff fallback.")

    # ── Kalman helpers ─────────────────────────────────────────────────────────
    def _kf_set(self, x, y, vx=0.0, vy=0.0):
        self.kf.statePost = np.array([[x],[y],[vx],[vy]], dtype=np.float32)

    def _predict_state(self):
        p = self.kf.predict()
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    def _correct_state(self, x, y):
        m = np.array([[np.float32(x)],[np.float32(y)]])
        c = self.kf.correct(m)
        return float(c[0]), float(c[1])

    def initialize(self, x, y, conf=1.0):
        self._kf_set(x, y, 0.0, 0.0)
        self.initialized = True
        self.lost_count  = 0
        obs = BallObservation(self.frame_idx, x, y, conf, is_predicted=False)
        self.last_obs = obs
        self.track_buffer.append(obs)

    # ── TrackNet inference ─────────────────────────────────────────────────────
    def _run_tracknet(self, orig_h: int, orig_w: int) -> Optional[Tuple[float, float, float]]:
        """
        Stack the last 9 frames into a (1, 27, 288, 512) tensor,
        run PyTorch model, extract heatmap channel 7 (last frame in sequence).
        Returns (x_orig, y_orig, confidence) or None.
        """
        if not self._tf_ok or self._model is None or len(self._buf) < TRACKNET_FRAMES:
            return None

        try:
            import torch
            
            # Preprocess: resize to (W×H), normalize
            frames_resized = []
            for f in self._buf:
                # TrackNetV3 expects RGB or BGR. (assuming BGR from cv2)
                r = cv2.resize(f, (TRACKNET_W, TRACKNET_H)).astype(np.float32) / 255.0
                frames_resized.append(r)          # each: (288, 512, 3)

            stacked = np.concatenate(frames_resized, axis=-1)   # (288, 512, 27)
            
            # Convert to NCHW format
            stacked = np.transpose(stacked, (2, 0, 1))          # (27, 288, 512)
            inp     = stacked[np.newaxis, ...]                  # (1, 27, 288, 512)
            
            inp_tensor = torch.from_numpy(inp).float().to(self.device)
            
            with torch.no_grad():
                out = self._model(inp_tensor)                   # (1, 8, 288, 512)
            
            out = out.cpu().numpy()

            # The output has 8 heatmaps (for frames 1 to 8 in the sequence).
            # The background frame is frame 0.
            # We want the prediction for the last frame, which is index 7.
            heatmap   = out[0, 7, :, :]                         # (288, 512)
            heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap_u8)
            if max_val < self.heatmap_threshold:
                return None

            x_orig = max_loc[0] / TRACKNET_W  * orig_w
            y_orig = max_loc[1] / TRACKNET_H  * orig_h
            conf   = float(max_val) / 255.0

            return x_orig, y_orig, conf

        except Exception as e:
            print(f"[TrackNet] Inference error: {e}")
            return None

    # ── Frame-diff fallback ────────────────────────────────────────────────────
    def _run_diff_fallback(self, pred_xy) -> Optional[Tuple[float, float, float]]:
        if len(self._gray_buf) < 3:
            return None
        f0, f1, f2 = self._gray_buf[0], self._gray_buf[1], self._gray_buf[2]
        d1 = cv2.absdiff(f0, f2)
        d2 = cv2.absdiff(f1, f2)
        combined = cv2.bitwise_and(d1, d2)
        _, thresh = cv2.threshold(combined, 8, 255, cv2.THRESH_BINARY)
        heatmap   = cv2.GaussianBlur(thresh, (7, 7), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < 30:
            return None
        return float(max_loc[0]), float(max_loc[1]), float(max_val) / 255.0

    # ── YOLO seed ──────────────────────────────────────────────────────────────
    def _seed_from_heavy_detector(self, frame):
        if self.heavy_detector is None:
            return None
        try:
            res = self.heavy_detector.predict(frame, imgsz=640, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return None
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            best, best_conf = None, -1
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b.tolist()
                area = (x2-x1) * (y2-y1)
                if area < 4 or area > 4000:
                    continue
                if confs[i] > best_conf:
                    best_conf = float(confs[i])
                    best = ((x1+x2)*0.5, (y1+y2)*0.5, best_conf)
            return best
        except Exception:
            return None

    # ── Gap interpolation helper ───────────────────────────────────────────────
    def _repair_gap(self, dest_x: float, dest_y: float):
        if self.lost_count == 0 or self.last_obs is None:
            return
        gap = self.lost_count
        src_x, src_y = self.last_obs.x, self.last_obs.y
        filled = 0
        for i in range(len(self.track_buffer) - 1, -1, -1):
            if filled >= gap:
                break
            if self.track_buffer[i].is_predicted:
                t   = (gap - filled) / gap
                px  = src_x + (dest_x - src_x) * t
                py  = src_y + (dest_y - src_y) * t
                self.track_buffer[i].x            = px
                self.track_buffer[i].y            = py
                self.track_buffer[i].is_predicted = False
                filled += 1

    # ── Main update ────────────────────────────────────────────────────────────
    def update(self, frame, frame_idx: int):
        self.frame_idx  = frame_idx
        orig_h, orig_w  = frame.shape[:2]

        # Push into buffers
        self._buf.append(frame.copy())
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._gray_buf.append(gray)

        # Kalman predict
        pred_xy = None
        if self.initialized:
            px, py, _, _ = self._predict_state()
            pred_xy = (px, py)

        # ── Detect ────────────────────────────────────────────────────────────
        detection = self._run_tracknet(orig_h, orig_w)
        if detection is None:
            detection = self._run_diff_fallback(pred_xy)

        # ── Seeding (first track) ─────────────────────────────────────────────
        if not self.initialized:
            seed = self._seed_from_heavy_detector(frame)
            if seed:
                self.initialize(*seed)
                return {"x": seed[0], "y": seed[1], "conf": seed[2],
                        "predicted": False, "initialized": True}
            if detection and detection[2] >= self.conf_threshold:
                self.initialize(*detection)
                return {"x": detection[0], "y": detection[1], "conf": detection[2],
                        "predicted": False, "initialized": True}
            return None

        # ── Already tracking ──────────────────────────────────────────────────
        if detection and detection[2] >= self.conf_threshold:
            dx, dy, dc = detection
            
            # Outlier rejection: if the detection jumps too far from prediction, it's noise
            if pred_xy is not None:
                dist = ((dx - pred_xy[0])**2 + (dy - pred_xy[1])**2)**0.5
                if dist > 200:  # >200 pixels in 1 frame is physically impossible
                    detection = None
            
        if detection:
            dx, dy, dc = detection
            self._repair_gap(dx, dy)
            
            # Vector Reversal Detection (Fix for rubber-banding lag on hits)
            if len(self.track_buffer) >= 2 and self.last_obs is not None:
                prev_obs = self.track_buffer[-2]
                v_prev_x = self.last_obs.x - prev_obs.x
                v_prev_y = self.last_obs.y - prev_obs.y
                
                v_curr_x = dx - self.last_obs.x
                v_curr_y = dy - self.last_obs.y
                
                dot_product = (v_prev_x * v_curr_x) + (v_prev_y * v_curr_y)
                
                # If dot product is negative, vectors point in opposite directions
                if dot_product < -50 and (v_curr_x**2 + v_curr_y**2) > 25:
                    print(f"Hit detected at {frame_idx}! Resetting Kalman filter.")
                    self._kf_set(dx, dy, v_curr_x, v_curr_y)
                else:
                    self._correct_state(dx, dy)
            else:
                self._correct_state(dx, dy)
                
            self.lost_count = 0
            obs = BallObservation(frame_idx, dx, dy, dc, is_predicted=False)
            self.last_obs = obs
            self.track_buffer.append(obs)
            return {"x": dx, "y": dy, "conf": dc, "predicted": False, "initialized": True}

        # ── Lost — coast on Kalman ────────────────────────────────────────────
        self.lost_count += 1
        kx, ky, _, _ = self._predict_state()
        if self.lost_count <= self.lost_reset:
            obs = BallObservation(frame_idx, kx, ky, 0.10, is_predicted=True)
            self.last_obs = obs
            self.track_buffer.append(obs)
            return {"x": kx, "y": ky, "conf": 0.10, "predicted": True, "initialized": True}

        # ── Long loss — try YOLO re-seed ──────────────────────────────────────
        seed = self._seed_from_heavy_detector(frame)
        if seed:
            self.initialize(*seed)
            return {"x": seed[0], "y": seed[1], "conf": seed[2],
                    "predicted": False, "initialized": True}

        # Fully lost
        self.initialized = False
        return None