# track.py
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from TrackNetV4 import TrackNetV4


class TrackNetV4_Infer:
    def __init__(
        self,
        weights_path: str | Path,
        fusion_layer_type: str = "TypeA",
        input_width: int = 512,
        input_height: int = 288,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.weights_path = str(Path(weights_path))
        self.fusion_layer_type = fusion_layer_type

        print(
            f"Building TrackNetV4 Architecture ({fusion_layer_type}) and loading weights from {self.weights_path}..."
        )

        # Build architecture first, then load weights
        self.model = TrackNetV4(
            self.input_height,
            self.input_width,
            fusion_layer_type=self.fusion_layer_type,
        )

        # Force model build before loading weights
        dummy = np.zeros((1, 9, self.input_height, self.input_width), dtype=np.float32)
        _ = self.model(dummy, training=False)

        self.model.load_weights(self.weights_path)
        print("TrackNet weights loaded successfully!")

    def preprocess(self, f1, f2, f3):
        """
        Converts 3 BGR frames into channels-first tensor:
        (1, 9, 288, 512)
        """
        frames = [f1, f2, f3]
        processed_frames = []

        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)

        combined = np.concatenate(processed_frames, axis=-1)     # (H, W, 9)
        transposed = np.transpose(combined, (2, 0, 1))           # (9, H, W)
        return np.expand_dims(transposed, axis=0)                # (1, 9, H, W)

    def get_peak_from_heatmap(self, heatmap, original_w, original_h):
        """
        Extract the ball position from model output.
        Expected output is usually channels-first with 3 heatmaps.
        """
        heatmap = np.asarray(heatmap)

        if heatmap.ndim == 4:
            # (B, C, H, W) or (B, H, W, C)
            if heatmap.shape[1] in (1, 3):
                pred_map = heatmap[0, 2, :, :] if heatmap.shape[1] == 3 else heatmap[0, 0, :, :]
            elif heatmap.shape[-1] in (1, 3):
                pred_map = heatmap[0, :, :, 2] if heatmap.shape[-1] == 3 else heatmap[0, :, :, 0]
            else:
                pred_map = np.squeeze(heatmap[0])
        else:
            pred_map = np.squeeze(heatmap)

        pred_map = np.asarray(pred_map, dtype=np.float32)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(pred_map)

        # Confidence threshold
        if max_val < 0.5:
            return None

        x = int(max_loc[0] * (original_w / self.input_width))
        y = int(max_loc[1] * (original_h / self.input_height))

        return {"x": x, "y": y, "conf": float(max_val)}

    def predict(self, f1, f2, f3, orig_w, orig_h):
        input_tensor = self.preprocess(f1, f2, f3)
        heatmap = self.model.predict(input_tensor, verbose=0)
        return self.get_peak_from_heatmap(heatmap, orig_w, orig_h)


def process_sports_video(
    video_path: str | Path,
    pose_model_path: str | Path,
    tracknet_path: str | Path,
    output_dir: str | Path,
    fusion_layer_type: str = "TypeA",
):
    vid_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = out_dir / f"{vid_path.stem}_analyzed.mp4"
    out_json_path = out_dir / f"{vid_path.stem}_data.json"

    print(f"Loading Native PyTorch YOLO Pose: {pose_model_path}")
    pose_model = YOLO(str(pose_model_path), task="pose")

    ball_tracker = TrackNetV4_Infer(
        weights_path=tracknet_path,
        fusion_layer_type=fusion_layer_type,
    )

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    frame_buffer = deque(maxlen=3)
    reservoir_data = []
    frame_idx = 0

    print("Starting processing loop...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_buffer.append(frame)

        # --- A. POSE ESTIMATION ---
        pose_results = pose_model.predict(frame, imgsz=640, verbose=False)[0]

        players_data = []
        if pose_results.keypoints is not None:
            kpts = pose_results.keypoints.xy.cpu().numpy()
            for p_idx, person in enumerate(kpts):
                if len(person) > 10:
                    left_wrist = [float(person[9][0]), float(person[9][1])]
                    right_wrist = [float(person[10][0]), float(person[10][1])]
                else:
                    left_wrist, right_wrist = [0.0, 0.0], [0.0, 0.0]

                players_data.append(
                    {
                        "player_id": p_idx,
                        "left_wrist": left_wrist,
                        "right_wrist": right_wrist,
                    }
                )

        # --- B. BALL TRACKING ---
        ball_data = None
        if len(frame_buffer) == 3:
            f1, f2, f3 = frame_buffer[0], frame_buffer[1], frame_buffer[2]
            ball_data = ball_tracker.predict(f1, f2, f3, width, height)

        # --- C. EXPORT DATA ---
        reservoir_data.append(
            {
                "frame": frame_idx,
                "ball": ball_data,
                "players": players_data,
            }
        )

        # --- D. RENDER VISUALS ---
        annotated_frame = pose_results.plot(labels=False, boxes=False)

        if ball_data is not None:
            bx, by = ball_data["x"], ball_data["y"]
            cv2.circle(annotated_frame, (bx, by), 6, (0, 255, 0), -1)
            cv2.putText(
                annotated_frame,
                "BALL",
                (bx + 10, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(annotated_frame)

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}...")

    cap.release()
    writer.release()

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(reservoir_data, f, indent=2)

    print(f"Pipeline complete! Output saved to: {out_dir}")
    print(f"Video: {out_video_path}")
    print(f"JSON : {out_json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNetV4 + YOLO pose pipeline")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Input sports video path",
    )
    parser.add_argument(
        "--pose_model_path",
        type=str,
        default="models/yolov8n-pose.pt",
        help="Ultralytics pose model path",
    )
    parser.add_argument(
        "--tracknet_path",
        type=str,
        required=True,
        help="TrackNetV4 .keras weights path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory",
    )
    parser.add_argument(
        "--fusion_layer_type",
        type=str,
        default="TypeA",
        choices=["TypeA", "TypeB"],
        help="TrackNetV4 fusion type",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_sports_video(
        video_path=args.video_path,
        pose_model_path=args.pose_model_path,
        tracknet_path=Path(args.tracknet_path),
        output_dir=args.output_dir,
        fusion_layer_type=args.fusion_layer_type,
    )