from TrackNetV4 import TrackNetV4
import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# IMPORT THE ARCHITECTURE FROM YOUR UPLOADED FILE


# ==========================================
# 1. TrackNetV4 Wrapper
# ==========================================
class TrackNetV4_Infer:
    def __init__(self, weights_path):
        print(f"Building TrackNetV4 Architecture and loading weights from {weights_path}...")
        self.input_width = 512
        self.input_height = 288
        
        try:
            # 1. Build the empty architecture using the imported function
            # Note: "TypeA" is the default in their code. 
            self.model = TrackNetV4(self.input_height, self.input_width, fusion_layer_type="TypeA")
            
            # 2. Inject the weights from your downloaded .keras file
            self.model.load_weights(weights_path)
            print("TrackNet weights loaded successfully!")
            
        except Exception as e:
            print(f"Error loading TrackNet model/weights: {e}")
            print("Note: If 'best_model_base' is TrackNetV2, it may not match V4 architecture.")
            raise

    def preprocess(self, f1, f2, f3):
        """Converts 3 BGR frames into Channels-First (1, 9, 288, 512) format."""
        frames = [f1, f2, f3]
        processed_frames = []
        
        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
            
        # Combine frames -> shape: (288, 512, 9)
        combined = np.concatenate(processed_frames, axis=-1)
        
        # Transpose from (H, W, C) to (C, H, W) -> shape: (9, 288, 512)
        transposed = np.transpose(combined, (2, 0, 1))
        
        # Add batch dimension -> shape: (1, 9, 288, 512)
        return np.expand_dims(transposed, axis=0)

    def get_peak_from_heatmap(self, heatmap, original_w, original_h):
        """Extracts coordinates from the Channels-First heatmap output."""
        if len(heatmap.shape) == 4:
            if heatmap.shape[1] in [1, 3]:  
                if heatmap.shape[1] == 3:
                    pred_map = heatmap[0, 2, :, :] 
                else:
                    pred_map = heatmap[0, 0, :, :]
            else: 
                if heatmap.shape[-1] == 3:
                    pred_map = heatmap[0, :, :, 2]
                else:
                    pred_map = heatmap[0, :, :, 0]
        else:
            pred_map = np.squeeze(heatmap)
             
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(pred_map)
        
        if max_val < 0.5:  
            return None
            
        x = int(max_loc[0] * (original_w / self.input_width))
        y = int(max_loc[1] * (original_h / self.input_height))
        
        return {"x": x, "y": y, "conf": float(max_val)}

    def predict(self, f1, f2, f3, orig_w, orig_h):
        input_tensor = self.preprocess(f1, f2, f3)
        heatmap = self.model.predict(input_tensor, verbose=0)
        return self.get_peak_from_heatmap(heatmap, orig_w, orig_h)

# ==========================================
# 2. Main Processing Pipeline
# ==========================================
def process_sports_video(video_path, pose_model_path, tracknet_path, output_dir):
    vid_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_video_path = out_dir / f"{vid_path.stem}_analyzed.mp4"
    out_json_path = out_dir / f"{vid_path.stem}_data.json"

    print(f"Loading Native PyTorch YOLO Pose: {pose_model_path}")
    pose_model = YOLO(pose_model_path, task="pose")
    ball_tracker = TrackNetV4_Infer(tracknet_path)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
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
                    
                players_data.append({
                    "player_id": p_idx,
                    "left_wrist": left_wrist,
                    "right_wrist": right_wrist
                })

        # --- B. BALL TRACKING ---
        ball_data = None
        if len(frame_buffer) == 3:
            f1, f2, f3 = frame_buffer[0], frame_buffer[1], frame_buffer[2]
            ball_data = ball_tracker.predict(f1, f2, f3, width, height)

        # --- C. DATA EXPORT STORAGE ---
        reservoir_data.append({
            "frame": frame_idx,
            "ball": ball_data,
            "players": players_data
        })

        # --- D. RENDERING VISUALS ---
        annotated_frame = pose_results.plot(labels=False, boxes=False)
        
        if ball_data is not None:
            bx, by = ball_data["x"], ball_data["y"]
            cv2.circle(annotated_frame, (bx, by), 6, (0, 255, 0), -1)
            cv2.putText(annotated_frame, "BALL", (bx + 10, by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        writer.write(annotated_frame)

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}...")

    cap.release()
    writer.release()

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(reservoir_data, f, indent=2)
        
    print(f"Pipeline complete! Output saved to: {out_dir}")

# ==========================================
# 3. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    process_sports_video(
        video_path=r"C:\Users\GANAPATHI\Downloads\Ahsan - Setiawan vs. Liang - Wang, All England 2023, MD-SF - Shuttle Play (1080p, h264, youtube).mp4",
        pose_model_path="models/yolov8n-pose.pt", 
        tracknet_path="models/best_model_V2_NF_RIO_1m_e8.keras",
        output_dir="./output"
    )