from ultralytics import YOLO
import numpy as np

class PlayerTracker:
    def __init__(self, pose_model_path: str):
        # Load the YOLO pose model
        self.model = YOLO(pose_model_path)

    def update(self, frame, frame_idx=0): # frame_idx added to match existing call signature in pipeline.py
        # Use YOLO's built-in tracker (ByteTrack is default for robust tracking)
        # persist=True keeps IDs across frames
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=1280, conf=0.15)[0]
        
        tracked_players = []
        if results.boxes is None or results.boxes.id is None:
            return tracked_players

        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        
        # Capture all 17 keypoints (shape: N, 17, 2)
        keypoints = results.keypoints.xy.cpu().numpy() if results.keypoints is not None else []

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            tracked_players.append({
                "track_id": int(track_id),
                "bbox": box.tolist(),
                # Store the full 17 keypoints
                "keypoints": kpts.tolist(),
                # Add default missed for compatibility
                "missed": 0
            })

        return tracked_players