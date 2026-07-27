import cv2
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional

SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # legs
    (5, 11), (6, 12), (5, 6),  # torso
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # face/head
]

class VideoVisualizer:
    def __init__(self, homography_matrix: np.ndarray):
        """
        Handles advanced video rendering (minimap, trajectories, skeletons, footprints).
        """
        self.H = homography_matrix
        # Inverse homography to map from 2D court back to perspective (if needed)
        self.H_inv = np.linalg.inv(self.H) if self.H is not None else None
        
        self.ball_history = deque(maxlen=20) # Keep last 20 frames for the tail
        
        # Minimap Settings
        self.minimap_scale = 20 # 1 meter = 20 pixels
        self.minimap_w = int(6.10 * self.minimap_scale)  # ~122 px
        self.minimap_h = int(13.40 * self.minimap_scale) # ~268 px
        self.minimap_margin = 15
        
        # Colors (BGR)
        self.color_team1 = (50, 255, 50)   # Neon Green
        self.color_team2 = (0, 165, 255)   # Neon Orange
        self.color_ball = (255, 255, 0)    # Cyan/Yellowish
        self.color_trail = (255, 200, 50)  # Cyan-ish tail
        
    def _create_minimap_base(self) -> np.ndarray:
        """ Create the static background for the 2D minimap. """
        # Dark translucent blue/gray background
        mm = np.full((self.minimap_h, self.minimap_w, 3), (40, 40, 50), dtype=np.uint8)
        
        # Draw court lines
        line_color = (255, 255, 255)
        # Outer boundary
        cv2.rectangle(mm, (0, 0), (self.minimap_w-1, self.minimap_h-1), line_color, 1)
        # Net line (middle)
        mid_y = int(self.minimap_h / 2)
        cv2.line(mm, (0, mid_y), (self.minimap_w, mid_y), (150, 150, 150), 2)
        
        # Service lines (approx)
        # Short service line is 1.98m from net
        short_y_offset = int(1.98 * self.minimap_scale)
        cv2.line(mm, (0, mid_y - short_y_offset), (self.minimap_w, mid_y - short_y_offset), line_color, 1)
        cv2.line(mm, (0, mid_y + short_y_offset), (self.minimap_w, mid_y + short_y_offset), line_color, 1)
        
        # Center line
        mid_x = int(self.minimap_w / 2)
        cv2.line(mm, (mid_x, 0), (mid_x, mid_y - short_y_offset), line_color, 1)
        cv2.line(mm, (mid_x, mid_y + short_y_offset), (mid_x, self.minimap_h), line_color, 1)
        
        return mm

    def _overlay_image_alpha(self, img: np.ndarray, img_overlay: np.ndarray, x: int, y: int, alpha: float = 0.7):
        """ Overlay an image with transparency onto the main frame. """
        h, w = img_overlay.shape[:2]
        if y+h > img.shape[0] or x+w > img.shape[1]:
            return
        
        roi = img[y:y+h, x:x+w]
        cv2.addWeighted(img_overlay, alpha, roi, 1 - alpha, 0, roi)

    def draw_minimap(self, frame: np.ndarray, players: List[Dict], ball: Optional[Dict]):
        """ Draw the top-down minimap on the top-right corner of the frame. """
        minimap = self._create_minimap_base()
        
        # Plot players
        for p in players:
            rx, ry = p.get("real_world_x"), p.get("real_world_y")
            if rx is not None and ry is not None:
                mx = int(rx * self.minimap_scale)
                my = int(ry * self.minimap_scale)
                
                # Determine team based on Y position relative to the net (6.7m)
                color = self.color_team1 if ry > 6.7 else self.color_team2
                cv2.circle(minimap, (mx, my), 5, color, -1)
                cv2.circle(minimap, (mx, my), 5, (255, 255, 255), 1)

        # Plot ball (using homography to map its screen coords to real_world coords)
        # Note: This is an approximation since ball is not on the ground plane,
        # but it gives a general idea of its overhead position.
        if ball is not None and self.H is not None:
            bx, by = ball["x"], ball["y"]
            pt = np.array([[[bx, by]]], dtype=np.float32)
            rw = cv2.perspectiveTransform(pt, self.H)[0][0]
            mx = int(rw[0] * self.minimap_scale)
            my = int(rw[1] * self.minimap_scale)
            
            # Draw ball on minimap if it's within bounds
            if 0 <= mx < self.minimap_w and 0 <= my < self.minimap_h:
                cv2.circle(minimap, (mx, my), 3, self.color_ball, -1)
        
        # Overlay minimap on frame (top right corner)
        x_offset = frame.shape[1] - self.minimap_w - self.minimap_margin
        y_offset = self.minimap_margin
        self._overlay_image_alpha(frame, minimap, x_offset, y_offset, alpha=0.85)

    def draw_players(self, frame: np.ndarray, players: List[Dict]):
        """ Draw enhanced skeletons and ground footprints for players. """
        for p in players:
            kpts = p.get("keypoints", [])
            rx, ry = p.get("real_world_x"), p.get("real_world_y")
            
            color = self.color_team1 if (ry is not None and ry > 6.7) else self.color_team2

            # 1. Draw Ground Footprint (Perspective Ellipse)
            x1, y1, x2, y2 = p["bbox"]
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)
            
            # Simple flattened ellipse for pseudo-3D look
            radius_x = int((x2 - x1) * 0.6)
            radius_y = int(radius_x * 0.3)
            cv2.ellipse(frame, (feet_x, feet_y), (radius_x, radius_y), 0, 0, 360, color, 3)

            # 2. Draw Skeleton
            if len(kpts) >= 17:
                # Edges
                for (u, v) in SKELETON_EDGES:
                    if kpts[u][0] > 0 and kpts[u][1] > 0 and kpts[v][0] > 0 and kpts[v][1] > 0:
                        pt1 = (int(kpts[u][0]), int(kpts[u][1]))
                        pt2 = (int(kpts[v][0]), int(kpts[v][1]))
                        cv2.line(frame, pt1, pt2, color, 3, lineType=cv2.LINE_AA)
                
                # Joints
                for i, kpt in enumerate(kpts):
                    if kpt[0] > 0 and kpt[1] > 0:
                        # Draw head joint slightly larger
                        radius = 5 if i in [0, 1, 2, 3, 4] else 3
                        cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius, (255, 255, 255), -1)

    def draw_ball(self, frame: np.ndarray, ball: Optional[Dict]):
        """ Draw ball with a fading trajectory tail. """
        if ball is not None:
            bx, by = int(ball["x"]), int(ball["y"])
            
            # If the ball teleports (e.g. tracking lost and restarted), clear the tail
            if len(self.ball_history) > 0:
                last_bx, last_by = self.ball_history[-1]
                dist = ((bx - last_bx)**2 + (by - last_by)**2)**0.5
                if dist > 250:
                    self.ball_history.clear()
                    
            self.ball_history.append((bx, by))
        else:
            # Gradually clear the tail if ball is completely lost
            if len(self.ball_history) > 0:
                self.ball_history.popleft()
        
        # Draw trail
        if len(self.ball_history) > 1:
            points = np.array(self.ball_history, dtype=np.int32)
            
            # Draw fading segments
            for i in range(1, len(points)):
                pt1 = tuple(points[i-1])
                pt2 = tuple(points[i])
                
                # Fade effect: older points are thinner and less opaque (simulated via thickness/color)
                progress = i / len(points)
                thickness = int(max(1, 4 * progress))
                
                # We can't do true alpha blending for lines easily without a separate layer,
                # but we can simulate it or just use thin/thick lines.
                cv2.line(frame, pt1, pt2, self.color_trail, thickness, lineType=cv2.LINE_AA)
        
        # Draw current ball
        if ball is not None:
            bx, by = int(ball["x"]), int(ball["y"])
            is_pred = ball.get("predicted", False)
            color = (0, 0, 255) if is_pred else (255, 255, 255)
            
            # Glow effect
            cv2.circle(frame, (bx, by), 8, self.color_trail, -1)
            cv2.circle(frame, (bx, by), 4, color, -1)

    def render(self, frame: np.ndarray, players: List[Dict], ball: Optional[Dict], events: List[Any]) -> np.ndarray:
        """ Master render function. Modifies frame in-place. """
        vis = frame.copy()
        
        self.draw_players(vis, players)
        self.draw_ball(vis, ball)
        self.draw_minimap(vis, players, ball)
        
        # Draw Events Overlay
        if events:
            y_pos = 50
            for ev in events:
                speed_str = ""
                if ev.details.get('speed_kmh') is not None:
                    speed_str = f" ({ev.details['speed_kmh']:.1f} km/h)"
                
                text = f"{ev.event_type.upper()}{speed_str}"
                
                # Text background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(vis, (20, y_pos - th - 10), (20 + tw + 20, y_pos + 10), (0, 0, 0), -1)
                
                cv2.putText(vis, text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_pos += 40
                
        return vis
