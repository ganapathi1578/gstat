import cv2
import numpy as np
import json

import argparse
from pathlib import Path

# =====================================================
# Configuration
# =====================================================
FRAME_NUMBER = 0
WINDOW_NAME = "Badminton Court Calibration - 13 Handles"

# =====================================================
# Point layout
# =====================================================
POINT_NAMES_31 = [
    "Top Left Outer",              # 0
    "Top Left Singles",            # 1
    "Top Center",                  # 2
    "Top Right Singles",           # 3
    "Top Right Outer",             # 4
    "Top Doubles Service Left",    # 5
    "Top Singles Service Left",    # 6
    "Top Center Line",             # 7
    "Top Singles Service Right",   # 8
    "Top Doubles Service Right",   # 9
    "Net Left",                    # 10
    "Net Singles Left",            # 11
    "Net Center",                  # 12
    "Net Singles Right",           # 13
    "Net Right",                   # 14
    "Bottom Doubles Service Left", # 15
    "Bottom Singles Service Left", # 16
    "Bottom Center Line",          # 17
    "Bottom Singles Service Right",# 18
    "Bottom Doubles Service Right",# 19
    "Bottom Left Outer",           # 20
    "Bottom Left Singles",         # 21
    "Bottom Center",               # 22
    "Bottom Right Singles",        # 23
    "Bottom Right Outer",          # 24
    "Top Long Service Center",     # 25
    "Bottom Long Service Center",  # 26
    "Court Center Left",           # 27
    "Court Center Right",          # 28
    "Net Pole Top Left",           # 29
    "Net Pole Top Right",          # 30
]
NUM_POINTS = len(POINT_NAMES_31)

# The 15 handles:
#   corners: 0, 4, 20, 24
#   center axis (1D sliding): 2, 7, 12, 17, 22, 25, 26
#   net singles (1D sliding along net): 11, 13
#   net poles (free dragging): 29, 30
DRAGGABLE_IDS = {0, 4, 20, 24, 2, 7, 11, 12, 13, 17, 22, 25, 26, 29, 30}

# Official court proportions
COURT_LENGTH_M = 13.40
COURT_WIDTH_M = 6.10
SHORT_SERVICE_FROM_NET_M = 1.98
LONG_SERVICE_FROM_BASELINE_M = 0.76
SINGLES_INSET_M = 0.46

# Fractions in [0, 1]
SHORT_SERVICE_FRAC = SHORT_SERVICE_FROM_NET_M / COURT_LENGTH_M
LONG_SERVICE_FRAC = LONG_SERVICE_FROM_BASELINE_M / COURT_LENGTH_M
SINGLES_INSET_FRAC = SINGLES_INSET_M / COURT_WIDTH_M

# =====================================================
# Basic geometry helpers
# =====================================================
def order_points_4(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def point_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

def lerp(a, b, t):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a * (1.0 - t) + b * t

def get_fraction_on_segment(pt, A, B):
    """Returns the fractional position 't' of pt projected onto the segment A->B."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    pt = np.asarray(pt, dtype=np.float32)
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-8:
        return 0.5
    return float(np.dot(pt - A, AB) / denom)

def draw_point(img, pt, idx, color=(0, 0, 255), radius=6, thickness=-1, show_label=True):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img, (x, y), radius, color, thickness)
    if show_label:
        cv2.putText(
            img, str(idx), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA,
        )

def draw_crosshair(img, pt, idx, color=(0, 255, 255), size=10, show_label=True):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, 2)
    cv2.line(img, (x, y - size), (x, y + size), color, 2)
    if show_label:
        cv2.putText(
            img, str(idx), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA,
        )

def draw_label(img, text, x, y, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def line_from_points(p1, p2):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return p1, p2 - p1

def intersect_lines(p, d, q, e, eps=1e-8):
    p = np.asarray(p, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    e = np.asarray(e, dtype=np.float32)
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(float(cross)) < eps:
        return None
    qp = q - p
    t = (qp[0] * e[1] - qp[1] * e[0]) / cross
    return p + t * d

def midpoint(a, b):
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) * 0.5

# =====================================================
# Court model (Parameterized)
# =====================================================
class CourtModel:
    def __init__(self, width, height, template=False):
        self.w = width
        self.h = height
        self.template = template

        # Fractional parameters determining the internal grid lines
        self.center_s_top = 0.5
        self.center_s_bot = 0.5
        self.net_t = 0.5
        self.short_serve_t_top = SHORT_SERVICE_FRAC
        self.short_serve_t_bot = SHORT_SERVICE_FRAC
        self.singles_left_s = SINGLES_INSET_FRAC
        self.singles_right_s = 1.0 - SINGLES_INSET_FRAC
        self.long_serve_t_top = LONG_SERVICE_FRAC
        self.long_serve_t_bot = LONG_SERVICE_FRAC

        # Outer corners state
        self.corners_arr = np.zeros((4, 2), dtype=np.float32)
        if self.template:
            # Isometric pseudo-3D parallelogram projection
            margin_x = int(width * 0.25)
            margin_y = int(height * 0.2)
            skew = int(width * 0.2)
            self.corners_arr[0] = [margin_x + skew, margin_y]              # Top Left
            self.corners_arr[1] = [width - margin_x + skew, margin_y]      # Top Right
            self.corners_arr[2] = [width - margin_x - skew, height - margin_y] # Bottom Right
            self.corners_arr[3] = [margin_x - skew, height - margin_y]     # Bottom Left
        else:
            # Flat projection for actual image calibration
            margin_x = int(width * 0.12)
            margin_y = int(height * 0.08)
            self.corners_arr[0] = [margin_x, margin_y]              # Top Left
            self.corners_arr[1] = [width - margin_x, margin_y]      # Top Right
            self.corners_arr[2] = [width - margin_x, height - margin_y] # Bottom Right
            self.corners_arr[3] = [margin_x, height - margin_y]     # Bottom Left

        # 3D Net Poles (default slightly above the net anchors)
        self.net_pole_left_top = [margin_x, int(height * 0.4)]
        self.net_pole_right_top = [width - margin_x, int(height * 0.4)]

        self.computed = np.zeros((NUM_POINTS, 2), dtype=np.float32)
        
        # Cached line anchors for handles to project onto
        self.center_top_pt = None
        self.center_bot_pt = None
        self.net_left_pt = None
        self.net_right_pt = None

        self.recompute()

    def corners(self):
        return self.corners_arr[0], self.corners_arr[1], self.corners_arr[2], self.corners_arr[3]

    def recompute(self):
        """Compute all 29 points based purely on the corners and the parameterized fractions."""
        tl, tr, br, bl = self.corners()
        lfp = line_from_points

        # 1. Main boundary axes
        top_line = lfp(tl, tr)
        bot_line = lfp(bl, br)
        left_line = lfp(tl, bl)
        right_line = lfp(tr, br)

        # 2. Key internal axes
        center_top = lerp(tl, tr, self.center_s_top)
        center_bot = lerp(bl, br, self.center_s_bot)
        center_col = lfp(center_top, center_bot)

        net_left = lerp(tl, bl, self.net_t)
        net_right = lerp(tr, br, self.net_t)
        net_row = lfp(net_left, net_right)

        # 3. Create arrays for the 5 Rows and 5 Columns
        row_lines = []
        row_lines.append(top_line)
        tss_l, tss_r = lerp(tl, bl, 0.5 - self.short_serve_t_top), lerp(tr, br, 0.5 - self.short_serve_t_top)
        row_lines.append(lfp(tss_l, tss_r))
        row_lines.append(net_row)
        bss_l, bss_r = lerp(tl, bl, 0.5 + self.short_serve_t_bot), lerp(tr, br, 0.5 + self.short_serve_t_bot)
        row_lines.append(lfp(bss_l, bss_r))
        row_lines.append(bot_line)

        col_lines = []
        col_lines.append(left_line)
        ls_t, ls_b = lerp(tl, tr, self.singles_left_s), lerp(bl, br, self.singles_left_s)
        col_lines.append(lfp(ls_t, ls_b))
        col_lines.append(center_col)
        rs_t, rs_b = lerp(tl, tr, self.singles_right_s), lerp(bl, br, self.singles_right_s)
        col_lines.append(lfp(rs_t, rs_b))
        col_lines.append(right_line)

        # 4. Intersect to generate core 25 points
        pts = np.zeros((NUM_POINTS, 2), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                idx = r * 5 + c
                p = intersect_lines(row_lines[r][0], row_lines[r][1], col_lines[c][0], col_lines[c][1])
                if p is None:
                    p = lerp(lerp(tl, tr, c/4.0), lerp(bl, br, c/4.0), r/4.0)
                pts[idx] = p

        # 5. Calculate floating Long Service handles (25, 26) on the center column
        tls_l, tls_r = lerp(tl, bl, self.long_serve_t_top), lerp(tr, br, self.long_serve_t_top)
        pts[25] = intersect_lines(center_col[0], center_col[1], tls_l, tls_r - tls_l)

        bls_l, bls_r = lerp(tl, bl, 1.0 - self.long_serve_t_bot), lerp(tr, br, 1.0 - self.long_serve_t_bot)
        pts[26] = intersect_lines(center_col[0], center_col[1], bls_l, bls_r - bls_l)

        # Save horizontal lines so they can be drawn connecting 25 & 26 to the outer edges
        self.long_serve_lines = [(tls_l, tls_r), (bls_l, bls_r)]

        # Fallbacks for extreme perspectives
        if pts[25] is None: pts[25] = lerp(pts[2], pts[12], 0.5)
        if pts[26] is None: pts[26] = lerp(pts[12], pts[22], 0.5)

        # Court Center Left/Right
        pts[27] = midpoint(pts[10], pts[11])
        pts[28] = midpoint(pts[13], pts[14])

        # Net poles (free floating for the left panel, vertical in the template)
        if self.template:
            pts[29] = pts[10] - np.array([0, 80], dtype=np.float32)
            pts[30] = pts[14] - np.array([0, 80], dtype=np.float32)
        else:
            pts[29] = self.net_pole_left_top
            pts[30] = self.net_pole_right_top

        self.computed = pts
        
        # Cache for drag constraint calculation
        self.center_top_pt = center_top
        self.center_bot_pt = center_bot
        self.net_left_pt = net_left
        self.net_right_pt = net_right
        return pts

    def update_handle(self, idx, mouse_pt):
        """Update fraction parameters directly—guaranteeing 1D handles don't jump awkwardly."""
        if idx == 0: self.corners_arr[0] = mouse_pt
        elif idx == 4: self.corners_arr[1] = mouse_pt
        elif idx == 24: self.corners_arr[2] = mouse_pt
        elif idx == 20: self.corners_arr[3] = mouse_pt
        elif idx == 29: self.net_pole_left_top = mouse_pt
        elif idx == 30: self.net_pole_right_top = mouse_pt
        else:
            tl, tr, br, bl = self.corners()
            c_top, c_bot = self.center_top_pt, self.center_bot_pt
            n_left, n_right = self.net_left_pt, self.net_right_pt

            # 1 DOF constraints updating the isolated fractions
            if idx == 2:
                self.center_s_top = get_fraction_on_segment(mouse_pt, tl, tr)
            elif idx == 22:
                self.center_s_bot = get_fraction_on_segment(mouse_pt, bl, br)
            elif idx == 12:
                self.net_t = get_fraction_on_segment(mouse_pt, c_top, c_bot)
            elif idx == 11:
                self.singles_left_s = get_fraction_on_segment(mouse_pt, n_left, n_right)
            elif idx == 13:
                self.singles_right_s = get_fraction_on_segment(mouse_pt, n_left, n_right)
            elif idx == 7:
                self.short_serve_t_top = 0.5 - get_fraction_on_segment(mouse_pt, c_top, c_bot)
            elif idx == 17:
                self.short_serve_t_bot = get_fraction_on_segment(mouse_pt, c_top, c_bot) - 0.5
            elif idx == 25:
                self.long_serve_t_top = get_fraction_on_segment(mouse_pt, c_top, c_bot)
            elif idx == 26:
                self.long_serve_t_bot = 1.0 - get_fraction_on_segment(mouse_pt, c_top, c_bot)

        self.recompute()

    def save(self, out_path):
        self.recompute()
        src = self.computed
        data = {
            "image_size": {"width": self.w, "height": self.h},
            "drag_handles": sorted(list(DRAGGABLE_IDS)),
            "points_31": [
                {
                    "id": i,
                    "name": POINT_NAMES_31[i],
                    "image": [float(src[i][0]), float(src[i][1])]
                }
                for i in range(NUM_POINTS)
            ]
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {NUM_POINTS} points to {out_path}")

    def render(self):
        self.recompute()
        return self.computed


# =====================================================
# Drawing
# =====================================================
def draw_court_lines(img, pts, model=None):
    line_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (20, 21), (21, 22), (22, 23), (23, 24),
        (0, 5), (5, 10), (10, 15), (15, 20),
        (4, 9), (9, 14), (14, 19), (19, 24),
        (1, 6), (6, 11), (11, 16), (16, 21),
        (3, 8), (8, 13), (13, 18), (18, 23),
        (10, 11), (11, 12), (12, 13), (13, 14),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (15, 16), (16, 17), (17, 18), (18, 19),
        (2, 7), (22, 17),
        (10, 29), (14, 30), (29, 30) # The 3D Net!
    ]

    seen = set()
    line_connections = [c for c in line_connections if not (c in seen or seen.add(c))]

    for i, j in line_connections:
        p1 = tuple(map(int, np.round(pts[i])))
        p2 = tuple(map(int, np.round(pts[j])))
        cv2.line(img, p1, p2, (0, 255, 255), 2)
        
    # Draw horizontal long service lines passing through 25 and 26
    if model is not None and hasattr(model, 'long_serve_lines'):
        for p1, p2 in model.long_serve_lines:
            pt1 = tuple(map(int, np.round(p1)))
            pt2 = tuple(map(int, np.round(p2)))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)


def draw_all_points(img, pts):
    for i, pt in enumerate(pts):
        is_draggable = i in DRAGGABLE_IDS
        
        # Color coding: Red if draggable, Gray if locked
        color = (0, 0, 255) if is_draggable else (150, 150, 150)
        radius = 6 if is_draggable else 4
        
        if i in {25, 26, 27, 28}:
            draw_crosshair(img, pt, i, color=color, size=10 if is_draggable else 6)
        else:
            draw_point(img, pt, i, color=color, radius=radius)

# =====================================================
# Template image
# =====================================================
def make_court_template(width=700, height=1100):
    model = CourtModel(width, height, template=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 140, 40)

    pts = model.render()
    draw_court_lines(img, pts, model)
    draw_all_points(img, pts)
    draw_label(img, "Canonical Court (13-handle model)", 20, 35, (255, 255, 255), 0.85, 2)
    draw_label(img, "Red points are draggable on the left panel", 20, height - 20, (255, 255, 255), 0.6, 2)
    return img, pts


# =====================================================
# Auto-estimate outer corners
# =====================================================
def estimate_outer_corners(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 25, 25], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = frame.shape[:2]
        inset_x = int(w * 0.15)
        inset_y = int(h * 0.12)
        return np.array([
            [inset_x, inset_y],
            [w - inset_x, inset_y],
            [w - inset_x, h - inset_y],
            [inset_x, h - inset_y],
        ], dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return order_points_4(box)

# =====================================================
# Interactive annotator
# =====================================================
class CourtAnnotator:
    def __init__(self, frame, save_path):
        self.frame = frame.copy()
        self.save_path = save_path
        self.h, self.w = frame.shape[:2]

        self.template_img, self.template_pts = make_court_template()
        self.template_h, self.template_w = self.template_img.shape[:2]

        self.model = CourtModel(self.w, self.h)
        corners = estimate_outer_corners(frame)
        self.model.corners_arr[0] = corners[0]
        self.model.corners_arr[1] = corners[1]
        self.model.corners_arr[2] = corners[2]
        self.model.corners_arr[3] = corners[3]
        self.model.recompute()

        self.drag_idx = None
        self.panel_gap = 10
        self.mouse_radius = 18

    def on_mouse(self, event, x, y, flags, param):
        if x >= self.w:
            return

        current = self.model.computed
        mouse = np.array([x, y], dtype=np.float32)

        if event == cv2.EVENT_LBUTTONDOWN:
            best_idx, best_dist = None, 1e9
            for i in DRAGGABLE_IDS:
                d = point_distance(mouse, current[i])
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx is not None and best_dist <= self.mouse_radius:
                self.drag_idx = best_idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                self.model.update_handle(self.drag_idx, mouse)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None

    def render(self):
        left = self.frame.copy()
        right = self.template_img.copy()

        pts = self.model.computed
        
        # Make sure to pass self.model so the long service lines render
        draw_court_lines(left, pts, self.model)
        draw_all_points(left, pts)
        
        draw_label(left, "Original frame (13 handles)", 20, 35, (255, 255, 255), 0.8, 2)
        draw_label(left, "Red points are draggable handles / 1D sliders", 20, 60, (255, 255, 255), 0.55, 2)

        draw_label(left, f"Dragging: {self.drag_idx}" if self.drag_idx is not None else "", 20, 85, (0, 255, 255), 0.55, 2)

        canvas_h = max(left.shape[0], right.shape[0])
        canvas_w = left.shape[1] + self.panel_gap + right.shape[1]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:left.shape[0], :left.shape[1]] = left
        x0 = left.shape[1] + self.panel_gap
        canvas[:right.shape[0], x0:x0 + right.shape[1]] = right

        cv2.line(canvas, (self.w + self.panel_gap // 2, 0), (self.w + self.panel_gap // 2, canvas_h - 1), (80, 80, 80), 2)
        draw_label(canvas, "Keys: [s] save   [r] reset   [q] quit", 20, canvas_h - 25, (255, 255, 255), 0.6, 2)

        return canvas

    def save(self, out_path):
        self.model.save(out_path)

    def reset(self):
        corners = estimate_outer_corners(self.frame)
        self.model.corners_arr[0] = corners[0]
        self.model.corners_arr[1] = corners[1]
        self.model.corners_arr[2] = corners[2]
        self.model.corners_arr[3] = corners[3]
        self.model.recompute()

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            canvas = self.render()
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('s'):
                self.save(self.save_path)
                print(f"Saved calibration to {self.save_path}")

        cv2.destroyAllWindows()


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Badminton Court Boundary Calibration Tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Derive the calibration JSON path
    save_path = video_path.parent / f"{video_path.stem}_calib.json"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Cannot read frame from video.")

    print(f"Calibration UI opened for {video_path.name}")
    print(f"Save destination: {save_path}")
    print("Press [s] to save the calibration, [r] to reset, [q] to quit.")

    annotator = CourtAnnotator(frame, str(save_path))
    annotator.run()

if __name__ == "__main__":
    main()