#!/usr/bin/env python3
"""
offline_pose_object_track.py

Offline video pipeline for sports analytics:
  1) Pose tracking for players
  2) Object tracking for racket / ball / shuttlecock
  3) Draw all detections and associations on video
  4) Save timing + FLOPs-style stats to JSON

Recommended tracker:
  - ocsort.yaml  -> good starting point for sports / abrupt motion
  - bytetrack.yaml -> simpler baseline
  - botsort.yaml -> stronger ID persistence in some cases

Examples:
  python tests/test_pose.py ^
    --source "C:\\Users\\GANAPATHI\\Downloads\\8053652-hd_1280_720_25fps.mp4" ^
    --pose-weights yolo26n-pose.pt ^
    --det-weights your_custom_racket_ball_shuttlecock_detector.pt ^
    --device 0 ^
    --tracker ocsort.yaml ^
    --det-imgsz 960 ^
    --pose-imgsz 640 ^
    --det-conf 0.10 ^
    --pose-conf 0.20

Notes:
  - Offline/local video files only.
  - Shuttlecock needs a detector that actually has a shuttlecock/birdie class.
  - If your detector is COCO-only, racket/ball may work, shuttlecock will not.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO

print("loaded libs")


# ----------------------------
# Helpers
# ----------------------------
def is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme.lower() in {"http", "https", "rtsp", "rtmp", "mms", "ftp"}
    except Exception:
        return False


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline pose + object tracking video annotator")

    parser.add_argument("--source", type=str, required=True, help="Local video file only")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--pose-weights", type=str, default="yolo26n-pose.pt", help="Pose weights")
    parser.add_argument(
        "--det-weights",
        type=str,
        required=True,
        help="Detector weights for racket / ball / shuttlecock. Use a custom detector for shuttlecock.",
    )

    parser.add_argument("--pose-imgsz", type=int, default=640, help="Pose input size")
    parser.add_argument("--det-imgsz", type=int, default=960, help="Detector input size. Larger helps small shuttlecock.")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Pose confidence threshold")
    parser.add_argument("--det-conf", type=float, default=0.10, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help='e.g. "cpu", "0", "0,1"')
    parser.add_argument("--tracker", type=str, default="ocsort.yaml", help="Tracker YAML: ocsort.yaml, bytetrack.yaml, botsort.yaml, ...")

    parser.add_argument(
        "--target-keywords",
        type=str,
        default="racket,tennis racket,sports ball,ball,shuttlecock,birdie",
        help="Comma-separated class-name keywords to keep from detector output",
    )
    parser.add_argument(
        "--pose-gflops",
        type=float,
        default=7.5,
        help="Fallback pose GFLOPs/frame if model.info() is unavailable",
    )
    parser.add_argument(
        "--det-gflops",
        type=float,
        default=None,
        help="Optional detector GFLOPs/frame override",
    )

    parser.add_argument("--fp16", action="store_true", help="Try FP16/quantized inference where supported")
    parser.add_argument("--no-fuse", action="store_true", help="Disable model.fuse()")
    parser.add_argument("--show", action="store_true", help="Show live annotated output")
    return parser.parse_args()


def parse_keywords(s: str) -> list[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def resolve_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, (list, tuple)):
        idx = int(class_id)
        return str(names[idx]) if 0 <= idx < len(names) else str(class_id)
    return str(class_id)


def model_gflops(model: YOLO, imgsz: int, fallback: float | None = None) -> float:
    try:
        info = model.info(verbose=False, imgsz=imgsz)
        if isinstance(info, tuple) and len(info) >= 4:
            val = safe_float(info[3], None)
            if val is not None:
                return val
    except Exception:
        pass
    return float(fallback) if fallback is not None else float("nan")


def draw_text_block(img, lines, x=10, y=30, line_gap=26):
    for i, line in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def bucket_from_name(name: str) -> str:
    n = name.lower()
    if "racket" in n:
        return "racket"
    if "sports ball" in n or n == "ball" or " ball" in n:
        return "ball"
    if "shuttlecock" in n or "birdie" in n:
        return "shuttlecock"
    return "other"


def center_of_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def nearest_candidate(src_xy, candidates):
    if not candidates:
        return None, None

    sx, sy = src_xy
    best = None
    best_d = None

    for c in candidates:
        if "xyxy" in c:
            cx, cy = center_of_xyxy(c["xyxy"])
        else:
            cx, cy = float(c["x"]), float(c["y"])

        d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best_d = d
            best = c

    return best, best_d


def pick_frames_result(result):
    """
    Handle result from model.track(...).
    """
    if isinstance(result, list):
        return result[0]
    return result


def predict_track(model: YOLO, frame, imgsz: int, conf: float, device, tracker: str, fp16: bool):
    """
    Track one frame, preserving state across frames.
    """
    kwargs = dict(imgsz=imgsz, conf=conf, device=device, tracker=tracker, persist=True, verbose=False, save=False)

    if fp16:
        try:
            return model.track(frame, quantize="fp16", **kwargs)
        except TypeError:
            try:
                return model.track(frame, half=True, **kwargs)
            except TypeError:
                pass

    return model.track(frame, **kwargs)


def extract_pose_people(pose_result):
    """
    Returns per-person data with keypoints and track IDs if present.
    COCO wrist indices:
      9 = left wrist
      10 = right wrist
    """
    people = []
    if getattr(pose_result, "keypoints", None) is None:
        return people

    xy = pose_result.keypoints.xy
    conf = getattr(pose_result.keypoints, "conf", None)

    if hasattr(xy, "cpu"):
        xy = xy.cpu().numpy()
    if conf is not None and hasattr(conf, "cpu"):
        conf = conf.cpu().numpy()

    track_ids = None
    if getattr(pose_result, "boxes", None) is not None and getattr(pose_result.boxes, "id", None) is not None:
        try:
            track_ids = pose_result.boxes.id.int().cpu().tolist()
        except Exception:
            track_ids = None

    if xy is None or len(xy) == 0:
        return people

    for p_idx in range(xy.shape[0]):
        kp = xy[p_idx]
        kp_conf = conf[p_idx] if conf is not None else None
        tid = track_ids[p_idx] if track_ids is not None and p_idx < len(track_ids) else None

        wrists = []
        for kpt_idx in (9, 10):
            x, y = kp[kpt_idx]
            if np.isnan(x) or np.isnan(y):
                continue
            c = float(kp_conf[kpt_idx]) if kp_conf is not None else 1.0
            wrists.append({"x": float(x), "y": float(y), "conf": c, "kpt_idx": kpt_idx})

        people.append(
            {
                "person_idx": p_idx,
                "track_id": tid,
                "wrists": wrists,
            }
        )
    return people


def extract_tracked_objects(det_result, names, target_keywords: list[str]):
    """
    Returns tracked object list:
      {
        xyxy, conf, cls_id, cls_name, track_id, bucket
      }
    """
    out = []
    if det_result is None or getattr(det_result, "boxes", None) is None:
        return out

    boxes = det_result.boxes
    if boxes.xyxy is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    cls = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), dtype=int)
    conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy), dtype=float)

    track_ids = None
    if getattr(boxes, "id", None) is not None:
        try:
            track_ids = boxes.id.int().cpu().tolist()
        except Exception:
            track_ids = None

    for i in range(len(xyxy)):
        class_name = resolve_name(names, int(cls[i]))
        class_name_l = class_name.lower()
        keep = True if not target_keywords else any(k in class_name_l for k in target_keywords)
        if not keep:
            continue

        tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None

        out.append(
            {
                "xyxy": xyxy[i].astype(float).tolist(),
                "conf": float(conf[i]),
                "cls_id": int(cls[i]),
                "cls_name": class_name,
                "track_id": tid,
                "bucket": bucket_from_name(class_name),
            }
        )
    return out


def color_for_bucket(bucket: str):
    if bucket == "racket":
        return (0, 255, 255)
    if bucket == "ball":
        return (0, 255, 0)
    if bucket == "shuttlecock":
        return (255, 0, 255)
    return (200, 200, 200)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    keywords = parse_keywords(args.target_keywords)

    source = Path(args.source)
    if is_url(args.source):
        raise ValueError("Offline/local video files only. URLs are disabled.")
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_pose_objects.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output.with_suffix(".json")

    pose_model = YOLO(args.pose_weights)
    det_model = YOLO(args.det_weights)

    if not args.no_fuse:
        pose_model.fuse()
        det_model.fuse()

    pose_gflops = model_gflops(pose_model, args.pose_imgsz, fallback=args.pose_gflops)
    det_gflops = model_gflops(det_model, args.det_imgsz, fallback=args.det_gflops)

    if math.isnan(pose_gflops):
        pose_gflops = float(args.pose_gflops)

    cap_meta = cv2.VideoCapture(str(source))
    if not cap_meta.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {source}")

    src_fps = safe_float(cap_meta.get(cv2.CAP_PROP_FPS), 0.0) or 0.0
    src_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_frame_count = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_meta.release()

    if src_fps <= 0:
        src_fps = 30.0

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {source}")

    writer = None
    frame_idx = 0

    total_read_ms = 0.0
    total_pose_ms = 0.0
    total_det_ms = 0.0
    total_draw_ms = 0.0
    total_write_ms = 0.0
    total_pipeline_ms = 0.0

    counts = defaultdict(int)
    assoc_counts = defaultdict(int)

    wall_start = perf_counter()

    while True:
        t_read0 = perf_counter()
        ret, frame = cap.read()
        t_read1 = perf_counter()
        total_read_ms += (t_read1 - t_read0) * 1000.0

        if not ret:
            break

        frame_idx += 1

        # Track people
        t_pose0 = perf_counter()
        pose_results = predict_track(
            pose_model,
            frame,
            imgsz=args.pose_imgsz,
            conf=args.pose_conf,
            device=args.device,
            tracker=args.tracker,
            fp16=args.fp16,
        )
        pose_result = pick_frames_result(pose_results)
        t_pose1 = perf_counter()
        pose_ms = (t_pose1 - t_pose0) * 1000.0
        total_pose_ms += pose_ms

        # Track racket / ball / shuttlecock
        t_det0 = perf_counter()
        det_results = predict_track(
            det_model,
            frame,
            imgsz=args.det_imgsz,
            conf=args.det_conf,
            device=args.device,
            tracker=args.tracker,
            fp16=args.fp16,
        )
        det_result = pick_frames_result(det_results)
        t_det1 = perf_counter()
        det_ms = (t_det1 - t_det0) * 1000.0
        total_det_ms += det_ms

        # Draw
        t_draw0 = perf_counter()

        annotated = pose_result.plot(
            conf=False,
            labels=False,
            boxes=False,
            masks=False,
            probs=False,
            kpt_radius=4,
            kpt_line=True,
            show=False,
        )

        people = extract_pose_people(pose_result)
        objects = extract_tracked_objects(det_result, getattr(det_model, "names", None), keywords)

        rackets = [o for o in objects if o["bucket"] == "racket"]
        balls = [o for o in objects if o["bucket"] == "ball"]
        shuttlecocks = [o for o in objects if o["bucket"] == "shuttlecock"]

        counts["people"] += len(people)
        counts["rackets"] += len(rackets)
        counts["balls"] += len(balls)
        counts["shuttlecocks"] += len(shuttlecocks)
        counts["other"] += max(0, len(objects) - len(rackets) - len(balls) - len(shuttlecocks))

        # Draw people IDs if available
        if getattr(pose_result, "boxes", None) is not None and getattr(pose_result.boxes, "xyxy", None) is not None:
            try:
                p_xyxy = pose_result.boxes.xyxy.cpu().numpy()
                p_ids = pose_result.boxes.id.int().cpu().tolist() if getattr(pose_result.boxes, "id", None) is not None else [None] * len(p_xyxy)
                for i in range(len(p_xyxy)):
                    x1, y1, x2, y2 = p_xyxy[i]
                    pid = p_ids[i] if i < len(p_ids) else None
                    if pid is not None:
                        cv2.putText(
                            annotated,
                            f"person#{pid}",
                            (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
            except Exception:
                pass

        # Precompute racket centers for association
        racket_centers = []
        for r in rackets:
            cx, cy = center_of_xyxy(r["xyxy"])
            racket_centers.append({"x": cx, "y": cy, "xyxy": r["xyxy"], "track_id": r["track_id"], "cls_name": r["cls_name"]})

        # Draw tracked objects
        for obj in objects:
            x1, y1, x2, y2 = obj["xyxy"]
            conf = obj["conf"]
            cls_name = obj["cls_name"]
            tid = obj["track_id"]
            bucket = obj["bucket"]
            color = color_for_bucket(bucket)

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))

            cv2.rectangle(annotated, p1, p2, color, 2)
            label = f"{cls_name}"
            if tid is not None:
                label += f"#{tid}"
            label += f" {conf:.2f}"

            cv2.putText(
                annotated,
                label,
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            cx, cy = center_of_xyxy(obj["xyxy"])
            cxy = (int(round(cx)), int(round(cy)))
            cv2.circle(annotated, cxy, 3, color, -1)

            # Associations
            if bucket == "racket":
                # associate racket to nearest wrist across all people
                wrist_candidates = []
                for person in people:
                    for w in person["wrists"]:
                        wrist_candidates.append(
                            {
                                "x": w["x"],
                                "y": w["y"],
                                "person_idx": person["person_idx"],
                                "track_id": person["track_id"],
                                "kpt_idx": w["kpt_idx"],
                            }
                        )

                wrist, _ = nearest_candidate(cxy, wrist_candidates)
                if wrist is not None:
                    assoc_counts["racket_to_wrist"] += 1
                    wx, wy = int(round(wrist["x"])), int(round(wrist["y"]))
                    cv2.line(annotated, (wx, wy), cxy, (255, 0, 255), 2)
                    cv2.circle(annotated, (wx, wy), 4, (255, 0, 255), -1)

            elif bucket in {"ball", "shuttlecock"}:
                racket, _ = nearest_candidate(cxy, racket_centers)
                if racket is not None:
                    if bucket == "ball":
                        assoc_counts["ball_to_racket"] += 1
                    else:
                        assoc_counts["shuttlecock_to_racket"] += 1
                    rx, ry = int(round(racket["x"])), int(round(racket["y"]))
                    cv2.line(annotated, (rx, ry), cxy, (0, 165, 255), 2)
                    cv2.circle(annotated, (rx, ry), 4, (0, 165, 255), -1)

        pipeline_ms = pose_ms + det_ms
        total_pipeline_ms += pipeline_ms

        overlay_lines = [
            f"frame: {frame_idx}/{src_frame_count if src_frame_count > 0 else '?'}",
            f"pose: {pose_ms:.1f} ms",
            f"detect: {det_ms:.1f} ms",
            f"pipeline: {pipeline_ms:.1f} ms  |  FPS: {1000.0 / pipeline_ms:.2f}" if pipeline_ms > 0 else "pipeline: n/a",
            f"people: {len(people)}  rackets: {len(rackets)}  balls: {len(balls)}  shuttlecocks: {len(shuttlecocks)}",
            f"tracker: {args.tracker}",
        ]
        draw_text_block(annotated, overlay_lines, x=10, y=30, line_gap=26)

        t_draw1 = perf_counter()
        total_draw_ms += (t_draw1 - t_draw0) * 1000.0

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, src_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for output: {output}")

        t_write0 = perf_counter()
        writer.write(annotated)
        t_write1 = perf_counter()
        total_write_ms += (t_write1 - t_write0) * 1000.0

        if args.show:
            cv2.imshow("Pose + Racket + Ball + Shuttlecock", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    wall_end = perf_counter()

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed_s = wall_end - wall_start
    processed_frames = frame_idx
    processed_fps = (processed_frames / elapsed_s) if elapsed_s > 0 else 0.0

    avg_read_ms = total_read_ms / processed_frames if processed_frames else 0.0
    avg_pose_ms = total_pose_ms / processed_frames if processed_frames else 0.0
    avg_det_ms = total_det_ms / processed_frames if processed_frames else 0.0
    avg_draw_ms = total_draw_ms / processed_frames if processed_frames else 0.0
    avg_write_ms = total_write_ms / processed_frames if processed_frames else 0.0
    avg_pipeline_ms = total_pipeline_ms / processed_frames if processed_frames else 0.0

    video_duration_s = (src_frame_count / src_fps) if (src_fps > 0 and src_frame_count > 0) else None

    combined_gflops = 0.0
    has_any = False
    for v in (pose_gflops, det_gflops):
        if v is not None and not math.isnan(v):
            combined_gflops += float(v)
            has_any = True
    if not has_any:
        combined_gflops = float("nan")

    pose_flops_per_frame = pose_gflops * 1e9 if not math.isnan(pose_gflops) else float("nan")
    det_flops_per_frame = det_gflops * 1e9 if not math.isnan(det_gflops) else float("nan")
    combined_flops_per_frame = combined_gflops * 1e9 if not math.isnan(combined_gflops) else float("nan")

    if not math.isnan(combined_flops_per_frame):
        flops_per_second_of_video = combined_flops_per_frame * src_fps
        flops_per_minute_of_video = flops_per_second_of_video * 60.0
        flops_for_full_video = combined_flops_per_frame * processed_frames
    else:
        flops_per_second_of_video = float("nan")
        flops_per_minute_of_video = float("nan")
        flops_for_full_video = float("nan")

    wall_seconds_for_1_second_of_video = (avg_pipeline_ms / 1000.0) * src_fps
    wall_seconds_for_1_minute_of_video = wall_seconds_for_1_second_of_video * 60.0

    stats = {
        "input": str(source),
        "output": str(output),
        "pose_weights": args.pose_weights,
        "det_weights": args.det_weights,
        "tracker": args.tracker,
        "pose_imgsz": args.pose_imgsz,
        "det_imgsz": args.det_imgsz,
        "pose_conf": args.pose_conf,
        "det_conf": args.det_conf,
        "device": args.device,
        "fp16_requested": bool(args.fp16),
        "target_keywords": keywords,
        "source_metadata": {
            "fps": src_fps,
            "width": src_w,
            "height": src_h,
            "frame_count": src_frame_count,
            "duration_s_estimated": video_duration_s,
        },
        "model_info": {
            "pose_gflops_per_frame": pose_gflops,
            "det_gflops_per_frame": det_gflops,
            "combined_gflops_per_frame": combined_gflops,
            "pose_flops_per_frame": pose_flops_per_frame,
            "det_flops_per_frame": det_flops_per_frame,
            "combined_flops_per_frame": combined_flops_per_frame,
        },
        "runtime": {
            "processed_frames": processed_frames,
            "elapsed_wall_s": elapsed_s,
            "processed_fps": processed_fps,
            "avg_read_ms_per_frame": avg_read_ms,
            "avg_pose_ms_per_frame": avg_pose_ms,
            "avg_det_ms_per_frame": avg_det_ms,
            "avg_draw_ms_per_frame": avg_draw_ms,
            "avg_write_ms_per_frame": avg_write_ms,
            "avg_pipeline_ms_per_frame": avg_pipeline_ms,
            "wall_seconds_for_1_second_of_video": wall_seconds_for_1_second_of_video,
            "wall_seconds_for_1_minute_of_video": wall_seconds_for_1_minute_of_video,
        },
        "theoretical_compute": {
            "flops_per_second_of_video": flops_per_second_of_video,
            "flops_per_minute_of_video": flops_per_minute_of_video,
            "flops_for_full_video": flops_for_full_video,
        },
        "counts": dict(counts),
        "associations": dict(assoc_counts),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"Output video : {output}")
    print(f"Stats JSON   : {stats_path}")
    print(f"Frames       : {processed_frames}")
    print(f"Elapsed      : {elapsed_s:.3f} s")
    print(f"Wall/frame   : {avg_pipeline_ms:.3f} ms")
    print(f"Processed FPS: {processed_fps:.2f}")
    print(f"Pose GFLOPs/frame: {pose_gflops:.3f}")
    print(f"Det GFLOPs/frame : {det_gflops:.3f}" if not math.isnan(det_gflops) else "Det GFLOPs/frame : unavailable")
    print(f"Combined GFLOPs/frame: {combined_gflops:.3f}" if not math.isnan(combined_gflops) else "Combined GFLOPs/frame: unavailable")
    if not math.isnan(combined_flops_per_frame):
        print(f"FLOPs/frame   : {combined_flops_per_frame:.3e}")
        print(f"FLOPs/sec vid : {flops_per_second_of_video:.3e}")
        print(f"FLOPs/min vid : {flops_per_minute_of_video:.3e}")
        print(f"FLOPs full vid: {flops_for_full_video:.3e}")


if __name__ == "__main__":
    main()