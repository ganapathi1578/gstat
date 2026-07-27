#!/usr/bin/env python3
"""
offline_yolopose_racket_video.py

Offline video-only pipeline:
  1) YOLO pose for human skeleton/keypoints
  2) YOLO detect model for racket boxes
  3) Draw both on the same output video
  4) Save timing + FLOPs-style stats to JSON

Usage:
  python tests/test_pose.py \
    --source "C:\\Users\\GANAPATHI\\Downloads\\8053652-hd_1280_720_25fps.mp4" \
    --pose-weights yolo26n-pose.pt \
    --racket-weights yolo26n.pt \
    --device 0

If your racket detector is custom, pass its path instead of yolo26n.pt.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Offline YOLO pose + racket annotator")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a local video file (.mp4, .avi, .mov, etc.). Offline/local only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Default: <source_stem>_pose_racket.mp4",
    )
    parser.add_argument(
        "--pose-weights",
        type=str,
        default="yolo26n-pose.pt",
        help="Pose weights path/name.",
    )
    parser.add_argument(
        "--racket-weights",
        type=str,
        default="yolo26s.pt",
        help="Detect weights path/name for racket detection.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="Pose confidence threshold")
    parser.add_argument("--racket-conf", type=float, default=0.25, help="Racket confidence threshold")
    parser.add_argument("--device", type=str, default=None, help='e.g. "cpu", "0", "0,1"')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Try FP16 on CUDA if supported by your Ultralytics version.",
    )
    parser.add_argument(
        "--pose-gflops",
        type=float,
        default=7.5,
        help="Fallback pose GFLOPs per frame if model.info() cannot provide it.",
    )
    parser.add_argument(
        "--racket-gflops",
        type=float,
        default=None,
        help="Optional racket detector GFLOPs per frame.",
    )
    parser.add_argument(
        "--racket-keyword",
        type=str,
        default="racket",
        help='Keep detections whose class name contains this keyword (case-insensitive).',
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Disable model.fuse() for faster load if needed.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames while processing.",
    )
    return parser.parse_args()


def model_gflops(model: YOLO, imgsz: int, fallback: float | None = None) -> float:
    """
    Try to read GFLOPs from Ultralytics model.info().
    If unavailable, use fallback.
    """
    try:
        info = model.info(verbose=False, imgsz=imgsz)
        # Different Ultralytics versions return different shapes.
        if isinstance(info, tuple) and len(info) >= 4:
            val = safe_float(info[3], None)
            if val is not None:
                return val
    except Exception:
        pass
    return float(fallback) if fallback is not None else float("nan")


def resolve_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, (list, tuple)):
        idx = int(class_id)
        return str(names[idx]) if 0 <= idx < len(names) else str(class_id)
    return str(class_id)


def draw_text_block(img, lines, x=10, y=30, line_gap=28):
    for i, line in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            img,
            line,
            (x + 1, yy + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def extract_wrists(pose_result):
    """
    Return list of wrist points from all detected persons.
    COCO wrist indices: 9 = left wrist, 10 = right wrist.
    Output item: (x, y, conf, person_idx, keypoint_idx)
    """
    wrists = []
    if not getattr(pose_result, "keypoints", None):
        return wrists

    xy = pose_result.keypoints.xy
    conf = getattr(pose_result.keypoints, "conf", None)

    if hasattr(xy, "cpu"):
        xy = xy.cpu().numpy()
    if conf is not None and hasattr(conf, "cpu"):
        conf = conf.cpu().numpy()

    if xy is None or len(xy) == 0:
        return wrists

    for p_idx in range(xy.shape[0]):
        for kpt_idx in (9, 10):
            x, y = xy[p_idx, kpt_idx]
            if np.isnan(x) or np.isnan(y):
                continue
            c = float(conf[p_idx, kpt_idx]) if conf is not None else 1.0
            wrists.append((float(x), float(y), c, p_idx, kpt_idx))
    return wrists


def get_racket_boxes(det_result, names, keyword: str):
    """
    Keep boxes whose class name contains keyword.
    Returns list of dicts:
      {xyxy, conf, cls_id, cls_name}
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

    key = (keyword or "").strip().lower()

    for i in range(len(xyxy)):
        class_name = resolve_name(names, int(cls[i]))
        if key and key not in class_name.lower():
            continue
        out.append(
            {
                "xyxy": xyxy[i].astype(float).tolist(),
                "conf": float(conf[i]),
                "cls_id": int(cls[i]),
                "cls_name": class_name,
            }
        )

    return out


def nearest_wrist_center(box_xyxy, wrists):
    """
    Find nearest wrist to the center of a racket box.
    Returns (wrist_point, distance) or (None, None).
    """
    if not wrists:
        return None, None

    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    best = None
    best_d = None
    for wx, wy, wconf, p_idx, kpt_idx in wrists:
        d = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best_d = d
            best = (wx, wy, wconf, p_idx, kpt_idx)
    return best, best_d


def try_predict(model: YOLO, frame, imgsz: int, conf: float, device, fp16: bool):
    """
    Ultralytics API differs a bit across versions.
    Try fp16/quantize if requested, otherwise normal predict.
    """
    kwargs = dict(imgsz=imgsz, conf=conf, device=device, verbose=False, save=False)

    if fp16:
        # Newer versions may accept quantize="fp16".
        # Older versions may accept half=True.
        try:
            return model.predict(frame, quantize="fp16", **kwargs)
        except TypeError:
            try:
                return model.predict(frame, half=True, **kwargs)
            except TypeError:
                pass

    return model.predict(frame, **kwargs)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    source = Path(args.source)
    if is_url(args.source):
        raise ValueError("This script is for offline/local video files only. URLs are disabled.")
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_pose_racket.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output.with_suffix(".json")

    # Load models
    pose_model = YOLO(args.pose_weights)
    racket_model = YOLO(args.racket_weights) if args.racket_weights else None

    if not args.no_fuse:
        pose_model.fuse()
        if racket_model is not None:
            racket_model.fuse()

    # Model FLOPs
    pose_gflops = model_gflops(pose_model, args.imgsz, fallback=args.pose_gflops)
    racket_gflops = float(args.racket_gflops) if args.racket_gflops is not None else (
        model_gflops(racket_model, args.imgsz, fallback=None) if racket_model is not None else float("nan")
    )

    if math.isnan(pose_gflops):
        pose_gflops = float(args.pose_gflops)

    # Video metadata
    cap_meta = cv2.VideoCapture(str(source))
    if not cap_meta.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {source}")

    src_fps = safe_float(cap_meta.get(cv2.CAP_PROP_FPS), 0.0) or 0.0
    src_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_frame_count = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_meta.release()

    if src_fps <= 0:
        src_fps = 30.0  # fallback for compute estimates and writer

    # Actual read loop
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {source}")

    writer = None
    frame_idx = 0

    total_read_ms = 0.0
    total_pose_ms = 0.0
    total_racket_ms = 0.0
    total_draw_ms = 0.0
    total_write_ms = 0.0
    total_frame_ms = 0.0

    total_racket_boxes = 0
    total_racket_associations = 0

    wall_start = perf_counter()

    while True:
        t_read0 = perf_counter()
        ret, frame = cap.read()
        t_read1 = perf_counter()
        total_read_ms += (t_read1 - t_read0) * 1000.0

        if not ret:
            break

        frame_idx += 1

        # Pose
        t_pose0 = perf_counter()
        pose_results = try_predict(
            pose_model,
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            fp16=args.fp16,
        )
        pose_result = pose_results[0]
        t_pose1 = perf_counter()
        pose_ms = (t_pose1 - t_pose0) * 1000.0
        total_pose_ms += pose_ms

        # Racket detect
        racket_result = None
        racket_ms = 0.0
        if racket_model is not None:
            t_racket0 = perf_counter()
            racket_results = try_predict(
                racket_model,
                frame,
                imgsz=args.imgsz,
                conf=args.racket_conf,
                device=args.device,
                fp16=args.fp16,
            )
            racket_result = racket_results[0]
            t_racket1 = perf_counter()
            racket_ms = (t_racket1 - t_racket0) * 1000.0
            total_racket_ms += racket_ms

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

        wrists = extract_wrists(pose_result)
        racket_boxes = get_racket_boxes(racket_result, getattr(racket_model, "names", None), args.racket_keyword)
        total_racket_boxes += len(racket_boxes)

        for rb in racket_boxes:
            x1, y1, x2, y2 = rb["xyxy"]
            conf = rb["conf"]
            cls_name = rb["cls_name"]

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))

            cv2.rectangle(annotated, p1, p2, (0, 255, 255), 2)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            wrist, dist = nearest_wrist_center(rb["xyxy"], wrists)
            if wrist is not None:
                wx, wy, wconf, p_idx, kpt_idx = wrist
                total_racket_associations += 1
                center = (int(round(0.5 * (x1 + x2))), int(round(0.5 * (y1 + y2))))
                wrist_pt = (int(round(wx)), int(round(wy)))
                cv2.line(annotated, wrist_pt, center, (255, 0, 255), 2)
                cv2.circle(annotated, wrist_pt, 4, (255, 0, 255), -1)

        # Overlay stats on frame
        total_ms_this_frame = pose_ms + racket_ms
        total_frame_ms += total_ms_this_frame
        draw_overlay = [
            f"frame: {frame_idx}/{src_frame_count if src_frame_count > 0 else '?'}",
            f"pose: {pose_ms:.1f} ms",
            f"racket: {racket_ms:.1f} ms" if racket_model is not None else "racket: disabled",
            f"pipeline: {total_ms_this_frame:.1f} ms  |  approx FPS: {1000.0 / total_ms_this_frame:.2f}" if total_ms_this_frame > 0 else "pipeline: n/a",
            f"racket boxes: {len(racket_boxes)}",
        ]
        draw_text_block(annotated, draw_overlay, x=10, y=30, line_gap=26)

        t_draw1 = perf_counter()
        total_draw_ms += (t_draw1 - t_draw0) * 1000.0

        # Writer init on first annotated frame
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, src_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for output: {output}")

        # Write
        t_write0 = perf_counter()
        writer.write(annotated)
        t_write1 = perf_counter()
        total_write_ms += (t_write1 - t_write0) * 1000.0

        if args.show:
            cv2.imshow("YOLO Pose + Racket", annotated)
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
    avg_racket_ms = total_racket_ms / processed_frames if processed_frames else 0.0
    avg_draw_ms = total_draw_ms / processed_frames if processed_frames else 0.0
    avg_write_ms = total_write_ms / processed_frames if processed_frames else 0.0
    avg_total_ms = total_frame_ms / processed_frames if processed_frames else 0.0

    video_duration_s = (src_frame_count / src_fps) if (src_fps > 0 and src_frame_count > 0) else None

    combined_gflops = 0.0
    has_any_flops = False
    for v in (pose_gflops, racket_gflops):
        if v is not None and not math.isnan(v):
            combined_gflops += float(v)
            has_any_flops = True
    if not has_any_flops:
        combined_gflops = float("nan")

    pose_flops_per_frame = pose_gflops * 1e9 if not math.isnan(pose_gflops) else float("nan")
    racket_flops_per_frame = racket_gflops * 1e9 if not math.isnan(racket_gflops) else float("nan")
    combined_flops_per_frame = combined_gflops * 1e9 if not math.isnan(combined_gflops) else float("nan")

    if not math.isnan(combined_flops_per_frame):
        flops_per_second_of_video = combined_flops_per_frame * src_fps
        flops_per_minute_of_video = flops_per_second_of_video * 60.0
        flops_for_full_video = combined_flops_per_frame * processed_frames
    else:
        flops_per_second_of_video = float("nan")
        flops_per_minute_of_video = float("nan")
        flops_for_full_video = float("nan")

    # Wall-time equivalents requested
    wall_seconds_for_1_second_of_video = (avg_total_ms / 1000.0) * src_fps
    wall_seconds_for_1_minute_of_video = wall_seconds_for_1_second_of_video * 60.0

    stats = {
        "input": str(source),
        "output": str(output),
        "pose_weights": args.pose_weights,
        "racket_weights": args.racket_weights,
        "imgsz": args.imgsz,
        "pose_conf": args.pose_conf,
        "racket_conf": args.racket_conf,
        "device": args.device,
        "fp16_requested": bool(args.fp16),
        "source_metadata": {
            "fps": src_fps,
            "width": src_w,
            "height": src_h,
            "frame_count": src_frame_count,
            "duration_s_estimated": video_duration_s,
        },
        "model_info": {
            "pose_gflops_per_frame": pose_gflops,
            "racket_gflops_per_frame": racket_gflops,
            "combined_gflops_per_frame": combined_gflops,
            "pose_flops_per_frame": pose_flops_per_frame,
            "racket_flops_per_frame": racket_flops_per_frame,
            "combined_flops_per_frame": combined_flops_per_frame,
        },
        "runtime": {
            "processed_frames": processed_frames,
            "elapsed_wall_s": elapsed_s,
            "processed_fps": processed_fps,
            "avg_read_ms_per_frame": avg_read_ms,
            "avg_pose_ms_per_frame": avg_pose_ms,
            "avg_racket_ms_per_frame": avg_racket_ms,
            "avg_draw_ms_per_frame": avg_draw_ms,
            "avg_write_ms_per_frame": avg_write_ms,
            "avg_total_pipeline_ms_per_frame": avg_total_ms,
            "wall_seconds_for_1_second_of_video": wall_seconds_for_1_second_of_video,
            "wall_seconds_for_1_minute_of_video": wall_seconds_for_1_minute_of_video,
        },
        "theoretical_compute": {
            "flops_per_second_of_video": flops_per_second_of_video,
            "flops_per_minute_of_video": flops_per_minute_of_video,
            "flops_for_full_video": flops_for_full_video,
        },
        "detections": {
            "total_racket_boxes": total_racket_boxes,
            "total_racket_wrist_associations": total_racket_associations,
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"Output video : {output}")
    print(f"Stats JSON   : {stats_path}")
    print(f"Frames       : {processed_frames}")
    print(f"Elapsed      : {elapsed_s:.3f} s")
    print(f"Wall/frame   : {avg_total_ms:.3f} ms")
    print(f"Processed FPS: {processed_fps:.2f}")
    print(f"Pose GFLOPs/frame   : {pose_gflops:.3f}")
    print(f"Racket GFLOPs/frame : {racket_gflops:.3f}" if not math.isnan(racket_gflops) else "Racket GFLOPs/frame : unavailable")
    print(f"Combined GFLOPs/frame: {combined_gflops:.3f}" if not math.isnan(combined_gflops) else "Combined GFLOPs/frame: unavailable")
    if not math.isnan(combined_flops_per_frame):
        print(f"FLOPs/frame   : {combined_flops_per_frame:.3e}")
        print(f"FLOPs/sec vid : {flops_per_second_of_video:.3e}")
        print(f"FLOPs/min vid : {flops_per_minute_of_video:.3e}")
        print(f"FLOPs full vid: {flops_for_full_video:.3e}")


if __name__ == "__main__":
    main()