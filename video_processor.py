"""
video_processor.py
------------------
CCTV Video Processing Pipeline for night surveillance footage.

Features:
  1. Frame-by-frame streaming enhancement using YCrCb + CLAHE / Gamma
  2. Real-time performance benchmarking (FPS, frame latency in ms)
  3. Realistic CCTV HUD overlay burning (Camera ID, Timestamp, REC badge)
  4. Built-in synthetic night video generator for instant testing
"""

import os
import time
import cv2
import numpy as np
from datetime import datetime
from enhancement import enhance_color_image, clahe_enhancement, gamma_correction


def create_browser_compatible_videowriter(output_path: str, fps: float, width: int, height: int):
    """
    Creates an OpenCV VideoWriter optimized for HTML5 browser playback.
    Prefers H.264 (avc1), which is supported natively across all modern web browsers.
    """
    for codec in ['avc1', 'H264', 'mp4v']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, codec
        except Exception:
            continue
    raise RuntimeError("Could not initialize VideoWriter with any supported codec.")


def extract_video_frame(video_path: str, frame_idx: int = 0) -> np.ndarray | None:
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_metadata(video_path: str) -> dict:
    """Retrieve metadata of a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "resolution": f"{width} × {height}",
    }


def add_video_cctv_hud(frame: np.ndarray,
                       camera_id: str = "CAM-01 [NIGHT-MODE]",
                       timestamp_str: str = None,
                       is_enhanced: bool = False,
                       method_label: str = "ENHANCED") -> np.ndarray:
    """Burn realistic CCTV on-screen display (OSD) HUD onto video frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = cv2.FONT_HERSHEY_PLAIN

    # Top-Left: Camera ID & Mode
    mode_text = f"{camera_id} - {method_label.upper() if is_enhanced else 'RAW CAPTURE'}"
    color_banner = (0, 255, 128) if is_enhanced else (200, 200, 200)
    cv2.putText(out, mode_text, (12, 25), font, 0.55, color_banner, 1, cv2.LINE_AA)

    # Top-Right: REC badge
    cv2.circle(out, (w - 60, 18), 5, (0, 0, 255), -1)
    cv2.putText(out, "LIVE", (w - 48, 23), font_small, 1.1, (0, 0, 255), 1, cv2.LINE_AA)

    # Bottom-Left: Live Timestamp
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(out, timestamp_str, (12, h - 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom-Right: FPS / Resolution tag
    cv2.putText(out, f"{w}x{h} HD", (w - 85, h - 14), font_small, 1.0, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def process_video(input_path: str,
                  output_path: str,
                  method: str = "clahe",
                  max_frames: int | None = None,
                  burn_hud: bool = True,
                  progress_callback = None,
                  **kwargs) -> dict:
    """
    Enhance a video stream frame-by-frame and compute real-time throughput metrics.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    # Setup Video Writer (Optimized for HTML5 Web Browsers: H.264/avc1)
    out_writer, used_codec = create_browser_compatible_videowriter(output_path, orig_fps, width, height)

    processed_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Universal frame enhancement (works on both color and grayscale)
        enhanced = enhance_color_image(frame, method, **kwargs)

        if burn_hud:
            enhanced = add_video_cctv_hud(enhanced, is_enhanced=True, method_label=f"ENHANCED [{method.upper()}]")

        out_writer.write(enhanced)
        processed_count += 1

        if progress_callback and total_frames > 0:
            progress_callback(processed_count / total_frames)

        if max_frames and processed_count >= max_frames:
            break

    total_time = time.perf_counter() - start_time
    cap.release()
    out_writer.release()

    effective_fps = processed_count / total_time if total_time > 0 else 0
    latency_ms = (total_time / processed_count * 1000) if processed_count > 0 else 0

    return {
        "processed_frames": processed_count,
        "elapsed_time_s": round(total_time, 2),
        "processing_fps": round(effective_fps, 1),
        "latency_ms": round(latency_ms, 2),
        "realtime_capable": effective_fps >= 24.0,
        "output_path": output_path,
    }


def generate_synthetic_cctv_video(output_path: str,
                                  num_frames: int = 75,
                                  width: int = 640,
                                  height: int = 480,
                                  fps: float = 25.0) -> str:
    """
    Generate a realistic night CCTV surveillance clip:
    Simulates a dimly lit parking lot with a car slowly driving and headlight motion.
    """
    writer, used_codec = create_browser_compatible_videowriter(output_path, fps, width, height)

    np.random.seed(42)

    for i in range(num_frames):
        # Dark background with mild sensor noise
        frame = np.random.randint(4, 25, (height, width), dtype=np.uint8)

        # Distant faint streetlamp light pool
        cv2.circle(frame, (180, 100), 110, 45, -1)
        cv2.circle(frame, (180, 100), 50, 75, -1)

        # Moving car across the frame
        car_x = int(80 + (i / num_frames) * (width - 240))
        car_y = 260

        # Car body (dark metal)
        cv2.rectangle(frame, (car_x, car_y), (car_x + 130, car_y + 45), 38, -1)
        cv2.rectangle(frame, (car_x + 25, car_y - 20), (car_x + 95, car_y), 28, -1)

        # Wheels
        cv2.circle(frame, (car_x + 25, car_y + 45), 10, 15, -1)
        cv2.circle(frame, (car_x + 105, car_y + 45), 10, 15, -1)

        # Headlight beam casting light ahead
        beam_pts = np.array([
            [car_x + 130, car_y + 20],
            [car_x + 260, car_y - 15],
            [car_x + 270, car_y + 55],
            [car_x + 130, car_y + 35]
        ], np.int32)
        cv2.fillPoly(frame, [beam_pts], 60)

        # Moving pedestrian silhouette in shadows
        ped_x = 480 - int(i * 0.8)
        ped_y = 220
        # Head
        cv2.circle(frame, (ped_x, ped_y), 7, 20, -1)
        # Body
        cv2.rectangle(frame, (ped_x - 5, ped_y + 7), (ped_x + 5, ped_y + 32), 18, -1)

        # Blur slightly to simulate optics
        frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Add CCTV HUD
        hud_frame = add_video_cctv_hud(bgr, "CAM-04 PARKING-LOT", is_enhanced=False)
        writer.write(hud_frame)

    writer.release()
    return output_path

