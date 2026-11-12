"""
setup_dataset.py
----------------
Setup script for ExDark (Exclusively Dark) dataset and CCTV samples.

Ensures all authentic ExDark dataset images and synthetic CCTV
samples are in place for the application.

Run with:
    python setup_dataset.py
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime
from download_exdark import download_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")


def add_cctv_overlay(image, camera_id="CAM-01", timestamp=None):
    """
    Add a realistic CCTV-style overlay to an image.
    Includes camera ID, timestamp, and REC indicator.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    h, w = image.shape[:2]
    overlay = image.copy()

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = cv2.FONT_HERSHEY_PLAIN

    # Top-left: Camera ID
    cv2.putText(overlay, camera_id, (10, 25),
                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-right: REC indicator with red circle
    cv2.circle(overlay, (w - 55, 18), 6, (0, 0, 255), -1)
    cv2.putText(overlay, "REC", (w - 45, 25),
                font_small, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

    # Bottom-left: Timestamp
    cv2.putText(overlay, timestamp, (10, h - 12),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom-right: Resolution label
    res_text = f"{w}x{h}"
    cv2.putText(overlay, res_text, (w - 90, h - 12),
                font_small, 1.0, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def generate_cctv_samples():
    """Generate synthetic CCTV-style dark images with timestamps."""
    H, W = 480, 640
    np.random.seed(42)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # 1. Dark street
    img = np.random.randint(3, 28, (H, W), dtype=np.uint8)
    cv2.ellipse(img, (W // 3, H // 4), (70, 90), 0, 0, 360, 65, -1)
    cv2.ellipse(img, (W // 3, H // 4), (35, 45), 0, 0, 360, 105, -1)
    cv2.rectangle(img, (380, 60), (560, 380), 18, -1)
    cv2.rectangle(img, (400, 80), (540, 360), 28, -1)
    for wy in range(100, 340, 60):
        for wx in range(420, 530, 50):
            cv2.rectangle(img, (wx, wy), (wx + 20, wy + 25),
                          np.random.randint(35, 65), -1)
    cv2.rectangle(img, (0, 340), (W, H), 12, -1)
    cv2.line(img, (80, 390), (560, 390), 35, 2)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    overlay = add_cctv_overlay(img, "CAM-01  STREET", "2024-11-12  23:47:15")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "cctv_street.png"), overlay)

    # 2. Dark parking lot
    img = np.random.randint(2, 22, (H, W), dtype=np.uint8)
    for x in range(40, 520, 110):
        cv2.rectangle(img, (x, 180), (x + 85, 300), 30, -1)
        cv2.rectangle(img, (x + 5, 185), (x + 80, 295), 22, -1)
        cv2.circle(img, (x + 10, 290), 4, 50, -1)
        cv2.circle(img, (x + 75, 290), 4, 50, -1)
    for cx in [150, 350, 550]:
        cv2.circle(img, (cx, 50), 55, 50, -1)
        cv2.circle(img, (cx, 50), 28, 85, -1)
    for x in range(40, 560, 110):
        cv2.line(img, (x, 310), (x, 400), 25, 1)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    overlay = add_cctv_overlay(img, "CAM-02  PARKING", "2024-11-12  23:47:15")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "cctv_parking.png"), overlay)

    # 3. Dark alley
    img = np.random.randint(1, 16, (H, W), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (90, H), 10, -1)
    cv2.rectangle(img, (W - 90, 0), (W, H), 10, -1)
    cv2.circle(img, (W // 2, 40), 25, 130, -1)
    cv2.circle(img, (W // 2, 40), 50, 55, -1)
    cv2.circle(img, (W // 2, 40), 90, 25, -1)
    for y in range(H):
        brightness = max(0, int(28 * (1 - y / H)))
        img[y, 100:W - 100] = np.clip(
            img[y, 100:W - 100].astype(np.int16) + brightness, 0, 255
        ).astype(np.uint8)
    cv2.ellipse(img, (320, 250), (15, 40), 0, 0, 360, 8, -1)
    cv2.ellipse(img, (320, 200), (10, 12), 0, 0, 360, 8, -1)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    overlay = add_cctv_overlay(img, "CAM-03  ALLEY", "2024-11-12  23:47:15")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "cctv_alley.png"), overlay)


if __name__ == "__main__":
    print("=" * 60)
    print("Setting up CCTV Night Footage Enhancer Datasets...")
    print("=" * 60)
    print("1. Generating synthetic CCTV surveillance images...")
    generate_cctv_samples()
    print("2. Downloading authentic ExDark low-light images...")
    download_dataset()
    print("\n✅ Dataset setup complete! Launch with: streamlit run app.py")
