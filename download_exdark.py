"""
download_exdark.py
------------------
Automatically downloads authentic low-light images from the official
ExDark (Exclusively Dark) dataset hosted on Hugging Face.
"""

import os
import urllib.request
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images", "exdark")

# Curated selections of real ExDark images representing low-light CCTV/surveillance scenarios
# (Low, Weak, Ambient, and Shadow conditions across key classes)
TARGET_IMAGES = {
    "People": [
        ("2015_06247.jpg", "Weak_Light_Street"),
        ("2015_06248.jpg", "Shadow_Night_Alley"),
        ("2015_06253.jpg", "Low_Light_Pedestrian"),
        ("2015_06260.jpg", "Ambient_Night_Walk"),
    ],
    "Car": [
        ("2015_02407.jpg", "Shadow_Parking"),
        ("2015_02408.jpg", "Ambient_Road"),
        ("2015_02410.jpg", "Low_Light_Sedan"),
        ("2015_02415.jpg", "Night_Traffic"),
    ],
    "Motorbike": [
        ("2015_05745.jpg", "Shadow_Night_Bike"),
        ("2015_05747.png", "Ambient_Parking"),
        ("2015_05750.jpg", "Low_Light_Road"),
    ],
    "Bus": [
        ("2015_01882.jpg", "Low_Light_Bus_Stop"),
        ("2015_01886.jpg", "Night_Transit"),
        ("2015_01890.jpg", "Shadow_Vehicle"),
    ],
    "Bicycle": [
        ("2015_00001.png", "Ambient_Night_Street"),
        ("2015_00003.png", "Weak_Light_Rider"),
        ("2015_00007.jpg", "Low_Light_Bicycle"),
    ]
}

HF_BASE_URL = "https://huggingface.co/datasets/fuhao7i/ExDark/resolve/main/JPEGImages/IMGS"


def download_dataset():
    print("=" * 60)
    print("Downloading authentic ExDark low-light images from Hugging Face...")
    print("=" * 60)

    total_downloaded = 0

    for category, images in TARGET_IMAGES.items():
        cat_dir = os.path.join(SAMPLE_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        for filename, desc in images:
            ext = os.path.splitext(filename)[1]
            out_filename = f"{desc}_{filename}"
            out_path = os.path.join(cat_dir, out_filename)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                print(f"  [Exists] {category}/{out_filename}")
                total_downloaded += 1
                continue

            url = f"{HF_BASE_URL}/{filename}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            try:
                print(f"  Downloading {category} -> {filename} ({desc})...", end=" ")
                with urllib.request.urlopen(req, timeout=20) as resp, open(out_path, "wb") as f:
                    f.write(resp.read())
                print(f"Done ({os.path.getsize(out_path) // 1024} KB)")
                total_downloaded += 1
            except Exception as e:
                print(f"Failed: {e}")

    print("=" * 60)
    print(f"Successfully loaded {total_downloaded} authentic ExDark images into:")
    print(f"  {SAMPLE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    download_dataset()

