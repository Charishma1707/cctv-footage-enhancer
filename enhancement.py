"""
enhancement.py
--------------
Image enhancement techniques for low-light / CCTV footage.

Point Transformations:
  1. Image Negative
  2. Logarithmic Transformation
  3. Gamma Correction (Power-Law)

Histogram-Based:
  4. Histogram Equalization
  5. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Sharpening:
  6. Unsharp Masking

All functions work on single-channel (grayscale) images.
For color images, use the color-space wrapper `enhance_color_image()`
which processes only the luminance channel and preserves chrominance.
"""

import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════
#  POINT TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════

def negative_transform(image: np.ndarray) -> np.ndarray:
    """
    Compute the negative (complement) of a grayscale image.

    Formula:  s = 255 - r

    Reverses intensity levels — bright becomes dark and vice-versa.
    Useful for revealing details hidden in dark regions, especially
    in overexposed highlights of night CCTV footage.
    """
    return (255 - image).astype(np.uint8)


def log_transform(image: np.ndarray) -> np.ndarray:
    """
    Apply a logarithmic intensity transformation.

    Formula:  s = c × log(1 + r)
      where c = 255 / log(1 + 255) to normalize the output to [0, 255].

    The log function compresses the dynamic range:
      - Expands dark (low-intensity) pixels   → makes dark details visible
      - Compresses bright (high-intensity) pixels → prevents washout

    Particularly effective for CCTV images where dark areas contain
    important details (faces, license plates) that are barely visible.
    """
    img_float = image.astype(np.float64)
    # c normalizes so that log(1 + 255) maps to 255
    c = 255.0 / np.log1p(255.0)
    result = c * np.log1p(img_float)
    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma (power-law) correction.

    Formula:  s = 255 × (r / 255) ^ gamma

    - gamma < 1  →  brightens the image  (essential for low-light CCTV)
    - gamma = 1  →  no change (identity)
    - gamma > 1  →  darkens the image

    This is the most commonly used technique in real-world CCTV
    enhancement systems.  Gamma values of 0.3–0.6 typically work
    best for night-time surveillance footage.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be a positive number.")

    normalized = image.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
#  HISTOGRAM-BASED ENHANCEMENT
# ═══════════════════════════════════════════════════════════

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization.

    Redistributes pixel intensities so that the histogram becomes
    approximately uniform across the full 0–255 range.

    Effect:
      - Significantly increases contrast in low-contrast images.
      - Can over-amplify noise in very dark regions (a known limitation).

    Uses OpenCV's optimized implementation which computes the
    cumulative distribution function (CDF) internally.
    """
    return cv2.equalizeHist(image)


def clahe_enhancement(image: np.ndarray,
                      clip_limit: float = 2.0,
                      grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Unlike global histogram equalization, CLAHE:
      1. Divides the image into small tiles (grid_size × grid_size).
      2. Equalizes each tile's histogram independently.
      3. Limits contrast amplification with `clip_limit` to avoid
         noise amplification.
      4. Interpolates between tiles to eliminate boundary artifacts.

    This is the BEST traditional (non-ML) technique for enhancing
    low-light CCTV footage because it:
      - Enhances local contrast without washing out bright areas.
      - Controls noise amplification via the clip limit.
      - Preserves natural-looking results.

    Parameters
    ----------
    clip_limit : float  – Threshold for contrast limiting (1.0–4.0 typical).
    grid_size  : int    – Size of the tile grid (8 is standard).
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    return clahe.apply(image)


# ═══════════════════════════════════════════════════════════
#  SHARPENING
# ═══════════════════════════════════════════════════════════

def unsharp_mask(image: np.ndarray,
                 sigma: float = 1.0,
                 strength: float = 1.5) -> np.ndarray:
    """
    Apply unsharp masking to sharpen the image.

    Process:
      1. Blur the image with a Gaussian filter.
      2. Subtract the blurred image from the original to get the
         "detail" (high-frequency) layer.
      3. Add the detail layer back (scaled by `strength`) to sharpen.

    Formula:  sharpened = original + strength × (original − blurred)

    Useful as a post-processing step after brightness enhancement
    to recover edge detail that may have been softened.

    Parameters
    ----------
    sigma    : float – Gaussian blur sigma (controls what counts as "detail").
    strength : float – How much to amplify the detail layer (1.0–3.0 typical).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
#  COLOR IMAGE SUPPORT
# ═══════════════════════════════════════════════════════════

def enhance_color_image(image_bgr: np.ndarray,
                        method: str,
                        **kwargs) -> np.ndarray:
    """
    Apply an enhancement method to a color image while preserving colors.

    Strategy (YCrCb color-space approach):
      1. Convert BGR → YCrCb  (Y = luminance, Cr/Cb = chrominance).
      2. Apply the enhancement to the Y (brightness) channel ONLY.
      3. Merge the enhanced Y with the original Cr, Cb channels.
      4. Convert back to BGR.

    This preserves the original color information while enhancing
    brightness/contrast — exactly what CCTV enhancement needs.
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    # Map method name to function
    methods = {
        "negative":   lambda img: negative_transform(img),
        "log":        lambda img: log_transform(img),
        "gamma":      lambda img: gamma_correction(img, kwargs.get("gamma", 0.5)),
        "he":         lambda img: histogram_equalization(img),
        "clahe":      lambda img: clahe_enhancement(
                          img,
                          kwargs.get("clip_limit", 2.0),
                          kwargs.get("grid_size", 8)),
        "unsharp":    lambda img: unsharp_mask(
                          img,
                          kwargs.get("sigma", 1.0),
                          kwargs.get("strength", 1.5)),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}")

    # Enhance only the luminance channel
    y_enhanced = methods[method](y_channel)

    # Merge and convert back
    ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
    result_bgr = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    return result_bgr


# ═══════════════════════════════════════════════════════════
#  HISTOGRAM COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Return a 256-bin histogram of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()
