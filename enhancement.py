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
#  FREQUENCY-DOMAIN ENHANCEMENT (HOMOMORPHIC FILTERING)
# ═══════════════════════════════════════════════════════════

def homomorphic_filter(image: np.ndarray,
                       gamma_low: float = 0.5,
                       gamma_high: float = 2.0,
                       d0: float = 30.0,
                       c: float = 1.0) -> np.ndarray:
    """
    Apply Frequency-Domain Homomorphic Filtering based on Retinex illumination-reflectance model.

    Model:
      f(x, y) = i(x, y) * r(x, y)  (Illumination * Reflectance)
      ln(f) = ln(i) + ln(r)        (Converts multiplicative model to additive)

    Process:
      1. Natural logarithm: z = ln(1 + image / 255)
      2. 2D Fast Fourier Transform (FFT) to shift into frequency domain
      3. High-frequency emphasis transfer function:
         H(u,v) = (gamma_high - gamma_low) * [1 - exp(-c * (D^2 / D0^2))] + gamma_low
         - Suppresses low frequencies (gamma_low < 1: evens out illumination/shadows)
         - Amplifies high frequencies (gamma_high > 1: sharpens details and reflectance)
      4. Inverse 2D FFT (IFFT) back to spatial domain
      5. Exponential mapping: exp(Re(ifft)) - 1
      6. Percentile-based dynamic range restoration to [0, 255]

    Parameters:
      gamma_low  : float - Low-frequency gain (0.2 - 0.9, default 0.5)
      gamma_high : float - High-frequency gain (1.2 - 2.5, default 2.0)
      d0         : float - Cutoff frequency radius in pixels (10 - 100, default 30.0)
      c          : float - Sharpness coefficient of filter transition (default 1.0)
    """
    # Normalize channels
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # For 3-channel color, process luminance channel in YCrCb
    if image.ndim == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_filt = homomorphic_filter(y, gamma_low, gamma_high, d0, c)
        return cv2.cvtColor(cv2.merge([y_filt, cr, cb]), cv2.COLOR_YCrCb2BGR)

    img_float = image.astype(np.float64) / 255.0
    img_log = np.log1p(img_float)

    # 2D Fast Fourier Transform
    fft = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(u, v, indexing='ij')
    D2 = U**2 + V**2

    # High-frequency emphasis transfer function H(u,v)
    H = (gamma_high - gamma_low) * (1.0 - np.exp(-c * (D2 / (d0**2 + 1e-6)))) + gamma_low

    # Apply filter in frequency domain
    filtered = fft_shift * H

    # Inverse 2D FFT back to spatial domain
    ifft_shift = np.fft.ifftshift(filtered)
    ifft = np.fft.ifft2(ifft_shift)
    real_part = np.real(ifft)

    # Invert log transform via exponential
    exp_img = np.expm1(real_part)

    # Robust normalization across dynamic range
    p1, p99 = np.percentile(exp_img, (1.0, 99.0))
    if p99 > p1:
        norm_img = np.clip((exp_img - p1) / (p99 - p1 + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm_img = np.clip(exp_img * 255.0, 0, 255).astype(np.uint8)

    return norm_img


def compute_fft_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute centered 2D Fourier log-magnitude spectrum for visualization."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray.astype(np.float64))
    fft_shift = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fft_shift))
    norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_mag


# ═══════════════════════════════════════════════════════════
#  COMBINED HYBRID ENHANCEMENT
# ═══════════════════════════════════════════════════════════

def combined_enhancement(image: np.ndarray,
                         gamma: float = 0.5,
                         clip_limit: float = 2.0,
                         grid_size: int = 8,
                         sharpen_strength: float = 1.2) -> np.ndarray:
    """
    Multi-stage hybrid enhancement pipeline:
      Stage 1: Gamma correction (power-law) to elevate underexposed dark shadows.
      Stage 2: CLAHE (Contrast-Limited Adaptive Histogram Equalization) for tile-based contrast.
      Stage 3: Unsharp masking to restore high-frequency edges and contours.

    This multi-stage hybrid approach produces superior forensic clarity
    by combining brightness expansion, local contrast normalization, and edge sharpening.
    """
    # Stage 1: Gamma correction
    s1 = gamma_correction(image, gamma=gamma)
    # Stage 2: CLAHE
    s2 = clahe_enhancement(s1, clip_limit=clip_limit, grid_size=grid_size)
    # Stage 3: Edge sharpening
    s3 = unsharp_mask(s2, sigma=1.0, strength=sharpen_strength)
    return s3


# ═══════════════════════════════════════════════════════════
#  UNIVERSAL COLOR & GRAYSCALE IMAGE ENHANCEMENT
# ═══════════════════════════════════════════════════════════

def enhance_color_image(image: np.ndarray,
                        method: str,
                        **kwargs) -> np.ndarray:
    """
    Apply an enhancement method to an image (supports both color BGR and grayscale).

    For color images:
      1. Convert BGR → YCrCb  (Y = luminance, Cr/Cb = chrominance).
      2. Apply the enhancement to the Y (brightness) channel ONLY.
      3. Merge the enhanced Y with the original Cr, Cb channels.
      4. Convert back to BGR.

    For grayscale images:
      Applies the enhancement directly without channel conversion.
    """
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
        "combined":   lambda img: combined_enhancement(
                          img,
                          gamma=kwargs.get("gamma", 0.5),
                          clip_limit=kwargs.get("clip_limit", 2.0),
                          grid_size=kwargs.get("grid_size", 8),
                          sharpen_strength=kwargs.get("sharpen_strength", 1.2)),
        "homomorphic": lambda img: homomorphic_filter(
                          img,
                          gamma_low=kwargs.get("gamma_low", 0.5),
                          gamma_high=kwargs.get("gamma_high", 2.0),
                          d0=kwargs.get("d0", 30.0),
                          c=kwargs.get("c", 1.0)),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Supported: {list(methods.keys())}")

    # Normalize image dimensions defensively
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # If single-channel grayscale (2D array), apply directly
    if image.ndim == 2:
        return methods[method](image)

    # If 3-channel color image, enhance only the luminance (Y) in YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    # Enhance luminance channel
    y_enhanced = methods[method](y_channel)

    # Merge and convert back to BGR
    ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
    return cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)


# ═══════════════════════════════════════════════════════════
#  HISTOGRAM COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Return a 256-bin histogram of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()
