"""
evaluation.py
-------------
Image quality evaluation metrics for comparing enhanced/denoised
images against their originals.

Implements:
  1. MSE  – Mean Squared Error
  2. PSNR – Peak Signal-to-Noise Ratio
  3. SSIM – Structural Similarity Index (manual implementation)

No external dependencies beyond NumPy and OpenCV.
"""

import numpy as np
import cv2


def compute_mse(original: np.ndarray, compared: np.ndarray) -> float:
    """
    Mean Squared Error between two images.

    Formula:
        MSE = (1 / M×N) × Σ (original[i,j] - compared[i,j])²

    Interpretation:
      MSE = 0     → identical images
      Higher MSE  → more distortion
    """
    err = original.astype(np.float64) - compared.astype(np.float64)
    return float(np.mean(err ** 2))


def compute_psnr(original: np.ndarray, compared: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Formula:
        PSNR = 10 × log₁₀(MAX² / MSE)

    where MAX = 255 for 8-bit images.

    Interpretation:
      PSNR > 40 dB  → excellent quality
      30–40 dB      → good quality
      20–30 dB      → acceptable
      < 20 dB       → poor quality
      ∞             → identical images
    """
    mse = compute_mse(original, compared)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(original: np.ndarray, compared: np.ndarray) -> float:
    """
    Structural Similarity Index (manual implementation).

    SSIM considers three components:
      - Luminance:  comparison of mean intensities
      - Contrast:   comparison of standard deviations
      - Structure:  comparison of correlation

    Formula (per window, then averaged):
        SSIM(x, y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂)
                     ─────────────────────────────
                     (μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)

    Uses an 11×11 Gaussian window (standard SSIM practice).

    Returns a value in [-1, 1] where 1 = identical images.
    Typical good denoising produces SSIM > 0.85.

    Handles both grayscale and color images (converts color to gray).
    """
    # Ensure grayscale
    if original.ndim == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if compared.ndim == 3:
        compared = cv2.cvtColor(compared, cv2.COLOR_BGR2GRAY)

    img1 = original.astype(np.float64)
    img2 = compared.astype(np.float64)

    # Constants from the original SSIM paper (Wang et al., 2004)
    L = 255.0
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2        # ≈ 6.5025
    C2 = (K2 * L) ** 2        # ≈ 58.5225

    # 11×11 Gaussian-weighted local statistics
    ksize = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, ksize, sigma)
    mu2 = cv2.GaussianBlur(img2, ksize, sigma)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, ksize, sigma) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1 * img2, ksize, sigma) - mu1_mu2

    # SSIM map
    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_all_metrics(original: np.ndarray,
                        compared: np.ndarray) -> dict:
    """
    Compute all quality metrics at once.

    Returns dict with keys: 'MSE', 'PSNR', 'SSIM'.
    """
    return {
        "MSE":  round(compute_mse(original, compared), 2),
        "PSNR": round(compute_psnr(original, compared), 2),
        "SSIM": round(compute_ssim(original, compared), 4),
    }
