"""
denoising.py
------------
Noise generation and spatial-domain filtering for image denoising.

Noise types:
  1. Gaussian noise   — additive random noise (sensor noise in CCTV cameras)
  2. Salt-and-pepper   — impulse noise (transmission errors, dead pixels)

Spatial Filters:
  1. Mean (averaging) filter
  2. Gaussian filter
  3. Median filter

Each filter supports 3×3 and 5×5 kernel sizes.
"""

import numpy as np
import cv2


# ===================================================================
#  NOISE GENERATION
# ===================================================================

def add_gaussian_noise(image: np.ndarray,
                       mean: float = 0.0,
                       sigma: float = 25.0) -> np.ndarray:
    """
    Add additive Gaussian noise to simulate CCTV sensor noise.

    In real CCTV cameras, sensor noise increases dramatically in
    low-light conditions because the camera amplifies the signal
    (higher ISO / gain), which also amplifies the noise.

    Each pixel gets a random value drawn from N(mean, sigma²) added.
    Higher sigma → more noise (simulates worse lighting / cheaper camera).

    Parameters
    ----------
    image : np.ndarray   – Input image (uint8, grayscale or color).
    mean  : float        – Mean of the Gaussian (usually 0).
    sigma : float        – Standard deviation (noise intensity).
                           Typical: 15-30 for moderate, 40-60 for heavy.
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray,
                              amount: float = 0.05) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise.

    Simulates pixel-level corruption from transmission errors or
    dead/stuck pixels — common in aging CCTV hardware.

    A fraction `amount` of pixels are randomly set to either
    0 (pepper / black) or 255 (salt / white).

    Parameters
    ----------
    image  : np.ndarray – Input image (uint8).
    amount : float      – Fraction of pixels affected (0.0 to 1.0).
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    num_salt = int(total_pixels * amount / 2)
    num_pepper = int(total_pixels * amount / 2)

    # Salt (white pixels)
    rows = np.random.randint(0, image.shape[0], num_salt)
    cols = np.random.randint(0, image.shape[1], num_salt)
    if image.ndim == 3:
        noisy[rows, cols, :] = 255
    else:
        noisy[rows, cols] = 255

    # Pepper (black pixels)
    rows = np.random.randint(0, image.shape[0], num_pepper)
    cols = np.random.randint(0, image.shape[1], num_pepper)
    if image.ndim == 3:
        noisy[rows, cols, :] = 0
    else:
        noisy[rows, cols] = 0

    return noisy


# ===================================================================
#  SPATIAL-DOMAIN FILTERS
# ===================================================================

def apply_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a mean (averaging / box) filter.

    Replaces each pixel with the arithmetic average of its neighbourhood.

    Pros:  Simple, reduces Gaussian noise.
    Cons:  Blurs edges — problematic for CCTV where we need to
           read license plates or identify faces.
    """
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_filter(image: np.ndarray,
                          kernel_size: int = 3) -> np.ndarray:
    """
    Apply a Gaussian low-pass filter.

    Uses a Gaussian-weighted kernel where centre pixels contribute
    more than edge pixels.

    Better than mean filter for CCTV because it:
      - Preserves edges better (gradual falloff vs. uniform weights).
      - Provides smoother, more natural denoising.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_filter(image: np.ndarray,
                        kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median filter.

    Replaces each pixel with the median of its neighbourhood.

    The BEST filter for salt-and-pepper noise because:
      - The median ignores extreme outlier values (the salt/pepper pixels).
      - Preserves edges much better than mean or Gaussian filters.
      - Non-linear — does not introduce new intensity values.

    This is the recommended first-pass filter for old CCTV footage
    with pixel-level artifacts.
    """
    return cv2.medianBlur(image, kernel_size)
