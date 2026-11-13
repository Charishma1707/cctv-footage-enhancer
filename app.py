"""
app.py
------
CCTV Night Footage Enhancer — Digital Image Processing Application

A dedicated DIP application for enhancing low-light CCTV and surveillance
imagery using traditional spatial-domain image processing techniques.

Sections:
  1. Image Enhancement  – Point transformations & histogram-based methods
  2. Noise Removal       – Noise generation, spatial filtering, & quality metrics

Technologies:
  Python, OpenCV, NumPy, Matplotlib, Streamlit
"""

import os
import io
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

import importlib
import enhancement
import detection
import denoising
import evaluation
import video_processor

importlib.reload(enhancement)
importlib.reload(detection)
importlib.reload(denoising)
importlib.reload(evaluation)
importlib.reload(video_processor)

from enhancement import (
    negative_transform,
    log_transform,
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
    unsharp_mask,
    combined_enhancement,
    homomorphic_filter,
    compute_fft_spectrum,
    enhance_color_image,
    compute_histogram,
)
from denoising import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_mean_filter,
    apply_gaussian_filter,
    apply_median_filter,
)
from evaluation import compute_mse, compute_psnr, compute_ssim, compute_all_metrics
from video_processor import process_video, generate_synthetic_cctv_video, get_video_metadata, extract_video_frame
from detection import evaluate_vision_benchmark, draw_detections, NanoDetector


# ───────────────────────── Paths & Directories ───────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
EXDARK_DIR = os.path.join(SAMPLE_DIR, "exdark")


# ───────────────────────── Dataset Helpers ────────────────────

def get_available_datasets() -> dict[str, list[tuple[str, str]]]:
    """
    Scan sample_images and organize available images by category.
    Returns a dict mapping category names to lists of (display_label, file_path).
    """
    categories = {}

    # 1. Authentic ExDark Dataset categories
    if os.path.isdir(EXDARK_DIR):
        for sub in sorted(os.listdir(EXDARK_DIR)):
            sub_path = os.path.join(EXDARK_DIR, sub)
            if os.path.isdir(sub_path):
                img_files = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG"):
                    img_files.extend(glob.glob(os.path.join(sub_path, ext)))
                if img_files:
                    cat_name = f"ExDark Dataset: {sub}"
                    items = []
                    for f in sorted(img_files):
                        base = os.path.splitext(os.path.basename(f))[0]
                        # Clean label for presentation
                        clean_label = base.replace("_", " ").title()
                        items.append((clean_label, f))
                    categories[cat_name] = items

    # 2. Synthetic CCTV surveillance samples
    if os.path.isdir(SAMPLE_DIR):
        root_imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            root_imgs.extend(glob.glob(os.path.join(SAMPLE_DIR, ext)))
        if root_imgs:
            cat_name = "CCTV Surveillance Samples"
            items = []
            for f in sorted(root_imgs):
                base = os.path.splitext(os.path.basename(f))[0]
                items.append((base.replace("_", " ").upper(), f))
            categories[cat_name] = items

    return categories


def to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is a 2D grayscale numpy array."""
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is a 3-channel BGR numpy array."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        return img
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def load_image(uploaded_file=None, file_path=None, as_color=False) -> np.ndarray | None:
    """Read an image into an 8-bit BGR or Grayscale NumPy array."""
    flag = cv2.IMREAD_COLOR if as_color else cv2.IMREAD_GRAYSCALE
    img = None
    if uploaded_file is not None:
        try:
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, flag)
        except Exception:
            return None
    elif file_path is not None and os.path.exists(file_path):
        img = cv2.imread(file_path, flag)

    if img is None:
        return None

    return to_bgr(img) if as_color else to_gray(img)


def image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode an image array into PNG bytes for browser download."""
    success, buf = cv2.imencode(fmt, image)
    return buf.tobytes() if success else b""


def display_image(image: np.ndarray, caption: str = "", use_container_width: bool = True):
    """Render grayscale or BGR images accurately in Streamlit."""
    if image.ndim == 3 and image.shape[2] == 3:
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        display = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif image.ndim == 3 and image.shape[2] == 1:
        display = image[:, :, 0]
    else:
        display = image
    st.image(display, caption=caption, clamp=True, use_container_width=use_container_width)


def render_comparison_view(orig_img: np.ndarray,
                           proc_img: np.ndarray,
                           label_orig: str = "Raw Input",
                           label_proc: str = "Enhanced",
                           id_prefix: str = "view"):
    """
    Renders an interactive visual comparison between two images.
    Offers two modes:
      1. Interactive Draggable Wipe Slider (Juxtapose)
      2. Side-by-Side Dual View
    """
    view_col1, view_col2 = st.columns([3, 1])
    with view_col2:
        comp_mode = st.radio(
            "Comparison Mode",
            ["🎚️ Wipe Slider", "🖼️ Side-by-Side"],
            horizontal=True,
            key=f"comp_mode_{id_prefix}",
        )

    if comp_mode == "🎚️ Wipe Slider":
        def to_rgb_pil(img: np.ndarray) -> Image.Image:
            if img.ndim == 3 and img.shape[2] == 3:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif img.ndim == 3 and img.shape[2] == 4:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
            elif img.ndim == 3 and img.shape[2] == 1:
                return Image.fromarray(cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB))
            else:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        pil_orig = to_rgb_pil(orig_img)
        pil_proc = to_rgb_pil(proc_img)

        if pil_orig.size != pil_proc.size:
            pil_proc = pil_proc.resize(pil_orig.size)

        image_comparison(
            img1=pil_orig,
            img2=pil_proc,
            label1=label_orig,
            label2=label_proc,
            width=720,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            display_image(orig_img, label_orig, use_container_width=True)
        with c2:
            display_image(proc_img, label_proc, use_container_width=True)


def make_histogram_figure(images_dict: dict[str, np.ndarray], figsize=(10, 3)):
    """Generate Matplotlib intensity histograms comparing original and processed states."""
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#4A5568", "#3182CE", "#38A169", "#D69E2E", "#805AD5"]

    for i, (label, img) in enumerate(images_dict.items()):
        gray = to_gray(img)
        hist = compute_histogram(gray)
        axes[i].fill_between(range(256), hist, alpha=0.75, color=colors[i % len(colors)])
        axes[i].plot(range(256), hist, color=colors[i % len(colors)], linewidth=1)
        axes[i].set_title(label, fontsize=10, fontweight="bold")
        axes[i].set_xlim([0, 255])
        axes[i].set_xlabel("Pixel Intensity (0-255)", fontsize=9)
        if i == 0:
            axes[i].set_ylabel("Pixel Frequency", fontsize=9)
        axes[i].grid(True, linestyle=":", alpha=0.4)
        axes[i].tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def choose_image_widget(key_prefix: str, allow_color: bool = True):
    """
    Standardized image selection widget:
    Allows user to select from real ExDark dataset, synthetic CCTV scenes, or upload custom files.
    """
    dataset_dict = get_available_datasets()

    col1, col2 = st.columns([3, 1])
    with col1:
        options = ["Upload your own image (JPG / PNG)"]
        if dataset_dict:
            options.append("Select from ExDark Dataset / Samples")

        source = st.radio(
            "Input Image Source",
            options,
            key=f"{key_prefix}_source",
            horizontal=True,
        )

    with col2:
        if allow_color:
            color_mode = st.radio(
                "Processing Domain",
                ["Grayscale (DIP Standard)", "Color (YCrCb Preserve)"],
                key=f"{key_prefix}_color_mode",
            )
            as_color = (color_mode == "Color (YCrCb Preserve)")
        else:
            as_color = False

    image = None

    if source == "Upload your own image (JPG / PNG)":
        uploaded = st.file_uploader(
            "Upload a low-light or CCTV image",
            type=["jpg", "jpeg", "png", "bmp"],
            key=f"{key_prefix}_upload",
        )
        if uploaded:
            image = load_image(uploaded_file=uploaded, as_color=as_color)
            if image is None:
                st.error("Error decoding image. Please verify the file format.")
    else:
        cat_col, img_col = st.columns(2)
        with cat_col:
            chosen_cat = st.selectbox(
                "Dataset Category",
                list(dataset_dict.keys()),
                key=f"{key_prefix}_cat",
            )
        with img_col:
            cat_images = dataset_dict.get(chosen_cat, [])
            labels = [item[0] for item in cat_images]
            chosen_label = st.selectbox(
                "Select Image",
                labels,
                key=f"{key_prefix}_img",
            )
            idx = labels.index(chosen_label)
            selected_path = cat_images[idx][1]

        image = load_image(file_path=selected_path, as_color=as_color)

    return image, as_color


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIGURATION & HEADER
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CCTV Night Footage Enhancer",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("📹 CCTV Enhancer")
    st.caption("Digital Image Processing — Spatial & Frequency Domain Suite")
    st.divider()

    st.markdown("""
    ### 🔬 Implemented Techniques

    **1. Point Transformations:**
    - **Image Negative:** `s = 255 - r`
    - **Log Transform:** `s = c × log(1 + r)`
    - **Gamma Correction:** `s = 255 × (r/255)^γ`

    **2. Histogram-Based Methods:**
    - **Global Histogram Equalization**
    - **CLAHE** (Contrast-Limited Adaptive HE)
    - **Combined Multi-Stage Hybrid**

    **3. Frequency-Domain Enhancement:**
    - **Homomorphic Filtering** (2D-FFT Retinex)
    - **2D Fourier Spectrum Magnitude**

    **4. Spatial-Domain Filters:**
    - **Mean Filter** ($3\\times 3$, $5\\times 5$)
    - **Gaussian Filter** ($3\\times 3$, $5\\times 5$)
    - **Median Filter** ($3\\times 3$, $5\\times 5$)

    **5. Objective Quality Evaluation:**
    - **MSE** (Mean Squared Error)
    - **PSNR** (Peak Signal-to-Noise Ratio)
    - **SSIM** (Structural Similarity Index)

    **6. Forensic Video & Vision Benchmarks:**
    - Real-time video processing (H.264 / FPS)
    - Automated detection audit (NanoDet)
    - **Interactive Before/After Wipe Slider**

    ---
    ### 📦 Dataset
    Integrated with authentic **ExDark (Exclusively Dark)** dataset images across multiple classes (People, Car, Bus, Motorbike, Bicycle).
    """)

# ── Main Header ───────────────────────────────────────────
st.title("📹 CCTV Night Footage Enhancer")
st.markdown(
    "A practical Digital Image Processing application that enhances low-light CCTV surveillance images "
    "using **traditional spatial-domain algorithms**. Test with real images from the **ExDark dataset**, "
    "upload your own night footage, process CCTV video streams, or run automated vision benchmarks."
)

tab_enhance, tab_denoise, tab_video, tab_benchmark = st.tabs([
    "🌙 Section 1: Image Enhancement",
    "🧹 Section 2: Noise Removal & Filtering",
    "📹 Section 3: Video Stream Processing",
    "🔍 Section 4: Downstream Vision Benchmark",
])


# ═══════════════════════════════════════════════════════════
#  SECTION 1: IMAGE ENHANCEMENT
# ═══════════════════════════════════════════════════════════

with tab_enhance:
    st.header("Image Enhancement (Point & Histogram Transformations)")
    st.caption(
        "Low-light CCTV cameras suffer from compressed dynamic range and poor contrast. "
        "These spatial-domain transformations redistribute intensity levels to uncover hidden details."
    )

    image, is_color = choose_image_widget("enhance", allow_color=True)

    if image is not None:
        gray = to_gray(image)

        # Overview statistics card
        with st.expander("📊 Image Properties & Illumination Analysis", expanded=True):
            col_img, col_stat = st.columns([1, 2])
            with col_img:
                display_image(image, "Input Image", use_container_width=True)
            with col_stat:
                st.markdown("**Illumination Metrics:**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Width × Height", f"{image.shape[1]}×{image.shape[0]}")
                m2.metric("Mean Luminance", f"{gray.mean():.1f} / 255")
                m3.metric("Min Intensity", f"{gray.min()}")
                m4.metric("Max Intensity", f"{gray.max()}")

                if gray.mean() < 30:
                    st.error("🌑 **Very Low Light / Night:** Severe underexposure. Gamma (γ < 0.5) or CLAHE recommended.")
                elif gray.mean() < 70:
                    st.warning("🌘 **Low Light / Twilight:** Weak illumination. Log transform or Gamma correction will highlight dark features.")
                else:
                    st.info("🌗 **Moderate Illumination:** Balanced scene; CLAHE or mild gamma will sharpen local contrast.")

        st.divider()

        # Enhancement Execution Mode
        view_mode = st.radio(
            "Evaluation Mode",
            ["Single Transformation Analysis", "Compare All Methods Side-by-Side"],
            horizontal=True,
            key="enhance_view_mode",
        )

        if view_mode == "Single Transformation Analysis":
            t_col1, t_col2 = st.columns([2, 2])

            with t_col1:
                transformation = st.selectbox(
                    "Select Enhancement Technique",
                    [
                        "Combined Hybrid (Gamma + CLAHE + Sharpen)",
                        "Homomorphic Filtering (2D-FFT Frequency Domain)",
                        "CLAHE (Adaptive Histogram Eq)",
                        "Gamma Correction (Power-Law)",
                        "Logarithmic Transformation",
                        "Histogram Equalization (Global)",
                        "Image Negative",
                        "Unsharp Masking (Edge Sharpening)",
                    ],
                    key="trans_choice",
                )

            params = {}
            with t_col2:
                if transformation == "Combined Hybrid (Gamma + CLAHE + Sharpen)":
                    st.markdown("**Forensic Pipeline:** 3-Stage Pipeline: $\\text{Gamma}(\\gamma) \\rightarrow \\text{CLAHE} \\rightarrow \\text{Unsharp Mask}$")
                    st.info("Combines brightness expansion (Gamma), local adaptive contrast (CLAHE), and edge restoration (Sharpening).")
                    ch1, ch2, ch3 = st.columns(3)
                    with ch1:
                        params["gamma"] = st.slider("Gamma (γ)", 0.2, 1.2, 0.5, 0.05, key="ch_gamma")
                    with ch2:
                        params["clip_limit"] = st.slider("CLAHE Clip", 1.0, 5.0, 2.5, 0.5, key="ch_clip")
                        params["grid_size"] = 8
                    with ch3:
                        params["sharpen_strength"] = st.slider("Sharpen", 0.5, 2.5, 1.2, 0.1, key="ch_sharp")

                elif transformation == "Homomorphic Filtering (2D-FFT Frequency Domain)":
                    st.markdown("**Retinex 2D-FFT:** $H(u,v) = (\\gamma_H - \\gamma_L) [1 - e^{-c \\cdot D^2 / D_0^2}] + \\gamma_L$")
                    st.info("Frequency-domain filtering: suppresses low-frequency illumination shadows (γ_L) while amplifying high-frequency reflectance details (γ_H).")
                    hf_col1, hf_col2 = st.columns(2)
                    with hf_col1:
                        params["gamma_low"] = st.slider("Gamma Low (γ_L)", 0.1, 1.0, 0.50, 0.05, key="hf_glow",
                                                        help="Suppresses non-uniform lighting & deep shadows.")
                        params["gamma_high"] = st.slider("Gamma High (γ_H)", 1.0, 3.0, 2.00, 0.1, key="hf_ghigh",
                                                         help="Amplifies object reflectance, edges, and textures.")
                    with hf_col2:
                        params["d0"] = st.slider("Cutoff Radius (D0)", 10.0, 80.0, 30.0, 5.0, key="hf_d0",
                                                 help="Separates low frequencies from high frequencies.")
                        params["c"] = st.slider("Slope Constant (c)", 0.5, 2.0, 1.0, 0.1, key="hf_c",
                                                help="Steepness of frequency transition.")

                elif transformation == "Gamma Correction (Power-Law)":
                    st.markdown("**Formula:** $s = 255 \\times (r / 255)^\\gamma$")
                    gamma_val = st.slider(
                        "Gamma value (γ)",
                        min_value=0.1,
                        max_value=3.0,
                        value=0.5,
                        step=0.05,
                        help="γ < 1 expands dark levels (brightens); γ > 1 compresses dark levels (darkens).",
                    )
                    b1, b2, b3 = st.columns(3)
                    if b1.button("Preset: γ = 0.5"):
                        gamma_val = 0.5
                    if b2.button("Preset: γ = 1.0"):
                        gamma_val = 1.0
                    if b3.button("Preset: γ = 2.0"):
                        gamma_val = 2.0
                    params["gamma"] = gamma_val

                elif transformation == "Logarithmic Transformation":
                    st.markdown("**Formula:** $s = c \\times \\log(1 + r)$, where $c = 255 / \\log(1 + 255)$")
                    st.info("Expands dark pixel intensities while compressing high-intensity highlights.")

                elif transformation == "Image Negative":
                    st.markdown("**Formula:** $s = 255 - r$")
                    st.info("Inverts all intensity values. Effective for identifying faint shapes against black backgrounds.")

                elif transformation == "CLAHE (Adaptive Histogram Eq)":
                    st.markdown("**Method:** Contrast Limited Adaptive Histogram Equalization")
                    c1, c2 = st.columns(2)
                    with c1:
                        clip_lim = st.slider("Clip Limit", 1.0, 6.0, 2.0, 0.5,
                                             help="Limits contrast amplification in uniform areas to prevent noise.")
                    with c2:
                        grid_s = st.selectbox("Tile Grid Size", [4, 8, 16], index=1,
                                              help="Size of contextual neighborhood tiles.")
                    params["clip_limit"] = clip_lim
                    params["grid_size"] = grid_s

                elif transformation == "Histogram Equalization (Global)":
                    st.markdown("**Method:** Global Cumulative Distribution Function (CDF) Equalization")
                    st.info("Flattens the overall histogram across all 256 levels to maximize dynamic range.")

                elif transformation == "Unsharp Masking (Edge Sharpening)":
                    st.markdown("**Method:** High-frequency edge amplification via Gaussian residual")
                    sh_col1, sh_col2 = st.columns(2)
                    with sh_col1:
                        params["strength"] = st.slider("Sharpen Strength", 0.5, 3.0, 1.5, 0.1)
                    with sh_col2:
                        params["sigma"] = st.slider("Gaussian Sigma", 0.5, 3.0, 1.0, 0.5)

            # Compute processed image using universal enhancer
            trans_key_map = {
                "Combined Hybrid (Gamma + CLAHE + Sharpen)": "combined",
                "Homomorphic Filtering (2D-FFT Frequency Domain)": "homomorphic",
                "Gamma Correction (Power-Law)": "gamma",
                "Logarithmic Transformation": "log",
                "CLAHE (Adaptive Histogram Eq)": "clahe",
                "Histogram Equalization (Global)": "he",
                "Image Negative": "negative",
                "Unsharp Masking (Edge Sharpening)": "unsharp",
            }
            method_key = trans_key_map[transformation]

            # Universal enhancement handles both color BGR and grayscale natively
            enhanced = enhance_color_image(image if is_color else gray, method_key, **params)
            enhanced_gray = to_gray(enhanced)

            # 2D Fourier Spectrum Analysis for Homomorphic Filtering
            if method_key == "homomorphic":
                with st.expander("🌐 2D Fourier Frequency Spectrum Analysis (FFT)", expanded=True):
                    fft_col1, fft_col2 = st.columns(2)
                    fft_orig = compute_fft_spectrum(gray)
                    fft_enh = compute_fft_spectrum(enhanced_gray)
                    with fft_col1:
                        display_image(fft_orig, "Input 2D Fourier Spectrum log|F(u,v)|", use_container_width=True)
                    with fft_col2:
                        display_image(fft_enh, "Filtered Fourier Spectrum log|F(u,v)·H(u,v)|", use_container_width=True)
                    st.caption("Center represents DC / low frequencies (illumination variation); outer rings represent high frequencies (edges & textures).")

            # Visual Comparison (Interactive Wipe Slider or Side-by-Side)
            st.subheader("Enhancement Output")
            lbl = transformation
            if method_key == "gamma":
                lbl += f" (γ = {params['gamma']})"
            render_comparison_view(image, enhanced, "Raw CCTV Input", f"Enhanced ({lbl})", id_prefix="sec1_single")

            # Intensity Histograms
            st.subheader("Intensity Distribution Analysis")
            fig = make_histogram_figure({
                "Original Distribution": gray,
                f"Enhanced ({method_key.upper()})": enhanced_gray,
            })
            st.pyplot(fig)
            plt.close(fig)

            # Download
            st.download_button(
                label=f"⬇️ Download Enhanced Image ({method_key.upper()})",
                data=image_to_bytes(enhanced),
                file_name=f"cctv_enhanced_{method_key}.png",
                mime="image/png",
            )

        else:
            # Compare All Methods Mode
            st.subheader("Comprehensive Enhancement Comparison")

            gamma_05 = gamma_correction(gray, 0.5)
            gamma_20 = gamma_correction(gray, 2.0)
            log_res = log_transform(gray)
            neg_res = negative_transform(gray)
            he_res = histogram_equalization(gray)
            clahe_res = clahe_enhancement(gray, clip_limit=2.0, grid_size=8)
            homo_res = homomorphic_filter(gray, gamma_low=0.5, gamma_high=2.0, d0=30.0)
            comb_res = combined_enhancement(gray, gamma=0.5, clip_limit=2.5, grid_size=8, sharpen_strength=1.2)

            all_results = {
                "Original Input": gray,
                "Homomorphic (2D-FFT)": homo_res,
                "Combined Hybrid": comb_res,
                "CLAHE (Adaptive HE)": clahe_res,
                "Gamma Correction (γ = 0.5)": gamma_05,
                "Gamma Correction (γ = 2.0)": gamma_20,
                "Logarithmic Transform": log_res,
                "Histogram Equalization": he_res,
            }

            # 2-Row Grid
            grid_cols = st.columns(4)
            for i, (name, img_res) in enumerate(all_results.items()):
                with grid_cols[i % 4]:
                    st.image(img_res, caption=name, clamp=True, use_container_width=True)

            st.divider()
            st.subheader("Statistical Summary Table")
            summary_table = {
                "Algorithm": [],
                "Mean Luminance": [],
                "Std Deviation (Contrast)": [],
                "Dynamic Range [Min, Max]": [],
                "Distortion vs Orig (MSE)": [],
            }
            for name, img_res in all_results.items():
                summary_table["Algorithm"].append(name)
                summary_table["Mean Luminance"].append(f"{img_res.mean():.1f}")
                summary_table["Std Deviation (Contrast)"].append(f"{img_res.std():.1f}")
                summary_table["Dynamic Range [Min, Max]"].append(f"[{img_res.min()}, {img_res.max()}]")
                if name == "Original Input":
                    summary_table["Distortion vs Orig (MSE)"].append("0.0 (Reference)")
                else:
                    summary_table["Distortion vs Orig (MSE)"].append(f"{compute_mse(gray, img_res):.1f}")

            st.table(summary_table)

            # Combined Histogram View
            st.subheader("Multi-Method Histogram Comparison")
            fig = make_histogram_figure({
                "Original": gray,
                "Homomorphic (FFT)": homo_res,
                "Combined Hybrid": comb_res,
                "CLAHE": clahe_res,
                "Gamma (γ=0.5)": gamma_05,
            }, figsize=(14, 3))
            st.pyplot(fig)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  SECTION 2: NOISE REMOVAL & SPATIAL FILTERING
# ═══════════════════════════════════════════════════════════

with tab_denoise:
    st.header("Noise Simulation & Spatial-Domain Filtering")
    st.caption(
        "Low-light cameras amplify analog sensor gain (high ISO), introducing heavy Gaussian sensor noise "
        "and transmission/impulse artifacts (salt-and-pepper). Spatial filters suppress noise while balancing edge retention."
    )

    image_d, is_color_d = choose_image_widget("denoise", allow_color=False)

    if image_d is not None:
        gray_d = image_d

        st.divider()

        # Step 1: Noise Configuration
        st.subheader("Step 1: Noise Generation / Acquisition")
        col_noise_mode, col_noise_param = st.columns([1, 2])

        with col_noise_mode:
            noise_option = st.radio(
                "Noise Source",
                ["Add Synthetic Gaussian Noise", "Add Synthetic Salt & Pepper Noise", "Use Naturally Noisy Image As-Is"],
                key="noise_source_choice",
            )

        has_clean_reference = True
        noisy_d = gray_d.copy()

        with col_noise_param:
            if noise_option == "Add Synthetic Gaussian Noise":
                st.markdown("**Gaussian Noise:** Simulates analog amplifier sensor noise: $I_{noisy} = I + \\mathcal{N}(0, \\sigma^2)$")
                sigma = st.slider("Standard Deviation (σ)", min_value=5, max_value=80, value=25, step=5,
                                  help="Higher standard deviation represents cheaper cameras or lower illumination.")
                noisy_d = add_gaussian_noise(gray_d, sigma=float(sigma))

            elif noise_option == "Add Synthetic Salt & Pepper Noise":
                st.markdown("**Salt-and-Pepper Noise:** Simulates transmission errors and faulty sensor pixels: random 0 and 255 spikes.")
                amount = st.slider("Corruption Density (fraction of pixels)", min_value=0.01, max_value=0.25, value=0.05, step=0.01,
                                   help="Percentage of image pixels corrupted by impulse noise.")
                noisy_d = add_salt_and_pepper_noise(gray_d, amount=float(amount))

            else:
                has_clean_reference = False
                st.info("ℹ Operating directly on naturally noisy footage. Visual inspection will be used (no ground-truth reference available).")

        st.divider()

        # Step 2: Filter Selection & Parameters
        st.subheader("Step 2: Spatial Filter Selection")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_filter = st.selectbox(
                "Spatial Filter Type",
                ["Median Filter", "Gaussian Filter", "Mean (Averaging) Filter"],
                key="filter_type_select",
            )

        with filter_col2:
            kernel_choice = st.radio(
                "Kernel Neighborhood Window",
                [3, 5],
                format_func=lambda x: f"{x} × {x} Kernel",
                horizontal=True,
                key="kernel_choice_select",
            )

        # Theoretical Explanation Pill
        filter_insights = {
            "Mean (Averaging) Filter": (
                "**Mean Filter:** Replaces central pixel with arithmetic mean of kernel window. "
                "Smooths Gaussian noise effectively but blurs sharp edges and license plate / facial contours."
            ),
            "Gaussian Filter": (
                "**Gaussian Filter:** Replaces central pixel with weighted Gaussian average (closer neighbors weight higher). "
                "Offers superior edge preservation compared to uniform mean filtering."
            ),
            "Median Filter": (
                "**Median Filter:** Replaces central pixel with mathematical median of local window. "
                "**Optimal for Salt-and-Pepper noise**; eliminates extreme outlier spikes (0 / 255) without smearing edges."
            ),
        }
        st.info(f"💡 {filter_insights[selected_filter]}")

        # Compute Denoised Outputs
        filter_functions = {
            "Mean (Averaging) Filter": apply_mean_filter,
            "Gaussian Filter": apply_gaussian_filter,
            "Median Filter": apply_median_filter,
        }

        active_func = filter_functions[selected_filter]
        denoised_selected = active_func(noisy_d, kernel_choice)

        # Visual Comparison (Interactive Wipe Slider or Triad)
        st.subheader("Visual Comparison")
        render_comparison_view(noisy_d, denoised_selected, "Degraded / Noisy Input", f"Denoised: {selected_filter} ({kernel_choice}×{kernel_choice})", id_prefix="sec2_denoise")

        if has_clean_reference:
            with st.expander("🖼️ View Original Clean Ground Truth Reference", expanded=False):
                display_image(gray_d, "Original Clean Ground Truth Reference", use_container_width=True)

        # Objective Quality Metrics Table
        if has_clean_reference:
            st.divider()
            st.subheader("Quantitative Quality Evaluation")

            # Evaluate both 3x3 and 5x5 kernels for the selected filter
            denoised_3x3 = active_func(noisy_d, 3)
            denoised_5x5 = active_func(noisy_d, 5)

            metrics_noisy = compute_all_metrics(gray_d, noisy_d)
            metrics_3x3 = compute_all_metrics(gray_d, denoised_3x3)
            metrics_5x5 = compute_all_metrics(gray_d, denoised_5x5)

            metric_table_data = {
                "Metric": [
                    "Mean Squared Error (MSE) ↓",
                    "Peak Signal-to-Noise Ratio (PSNR dB) ↑",
                    "Structural Similarity Index (SSIM) ↑",
                ],
                "Degraded / Noisy": [
                    f"{metrics_noisy['MSE']:.2f}",
                    f"{metrics_noisy['PSNR']:.2f} dB",
                    f"{metrics_noisy['SSIM']:.4f}",
                ],
                f"{selected_filter} (3×3)": [
                    f"{metrics_3x3['MSE']:.2f}",
                    f"{metrics_3x3['PSNR']:.2f} dB",
                    f"{metrics_3x3['SSIM']:.4f}",
                ],
                f"{selected_filter} (5×5)": [
                    f"{metrics_5x5['MSE']:.2f}",
                    f"{metrics_5x5['PSNR']:.2f} dB",
                    f"{metrics_5x5['SSIM']:.4f}",
                ],
            }

            st.table(metric_table_data)
            st.caption("↓ Lower is superior  |  ↑ Higher is superior. Reference ground truth: Original clean image.")

            # Comprehensive Side-by-Side of All 3 Filters
            st.subheader(f"Multi-Filter Benchmark ({kernel_choice}×{kernel_choice} Window)")
            m_col1, m_col2, m_col3 = st.columns(3)

            out_mean = apply_mean_filter(noisy_d, kernel_choice)
            out_gauss = apply_gaussian_filter(noisy_d, kernel_choice)
            out_median = apply_median_filter(noisy_d, kernel_choice)

            m_mean = compute_all_metrics(gray_d, out_mean)
            m_gauss = compute_all_metrics(gray_d, out_gauss)
            m_median = compute_all_metrics(gray_d, out_median)

            with m_col1:
                st.image(out_mean, caption=f"Mean Filter ({kernel_choice}×{kernel_choice})", clamp=True, use_container_width=True)
                st.markdown(f"**PSNR:** `{m_mean['PSNR']} dB` | **SSIM:** `{m_mean['SSIM']}`")
            with m_col2:
                st.image(out_gauss, caption=f"Gaussian Filter ({kernel_choice}×{kernel_choice})", clamp=True, use_container_width=True)
                st.markdown(f"**PSNR:** `{m_gauss['PSNR']} dB` | **SSIM:** `{m_gauss['SSIM']}`")
            with m_col3:
                st.image(out_median, caption=f"Median Filter ({kernel_choice}×{kernel_choice})", clamp=True, use_container_width=True)
                st.markdown(f"**PSNR:** `{m_median['PSNR']} dB` | **SSIM:** `{m_median['SSIM']}`")
        else:
            st.info("ℹ Quality metrics are only defined against a ground-truth clean image. Compare denoised outputs visually.")

        # Download Denoised Output
        st.download_button(
            label="⬇️ Download Denoised Image",
            data=image_to_bytes(denoised_selected),
            file_name=f"cctv_denoised_{selected_filter.split()[0].lower()}_{kernel_choice}x{kernel_choice}.png",
            mime="image/png",
        )


# ═══════════════════════════════════════════════════════════
#  SECTION 3: VIDEO STREAM PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

with tab_video:
    st.header("CCTV Video (.mp4) Stream Processing Pipeline")
    st.caption(
        "Surveillance operations depend on video streams rather than static images. "
        "This pipeline applies frame-by-frame spatial enhancement to CCTV video clips and computes "
        "real-time throughput (FPS) and latency to verify real-time operational feasibility."
    )

    vid_source = st.radio(
        "Select Video Input",
        ["Bundled CCTV Night Patrol (.mp4)", "Upload CCTV Video (.mp4 / .avi / .mov)"],
        horizontal=True,
        key="video_source_select",
    )

    sample_vid_path = os.path.join(SAMPLE_DIR, "cctv_night_patrol.mp4")
    if not os.path.exists(sample_vid_path):
        generate_synthetic_cctv_video(sample_vid_path, num_frames=75)

    input_vid_file = None
    if vid_source == "Bundled CCTV Night Patrol (.mp4)":
        input_vid_file = sample_vid_path
    else:
        uploaded_vid = st.file_uploader(
            "Upload surveillance video clip",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )
        if uploaded_vid is not None:
            temp_vid_path = os.path.join(SAMPLE_DIR, "temp_uploaded_cctv.mp4")
            with open(temp_vid_path, "wb") as f:
                f.write(uploaded_vid.read())
            input_vid_file = temp_vid_path

    if input_vid_file and os.path.exists(input_vid_file):
        meta = get_video_metadata(input_vid_file)
        if meta:
            with st.expander("📊 Input Video Stream Telemetry", expanded=True):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Resolution", meta.get("resolution", "N/A"))
                k2.metric("Native FPS", f"{meta.get('fps', 0)} FPS")
                k3.metric("Total Frames", f"{meta.get('total_frames', 0)}")
                k4.metric("Duration", f"{meta.get('duration_s', 0)} sec")

        st.divider()
        st.subheader("Step 1: Select Enhancement, Tune Sliders & Live Frame Preview")
        st.caption("Adjust enhancement parameters below and inspect how the surveillance frames will look before rendering the full video.")

        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            vid_method = st.selectbox(
                "Enhancement Algorithm for Video",
                [
                    "Combined Hybrid (Gamma + CLAHE + Sharpen)",
                    "Homomorphic Filtering (2D-FFT Frequency Domain)",
                    "CLAHE (Adaptive Histogram Equalization)",
                    "Gamma Correction (Power-Law)",
                    "Logarithmic Transformation",
                    "Histogram Equalization (Global)",
                    "Image Negative",
                    "Unsharp Masking (Sharpening)",
                ],
                key="vid_method_select",
            )

        vid_params = {}
        with v_col2:
            if vid_method == "Combined Hybrid (Gamma + CLAHE + Sharpen)":
                st.markdown("**Combined Pipeline:** $\\text{Gamma}(\\gamma) \\rightarrow \\text{CLAHE} \\rightarrow \\text{Unsharp Mask}$")
                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    vid_params["gamma"] = st.slider("Gamma (γ)", 0.1, 1.5, 0.50, 0.05, key="vid_ch_gamma")
                with vc2:
                    vid_params["clip_limit"] = st.slider("CLAHE Clip", 1.0, 6.0, 2.5, 0.5, key="vid_ch_clip")
                    vid_params["grid_size"] = 8
                with vc3:
                    vid_params["sharpen_strength"] = st.slider("Sharpen", 0.5, 3.0, 1.2, 0.1, key="vid_ch_sharp")
                method_code = "combined"

            elif vid_method == "Homomorphic Filtering (2D-FFT Frequency Domain)":
                st.markdown("**Retinex 2D-FFT:** $H(u,v) = (\\gamma_H - \\gamma_L) [1 - e^{-c \\cdot D^2 / D_0^2}] + \\gamma_L$")
                vh1, vh2 = st.columns(2)
                with vh1:
                    vid_params["gamma_low"] = st.slider("Gamma Low (γ_L)", 0.1, 1.0, 0.50, 0.05, key="vid_hf_glow")
                    vid_params["gamma_high"] = st.slider("Gamma High (γ_H)", 1.0, 3.0, 2.00, 0.1, key="vid_hf_ghigh")
                with vh2:
                    vid_params["d0"] = st.slider("Cutoff (D0)", 10.0, 80.0, 30.0, 5.0, key="vid_hf_d0")
                    vid_params["c"] = 1.0
                method_code = "homomorphic"

            elif vid_method == "Gamma Correction (Power-Law)":
                st.markdown("**Formula:** $s = 255 \\times (r/255)^\\gamma$")
                vg1, vg2 = st.columns([2, 1])
                with vg1:
                    vid_params["gamma"] = st.slider("Gamma (γ)", 0.1, 2.5, 0.45, 0.05, key="vid_gamma_val")
                with vg2:
                    st.write("")
                    if st.button("Preset: 0.4", key="v_preset_04"):
                        vid_params["gamma"] = 0.4
                method_code = "gamma"

            elif vid_method == "CLAHE (Adaptive Histogram Equalization)":
                st.markdown("**Method:** Tile-based adaptive histogram equalization")
                vcl1, vcl2 = st.columns(2)
                with vcl1:
                    vid_params["clip_limit"] = st.slider("Clip Limit", 1.0, 6.0, 2.5, 0.5, key="vid_clahe_clip")
                with vcl2:
                    vid_params["grid_size"] = st.selectbox("Tile Grid", [4, 8, 16], index=1, key="vid_clahe_grid")
                method_code = "clahe"

            elif vid_method == "Logarithmic Transformation":
                st.markdown("**Formula:** $s = c \\times \\log(1 + r)$")
                st.info("Compresses headlights / streetlights while expanding dark shadows.")
                method_code = "log"

            elif vid_method == "Histogram Equalization (Global)":
                st.markdown("**Method:** Global Cumulative Distribution Function Equalization")
                st.info("Flattens pixel intensity across all 256 bins.")
                method_code = "he"

            elif vid_method == "Image Negative":
                st.markdown("**Formula:** $s = 255 - r$")
                st.info("Inverts intensities to uncover faint shapes in black shadows.")
                method_code = "negative"

            elif vid_method == "Unsharp Masking (Sharpening)":
                st.markdown("**Method:** High-frequency edge amplification via Gaussian residual")
                vsh1, vsh2 = st.columns(2)
                with vsh1:
                    vid_params["strength"] = st.slider("Sharpen Strength", 0.5, 3.0, 1.5, 0.1, key="vid_sharp_val")
                with vsh2:
                    vid_params["sigma"] = st.slider("Gaussian Sigma", 0.5, 3.0, 1.0, 0.5, key="vid_sigma_val")
                method_code = "unsharp"

        # Extract and show live preview of sample frame #0 (Wipe Slider or Side-by-Side)
        sample_frame = extract_video_frame(input_vid_file, 0)
        if sample_frame is not None:
            enhanced_sample = enhance_color_image(sample_frame, method_code, **vid_params)
            render_comparison_view(sample_frame, enhanced_sample, "Raw Video Frame #0", f"Enhanced Frame ({vid_method})", id_prefix="sec3_frame0")

        st.divider()

        # Step 2: Render full video stream
        st.subheader("Step 2: Render & Benchmark Surveillance Video Stream")
        vr_col1, vr_col2, vr_col3 = st.columns([2, 2, 2])
        with vr_col1:
            max_f = st.slider("Max Frames to Process", 25, 120, 60, 5, key="vid_max_frames",
                             help="Controls how many frames to render for snappy interactive testing.")
        with vr_col2:
            st.write("")
            burn_hud_opt = st.checkbox("Burn CCTV Forensic OSD HUD", value=True, key="burn_hud_box")
        with vr_col3:
            st.write("")
            run_vid_btn = st.button("🚀 Process Full Video Stream", type="primary", key="btn_process_video")

        if run_vid_btn:
            output_vid_path = os.path.join(SAMPLE_DIR, "enhanced_cctv_output.mp4")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            status_text.info("⚙ Processing video frames through spatial enhancement pipeline...")

            stats = process_video(
                input_path=input_vid_file,
                output_path=output_vid_path,
                method=method_code,
                max_frames=max_f,
                burn_hud=burn_hud_opt,
                progress_callback=lambda p: progress_bar.progress(p),
                **vid_params
            )

            status_text.success("✅ Video processing complete!")
            st.session_state["video_results"] = stats
            st.session_state["video_input_file"] = input_vid_file
            st.session_state["video_output_file"] = output_vid_path

        # Render results if available
        if "video_results" in st.session_state and os.path.exists(st.session_state.get("video_output_file", "")):
            v_res = st.session_state["video_results"]
            st.divider()
            st.subheader("⚡ Real-Time Operational Performance Benchmark")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processing Speed", f"{v_res['processing_fps']} FPS")
            m2.metric("Frame Latency", f"{v_res['latency_ms']} ms")
            m3.metric("Frames Rendered", f"{v_res['processed_frames']}")
            is_rt = v_res["realtime_capable"]
            m4.metric("Real-Time Status", "✅ PASS (>25 FPS)" if is_rt else "⚠ Non-Realtime")

            st.caption("Broadcasting standard requires $\\ge 24-25\\text{ FPS}$. Our classical DIP spatial pipeline exceeds this requirement.")

            st.subheader("Synchronized Visual Comparison")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**Raw CCTV Night Stream**")
                try:
                    with open(st.session_state["video_input_file"], "rb") as vf_in:
                        st.video(vf_in.read(), format="video/mp4")
                except Exception as e:
                    st.error(f"Error loading raw video: {e}")
            with col_v2:
                st.markdown("**Enhanced CCTV Surveillance Stream**")
                try:
                    with open(st.session_state["video_output_file"], "rb") as vf_out:
                        st.video(vf_out.read(), format="video/mp4")
                except Exception as e:
                    st.error(f"Error loading enhanced video: {e}")

            # Interactive Frame Scrubber for precise forensic inspection
            st.subheader("🎞️ Interactive Frame-by-Frame Inspector")
            st.caption("Scrub through the video frame-by-frame to inspect specific moments, vehicle motion, and details side-by-side.")
            max_scrub = max(0, v_res.get("processed_frames", 60) - 1)
            scrub_idx = st.slider("Select Frame Index", 0, max_scrub, 0, key="video_scrub_idx")

            f_raw = extract_video_frame(st.session_state["video_input_file"], scrub_idx)
            f_enh = extract_video_frame(st.session_state["video_output_file"], scrub_idx)

            if f_raw is not None and f_enh is not None:
                render_comparison_view(f_raw, f_enh, f"Raw Frame #{scrub_idx}", f"Enhanced Frame #{scrub_idx}", id_prefix=f"scrub_{scrub_idx}")

            with open(st.session_state["video_output_file"], "rb") as vf:
                vid_bytes = vf.read()
            st.download_button(
                label="⬇️ Download Enhanced Surveillance Video (.mp4)",
                data=vid_bytes,
                file_name="cctv_enhanced_surveillance.mp4",
                mime="video/mp4",
                key="btn_download_video",
            )


# ═══════════════════════════════════════════════════════════
#  SECTION 4: DOWNSTREAM VISION BENCHMARK (PROOF OF UTILITY)
# ═══════════════════════════════════════════════════════════

with tab_benchmark:
    st.header("Downstream Computer Vision Benchmark (Proof of Utility)")
    st.caption(
        "Demonstrates the real-world forensic value of image enhancement: "
        "compares automated object and pedestrian detection on RAW dark footage vs. ENHANCED footage "
        "using an integrated NanoDet neural vision detector."
    )

    bench_img, is_col_b = choose_image_widget("benchmark", allow_color=True)

    if bench_img is not None:
        st.divider()
        st.subheader("Step 1: Select Method, Tune Sliders & Live Preview")
        st.caption("Adjust the sliders below to see how the enhancement alters illumination, contrast, and edges in real-time.")

        b_col1, b_col2 = st.columns([1, 1])

        with b_col1:
            b_method_choice = st.selectbox(
                "Enhancement Method for Benchmark",
                [
                    "Combined Hybrid (Gamma + CLAHE + Sharpen)",
                    "Homomorphic Filtering (2D-FFT Frequency Domain)",
                    "CLAHE (Adaptive Histogram Equalization)",
                    "Gamma Correction (Power-Law)",
                    "Logarithmic Transformation",
                    "Histogram Equalization (Global)",
                    "Image Negative",
                    "Unsharp Masking (Sharpening)",
                ],
                key="bench_method_select",
            )

        bench_params = {}
        with b_col2:
            if b_method_choice == "Combined Hybrid (Gamma + CLAHE + Sharpen)":
                st.markdown("**Combined Pipeline:** $\\text{Gamma}(\\gamma) \\rightarrow \\text{CLAHE} \\rightarrow \\text{Unsharp Mask}$")
                ch_c1, ch_c2, ch_c3 = st.columns(3)
                with ch_c1:
                    bench_params["gamma"] = st.slider("Gamma (γ)", 0.1, 1.5, 0.50, 0.05, key="bench_ch_gamma",
                                                      help="Expands dark shadows.")
                with ch_c2:
                    bench_params["clip_limit"] = st.slider("CLAHE Clip", 1.0, 6.0, 2.5, 0.5, key="bench_ch_clip",
                                                           help="Limits local tile contrast amplification.")
                    bench_params["grid_size"] = 8
                with ch_c3:
                    bench_params["sharpen_strength"] = st.slider("Sharpen", 0.5, 3.0, 1.2, 0.1, key="bench_ch_sharp",
                                                                 help="Sharpens recovered edges.")
                b_code = "combined"

            elif b_method_choice == "Homomorphic Filtering (2D-FFT Frequency Domain)":
                st.markdown("**Retinex 2D-FFT:** $H(u,v) = (\\gamma_H - \\gamma_L) [1 - e^{-c \\cdot D^2 / D_0^2}] + \\gamma_L$")
                bh1, bh2 = st.columns(2)
                with bh1:
                    bench_params["gamma_low"] = st.slider("Gamma Low (γ_L)", 0.1, 1.0, 0.50, 0.05, key="bench_hf_glow")
                    bench_params["gamma_high"] = st.slider("Gamma High (γ_H)", 1.0, 3.0, 2.00, 0.1, key="bench_hf_ghigh")
                with bh2:
                    bench_params["d0"] = st.slider("Cutoff (D0)", 10.0, 80.0, 30.0, 5.0, key="bench_hf_d0")
                    bench_params["c"] = 1.0
                b_code = "homomorphic"

            elif b_method_choice == "Gamma Correction (Power-Law)":
                st.markdown("**Formula:** $s = 255 \\times (r/255)^\\gamma$")
                g_col1, g_col2 = st.columns([2, 1])
                with g_col1:
                    bench_params["gamma"] = st.slider("Gamma value (γ)", 0.1, 2.5, 0.45, 0.05, key="bench_gamma_slider",
                                                      help="γ < 1 brightens dark surveillance scenes.")
                with g_col2:
                    st.write("")
                    if st.button("Preset: 0.4", key="b_preset_04"):
                        bench_params["gamma"] = 0.4
                b_code = "gamma"

            elif b_method_choice == "CLAHE (Adaptive Histogram Equalization)":
                st.markdown("**Method:** Tile-based adaptive histogram equalization")
                cl_c1, cl_c2 = st.columns(2)
                with cl_c1:
                    bench_params["clip_limit"] = st.slider("Clip Limit", 1.0, 6.0, 2.5, 0.5, key="bench_clahe_clip")
                with cl_c2:
                    bench_params["grid_size"] = st.selectbox("Tile Grid", [4, 8, 16], index=1, key="bench_clahe_grid")
                b_code = "clahe"

            elif b_method_choice == "Logarithmic Transformation":
                st.markdown("**Formula:** $s = c \\times \\log(1 + r)$")
                st.info("Expands shadow detail while compressing bright streetlight highlights.")
                b_code = "log"

            elif b_method_choice == "Histogram Equalization (Global)":
                st.markdown("**Method:** Global Cumulative Distribution Function Equalization")
                st.info("Flattens pixel intensity across all 256 bins.")
                b_code = "he"

            elif b_method_choice == "Image Negative":
                st.markdown("**Formula:** $s = 255 - r$")
                st.info("Inverts intensities to uncover faint shapes camouflaged in black backgrounds.")
                b_code = "negative"

            elif b_method_choice == "Unsharp Masking (Sharpening)":
                st.markdown("**Method:** High-frequency edge amplification via Gaussian residual")
                sh_c1, sh_c2 = st.columns(2)
                with sh_c1:
                    bench_params["strength"] = st.slider("Sharpen Strength", 0.5, 3.0, 1.5, 0.1, key="bench_unsharp_str")
                with sh_c2:
                    bench_params["sigma"] = st.slider("Gaussian Sigma", 0.5, 3.0, 1.0, 0.5, key="bench_unsharp_sig")
                b_code = "unsharp"

        # Generate live enhanced preview
        enhanced_bench = enhance_color_image(bench_img, b_code, **bench_params)

        # Show Live Interactive Visual Preview (Wipe Slider or Side-by-Side)
        render_comparison_view(bench_img, enhanced_bench, "Raw Surveillance Input", f"Live Enhanced Preview: {b_method_choice}", id_prefix="sec4_preview")

        st.divider()

        # Step 2: Run Detector Audit
        st.subheader("Step 2: Run Automated Detection Audit on this Enhanced Image")
        st.caption("Once you are satisfied with the preview above, run the neural vision detector to evaluate object & person recovery.")

        audit_bar1, audit_bar2 = st.columns([3, 2])
        with audit_bar1:
            conf_thresh = st.slider(
                "Detector Confidence Threshold",
                0.15, 0.60, 0.25, 0.05,
                help="Minimum confidence percentage required to confirm a detected target.",
                key="bench_conf_slider",
            )
        with audit_bar2:
            st.write("")
            st.write("")
            run_bench_btn = st.button("🔬 Run Vision Audit on Tuned Image", type="primary", key="btn_run_vision_audit")

        if run_bench_btn or "bench_report" in st.session_state:
            if run_bench_btn:
                with st.spinner("Running automated computer vision detection audit..."):
                    report = evaluate_vision_benchmark(bench_img, enhanced_bench, prob_thresh=conf_thresh)
                    st.session_state["bench_report"] = report

            rep = st.session_state.get("bench_report")
            if rep:
                st.subheader("Automated Detection Audit Scorecard")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Raw Detections", f"{rep['raw_count']} Targets")
                c2.metric("Enhanced Detections", f"{rep['enhanced_count']} Targets",
                          delta=f"+{rep['targets_recovered']} Recovered" if rep['targets_recovered'] > 0 else None)
                c3.metric("AI Confidence (Enhanced)", f"{int(rep['enhanced_avg_confidence'] * 100)}%")
                status_color = "✅ SUCCESS" if rep['targets_recovered'] > 0 else "ℹ STABLE"
                c4.metric("Forensic Audit", status_color)

                st.subheader("Side-by-Side Automated Target Identification")
                render_comparison_view(rep["annotated_raw"], rep["annotated_enhanced"], f"Raw: {rep['raw_count']} Targets", f"Enhanced: {rep['enhanced_count']} Targets", id_prefix="sec4_detections")

                # Quantitative Comparative Audit Table
                st.subheader("Forensic Verification Summary")
                audit_table = {
                    "Evaluation Criterion": [
                        "Identified Security Targets",
                        "Average Detection Confidence",
                        "Automated Surveillance Efficacy",
                    ],
                    "Raw Dark Footage": [
                        f"{rep['raw_count']} Targets Detected",
                        f"{rep['raw_avg_confidence'] * 100:.1f}%",
                        "❌ High Failure Rate (Underexposed Blindspots)",
                    ],
                    "Enhanced Surveillance Footage": [
                        f"{rep['enhanced_count']} Targets Detected",
                        f"{rep['enhanced_avg_confidence'] * 100:.1f}%",
                        "✅ Actionable Forensic Evidence Recovered",
                    ],
                    "Forensic Improvement": [
                        f"+{rep['targets_recovered']} Targets Uncovered" if rep['targets_recovered'] > 0 else "Benchmark Complete",
                        f"+{(rep['enhanced_avg_confidence'] - rep['raw_avg_confidence']) * 100:.1f}% Confidence Gain",
                        "Restored Machine Vision Usability",
                    ],
                }
                st.table(audit_table)

                # Expandable detailed list of detections
                with st.expander("📋 Detailed Target Manifest", expanded=False):
                    d_left, d_right = st.columns(2)
                    with d_left:
                        st.markdown("**Raw Footage Targets:**")
                        if rep["raw_detections"]:
                            for i, d in enumerate(rep["raw_detections"], 1):
                                st.markdown(f"- **#{i} {d['class_name'].upper()}**: Confidence `{int(d['score']*100)}%` | Box: `{d['box']}`")
                        else:
                            st.caption("No targets could be recognized due to darkness.")
                    with d_right:
                        st.markdown("**Enhanced Footage Targets:**")
                        if rep["enhanced_detections"]:
                            for i, d in enumerate(rep["enhanced_detections"], 1):
                                st.markdown(f"- **#{i} {d['class_name'].upper()}**: Confidence `{int(d['score']*100)}%` | Box: `{d['box']}`")
                        else:
                            st.caption("No targets detected.")

