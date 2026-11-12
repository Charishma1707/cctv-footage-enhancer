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

from enhancement import (
    negative_transform,
    log_transform,
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
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


def load_image(uploaded_file=None, file_path=None, as_color=False) -> np.ndarray | None:
    """Read an image into an 8-bit BGR or Grayscale NumPy array."""
    flag = cv2.IMREAD_COLOR if as_color else cv2.IMREAD_GRAYSCALE
    if uploaded_file is not None:
        try:
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            return cv2.imdecode(file_bytes, flag)
        except Exception:
            return None
    elif file_path is not None and os.path.exists(file_path):
        return cv2.imread(file_path, flag)
    return None


def image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode an image array into PNG bytes for browser download."""
    success, buf = cv2.imencode(fmt, image)
    return buf.tobytes() if success else b""


def display_image(image: np.ndarray, caption: str = "", use_container_width: bool = True):
    """Render grayscale or BGR images accurately in Streamlit."""
    if image.ndim == 3:
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display = image
    st.image(display, caption=caption, clamp=True, use_container_width=use_container_width)


def make_histogram_figure(images_dict: dict[str, np.ndarray], figsize=(10, 3)):
    """Generate Matplotlib intensity histograms comparing original and processed states."""
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#4A5568", "#3182CE", "#38A169", "#D69E2E", "#805AD5"]

    for i, (label, img) in enumerate(images_dict.items()):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    st.caption("Digital Image Processing — Spatial Domain Techniques")
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

    **3. Spatial-Domain Filters:**
    - **Mean Filter** ($3\\times 3$, $5\\times 5$)
    - **Gaussian Filter** ($3\\times 3$, $5\\times 5$)
    - **Median Filter** ($3\\times 3$, $5\\times 5$)

    **4. Objective Evaluation:**
    - **MSE** (Mean Squared Error)
    - **PSNR** (Peak Signal-to-Noise Ratio)
    - **SSIM** (Structural Similarity Index)

    ---
    ### 📦 Dataset
    Integrated with authentic **ExDark (Exclusively Dark)** dataset images across multiple classes (People, Car, Bus, Motorbike, Bicycle).
    """)

# ── Main Header ───────────────────────────────────────────
st.title("📹 CCTV Night Footage Enhancer")
st.markdown(
    "A practical Digital Image Processing application that enhances low-light CCTV surveillance images "
    "using **traditional spatial-domain algorithms**. Test with real images from the **ExDark dataset** "
    "or upload your own night footage."
)

tab_enhance, tab_denoise = st.tabs([
    "🌙 Section 1: Image Enhancement",
    "🧹 Section 2: Noise Removal & Filtering",
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

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
                        "Gamma Correction (Power-Law)",
                        "Logarithmic Transformation",
                        "CLAHE (Adaptive Histogram Eq)",
                        "Histogram Equalization (Global)",
                        "Image Negative",
                    ],
                    key="trans_choice",
                )

            params = {}
            with t_col2:
                if transformation == "Gamma Correction (Power-Law)":
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

            # Compute processed image
            trans_key_map = {
                "Gamma Correction (Power-Law)": "gamma",
                "Logarithmic Transformation": "log",
                "CLAHE (Adaptive Histogram Eq)": "clahe",
                "Histogram Equalization (Global)": "he",
                "Image Negative": "negative",
            }
            method_key = trans_key_map[transformation]

            if is_color:
                enhanced = enhance_color_image(image, method_key, **params)
                enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                if method_key == "gamma":
                    enhanced = gamma_correction(gray, params["gamma"])
                elif method_key == "log":
                    enhanced = log_transform(gray)
                elif method_key == "negative":
                    enhanced = negative_transform(gray)
                elif method_key == "he":
                    enhanced = histogram_equalization(gray)
                elif method_key == "clahe":
                    enhanced = clahe_enhancement(gray, params["clip_limit"], params["grid_size"])
                enhanced_gray = enhanced

            # Visual Comparison
            st.subheader("Enhancement Output")
            col_orig, col_proc = st.columns(2)
            with col_orig:
                display_image(image, "Original Input", use_container_width=True)
            with col_proc:
                lbl = transformation
                if method_key == "gamma":
                    lbl += f" (γ = {params['gamma']})"
                display_image(enhanced, f"Enhanced Output: {lbl}", use_container_width=True)

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

            all_results = {
                "Original Input": gray,
                "Gamma Correction (γ = 0.5)": gamma_05,
                "Gamma Correction (γ = 2.0)": gamma_20,
                "Logarithmic Transform": log_res,
                "CLAHE (Adaptive HE)": clahe_res,
                "Histogram Equalization": he_res,
                "Image Negative": neg_res,
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
                "Gamma (γ=0.5)": gamma_05,
                "Log Transform": log_res,
                "CLAHE": clahe_res,
                "Hist Equalized": he_res,
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

        # Visual Triad
        st.subheader("Visual Comparison")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.image(gray_d, caption="Original Reference", clamp=True, use_container_width=True)
        with col_c2:
            st.image(noisy_d, caption="Degraded / Noisy Input", clamp=True, use_container_width=True)
        with col_c3:
            st.image(denoised_selected, caption=f"Denoised: {selected_filter} ({kernel_choice}×{kernel_choice})",
                     clamp=True, use_container_width=True)

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
