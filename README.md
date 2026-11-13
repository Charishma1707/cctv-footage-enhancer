# 📹 CCTV Night Footage Enhancer Using Digital Image Processing

An interactive Digital Image Processing (DIP) application that enhances low-light CCTV and surveillance images using **traditional spatial-domain techniques** (no machine learning or deep learning).

Integrated with authentic images from the benchmark **ExDark (Exclusively Dark)** dataset and supports custom image uploads (JPG/PNG).

---

## 🌟 Key Features

### 1. Image Enhancement (Section 1)
- **Point Transformations:**
  - **Image Negative:** $s = 255 - r$ — Inverts intensities to uncover shapes hidden in dark regions.
  - **Logarithmic Transformation:** $s = c \cdot \log(1 + r)$ — Compresses dynamic range by expanding dark levels and compressing bright highlights.
  - **Gamma Correction (Power-Law):** $s = 255 \cdot (r/255)^\gamma$ — Interactive slider ($\gamma \in [0.1, 3.0]$) and presets ($\gamma = 0.5$, $\gamma = 1.0$, $\gamma = 2.0$).
- **Histogram-Based Enhancement:**
  - **Global Histogram Equalization:** Spreads intensities across the full $[0, 255]$ spectrum.
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Performs tile-based adaptive histogram equalization with contrast limiting to prevent noise amplification.
- **Frequency-Domain Enhancement (Fourier Transform):**
  - **Homomorphic Filtering (2D-FFT Retinex Model):** Decomposes image into illumination and reflectance in the 2D frequency domain, suppressing non-uniform lighting while boosting high-frequency object details.
  - **2D Fourier Spectrum Magnitude:** Real-time visual comparison of $\log(1 + |F(u,v)|)$ centered spectrums before and after filtering.
- **Combined Hybrid Pipeline:** Gamma + CLAHE + Unsharp Masking for comprehensive multi-stage enhancement.
- **Interactive Before/After Wipe Slider:** Draggable vertical split-screen slider (Juxtapose) for seamless visual comparison alongside standard side-by-side view.
- **Dual Processing Modes:** Grayscale (standard DIP) and Color (YCrCb luminance enhancement preserving color chrominance).
- **Comparison Views:** Single method deep-dive or **"Compare All Methods Side-by-Side"** grid with statistical tables and histograms.
- **Processed Image Download:** Instant 1-click PNG download.

### 2. Noise Removal & Spatial Filtering (Section 2)
- **Noise Simulation:**
  - **Gaussian Noise:** Simulates analog sensor amplifier noise ($I + \mathcal{N}(0, \sigma^2)$) common in cheap/night CCTV sensors.
  - **Salt-and-Pepper Noise:** Simulates impulse noise and dead sensor pixels.
  - **Naturally Noisy Footage:** Option to process noisy footage without synthetic degradation.
- **Spatial-Domain Filters:**
  - **Mean Filter:** Box averaging kernel ($3 \times 3$ and $5 \times 5$).
  - **Gaussian Filter:** Gaussian-weighted kernel ($3 \times 3$ and $5 \times 5$).
  - **Median Filter:** Non-linear order-statistic median ($3 \times 3$ and $5 \times 5$) — optimal for salt-and-pepper artifacts.
- **Quantitative Quality Evaluation:**
  - **MSE (Mean Squared Error)**: Average squared intensity error.
  - **PSNR (Peak Signal-to-Noise Ratio in dB)**: Logarithmic signal fidelity metric.
  - **SSIM (Structural Similarity Index)**: Measures structural distortion, luminance, and contrast consistency.
- **Benchmark Table:** Comparative table evaluating Noisy, Filter $3 \times 3$, and Filter $5 \times 5$ against the clean reference.
- **Multi-Filter Benchmark:** Side-by-side comparison of all 3 filters with real-time PSNR and SSIM metrics.

### 3. CCTV Video (.mp4) Stream Processing (Section 3)
- **Frame-by-Frame Streaming Pipeline:** Applies YCrCb luminance enhancement across moving video frames.
- **Real-Time Throughput Benchmarking:** Measures actual processing speed in **FPS (Frames Per Second)** and frame latency in **milliseconds (ms)**.
- **Forensic OSD HUD:** Burns timestamp, camera identification, resolution, and live status into output video.
- **Side-by-Side Synchronized Playback:** Instant dual-player visual comparison of raw night footage vs. enhanced stream.
- **Instant Video Download:** 1-click download of processed `.mp4` video.

### 4. Downstream Computer Vision Benchmark (Section 4)
- **Proof-of-Utility Validation:** Evaluates whether enhancement enables automated security AI to detect hidden suspects and vehicles.
- **Neural Detector Integration:** Powered by an ultra-fast, lightweight NanoDet neural network detecting 80 COCO classes (People, Cars, Bikes, Buses, etc.).
- **Automated Detection Scorecard:** Quantifies target recovery (e.g. 0 raw targets $\rightarrow$ 3 recovered targets) and confidence improvement.
- **Detailed Target Manifest:** Complete list of identified security entities with bounding box coordinates and detection confidence.

---

## 📂 Project Architecture

```
cctv-night-footage-enhancer/
├── app.py                  # Main Streamlit interactive web application (4 tabs)
├── enhancement.py          # Spatial-domain point & histogram enhancement algorithms
├── denoising.py            # Noise generation (Gaussian, S&P) & spatial filters
├── evaluation.py           # Objective metrics: MSE, PSNR, SSIM
├── video_processor.py      # Video stream processing, FPS benchmark & HUD engine
├── detection.py            # Downstream object & pedestrian detection audit engine
├── models/                 # Lightweight ONNX detection models (NanoDet)
├── download_exdark.py      # Automated downloader for authentic ExDark dataset images
├── setup_dataset.py        # Helper to initialize dataset images & CCTV samples
├── requirements.txt        # Minimal dependency list
├── README.md               # Documentation and viva presentation guide
└── sample_images/          # Benchmark image & video repository
    ├── cctv_night_patrol.mp4 # Pre-rendered CCTV night surveillance video clip
    ├── cctv_street.png     # Synthetic CCTV scenes with camera HUD/timestamps
    ├── cctv_parking.png
    ├── cctv_alley.png
    └── exdark/             # Authentic ExDark Dataset images
        ├── People/         # Low-light pedestrian and street images
        ├── Car/            # Night vehicles and parking lots
        ├── Bus/            # Night public transit
        ├── Motorbike/      # Low-light bikes
        └── Bicycle/        # Low-light street bikes
```

---

## 🚀 Quickstart & Execution

### Prerequisites
- Python 3.9+ (tested on Python 3.11 and 3.14)
- Web browser (Chrome, Edge, Firefox)

### 1. Installation
In your project directory:
```bash
pip install -r requirements.txt
```

### 2. Verify / Refresh ExDark Dataset
The project comes pre-loaded with authentic ExDark dataset images. If you ever want to re-download or add more:
```bash
python download_exdark.py
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🔬 Mathematical Formulas & Viva Reference

### 1. Image Negative
$$s = 255 - r$$
Inverts dynamic range. Useful in night CCTV when dark foreground details blend into black shadows.

### 2. Logarithmic Transformation
$$s = c \cdot \log(1 + r), \quad c = \frac{255}{\log(1 + 255)} \approx 45.986$$
Expands dark pixel values while compressing bright ones, preventing bright glare (e.g., street lamps) from saturating while brightening dark alleys.

### 3. Gamma Correction (Power-Law)
$$s = 255 \cdot \left(\frac{r}{255}\right)^\gamma$$
- $\gamma < 1$: Curve bows upward, expanding dark intensities (ideal for underexposed night CCTV).
- $\gamma = 1$: Identity (no change).
- $\gamma > 1$: Curve bows downward, compressing dark intensities (darkens).

### 4. Frequency-Domain Homomorphic Filtering (2D-FFT Retinex Model)
Based on the image formation model:
$$f(x, y) = i(x, y) \cdot r(x, y)$$
where $i(x,y)$ represents scene illumination (low spatial frequencies) and $r(x,y)$ represents object reflectance (high spatial frequencies).

Taking the natural logarithm converts multiplication to addition:
$$\ln f(x, y) = \ln i(x, y) + \ln r(x, y)$$

Applying 2D Fast Fourier Transform (FFT) and filtering with a high-frequency emphasis transfer function:
$$H(u, v) = (\gamma_H - \gamma_L) \cdot \left[1 - \exp\left(-c \cdot \frac{D^2(u,v)}{D_0^2}\right)\right] + \gamma_L$$
- $\gamma_L < 1$ (e.g., $0.5$): Attenuates low frequencies, eliminating uneven lighting and heavy shadows.
- $\gamma_H > 1$ (e.g., $2.0$): Amplifies high frequencies, enhancing object edges, textures, and details.
- $D_0$: Filter cutoff radius determining frequency transition bandwidth.
- $c$: Sharpness control parameter.

After inverse FFT ($\text{IFFT}$) and exponential mapping $\exp(\cdot) - 1$, dynamic range is restored via percentile scaling.

### 5. Mean Filter vs Gaussian Filter vs Median Filter
| Filter Type | Kernel Operation | Best Application | Edge Preservation |
|---|---|---|---|
| **Mean Filter** | Arithmetic average over $K \times K$ | Uniform / Gaussian noise | Low (causes blurring) |
| **Gaussian Filter** | Gaussian-weighted convolution | Sensor amplifier noise | Moderate (weighted falloff) |
| **Median Filter** | Order-statistic median replacement | Salt-and-Pepper impulse noise | **High** (edges are preserved) |

### 5. Quality Metrics
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I(x,y) - K(x,y)]^2$$
- **Peak Signal-to-Noise Ratio (PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
  - Higher PSNR ($> 30\text{ dB}$) indicates cleaner reconstruction.
- **Structural Similarity Index (SSIM):**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  - Range $[-1, 1]$, where $1.0$ indicates identical structural integrity.

---

## 📚 Dataset Attribution
- **ExDark (Exclusively Dark) Dataset:**
  - Paper: *Exclusively Dark (ExDark) Image Dataset*, Loh & Chan, IEEE TIP / GitHub: [cs-chan/Exclusively-Dark-Image-Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).
  - Images hosted on Hugging Face (`fuhao7i/ExDark`).
