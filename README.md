# Automatic License Plate Deskewing & Unblurring using Digital Image Processing

### A Practical Demonstration of Image Enhancement, Geometric Transformation, Denoising, and Image Restoration

---

## 📌 Academic Positioning Statement
> **This project is an interactive Digital Image Processing (DIP) laboratory application designed for college presentations and viva voce examinations.**  
> The license plate scenario serves purely as an **application context**. The primary objective is to demonstrate, explain, and evaluate fundamental Digital Image Processing algorithms from the academic syllabus in real-time.

---

## 🎯 Project Objectives & Problem Statement
Vehicle license plate images captured by outdoor traffic cameras frequently suffer from environmental and optical degradations:
1. **Geometric Tilt / Perspective Skew**: Camera mounting angles cause character rotation and shear.
2. **Poor Contrast**: Low lighting or glare concentrates pixel intensities into narrow bands.
3. **Sensor Noise**: Low-light sensor gain introduces Gaussian or Salt-and-Pepper impulse noise.
4. **Motion Blur**: Relative movement between vehicle and shutter creates linear motion blur.

This application provides a visual, interactive platform where a professor or student can test individual DIP syllabus algorithms or execute a complete multi-stage restoration pipeline.

---

## 🗺️ Syllabus Module Mapping

| Syllabus Module | DIP Concept | Project Implementation | Applied License Plate Scenario |
| :--- | :--- | :--- | :--- |
| **MODULE I** | **Geometric Transformations** | `dip/transformations.py` | |
| | 1. Rotation | `rotate_image()` | Correct camera rotation tilt angle |
| | 2. Scaling | `scale_image()` | Resizing plate images with interpolation |
| | 3. Affine Transformation | `affine_transform_points()`, `deskew_image()` | Deskewing character alignment via $M_{2\times 3}$ matrix |
| | 4. Matrix Representation | `format_matrix_latex()` | Displays $2\times 3$ and $3\times 3$ homogeneous matrices |
| | 5. Composition of Transformations | `composite_transformation()` | Combined matrix multiplication $M_{trans} \cdot M_{shear} \cdot M_{rot} \cdot M_{scale}$ |
| **MODULE II** | **Image Enhancement** | `dip/enhancement.py` & `filters.py` | |
| | 1. Histogram Equalization | `histogram_equalization()` | Spreading intensity CDF to fix dark/glare plates |
| | 2. Smoothing Filters | `mean_filter()`, `gaussian_filter()`, `median_filter()` | Spatial averaging vs non-linear order statistics |
| | 3. Order Statistic Filter | `median_filter()` | Selective elimination of Salt & Pepper noise |
| | 4. Sharpening Filters | `laplacian_sharpening()`, `unsharp_masking()` | High-frequency detail enhancement ($\nabla^2 f$) |
| **MODULE III**| **Restoration & Denoising** | `dip/restoration.py` & `noise.py` | |
| | 1. Degradation Model | $g(x,y) = h(x,y) * f(x,y) + n(x,y)$ | Mathematical framework for blur and noise |
| | 2. Noise Models | `add_gaussian_noise()`, `add_salt_pepper_noise()` | Synthetic AWGN and Impulse noise simulation |
| | 3. Bilateral Filtering | `bilateral_filter()` | Edge-preserving range & spatial Gaussian filtering |
| | 4. Motion Blur Simulation | `create_motion_blur_kernel()`, `apply_motion_blur()` | PSF kernel $h(x,y)$ generation for motion blur |
| | 5. Inverse Filtering | `inverse_filter()` | Frequency-domain 2D FFT division with LPF truncation |
| | 6. Wiener Filtering | `wiener_filter()` | Minimum Mean Square Error (MMSE) unblurring ($K = 1/SNR$) |

---

## 🏗️ Project Architecture

```text
DIP Project/
├── app.py                      # Interactive Streamlit Web Application UI
├── dip/                        # Modular DIP Algorithm Engine
│   ├── __init__.py
│   ├── transformations.py      # Module I: Rotation, Scaling, Affine, Deskewing, Matrix math
│   ├── enhancement.py          # Module II: Histogram Equalization & Sharpening Filters
│   ├── filters.py              # Module II: Spatial Smoothing & Order Statistic Filters
│   ├── noise.py                # Module III: Gaussian Noise & Salt-and-Pepper Noise Models
│   ├── restoration.py          # Module III: Motion blur PSF, Bilateral, Inverse & Wiener filtering
│   └── pipeline.py             # Full Demonstration Sequential Restoration Pipeline
├── utils/                      # Helper Utilities
│   ├── __init__.py
│   ├── image_utils.py          # Formatting, channel conversions, sample plate synthesis
│   └── visualization.py        # Matplotlib histogram plots, PSNR/MSE metrics, LaTeX formatting
├── data/                       # Pre-built Sample Images
│   └── sample_images/          # Clean, skewed, low-contrast, noisy, and blurred plates
├── tests/                      # Unit Tests
│   └── test_dip.py             # Pytest / Unittest validation suite
├── requirements.txt            # Dependencies
└── README.md                   # Project Documentation & Viva Guide
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.9+ installed on Windows, macOS, or Linux.

### 2. Clone / Open Directory
```bash
cd "DIP Project"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Unit Tests (Verification)
```bash
python3 -m unittest tests/test_dip.py
```

### 5. Launch Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Detailed DIP Syllabus Topic Explanations

### Module I: Geometric Transformations
- **Affine Transformation & Deskewing**: An affine transformation maps a point $(x,y)$ to $(x',y')$ via matrix multiplication:
  $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
  Affine transforms preserve collinearity and parallel lines, making them ideal for correcting perspective tilt in license plates without warping straight edges.

### Module II: Image Enhancement
- **Histogram Equalization**: Spreads narrow pixel intensity distributions over the full $[0, 255]$ dynamic range using the Cumulative Distribution Function (CDF):
  $$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j)$$
- **Median vs Mean Filtering**:
  - **Mean Filter**: Computes uniform neighborhood average, which smears impulse noise into blurry gray regions.
  - **Median Filter**: Replaces center pixel with the median of sorted neighbors. Because extreme salt ($255$) or pepper ($0$) values fall at the outer tails of the sorted list, they are discarded completely!
- **Laplacian Sharpening**: Enhances high-frequency spatial gradients using second-order derivative operator:
  $$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}, \quad f_{sharp}(x,y) = f(x,y) - \nabla^2 f(x,y)$$

### Module III: Image Restoration & Denoising
- **Degradation Model**: $g(x,y) = h(x,y) * f(x,y) + n(x,y)$, which in the frequency domain becomes:
  $$G(u,v) = H(u,v) \cdot F(u,v) + N(u,v)$$
- **Bilateral Filtering**: Combines spatial Gaussian distance weighting with intensity range Gaussian weighting:
  $$I_{denoised}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) g_s(\|x_i - x\|) g_r(\|I(x_i) - I(x)\|)$$
  This denoises flat background areas while preserving sharp character boundary transitions.
- **Inverse vs Wiener Filtering**:
  - **Inverse Filter**: $\hat{F}(u,v) = \frac{G(u,v)}{H(u,v)} = F(u,v) + \frac{N(u,v)}{H(u,v)}$. When $H(u,v) \to 0$ at high frequencies, noise ratio $\frac{N(u,v)}{H(u,v)}$ explodes to infinity, producing catastrophic noise artifacts.
  - **Wiener Filter**: Minimizes mean square error (MMSE):
    $$\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + K} \right] G(u,v) \quad \text{where } K = \frac{1}{SNR}$$
    The constant $K$ prevents division by zero and suppresses frequencies where signal-to-noise ratio is low.

---

## 🎤 2–3 Minute Presentation Sequence for Team Members

1. **Member 1 (Module I Leader)**:
   - Introduce problem statement: Traffic cameras capture tilted plates.
   - Select **Module I $\to$ Affine Transformation / Deskewing**.
   - Explain $M_{2\times 3}$ affine matrix and demonstrate how horizontal shear and rotation align character baselines.

2. **Member 2 (Module II Leader)**:
   - Select **Module II $\to$ Histogram Equalization**. Show low-contrast dark plate $\to$ equalized plate alongside before/after Matplotlib intensity histograms and CDF curves.
   - Select **Module II $\to$ Order Statistic Filter**. Show Salt & Pepper noisy plate $\to$ compare Mean vs Median filter. Explain why median filter removes impulses cleanly.

3. **Member 3 (Module III Leader)**:
   - Select **Module III $\to$ Motion Blur & Wiener Filtering**.
   - Demonstrate linear motion blur simulation $h(x,y)$.
   - Show why direct Inverse Filtering suffers from noise explosion, and demonstrate how Wiener Filtering restores blurred plate characters safely using MMSE estimation.

4. **Team Conclusion**:
   - Switch to **🚀 Full Demonstration Mode** and run the complete multi-stage pipeline to demonstrate end-to-end restoration quality (PSNR & MSE metrics).

---

## ⚠️ Limitations & Future Scope

### Current Limitations
- Hand-crafted geometric parameters (deskew sliders) are used instead of deep learning bounding box detectors to keep focus strictly on DIP core algorithms.
- Frequency domain Wiener restoration assumes uniform linear motion blur parameters ($L, \theta$).

### Future Scope (ANPR Extension)
- Integration of automatic Hough Transform line detection for automated deskew angle estimation.
- Character segmentation via Connected Component Analysis (CCA).
- Tesseract OCR engine integration for automatic text reading after DIP pre-processing.
