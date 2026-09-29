"""
Sequential Pipeline Engine for Full Demonstration Mode.
Runs multi-stage DIP enhancement and restoration sequence on license plate images
and records intermediate visual stages and metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from dip.transformations import deskew_image
from dip.enhancement import histogram_equalization, laplacian_sharpening
from dip.noise import add_salt_pepper_noise, add_gaussian_noise
from dip.filters import median_filter
from dip.restoration import bilateral_filter, apply_motion_blur, wiener_filter
from utils.visualization import compute_metrics


def run_full_pipeline(
    input_img: np.ndarray,
    reference_clean: Optional[np.ndarray] = None,
    deskew_angle: float = -12.0,
    deskew_shear: float = 0.1,
    eq_method: str = "Global Equalization",
    add_noise: bool = True,
    noise_type: str = "Salt & Pepper",
    add_blur: bool = True,
    blur_length: int = 11,
    blur_angle: float = 0.0,
    wiener_nsr: float = 0.01,
    sharpen_strength: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Execute complete end-to-end DIP License Plate Processing Pipeline.
    Returns list of stage records containing image, step description, DIP concept,
    and PSNR/MSE metrics.
    """
    pipeline_stages = []
    current_img = input_img.copy()
    
    def add_stage(name: str, module: str, concept: str, img: np.ndarray, desc: str):
        metrics = (0.0, 0.0)
        if reference_clean is not None:
            metrics = compute_metrics(reference_clean, img)
        pipeline_stages.append({
            "stage_name": name,
            "module": module,
            "concept": concept,
            "image": img.copy(),
            "description": desc,
            "mse": metrics[0],
            "psnr": metrics[1]
        })

    # Stage 0: Input Image
    add_stage(
        "Stage 0: Input Image",
        "Raw Input",
        "Original Degraded License Plate",
        current_img,
        "Initial degraded license plate image selected for DIP processing."
    )
    
    # Stage 1: Geometric Deskewing (Module I)
    current_img, M = deskew_image(current_img, angle_deg=deskew_angle, shear_x=deskew_shear)
    add_stage(
        "Stage 1: Geometric Deskewing",
        "Module I",
        "Affine Transformation & Rotation",
        current_img,
        f"Corrected geometric perspective distortion using Affine transformation matrix (Angle={deskew_angle}°, Shear={deskew_shear})."
    )
    
    # Stage 2: Histogram Equalization (Module II)
    current_img = histogram_equalization(current_img, method=eq_method)
    add_stage(
        "Stage 2: Contrast Enhancement",
        "Module II",
        "Histogram Equalization",
        current_img,
        "Equalized intensity distribution to dramatically improve license plate contrast and character visibility."
    )
    
    # Stage 3: Degradation Simulation (Module III - Optional for demonstration)
    if add_blur:
        current_img, psf = apply_motion_blur(current_img, length=blur_length, angle_deg=blur_angle)
        add_stage(
            "Stage 3A: Motion Blur Degradation",
            "Module III",
            "Degradation Model h(x,y)",
            current_img,
            f"Simulated camera movement motion blur using PSF kernel (Length={blur_length}px, Angle={blur_angle}°)."
        )
    else:
        psf = None

    if add_noise:
        if noise_type == "Salt & Pepper":
            current_img = add_salt_pepper_noise(current_img, amount=0.03)
        else:
            current_img = add_gaussian_noise(current_img, sigma=20.0)
        add_stage(
            "Stage 3B: Noise Injection",
            "Module III",
            f"Noise Model n(x,y) ({noise_type})",
            current_img,
            f"Injected synthetic {noise_type} noise to simulate sensor noise in license plate capture."
        )

    # Stage 4: Denoising (Module II / III)
    if noise_type == "Salt & Pepper" and add_noise:
        current_img = median_filter(current_img, kernel_size=5)
        denoise_concept = "Order Statistic Median Filter"
        denoise_desc = "Applied Median filter to cleanly eliminate impulsive salt-and-pepper noise spikes."
    else:
        current_img = bilateral_filter(current_img, diameter=9, sigma_color=75.0, sigma_space=75.0)
        denoise_concept = "Bilateral Filter"
        denoise_desc = "Applied Bilateral filtering for edge-preserving Gaussian smoothing."

    add_stage(
        "Stage 4: Denoising",
        "Module II / III",
        denoise_concept,
        current_img,
        denoise_desc
    )

    # Stage 5: Unblurring & Wiener Restoration (Module III)
    if add_blur and psf is not None:
        current_img, _ = wiener_filter(current_img, psf, nsr=wiener_nsr)
        add_stage(
            "Stage 5: Image Restoration",
            "Module III",
            "Wiener Filtering (MMSE)",
            current_img,
            f"Performed frequency-domain Wiener restoration to unblur plate characters while regulating noise (NSR={wiener_nsr})."
        )

    # Stage 6: High-Frequency Sharpening (Module II)
    current_img, _ = laplacian_sharpening(current_img, strength=sharpen_strength)
    add_stage(
        "Stage 6: Detail Sharpening",
        "Module II",
        "Laplacian Sharpening Filter",
        current_img,
        "Enhanced high-frequency spatial gradients to make license plate alphanumeric characters crisp and legible."
    )

    return pipeline_stages
