"""
Visualization Utilities for DIP Demonstration Application.
Includes Matplotlib histogram plotters, restoration metrics (PSNR, MSE),
and LaTeX mathematical string generators.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def compute_metrics(original: np.ndarray, processed: np.ndarray) -> Tuple[float, float]:
    """
    Compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
    Higher PSNR (dB) and lower MSE indicate higher fidelity restoration.
    """
    # Ensure channel match (convert to grayscale if one is 2D and other is 3D)
    if len(original.shape) != len(processed.shape):
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
    if original.shape != processed.shape:
        # Resize processed to match original for metric calculation if needed
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)
    
    mse = float(np.mean((orig_f - proc_f) ** 2))
    if mse == 0:
        return 0.0, float('inf') # Perfect reconstruction
        
    max_pixel = 255.0
    psnr = float(20.0 * np.log10(max_pixel / np.sqrt(mse)))
    return round(mse, 2), round(psnr, 2)


def plot_histogram(image: np.ndarray, title: str = "Intensity Histogram") -> plt.Figure:
    """
    Generate a clean Matplotlib figure of an image's pixel intensity histogram.
    Supports single-channel grayscale or 3-channel RGB.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    fig.patch.set_facecolor('#0E1117') # Dark mode background matching Streamlit
    ax.set_facecolor('#1E222A')
    
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='#00E5FF', linewidth=1.5, label='Grayscale')
        ax.fill_between(range(256), hist.ravel(), color='#00E5FF', alpha=0.25)
    else:
        colors = ('r', 'g', 'b')
        hex_colors = ('#FF5252', '#4CAF50', '#29B6F6')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=hex_colors[i], linewidth=1.2, label=color.upper())
            
    ax.set_title(title, color='#FFFFFF', fontsize=10, pad=8)
    ax.set_xlabel("Pixel Intensity (0 - 255)", color='#A0AAB0', fontsize=8)
    ax.set_ylabel("Pixel Frequency", color='#A0AAB0', fontsize=8)
    ax.set_xlim([0, 256])
    ax.tick_params(colors='#A0AAB0', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.2, color='#FFFFFF')
    
    for spine in ax.spines.values():
        spine.set_color('#333945')
        
    plt.tight_layout()
    return fig


def plot_side_by_side_histograms(
    img_before: np.ndarray,
    img_after: np.ndarray,
    title_before: str = "Before Equalization",
    title_after: str = "After Equalization"
) -> plt.Figure:
    """
    Generate side-by-side histogram comparison plots for Histogram Equalization demo.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor('#0E1117')
    
    for ax, img, title in zip([ax1, ax2], [img_before, img_after], [title_before, title_after]):
        ax.set_facecolor('#1E222A')
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        
        ax.plot(hist, color='#00E5FF', linewidth=1.5, label='Histogram')
        ax.fill_between(range(256), hist.ravel(), color='#00E5FF', alpha=0.2)
        ax.plot(cdf_normalized, color='#FFD54F', linestyle='--', linewidth=1.2, label='CDF (Equalized)')
        
        ax.set_title(title, color='#FFFFFF', fontsize=10, pad=8)
        ax.set_xlabel("Pixel Intensity", color='#A0AAB0', fontsize=8)
        ax.set_ylabel("Frequency", color='#A0AAB0', fontsize=8)
        ax.set_xlim([0, 256])
        ax.tick_params(colors='#A0AAB0', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.2, color='#FFFFFF')
        ax.legend(loc='upper left', fontsize=7, facecolor='#1E222A', edgecolor='#333945', labelcolor='#FFFFFF')
        for spine in ax.spines.values():
            spine.set_color('#333945')
            
    plt.tight_layout()
    return fig


def format_matrix_latex(M: np.ndarray) -> str:
    """Format a 2x3 or 3x3 NumPy transformation matrix into LaTeX code."""
    if M.shape == (2, 3):
        return r"\begin{bmatrix}" + f" {M[0,0]:.3f} & {M[0,1]:.3f} & {M[0,2]:.3f} \\\\ {M[1,0]:.3f} & {M[1,1]:.3f} & {M[1,2]:.3f} " + r"\end{bmatrix}"
    elif M.shape == (3, 3):
        return r"\begin{bmatrix}" + f" {M[0,0]:.3f} & {M[0,1]:.3f} & {M[0,2]:.3f} \\\\ {M[1,0]:.3f} & {M[1,1]:.3f} & {M[1,2]:.3f} \\\\ {M[2,0]:.3f} & {M[2,1]:.3f} & {M[2,2]:.3f} " + r"\end{bmatrix}"
    return str(M)
