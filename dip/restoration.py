"""
Module III: Image Restoration & Denoising.
Implements Motion Blur simulation, Bilateral Filtering, Frequency-Domain Inverse Filtering,
and Frequency-Domain Wiener Filtering using 2D Fast Fourier Transform (FFT).
"""

import cv2
import numpy as np
from typing import Tuple
from utils.image_utils import ensure_gray


def bilateral_filter(
    image: np.ndarray, diameter: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply Bilateral Filter for edge-preserving denoising.
    Combines spatial domain Gaussian weight with range (color similarity) Gaussian weight.
    Denoises smooth regions while retaining sharp license plate character edges.
    """
    return cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def create_motion_blur_kernel(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """
    Generate Point Spread Function (PSF) kernel h(x,y) representing linear motion blur.
    length: Blur length in pixels.
    angle_deg: Motion direction in degrees (0° = horizontal, 90° = vertical).
    """
    if length < 1:
        length = 1
        
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    
    # Calculate line endpoints for motion trajectory
    rad = np.radians(angle_deg)
    dx = np.cos(rad) * (length / 2.0)
    dy = np.sin(rad) * (length / 2.0)
    
    pt1 = (int(center - dx), int(center - dy))
    pt2 = (int(center + dx), int(center + dy))
    
    cv2.line(kernel, pt1, pt2, 1.0, 1)
    
    # Normalize kernel so sum of weights equals 1
    k_sum = np.sum(kernel)
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
        
    return kernel


def apply_motion_blur(image: np.ndarray, length: int = 15, angle_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply synthetic motion blur degradation to image: g(x,y) = h(x,y) * f(x,y).
    Returns degraded image and PSF motion kernel.
    """
    psf = create_motion_blur_kernel(length, angle_deg)
    blurred = cv2.filter2D(image, -1, psf)
    return blurred, psf


def inverse_filter(
    degraded_img: np.ndarray,
    psf: np.ndarray,
    cutoff_radius: float = 80.0,
    epsilon: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Frequency Domain Truncated/Regularized Inverse Filtering:
    F_hat(u,v) = G(u,v) * H*(u,v) / (|H(u,v)|^2 + epsilon) * LPF(u,v)
    
    Includes a Butterworth Low Pass Filter (LPF) cutoff radius to mitigate
    catastrophic high-frequency noise division.
    """
    # Convert to grayscale float32 for FFT operations
    gray = ensure_gray(degraded_img).astype(np.float32)
    h, w = gray.shape[:2]
    
    # Pad PSF kernel to full image size and center it
    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape[:2]
    psf_padded[:kh, :kw] = psf
    # Circular shift PSF to align center with origin (0,0)
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)
    
    # 2D FFT
    G = np.fft.fft2(gray)
    H = np.fft.fft2(psf_padded)
    
    # Magnitude squared |H(u,v)|^2
    H_mag_sq = np.abs(H) ** 2
    
    # Regularized Inverse Transfer Function
    # H_inv = H* / (|H|^2 + eps)
    H_inv = np.conj(H) / (H_mag_sq + epsilon)
    
    # Low-pass filter (Butterworth LPF order 2) to control high-frequency noise explode
    u = np.fft.fftfreq(h).reshape(-1, 1) * h
    v = np.fft.fftfreq(w).reshape(1, -1) * w
    D = np.sqrt(u**2 + v**2)
    LPF = 1.0 / (1.0 + (D / cutoff_radius) ** 4)
    
    # Restored Spectrum F_hat = G * H_inv * LPF
    F_hat = G * H_inv * LPF
    
    # Inverse FFT to get spatial image
    f_restored = np.real(np.fft.ifft2(F_hat))
    f_restored = np.clip(f_restored, 0, 255).astype(np.uint8)
    
    # Return restored image and log spectrum of H for display
    H_spectrum_log = np.log(1 + np.fft.fftshift(np.abs(H)))
    return f_restored, H_spectrum_log


def wiener_filter(
    degraded_img: np.ndarray,
    psf: np.ndarray,
    nsr: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Frequency Domain Wiener Filtering (MMSE Restoration):
    F_hat(u,v) = [ H*(u,v) / (|H(u,v)|^2 + K) ] * G(u,v)
    where K = nsr = 1 / SNR (Noise-to-Signal Power Ratio).
    
    Unlike Inverse Filter, Wiener Filter handles noise presence gracefully
    by attenuating frequencies where signal-to-noise ratio is low.
    """
    gray = ensure_gray(degraded_img).astype(np.float32)
    h, w = gray.shape[:2]
    
    # Pad PSF kernel to full image size and center it
    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape[:2]
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)
    
    # 2D FFT
    G = np.fft.fft2(gray)
    H = np.fft.fft2(psf_padded)
    
    # Wiener Transfer Function: W(u,v) = H*(u,v) / (|H(u,v)|^2 + K)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2
    W = H_conj / (H_mag_sq + nsr)
    
    # Restored Fourier Spectrum F_hat = G * W
    F_hat = G * W
    
    # Inverse FFT
    f_restored = np.real(np.fft.ifft2(F_hat))
    f_restored = np.clip(f_restored, 0, 255).astype(np.uint8)
    
    # Magnitude spectrum of Wiener filter transfer function
    W_spectrum = np.log(1 + np.fft.fftshift(np.abs(W)))
    return f_restored, W_spectrum
