"""
Module II: Smoothing Spatial Filters & Order Statistic Filters.
Implements Mean (Box) Filter, Gaussian Filter, and Median Filter.
"""

import cv2
import numpy as np
from typing import Tuple


def mean_filter(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Mean (Box) Spatial Smoothing Filter.
    Uniform spatial weighting: kernel K_ij = 1 / (k * k).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1 # Ensure odd kernel size
        
    filtered = cv2.blur(image, (kernel_size, kernel_size))
    
    # 2D Kernel for display
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return filtered, kernel


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian Smoothing Spatial Filter.
    2D Gaussian function kernel with standard deviation sigma.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    
    # Get 1D Gaussian kernel and compute 2D outer product kernel for visual display
    k1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = k1d @ k1d.T
    return filtered, kernel_2d


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Median Spatial Filter (Order Statistic Filter).
    Replaces center pixel with median value of neighbor intensities.
    Superior for Salt-and-Pepper / Impulse Noise reduction.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.medianBlur(image, kernel_size)
