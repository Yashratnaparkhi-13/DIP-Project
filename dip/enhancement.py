"""
Module II: Image Enhancement.
Implements Histogram Equalization (Global & CLAHE) and Sharpening Spatial Filters
(Laplacian Sharpening & Unsharp Masking).
"""

import cv2
import numpy as np
from typing import Tuple


def histogram_equalization(image: np.ndarray, method: str = "Global Equalization", clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Perform Histogram Equalization to enhance low-contrast license plate images.
    Supports Global Histogram Equalization and Adaptive CLAHE.
    """
    if len(image.shape) == 3:
        # Convert RGB to YCrCb / LAB so luminance channel is equalized
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        if method == "Global Equalization":
            y_equalized = cv2.equalizeHist(y_channel)
        else: # CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            y_equalized = clahe.apply(y_channel)
            
        ycrcb[:, :, 0] = y_equalized
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        if method == "Global Equalization":
            return cv2.equalizeHist(image)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            return clahe.apply(image)


def laplacian_sharpening(image: np.ndarray, strength: float = 1.0, kernel_type: str = "4-neighbor") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 2nd order derivative Laplacian Sharpening spatial filter.
    Enhances high-frequency edges and fine details.
    """
    if kernel_type == "4-neighbor":
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
    else: # 8-neighbor
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        
    img_float = image.astype(np.float32)
    laplacian = cv2.filter2D(img_float, -1, kernel)
    
    # Sharpened: f(x,y) + strength * Laplacian
    sharpened = img_float + strength * laplacian
    sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened_clipped, kernel


def unsharp_masking(image: np.ndarray, blur_ksize: int = 5, blur_sigma: float = 1.5, weight: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Unsharp Masking technique:
    1. Smooth original image: f_smooth
    2. Obtain high-pass mask: g_mask = f - f_smooth
    3. Add weighted mask to original: f_sharp = f + k * g_mask
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
        
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), blur_sigma)
    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    
    # High-pass detail mask
    mask = img_f - blur_f
    
    sharpened = img_f + weight * mask
    sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    mask_visual = np.clip(mask + 128, 0, 255).astype(np.uint8)
    return sharpened_clipped, mask_visual
