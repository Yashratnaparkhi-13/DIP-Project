"""
Image Utility Functions for DIP Demonstration Application.
Provides utilities for image I/O, format conversion, synthetic sample plate generation,
and synthetic degradation injection.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure the image is 3-channel RGB format."""
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image
    return image


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to 8-bit single-channel grayscale."""
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def generate_sample_license_plate(
    plate_text: str = "MH 12 AB 1234",
    width: int = 480,
    height: int = 160,
    bg_color: Tuple[int, int, int] = (240, 220, 50), # Yellow plate
    text_color: Tuple[int, int, int] = (20, 20, 20)  # Dark text
) -> np.ndarray:
    """
    Synthesize a clean high-quality license plate image with realistic border,
    emblem, state header, and registration text for DIP testing.
    """
    plate = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Outer black border
    cv2.rectangle(plate, (4, 4), (width - 5, height - 5), (10, 10, 10), 4)
    # Inner border line
    cv2.rectangle(plate, (10, 10), (width - 11, height - 11), (50, 50, 50), 2)
    
    # Left blue IND strip (European/Indian style)
    strip_w = 40
    cv2.rectangle(plate, (12, 12), (12 + strip_w, height - 12), (180, 50, 20), -1)
    
    # IND text on strip
    cv2.putText(
        plate, "IND", (16, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    # Chakra/circle symbol on strip
    cv2.circle(plate, (12 + strip_w // 2, 40), 10, (255, 255, 255), 1)

    # Main text formatting
    main_text_x = 12 + strip_w + 20
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.3
    thickness = 3
    
    # Vertical placement centered
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    text_y = (height + text_size[1]) // 2 - 5
    
    cv2.putText(
        plate, plate_text, (main_text_x, text_y),
        font, font_scale, text_color, thickness, cv2.LINE_AA
    )
    
    # Add minor realistic texture/gradient
    gradient = np.linspace(1.0, 0.88, height).reshape(height, 1, 1)
    plate = np.clip(plate * gradient, 0, 255).astype(np.uint8)
    
    return plate


def create_degraded_sample_set() -> Dict[str, np.ndarray]:
    """
    Generate a dictionary of pre-built sample images targeting different DIP modules:
    - Clean Plate
    - Skewed Plate (Module I)
    - Low Contrast Plate (Module II)
    - Salt & Pepper Noisy Plate (Module II / III)
    - Motion Blurred Plate (Module III)
    """
    clean = generate_sample_license_plate("KA 01 MJ 9999")
    h, w = clean.shape[:2]
    
    # 1. Skewed Plate
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[25, 20], [w - 10, 35], [5, h - 15], [w - 30, h - 5]])
    M_affine = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(clean, M_affine, (w, h), borderValue=(200, 200, 200))
    
    # 2. Low Contrast Dark Plate
    low_contrast = (clean.astype(np.float32) * 0.35 + 40).astype(np.uint8)
    
    # 3. Salt & Pepper Noisy Plate
    sp_noisy = clean.copy()
    num_salt = int(0.04 * h * w)
    num_pepper = int(0.04 * h * w)
    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    sp_noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    sp_noisy[coords[0], coords[1]] = 0
    
    # 4. Motion Blurred Plate
    kernel_size = 15
    kernel_motion = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel_motion[int((kernel_size - 1) / 2), :] = 1.0 / kernel_size
    motion_blurred = cv2.filter2D(clean, -1, kernel_motion)
    
    return {
        "Clean Sample Plate": clean,
        "Skewed Plate (Module I)": skewed,
        "Low Contrast Plate (Module II)": low_contrast,
        "Salt & Pepper Noisy Plate (Module II/III)": sp_noisy,
        "Motion Blurred Plate (Module III)": motion_blurred
    }
