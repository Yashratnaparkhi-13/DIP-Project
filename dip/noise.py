"""
Module III: Noise Models.
Implements synthetic noise injection (Gaussian Noise & Salt-and-Pepper Noise)
for controlled image restoration experiments.
"""

import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to an image.
    Noise Model: g(x,y) = f(x,y) + n(x,y), n ~ N(mean, sigma^2).
    """
    gauss_noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image.astype(np.float32) + gauss_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """
    Add Salt-and-Pepper (Impulse) Noise to an image.
    Salt pixels = 255 (white max intensity), Pepper pixels = 0 (black min intensity).
    """
    noisy_img = image.copy()
    num_noise = int(amount * image.shape[0] * image.shape[1])
    
    # Salt noise (white pixels)
    num_salt = int(num_noise * salt_vs_pepper)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy_img[coords_salt[0], coords_salt[1], :] = 255
    else:
        noisy_img[coords_salt[0], coords_salt[1]] = 255
        
    # Pepper noise (black pixels)
    num_pepper = int(num_noise * (1.0 - salt_vs_pepper))
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy_img[coords_pepper[0], coords_pepper[1], :] = 0
    else:
        noisy_img[coords_pepper[0], coords_pepper[1]] = 0
        
    return noisy_img
