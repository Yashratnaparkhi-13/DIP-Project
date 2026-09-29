"""
Unit tests for Digital Image Processing (DIP) functions.
Run with pytest or python -m unittest discover tests.
"""

import unittest
import numpy as np
import cv2

from utils.image_utils import generate_sample_license_plate, ensure_gray, ensure_rgb
from dip.transformations import rotate_image, scale_image, deskew_image, composite_transformation
from dip.enhancement import histogram_equalization, laplacian_sharpening, unsharp_masking
from dip.filters import mean_filter, gaussian_filter, median_filter
from dip.noise import add_gaussian_noise, add_salt_pepper_noise
from dip.restoration import (
    bilateral_filter, create_motion_blur_kernel, apply_motion_blur,
    inverse_filter, wiener_filter
)
from dip.pipeline import run_full_pipeline


class TestDIPModules(unittest.TestCase):
    
    def setUp(self):
        self.sample_plate = generate_sample_license_plate("TEST 123", width=300, height=100)
        self.gray_plate = ensure_gray(self.sample_plate)

    def test_image_utils(self):
        rgb = ensure_rgb(self.gray_plate)
        self.assertEqual(len(rgb.shape), 3)
        self.assertEqual(rgb.shape[2], 3)

    def test_geometric_transformations(self):
        # Rotation
        rot, M_rot = rotate_image(self.sample_plate, angle=15.0)
        self.assertEqual(rot.shape, self.sample_plate.shape)
        self.assertEqual(M_rot.shape, (2, 3))
        
        # Scaling
        scaled, M_scale = scale_image(self.sample_plate, fx=1.5, fy=1.5)
        self.assertEqual(scaled.shape[:2], (150, 450))
        
        # Deskew
        deskewed, M_deskew = deskew_image(self.sample_plate, angle_deg=10, shear_x=0.1)
        self.assertEqual(deskewed.shape, self.sample_plate.shape)
        
        # Composite
        comp, M_comp = composite_transformation(self.sample_plate, scale=1.1, angle_deg=5)
        self.assertEqual(comp.shape, self.sample_plate.shape)

    def test_enhancement_module(self):
        # Histogram Equalization
        eq = histogram_equalization(self.sample_plate, method="Global Equalization")
        self.assertEqual(eq.shape, self.sample_plate.shape)
        
        # Laplacian Sharpening
        sharp, kernel = laplacian_sharpening(self.sample_plate, strength=1.0)
        self.assertEqual(sharp.shape, self.sample_plate.shape)
        self.assertEqual(kernel.shape, (3, 3))
        
        # Unsharp Masking
        unsharp, mask = unsharp_masking(self.sample_plate)
        self.assertEqual(unsharp.shape, self.sample_plate.shape)

    def test_filters_module(self):
        mean_img, k_mean = mean_filter(self.sample_plate, kernel_size=5)
        self.assertEqual(mean_img.shape, self.sample_plate.shape)
        
        gauss_img, k_gauss = gaussian_filter(self.sample_plate, kernel_size=5, sigma=1.0)
        self.assertEqual(gauss_img.shape, self.sample_plate.shape)
        
        med_img = median_filter(self.sample_plate, kernel_size=5)
        self.assertEqual(med_img.shape, self.sample_plate.shape)

    def test_noise_module(self):
        gauss_noisy = add_gaussian_noise(self.sample_plate, sigma=20)
        self.assertEqual(gauss_noisy.shape, self.sample_plate.shape)
        
        sp_noisy = add_salt_pepper_noise(self.sample_plate, amount=0.05)
        self.assertEqual(sp_noisy.shape, self.sample_plate.shape)

    def test_restoration_module(self):
        # Bilateral
        bilat = bilateral_filter(self.sample_plate)
        self.assertEqual(bilat.shape, self.sample_plate.shape)
        
        # Motion blur & kernel
        blurred, psf = apply_motion_blur(self.sample_plate, length=11, angle_deg=0)
        self.assertEqual(blurred.shape, self.sample_plate.shape)
        self.assertEqual(psf.shape, (11, 11))
        
        # Inverse filtering
        inv_restored, spec_inv = inverse_filter(blurred, psf)
        self.assertEqual(inv_restored.shape[:2], self.sample_plate.shape[:2])
        
        # Wiener filtering
        wiener_restored, spec_wiener = wiener_filter(blurred, psf, nsr=0.01)
        self.assertEqual(wiener_restored.shape[:2], self.sample_plate.shape[:2])

    def test_full_pipeline(self):
        stages = run_full_pipeline(self.sample_plate, reference_clean=self.sample_plate)
        self.assertGreater(len(stages), 5)
        for stage in stages:
            self.assertIn("image", stage)
            self.assertIn("stage_name", stage)


if __name__ == '__main__':
    unittest.main()
