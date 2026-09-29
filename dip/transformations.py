"""
Module I: Geometric Transformations.
Implements image rotation, scaling, affine transformation, deskewing,
matrix representation, and composite matrix transformations.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def rotate_image(image: np.ndarray, angle: float, expand: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by specified angle (degrees) around its center.
    Returns transformed image and 2x3 affine rotation matrix.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # Get standard 2x3 rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if expand:
        # Calculate new bounding dimensions to prevent clipping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust translation parameters in matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        output_size = (new_w, new_h)
    else:
        output_size = (w, h)
        
    rotated = cv2.warpAffine(
        image, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return rotated, M


def scale_image(
    image: np.ndarray, fx: float = 1.0, fy: float = 1.0, interpolation_method: str = "Linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale/resize image by scaling factors fx and fy.
    Returns scaled image and 3x3 homogeneous scaling matrix.
    """
    interp_map = {
        "Nearest": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "LANCZOS": cv2.INTER_LANCZOS4
    }
    flag = interp_map.get(interpolation_method, cv2.INTER_LINEAR)
    scaled = cv2.resize(image, None, fx=fx, fy=fy, interpolation=flag)
    
    # 3x3 scaling matrix representation
    M = np.array([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return scaled, M


def affine_transform_points(
    image: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply affine transformation mapping 3 source control points to 3 destination points.
    Returns transformed image and 2x3 affine matrix.
    """
    h, w = image.shape[:2]
    M = cv2.getAffineTransform(pts_src, pts_dst)
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return transformed, M


def deskew_image(
    image: np.ndarray, angle_deg: float = 0.0, shear_x: float = 0.0, shear_y: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deskew license plate image combining rotation and shear affine transformation.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # 3x3 Homogeneous transformation matrices
    # 1. Translation to origin
    T1 = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 2. Rotation matrix
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 3. Shear matrix
    S = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 4. Translation back
    T2 = np.array([
        [1, 0, center[0]],
        [0, 1, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Composite homogeneous matrix: M = T2 * S * R * T1
    M_comp = T2 @ S @ R @ T1
    M_2x3 = M_comp[:2, :]
    
    deskewed = cv2.warpAffine(
        image, M_2x3, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed, M_2x3


def composite_transformation(
    image: np.ndarray,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    shear_x: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demonstrate composition of multiple geometric transformations:
    Scale -> Rotation -> Shear -> Translation
    Composite Matrix M = M_trans * M_shear * M_rot * M_scale
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    # Matrices in 3x3 homogeneous coordinates
    M_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    M_rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    
    M_shear = np.array([[1, shear_x, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    
    M_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    # Center origin shift
    T_to_orig = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    T_from_orig = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
    
    # Combined centered matrix: M = M_trans * T_from_orig * M_shear * M_rot * M_scale * T_to_orig
    M_composite = M_trans @ T_from_orig @ M_shear @ M_rot @ M_scale @ T_to_orig
    M_2x3 = M_composite[:2, :]
    
    transformed = cv2.warpAffine(
        image, M_2x3, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return transformed, M_composite
