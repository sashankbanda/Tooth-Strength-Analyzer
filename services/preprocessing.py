import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Step 1.1: Noise Removal using Gaussian Blur."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Step 1.2: Contrast Enhancement using CLAHE."""
    # CLAHE expects grayscale or L channel of LAB
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    """Step 1.3: Edge Sharpening using Unsharp Masking."""
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), sigma)
    unsharp_image = cv2.addWeighted(image, 1.0 + strength, gaussian_3, -strength, 0)
    return unsharp_image

def normalize_image(image):
    """Step 1.4: Image Normalization (0-255 range assurance)."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def preprocess_image(image_np):
    """
    Executes the full preprocessing pipeline:
    1. Noise Removal
    2. Contrast Enhancement
    3. Edge Sharpening
    4. Normalization
    """
    # 1. Noise Removal
    denoised = apply_gaussian_blur(image_np)
    
    # 2. Contrast Enhancement
    enhanced = apply_clahe(denoised)
    
    # 3. Edge Sharpening
    sharpened = apply_unsharp_mask(enhanced)
    
    # 4. Normalization
    final_image = normalize_image(sharpened)
    
    return final_image
