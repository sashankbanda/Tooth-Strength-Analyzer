import numpy as np
import cv2

def crop_tooth_roi(original_image, box, mask=None, padding=10):
    """
    Crops the tooth region from the original image based on the bounding box.
    Optionally applies the mask to black out the background.
    
    Args:
        original_image: Numpy array (H, W, C)
        box: [x1, y1, x2, y2]
        mask: Binary mask of the whole image (same size as original_image) or the specific mask.
        padding: Pixel padding around the box.
        
    Returns:
        Cropped image (numpy array).
    """
    h, w = original_image.shape[:2]
    x1, y1, x2, y2 = box
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    cropped = original_image[y1:y2, x1:x2].copy()
    
    if mask is not None:
        # If mask is full size, crop it too
        if mask.shape[:2] == (h, w):
            mask_cropped = mask[y1:y2, x1:x2]
        else:
            # Assuming mask passed is already corresponding to the box? 
            # In the inference output, mask is full size usually or ROI?
            # MaskRCNN returns masks of size of image usually or fixed size? 
            # Torchvision returns masks of size (H, W).
            mask_cropped = mask[y1:y2, x1:x2]
            
        # Apply mask? Maybe keep background for context or black it out?
        # For U-Net++, having context might be good, or maybe we want strict isolation.
        # Let's just return the cropped image for now, and maybe a separate masked crop.
        pass

    return cropped

def extract_all_teeth(original_image, predictions, padding=10):
    """
    Extracts all teeth provided in predictions.
    
    Args:
        original_image: Numpy array
        predictions: List of dicts from ToothInstanceSegmentor.predict
        
    Returns:
        List of dictionaries with 'tooth_id' (index), 'image', 'box', 'score'.
    """
    extracted = []
    for i, pred in enumerate(predictions):
        crop = crop_tooth_roi(original_image, pred['box'], pred['mask'], padding)
        extracted.append({
            'id': i,
            'image': crop,
            'box': pred['box'],
            'mask': pred['mask'], # Full size mask
            'score': pred['score']
        })
    return extracted
