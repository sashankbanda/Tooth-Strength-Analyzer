import pytest
import numpy as np
from services.measurement import extract_root_length, detect_bone_level, calculate_bone_loss_percentage

def create_synthetic_mask(shape=(100, 100), root_bbox=None, bone_bbox=None, cej_y=None):
    mask = np.zeros(shape, dtype=np.uint8)
    
    if root_bbox:
        x1, y1, x2, y2 = root_bbox
        mask[y1:y2, x1:x2] = 1 # Root
        
    if bone_bbox:
        x1, y1, x2, y2 = bone_bbox
        mask[y1:y2, x1:x2] = 2 # Bone
        
    if cej_y is not None:
        # CEJ as a line across the root width? 
        # Or just a point in the centroid calculation logic.
        # Let's draw a small line segment to be safe.
        if root_bbox:
            cx = (root_bbox[0] + root_bbox[2]) // 2
            mask[cej_y, cx-2:cx+2] = 3 # CEJ
            
    return mask

def test_extract_root_length():
    # Setup: Root from y=10 to y=60. Apex at 59 (max index). Top at 10.
    # CEJ at y=10.
    # Length should be 59 - 10 = 49 pixels approx.
    # (Centroid of bottom row 59 vs Centroid of CEJ row 10) -> Distance 49.0
    
    mask = create_synthetic_mask(
        root_bbox=(40, 10, 60, 60), # 20x50 box. y range 10-59 inclusive.
        cej_y=10
    )
    
    length_mm, details = extract_root_length(mask, pixel_spacing=1.0)
    
    # Apex y should be ~59.5 (center of bottom row) or 59.
    # CEJ y should be 10.
    # Distance ~49.5
    
    assert details['apex'][0] >= 59
    assert details['cej'][0] == 10
    assert 48 < length_mm < 51

def test_detect_bone_level():
    # Setup: CEJ at y=10.
    # Bone starts at y=30.
    # Loss distance = 30 - 10 = 20 pixels.
    
    mask = create_synthetic_mask(
        root_bbox=(40, 10, 60, 60),
        bone_bbox=(0, 30, 100, 100),
        cej_y=10
    )
    
    # We need CEJ point to call detect_bone_level
    # Let's mock it or extract it first
    _, root_details = extract_root_length(mask)
    cej_point = root_details['cej'] # [10, 50]
    
    loss_mm, details = detect_bone_level(mask, cej_point, pixel_spacing=1.0)
    
    # Bone crest y should be 30.
    assert details['bone_crest'][0] == 30
    # Distance (Euclidean) from (10, 50) to (30, 50) is 20.
    assert abs(loss_mm - 20.0) < 1.0

def test_healthy_bone_level():
    # Setup: CEJ at y=30.
    # Bone starts at y=20 (Above CEJ).
    # Loss should be 0.
    
    mask = create_synthetic_mask(
        root_bbox=(40, 20, 60, 70),
        bone_bbox=(0, 20, 100, 100),
        cej_y=30
    )
    
    cej_point = [30, 50]
    loss_mm, details = detect_bone_level(mask, cej_point)
    
    assert loss_mm == 0.0

def test_calculate_bone_loss_percentage():
    assert calculate_bone_loss_percentage(10.0, 5.0) == 50.0
    assert calculate_bone_loss_percentage(10.0, 0.0) == 0.0
    assert calculate_bone_loss_percentage(10.0, 15.0) == 100.0 # Clamped
    assert calculate_bone_loss_percentage(0.0, 5.0) == 0.0
