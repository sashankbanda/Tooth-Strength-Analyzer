import cv2
import numpy as np
from scipy.spatial import ConvexHull

def get_classification(bone_loss_pct):
    """
    Classifies the severity of periodontitis based on bone loss percentage.
    Thresholds (approximated from AAP guidelines):
    - Healthy: < 15%
    - Mild: 15% - 33%
    - Moderate: 33% - 50%
    - Severe: > 50%
    """
    if bone_loss_pct < 15:
        return "Healthy"
    elif bone_loss_pct < 33:
        return "Mild Periodontitis"
    elif bone_loss_pct < 50:
        return "Moderate Periodontitis"
    else:
        return "Severe Periodontitis"

def calculate_integrity_score(mask):
    """
    Calculates Structural Integrity Score based on the shape of the root.
    Heuristic: 'Solidity' = Contour Area / Convex Hull Area.
    A healthy root is generally smooth and convex-ish.
    Resorption or irregular fractures reduce solidity.
    """
    # Root label is 1
    root_mask = (mask == 1).astype(np.uint8)
    
    contours, _ = cv2.findContours(root_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0 # No root found
        
    # Get largest contour (assume it's the main root)
    root_contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(root_contour)
    if area < 10: # Too small to score
        return 0.0
        
    hull = cv2.convexHull(root_contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0.0
        
    solidity = float(area) / hull_area
    
    # Map solidity (0-1) to score (0-100)
    # Roots usually have high solidity (>0.85). 
    # If solidity < 0.7, it's very irregular (score drops).
    
    # Linear mapping: 0.7 -> 50, 0.95 -> 100
    score = np.interp(solidity, [0.7, 0.95], [50, 100])
    
    return round(score, 2)

def calculate_infection_score(image, mask, apex_point):
    """
    Calculates Infection Score based on radiolucency (darkness) around the apex.
    Heuristic: Compare mean intensity of a small ROI around Apex vs. surrounding Bone.
    Infection (periapical radiolucency) appears as a dark area at the root tip.
    """
    if apex_point is None:
        return 50.0 # Unknown
        
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
        
    y, x = apex_point
    h, w = gray_image.shape
    
    # Define ROI around apex (e.g., 20x20 pixels)
    roi_size = 20
    y1 = max(0, y - roi_size // 2)
    y2 = min(h, y + roi_size // 2)
    x1 = max(0, x - roi_size // 2)
    x2 = min(w, x + roi_size // 2)
    
    if (y2 - y1) < 5 or (x2 - x1) < 5:
        return 50.0
        
    apex_roi = gray_image[y1:y2, x1:x2]
    apex_intensity = np.mean(apex_roi)
    
    # Reference: Mean intensity of "Bone" in the whole image
    bone_mask = (mask == 2).astype(np.uint8)
    if np.sum(bone_mask) == 0:
        # Fallback: use whole image mean ??
        ref_intensity = np.mean(gray_image)
    else:
        ref_intensity = cv2.mean(gray_image, mask=bone_mask)[0]
        
    # Logic: 
    # If Apex is significantly darker than Bone -> Potential Infection.
    # darker means lower value.
    # Ratio = Apex / Ref
    # Healthy: Apex ~= Ref (Ratio ~ 1.0)
    # Infection: Apex < Ref (Ratio < 0.8)
    
    if ref_intensity == 0: 
        return 50.0
        
    ratio = apex_intensity / ref_intensity
    
    # Map ratio to score
    # Ratio < 0.6 -> Score 20 (Severe)
    # Ratio > 0.9 -> Score 100 (Healthy)
    
    score = np.interp(ratio, [0.6, 0.9], [20, 100])
    
    return round(score, 2)

def calculate_strength_score(bone_loss_pct, integrity_score, infection_score):
    """
    Calculates the final tooth strength score based on the weighted formula.
    
    Formula:
    Strength Score = (0.6 * Bone Support Score) + (0.25 * Structural Integrity Score) + (0.15 * Infection Score)
    
    Bone Support Score = 100 - Bone Loss %
    """
    bone_support_score = max(0, 100 - bone_loss_pct)
    
    # Ensure inputs are valid
    integrity_score = max(0, min(100, integrity_score))
    infection_score = max(0, min(100, infection_score))
    
    final_score = (0.6 * bone_support_score) + \
                  (0.25 * integrity_score) + \
                  (0.15 * infection_score)
                  
    return round(final_score, 2), {
        "bone_support_score": bone_support_score,
        "structural_integrity_score": integrity_score,
        "infection_score": infection_score
    }
