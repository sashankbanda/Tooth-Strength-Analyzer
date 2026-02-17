import numpy as np
from scipy.spatial.distance import euclidean

def get_mask_points(mask, label):
    """
    Returns indices (y, x) of points with specific label.
    """
    return np.argwhere(mask == label)

def calculate_centroid(coords):
    if len(coords) == 0:
        return None
    return np.mean(coords, axis=0).astype(int) # (y, x)

def extract_root_length(mask, pixel_spacing=1.0):
    """
    Calculates root length from CEJ to Apex.
    Args:
        mask: Segmentation mask (H, W) with labels.
        pixel_spacing: mm per pixel (calibrated).
    Returns:
        length_mm, details_dict
    """
    # Labels: 0=Bg, 1=Root, 2=Bone, 3=CEJ
    
    # 1. Identify Apex: Lowest point (max y) of Root (1)
    root_coords = get_mask_points(mask, 1)
    if len(root_coords) == 0:
        return 0.0, {"error": "No root detected"}
    
    # Sort by y (descending) to find bottom
    # Apex is the point with max y. 
    # To be more robust, take average of bottom-most pixels?
    max_y = np.max(root_coords[:, 0])
    apex_candidates = root_coords[root_coords[:, 0] == max_y]
    apex_point = calculate_centroid(apex_candidates) # (y, x)
    
    # 2. Identify CEJ (3)
    cej_coords = get_mask_points(mask, 3)
    if len(cej_coords) > 0:
        cej_point = calculate_centroid(cej_coords)
    else:
        # Fallback: Top of root? Or intersection of Root and Crown (if logic existed)
        # For now, let's use the highest point of the root if CEJ is missing, 
        # BUT this is inaccurate as it includes crown if segmentation is just "tooth".
        # If output is Root vs Bone, and we assume the crop is just root/bone...
        # Let's assume prediction is imperfect and return None or estimate.
        # Fallback: Top-most point of Root mask.
        min_y = np.min(root_coords[:, 0])
        top_candidates = root_coords[root_coords[:, 0] == min_y]
        cej_point = calculate_centroid(top_candidates)
    
    # Distance
    # Coordinates are (y, x)
    dist_pixels = euclidean(cej_point, apex_point)
    dist_mm = dist_pixels * pixel_spacing
    
    return dist_mm, {
        "apex": apex_point.tolist(),
        "cej": cej_point.tolist(),
        "root_length_pixels": dist_pixels
    }

def detect_bone_level(mask, cej_point, pixel_spacing=1.0):
    """
    Calculates bone loss distance.
    Args:
        mask: Segmentation mask.
        cej_point: (y, x) coordinates of CEJ.
        pixel_spacing: mm per pixel.
    Returns:
        loss_mm, details_dict
    """
    if cej_point is None:
        return 0.0, {"error": "No CEJ point provided"}
        
    # Bone is label 2
    bone_coords = get_mask_points(mask, 2)
    if len(bone_coords) == 0:
        return 0.0, {"error": "No bone detected"}
    
    # Find Bone Crest: Highest point (min y) of Bone mask
    # We should ideally look for bone that is "supporting" the tooth, 
    # i.e., close to the root.
    # Simple heuristic: Min Y of bone mask.
    min_y = np.min(bone_coords[:, 0])
    
    # Filter bone points at this min_y to find the one closest to the root/center?
    # Or just average y is enough for level?
    # Bone level is usually defined by the alveolar crest.
    
    # Let's assume the highest point of the bone mask in this crop IS the crest.
    # But we need to ensure it's not some noise at the top.
    # Maybe limit to calculating within the x-range of the CEJ?
    
    crest_candidates = bone_coords[bone_coords[:, 0] == min_y]
    bone_crest_point = calculate_centroid(crest_candidates)
    
    # Distance from CEJ to Bone Crest
    # Only vertical component matters? Or euclidean?
    # Clinical attachment loss is usually linear distance from CEJ to base of pocket,
    # Bone loss is CEJ to Bone Crest.
    # Usually measured vertically along the root axis.
    # Let's use Euclidean for general approximation, or just Y-diff.
    # PRD says "Distance(CEJ, Bone Crest)" which implies Euclidean.
    
    dist_pixels = euclidean(cej_point, bone_crest_point)
    
    # Clinical Check: If Bone Crest is ABOVE CEJ (y is smaller), then no bone loss.
    # In image coordinates, smaller y is higher up.
    # So if bone_crest_y < cej_y, it's "above" CEJ (towards crown) -> Healthy/No Loss.
    # Bone loss happens when crest moves DOWN (y increases).
    if bone_crest_point[0] < cej_point[0]:
        dist_pixels = 0.0
        
    dist_mm = dist_pixels * pixel_spacing
    
    return dist_mm, {
        "bone_crest": bone_crest_point.tolist(),
        "bone_loss_pixels": dist_pixels
    }

def calculate_bone_loss_percentage(root_length_mm, bone_loss_mm):
    if root_length_mm == 0:
        return 0.0
    loss_pct = (bone_loss_mm / root_length_mm) * 100
    return max(0.0, min(100.0, loss_pct))
