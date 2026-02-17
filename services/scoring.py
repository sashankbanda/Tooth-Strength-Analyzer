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
