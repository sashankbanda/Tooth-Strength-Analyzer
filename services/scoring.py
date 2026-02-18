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


def calculate_strength_score(bone_loss_pct):
    """
    Calculates the final tooth strength score based purely on Bone Support.
    
    Formula:
    Strength Score = Bone Support Score
    Bone Support Score = 100 - Bone Loss %
    """
    bone_support_score = max(0, 100 - bone_loss_pct)
    
    return round(bone_support_score, 2), {
        "bone_support_score": bone_support_score
    }

