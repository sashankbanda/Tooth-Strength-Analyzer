import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from models.mask_rcnn.inference import ToothInstanceSegmentor
from models.mask_rcnn.utils import extract_all_teeth
from models.unetpp.inference import ToothStructureSegmentor
from services.measurement import extract_root_length, detect_bone_level, calculate_bone_loss_percentage
from services.measurement import extract_root_length, detect_bone_level, calculate_bone_loss_percentage
from services.scoring import calculate_strength_score, calculate_integrity_score, calculate_infection_score, get_classification
from services.preprocessing import preprocess_image

def visualize_detections(image, detections, output_path):
    vis_img = image.copy()
    for det in detections:
        box = det['box']
        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(vis_img, f"Tooth: {det['score']:.2f}", (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(output_path), vis_img)

def visualize_mask_overlay(crop, mask, output_path):
    # Mask: 0=bg, 1=root, 2=bone, 3=cej
    overlay = crop.copy()
    
    # Colors: Root=Red, Bone=Blue, CEJ=Yellow
    colors = {
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (0, 255, 255)
    }
    
    alpha = 0.4
    for label, color in colors.items():
        binary_mask = (mask == label).astype(np.uint8)
        if np.sum(binary_mask) > 0:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1)
            
            # Blend
            colored_mask = np.zeros_like(crop)
            colored_mask[mask == label] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
            
    cv2.imwrite(str(output_path), overlay)
    
def visualize_analysis(crop, mask, root_details, bone_details, flow_details, output_path):
    vis = crop.copy()
    
    # Draw Apex (Red circle)
    if "apex" in root_details and root_details["apex"] is not None:
        y, x = root_details["apex"]
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(vis, "Apex", (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    # Draw CEJ (Yellow circle)
    if "cej" in root_details and root_details["cej"] is not None:
        y, x = root_details["cej"]
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.putText(vis, "CEJ", (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
    # Draw Bone Crest (Blue circle)
    if "bone_crest" in bone_details and bone_details["bone_crest"] is not None:
        y, x = bone_details["bone_crest"]
        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(vis, "Bone", (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Add text for scores
    scores = flow_details['scores']
    msg = [
        f"Strength: {scores['strength']}",
        f"Dia: {flow_details['diagnosis']}",
        f"Bone Loss: {flow_details['measurements']['bone_loss_percent']}%",
        f"Infection: {scores['infection']}",
        f"Integrity: {scores['integrity']}"
    ]
    
    y0, dy = 15, 15
    for i, line in enumerate(msg):
        y = y0 + i*dy
        cv2.putText(vis, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    cv2.imwrite(str(output_path), vis)

def run_debug_pipeline(image_path, output_dir):
    print(f"Running pipeline on {image_path}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
    img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    print("Stage 0: Preprocessing...")
    img_rgb = preprocess_image(img_rgb_raw)
    
    # Save comparison
    cv2.imwrite(str(output_dir / "01_original.jpg"), cv2.cvtColor(img_rgb_raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "01_preprocessed.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    # 1. Instance Segmentation
    print("Stage 1: Detecting teeth...")
    inst_seg = ToothInstanceSegmentor(pretrained=True) 
    # Use very low threshold since model is not trained on teeth yet
    detections = inst_seg.predict(img_rgb, confidence_threshold=0.0) 
    
    visualize_detections(img, detections, output_dir / "02_detections.jpg")
    
    # Extract Teeth
    teeth_crops = extract_all_teeth(img_rgb, detections)
    print(f"Detected {len(teeth_crops)} teeth.")
    
    # Fallback for debugging if model finds nothing (expected for untrained model)
    if not teeth_crops:
        print("No teeth detected (untrained model). Using dummy central crop for pipeline verification.")
        h, w = img_rgb.shape[:2]
        # Central square
        cx, cy = w // 2, h // 2
        size = 200
        x1, y1 = max(0, cx - size), max(0, cy - size)
        x2, y2 = min(w, cx + size), min(h, cy + size)
        
        dummy_crop = img_rgb[y1:y2, x1:x2].copy()
        teeth_crops.append({
            'id': 999,
            'image': dummy_crop,
            'box': np.array([x1, y1, x2, y2]),
            'mask': np.zeros((h, w), dtype=np.uint8),
            'score': 1.0
        })
    
    # 2. Structural Segmentation & Analysis
    print("Stage 2-4: Analyzing per tooth...")
    struct_seg = ToothStructureSegmentor(encoder_weights="imagenet")
    
    pixel_spacing = 0.1 # Mock value
    
    for tooth in teeth_crops:
        tid = tooth['id']
        crop = tooth['image'] # RGB numpy
        
        # Save Crop
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"03_tooth_{tid}_crop.jpg"), crop_bgr)
        
        # Structure Seg
        mask = struct_seg.predict(crop)
        
        # Save Mask Overlay
        visualize_mask_overlay(crop_bgr, mask, output_dir / f"04_tooth_{tid}_mask.jpg")
        
        # Measurements
        root_len_mm, root_details = extract_root_length(mask, pixel_spacing)
        cej = root_details.get("cej")
        bone_loss_mm, bone_details = detect_bone_level(mask, cej, pixel_spacing)
        bone_loss_pct = calculate_bone_loss_percentage(root_len_mm, bone_loss_mm)
        
        # Scoring
        apex_point = root_details.get("apex")
        integrity_score = calculate_integrity_score(mask)
        infection_score = calculate_infection_score(crop, mask, apex_point)
        
        strength_score, score_details = calculate_strength_score(
            bone_loss_pct, integrity_score, infection_score
        )
        
        diagnosis = get_classification(bone_loss_pct)
        
        flow_details = {
            "scores": {
                "strength": strength_score,
                "infection": infection_score,
                "integrity": integrity_score,
                "bone_support": score_details["bone_support_score"]
            },
            "measures": {
                "bone_loss_percent": round(bone_loss_pct, 2)
            },
            "diagnosis": diagnosis,
            "measurements": {
                "bone_loss_percent": round(bone_loss_pct, 2)
            }
        }
        
        # Visualize Analysis
        visualize_analysis(crop_bgr, mask, root_details, bone_details, flow_details, output_dir / f"05_tooth_{tid}_analysis.jpg")
        
        print(f"Tooth {tid}: Strength={strength_score}, Diagnosis={diagnosis}, Infection={infection_score}, Integrity={integrity_score}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="inputs/00011.jpg")
    parser.add_argument("--outdir", default="outputs/debug_00011")
    args = parser.parse_args()
    
    run_debug_pipeline(args.image, args.outdir)
