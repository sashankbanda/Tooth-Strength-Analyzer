import os
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from PIL import Image

# Add current directory to path to ensure imports work
import sys
sys.path.append(os.getcwd())

from models.mask_rcnn.inference import ToothInstanceSegmentor
from models.mask_rcnn.utils import extract_all_teeth
from models.unetpp.inference import ToothStructureSegmentor
from services.measurement import extract_root_length, detect_bone_level, calculate_bone_loss_percentage
from services.scoring import calculate_strength_score

def visualize_results(image, teeth_data, output_path):
    """
    Draws bounding boxes and scores on the image.
    """
    vis_img = image.copy()
    
    for tooth in teeth_data: 
        # Draw Bounding Box
        x1, y1, x2, y2 = tooth['box']
        color = (0, 255, 0) # Green
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        label = f"ID:{tooth['id']} Score:{tooth['strength_score']}"
        cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    cv2.imwrite(str(output_path), vis_img)

def main():
    input_dir = Path("inputs")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check if input dir exists
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist. Please create it and add images.")
        return

    # Load Models
    print("Loading models...")
    # Use pretrained=False for now as we don't have trained weights
    instance_segmentor = ToothInstanceSegmentor(pretrained=False)
    structure_segmentor = ToothStructureSegmentor(encoder_weights=None)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print("No images found in 'inputs/' directory.")
        return
        
    print(f"Found {len(image_files)} images. Processing...")
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        # Load Image
        # Open with PIL, convert to RGB, then to numpy (H, W, 3) for matching API flow
        pil_img = Image.open(img_path).convert("RGB")
        image_np = np.array(pil_img)
        # Convert RGB to BGR for OpenCV usage later if needed, but models expect RGB usually?
        # MaskRCNN (torchvision) expects 0-1 float or 0-255 byte tensor. Our inference handles it.
        # But OpenCV uses BGR.
        
        # 1. Instance Segmentation
        detections = instance_segmentor.predict(image_np)
        
        # 2. Extract Crops
        teeth_crops = extract_all_teeth(image_np, detections)
        
        results = []
        vis_data = []
        
        for tooth_data in teeth_crops:
            # 3. Structural Segmentation
            structure_mask = structure_segmentor.predict(tooth_data['image'])
            
            # 4. Measurement
            # Assuming 0.1mm pixel spacing
            pixel_spacing = 0.1 
            root_len, r_details = extract_root_length(structure_mask, pixel_spacing)
            cej = r_details.get('cej')
            bone_loss, b_details = detect_bone_level(structure_mask, cej, pixel_spacing)
            loss_pct = calculate_bone_loss_percentage(root_len, bone_loss)
            
            # 5. Scoring
            strength, s_details = calculate_strength_score(loss_pct, 100, 100)
            
            results.append({
                "tooth_id": tooth_data['id'],
                "strength_score": strength,
                "measurements": {
                    "root_length_mm": round(root_len, 2),
                    "bone_loss_pct": round(loss_pct, 2)
                },
                "details": s_details
            })
            
            vis_data.append({
                "id": tooth_data['id'],
                "box": tooth_data['box'],
                "strength_score": strength
            })
            
        # Save JSON Report
        report = {
            "filename": img_path.name,
            "total_teeth": len(results),
            "teeth_analysis": results
        }
        
        json_path = output_dir / f"{img_path.stem}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save Visualized Image
        # Convert RGB (from PIL/API flow) back to BGR for OpenCV
        vis_image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        vis_path = output_dir / f"{img_path.stem}_visualized.jpg"
        visualize_results(vis_image_bgr, vis_data, vis_path)
        
        print(f"Saved report to {json_path}")
        print(f"Saved visualization to {vis_path}")

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
