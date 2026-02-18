from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import numpy as np
import cv2
import io
from PIL import Image

from models.mask_rcnn.inference import ToothInstanceSegmentor
from models.mask_rcnn.utils import extract_all_teeth
from models.unetpp.inference import ToothStructureSegmentor
from services.measurement import extract_root_length, detect_bone_level, calculate_bone_loss_percentage
from services.scoring import calculate_strength_score, get_classification
from services.preprocessing import preprocess_image
from api.schemas import AnalysisReport, ToothResult, ToothScores, ToothMeasurements, ToothMasks

import base64

router = APIRouter()

# Global instances (loaded in main.py lifespan or lazily here)
# Ideally passed via dependency injection or global state
instance_segmentor = None
structure_segmentor = None

def get_segmentors():
    global instance_segmentor, structure_segmentor
    if instance_segmentor is None:
        print("Loading Mask R-CNN...")
        instance_segmentor = ToothInstanceSegmentor(weights_path=None, pretrained=False) # No weights for now
    if structure_segmentor is None:
        print("Loading U-Net++...")
        structure_segmentor = ToothStructureSegmentor(weights_path=None, encoder_weights=None) 
    return instance_segmentor, structure_segmentor

def numpy_to_base64_mask(mask):
    """Encodes a numpy mask to base64 png."""
    success, encoded_image = cv2.imencode('.png', (mask * 50).astype(np.uint8)) # Scale for visibility
    if not success:
        return ""
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

@router.post("/analyze", response_model=AnalysisReport)
async def analyze_scan(scan: UploadFile = File(...)):
    if not scan.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await scan.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np_raw = np.array(image)
        
        # Stage 1: Preprocessing
        image_np = preprocess_image(image_np_raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    inst_seg, struct_seg = get_segmentors()

    # Stage 2: Tooth Instance Segmentation
    # Run in threadpool to avoid blocking event loop if not fully async
    detections = await run_in_threadpool(inst_seg.predict, image_np)

    # Extract Teeth
    teeth_crops = extract_all_teeth(image_np, detections)
    
    analyzed_teeth = []
    
    for tooth_data in teeth_crops:
        tooth_id = tooth_data['id']
        crop = tooth_data['image']
        
        # Stage 3: Structural Segmentation
        structure_mask = await run_in_threadpool(struct_seg.predict, crop)
        
        # Stage 4: Measurement
        # Assuming pixel spacing is 1.0 for now, or derived from metadata
        pixel_spacing = 0.1 # Placeholder: 0.1mm per pixel
        
        root_len_mm, root_details = extract_root_length(structure_mask, pixel_spacing)
        
        # CEJ fallback from measurement if not found
        cej = root_details.get("cej")
        
        bone_loss_mm, bone_details = detect_bone_level(structure_mask, cej, pixel_spacing)
        
        bone_loss_pct = calculate_bone_loss_percentage(root_len_mm, bone_loss_mm)
        
        # Stage 5: Scoring & Classification
        # integrity_score = 100 # Default valid
        # infection_score = 100 # Default healthy
        
        apex_point = root_details.get("apex") # [y, x]
        
        # Stage 5: Scoring & Classification
        # Score is now purely based on Bone Support (Safety First)
        strength_score, score_details = calculate_strength_score(bone_loss_pct)
        
        diagnosis = get_classification(bone_loss_pct)
        
        # Encode masks for response (optional, can be heavy)
        # root_mask = (structure_mask == 1).astype(np.uint8)
        # bone_mask = (structure_mask == 2).astype(np.uint8)
        
        analyzed_teeth.append(ToothResult(
            tooth_id=tooth_id,
            scores=ToothScores(
                strength=strength_score,
                bone_support=score_details["bone_support_score"]
            ),
            measurements=ToothMeasurements(
                root_length_mm=round(root_len_mm, 2),
                bone_loss_percent=round(bone_loss_pct, 2)
            ),
            diagnosis=diagnosis,
            masks=None # Skip sending heavy base64 for now
        ))

    global_strength = 0
    if analyzed_teeth:
        global_strength = sum([t.scores.strength for t in analyzed_teeth]) / len(analyzed_teeth)

    return AnalysisReport(
        scan_id="scan_001", # Generate real ID
        teeth=analyzed_teeth,
        global_metrics={"average_strength": round(global_strength, 2)}
    )
