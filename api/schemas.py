from pydantic import BaseModel
from typing import List, Dict, Optional

class ToothScores(BaseModel):
    strength: float
    bone_support: float
    integrity: float
    infection: float

class ToothMeasurements(BaseModel):
    root_length_mm: float
    bone_loss_percent: float

class ToothMasks(BaseModel):
    full_tooth: str  # Base64 encoded RLE or similar
    root: str
    bone: str

class ToothResult(BaseModel):
    tooth_id: int
    scores: ToothScores
    measurements: ToothMeasurements
    diagnosis: Optional[str] = None
    masks: Optional[ToothMasks] = None

class AnalysisReport(BaseModel):
    scan_id: str
    teeth: List[ToothResult]
    global_metrics: Dict[str, float]
