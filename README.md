# Automated Tooth Strength Analysis System

## Overview
This project is a research-grade, automated system for analyzing tooth strength using panoramic dental X-rays. It employs a hybrid multi-model pipeline:
1.  **Tooth Instance Segmentation:** Mask R-CNN isolates meaningful tooth regions.
2.  **Structural Segmentation:** U-Net++ segments root and alveolar bone within each tooth.
3.  **Deterministic Measurement:** Mathematical analysis of segmentation masks calculates Root Length and Bone Loss %.
4.  **Scoring:** A multi-factor scoring model combines bone support, structural integrity, and biological health.

## Setup

1.  **Prerequisites:**
    - Python 3.10+
    - CUDA-capable GPU (recommended for inference)

2.  **Installation:**
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate virtual environment (Windows)
    .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Running the API:**
    ```bash
    uvicorn api.main:app --reload
    ```

## Directory Structure
- `api/`: FastAPI application and routes.
- `models/`: PyTorch model definitions (Mask R-CNN, U-Net++).
- `services/`: Business logic (Measurement Engine, Scoring).
- `config/`: Configuration settings.
- `tests/`: Unit and integration tests.
